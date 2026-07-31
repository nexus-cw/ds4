/*
 * Correctness test for the generalized routed-MoE dequant+GEMM fallback
 * dispatch in routed_moe_launch() / routed_moe_dequant_gemm_dispatch()
 * (ds4_cuda.cu), covering MIXED per-tensor quant types within a single
 * MoE layer -- the case the real target artifact
 * (DeepSeek-V4-Flash-MXFP4_MOE.gguf) actually exercises: gate/up experts
 * in one quant type (MXFP4, GGUF type 39) and down experts in a
 * DIFFERENT type (Q3_K, GGUF type 11, or Q5_K, GGUF type 13). ds4's
 * loader enforces gate_type == up_type at load time (see ds4.c's
 * "routed gate/up experts use different quant types" checks), so gate
 * and up always share a type; down is independently dispatched, which
 * is exactly what this test exercises.
 *
 * Structurally this is test_mxfp4_moe.c generalized: synthetic
 * gate/up (MXFP4) and down (Q3_K or Q5_K) expert weights in host memory,
 * run through the public ds4_gpu_routed_moe_batch_tensor() entry point,
 * checked against an independent CPU reference that dequantizes each
 * type with a standalone port of ds4.c's dequantize_row_mxfp4 /
 * dequantize_row_q3_K / dequantize_row_q5_K (themselves ported from
 * llama.cpp ggml-quants.c @5f55650, MIT) plus a plain float dot product
 * and the same SiLU(gate)*up*routing_weight combine used by the CUDA
 * path.
 *
 * Build (from ds4/ repo root, after `make cuda-spark` has produced
 * ds4_cuda.o):
 *   /usr/local/cuda/bin/nvcc -O2 -o research/gb10/test_mixed_moe \
 *       research/gb10/test_mixed_moe.c ds4_cuda.o -lcudart -lcublas -lm \
 *       -L/usr/local/cuda/targets/sbsa-linux/lib -L/usr/local/cuda/lib64
 * Run:
 *   ./research/gb10/test_mixed_moe
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "../../ds4_gpu.h"

#define QK_K 256

/* ---- MXFP4 (GGUF type 39) ---- */
#define QK_MXFP4 32
typedef struct {
    uint8_t e;
    uint8_t qs[QK_MXFP4 / 2];
} block_mxfp4;

static const int8_t kvalues_mxfp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

static float mxfp4_e8m0_to_fp32_half(uint8_t x) {
    uint32_t bits;
    if (x < 2) {
        bits = 0x00200000u << x;
    } else {
        bits = (uint32_t)(x - 1) << 23;
    }
    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

static void dequantize_row_mxfp4(const block_mxfp4 *x, float *y, int64_t k) {
    const int64_t nb = k / QK_MXFP4;
    for (int64_t i = 0; i < nb; i++) {
        const float d = mxfp4_e8m0_to_fp32_half(x[i].e);
        for (int64_t j = 0; j < QK_MXFP4 / 2; ++j) {
            const int8_t x0 = kvalues_mxfp4[x[i].qs[j] & 0x0F];
            const int8_t x1 = kvalues_mxfp4[x[i].qs[j] >> 4];
            y[i * QK_MXFP4 + j] = x0 * d;
            y[i * QK_MXFP4 + j + QK_MXFP4 / 2] = x1 * d;
        }
    }
}

/* ---- Q3_K (GGUF type 11) ---- */
typedef struct {
    uint8_t  hmask[QK_K / 8];
    uint8_t  qs[QK_K / 4];
    uint8_t  scales[12];
    uint16_t d;
} block_q3_K;

/* ---- Q5_K (GGUF type 13) ---- */
typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qh[QK_K / 8];
    uint8_t  qs[QK_K / 2];
} block_q5_K;

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 1u;
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign << 31;
        } else {
            /* subnormal */
            int e = -1;
            do { mant <<= 1; e++; } while ((mant & 0x400u) == 0);
            mant &= 0x3FFu;
            bits = (sign << 31) | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        bits = (sign << 31) | 0x7F800000u | (mant << 13);
    } else {
        bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static void q4_k_get_scale_min(int j, const uint8_t *q, uint8_t *sc, uint8_t *m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m  = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m  = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}

static void dequantize_row_q3_K(const block_q3_K *x, float *y, int64_t k) {
    const int64_t nb = k / QK_K;
    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    uint32_t aux[4];
    const int8_t *scales = (const int8_t *)aux;

    for (int64_t i = 0; i < nb; i++) {
        const float d_all = f16_to_f32(x[i].d);
        const uint8_t *q = x[i].qs;
        const uint8_t *hm = x[i].hmask;
        uint8_t m = 1;

        memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        float dl;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
                }
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                }
                shift += 2;
                m = (uint8_t)(m << 1);
            }
            q += 32;
        }
    }
}

static void dequantize_row_q5_K(const block_q5_K *x, float *y, int64_t k) {
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t *ql = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const float d = f16_to_f32(x[i].d);
        const float min = f16_to_f32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            q4_k_get_scale_min(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * (float)sc, m1 = min * (float)m;
            q4_k_get_scale_min(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * (float)sc, m2 = min * (float)m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 = (uint8_t)(u1 << 2); u2 = (uint8_t)(u2 << 2);
        }
    }
}

/* ---- shared test scaffolding ---- */

#define TYPE_MXFP4 39u
#define TYPE_Q3_K  11u
#define TYPE_Q5_K  13u

static uint32_t block_elems_of(uint32_t type) {
    switch (type) {
    case TYPE_MXFP4: return QK_MXFP4;
    case TYPE_Q3_K:  return QK_K;
    case TYPE_Q5_K:  return QK_K;
    default: abort();
    }
}

static uint64_t block_bytes_of(uint32_t type) {
    switch (type) {
    case TYPE_MXFP4: return sizeof(block_mxfp4);
    case TYPE_Q3_K:  return sizeof(block_q3_K);
    case TYPE_Q5_K:  return sizeof(block_q5_K);
    default: abort();
    }
}

static uint32_t g_rng = 0x1234abcdu;
static uint32_t rng_next(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}

static float rng_unit(void) {
    return (float)(rng_next() % 20001) / 10000.0f - 1.0f; /* [-1, 1] */
}

/* Fill one row (in blocks) of a given fallback type with random-but-sane
 * bytes: MXFP4 needs a modest E8M0 scale range to keep magnitudes sane;
 * Q3_K/Q5_K's f16 scale fields likewise need to avoid inf/nan bit
 * patterns, so scale bytes are drawn from a small positive-exponent
 * range and everything else is fully random. */
static void fill_random_row(uint32_t type, uint8_t *row_bytes, uint32_t blocks) {
    const uint64_t bb = block_bytes_of(type);
    for (uint32_t b = 0; b < blocks; b++) {
        uint8_t *blk = row_bytes + (uint64_t)b * bb;
        if (type == TYPE_MXFP4) {
            block_mxfp4 *m = (block_mxfp4 *)blk;
            m->e = (uint8_t)(120 + (rng_next() % 16));
            for (uint32_t j = 0; j < QK_MXFP4 / 2; j++) m->qs[j] = (uint8_t)(rng_next() & 0xFFu);
        } else if (type == TYPE_Q3_K) {
            block_q3_K *q = (block_q3_K *)blk;
            for (uint32_t j = 0; j < QK_K / 8; j++) q->hmask[j] = (uint8_t)(rng_next() & 0xFFu);
            for (uint32_t j = 0; j < QK_K / 4; j++) q->qs[j] = (uint8_t)(rng_next() & 0xFFu);
            for (uint32_t j = 0; j < 12; j++) q->scales[j] = (uint8_t)(rng_next() & 0xFFu);
            /* f16 "small positive" scale: exponent bits in [14,17], sign 0. */
            uint16_t exp = (uint16_t)(14 + (rng_next() % 4));
            uint16_t mant = (uint16_t)(rng_next() & 0x3FFu);
            q->d = (uint16_t)((exp << 10) | mant);
        } else { /* Q5_K */
            block_q5_K *q = (block_q5_K *)blk;
            uint16_t exp_d = (uint16_t)(14 + (rng_next() % 4));
            uint16_t mant_d = (uint16_t)(rng_next() & 0x3FFu);
            q->d = (uint16_t)((exp_d << 10) | mant_d);
            uint16_t exp_m = (uint16_t)(10 + (rng_next() % 4));
            uint16_t mant_m = (uint16_t)(rng_next() & 0x3FFu);
            q->dmin = (uint16_t)((exp_m << 10) | mant_m);
            for (uint32_t j = 0; j < 12; j++) q->scales[j] = (uint8_t)(rng_next() & 0xFFu);
            for (uint32_t j = 0; j < QK_K / 8; j++) q->qh[j] = (uint8_t)(rng_next() & 0xFFu);
            for (uint32_t j = 0; j < QK_K / 2; j++) q->qs[j] = (uint8_t)(rng_next() & 0xFFu);
        }
    }
}

static float dot_f32(const float *a, const float *b, uint32_t n) {
    float acc = 0.0f;
    for (uint32_t i = 0; i < n; i++) acc += a[i] * b[i];
    return acc;
}

/* Reference row-major dequant matvec for any of the three fallback
 * types: weight is [out_dim, in_dim] packed as out_dim rows of
 * (in_dim/block_elems) blocks, x is [in_dim]. */
static void ref_matvec(
        uint32_t type, float *out, const uint8_t *w_bytes, uint32_t in_dim,
        uint32_t out_dim, const float *x) {
    const uint32_t blocks = in_dim / block_elems_of(type);
    const uint64_t row_bytes = (uint64_t)blocks * block_bytes_of(type);
    float *row_f = (float *)malloc((size_t)in_dim * sizeof(float));
    for (uint32_t r = 0; r < out_dim; r++) {
        const uint8_t *row = w_bytes + (uint64_t)r * row_bytes;
        switch (type) {
        case TYPE_MXFP4: dequantize_row_mxfp4((const block_mxfp4 *)row, row_f, in_dim); break;
        case TYPE_Q3_K:  dequantize_row_q3_K((const block_q3_K *)row, row_f, in_dim); break;
        case TYPE_Q5_K:  dequantize_row_q5_K((const block_q5_K *)row, row_f, in_dim); break;
        default: abort();
        }
        out[r] = dot_f32(row_f, x, in_dim);
    }
    free(row_f);
}

static int g_fail = 0;

static void check_close(const char *what, float got, float want, float tol) {
    float diff = fabsf(got - want);
    float scale = fmaxf(1.0f, fabsf(want));
    if (diff > tol * scale) {
        fprintf(stderr, "FAIL %s: got %.6f want %.6f (diff %.6f)\n", what, got, want, diff);
        g_fail = 1;
    }
}

/* gate_type covers both gate and up (ds4's load-time invariant); down_type
 * may be a different type from the fallback table, exercising the mixed
 * per-tensor-type dispatch. */
static int run_case(
        uint32_t gate_type, uint32_t down_type,
        uint32_t n_tokens, uint32_t n_expert, uint32_t n_total_expert,
        uint32_t in_dim, uint32_t mid_dim, uint32_t out_dim) {
    fprintf(stderr,
            "-- case gate/up_type=%u down_type=%u n_tokens=%u n_expert=%u "
            "n_total_expert=%u in=%u mid=%u out=%u --\n",
            gate_type, down_type, n_tokens, n_expert, n_total_expert, in_dim, mid_dim, out_dim);

    const uint32_t gate_blocks = in_dim / block_elems_of(gate_type);
    const uint32_t down_blocks = mid_dim / block_elems_of(down_type);
    const uint64_t gate_row_bytes = (uint64_t)gate_blocks * block_bytes_of(gate_type);
    const uint64_t down_row_bytes = (uint64_t)down_blocks * block_bytes_of(down_type);
    const uint64_t gate_expert_bytes = gate_row_bytes * mid_dim;
    const uint64_t down_expert_bytes = down_row_bytes * out_dim;

    const uint64_t gate_bytes = gate_expert_bytes * n_total_expert;
    const uint64_t up_bytes = gate_expert_bytes * n_total_expert;
    const uint64_t down_bytes = down_expert_bytes * n_total_expert;
    const uint64_t model_size = gate_bytes + up_bytes + down_bytes;

    uint8_t *model = (uint8_t *)malloc((size_t)model_size);
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = gate_bytes;
    const uint64_t down_offset = gate_bytes + up_bytes;

    for (uint32_t e = 0; e < n_total_expert; e++) {
        fill_random_row(gate_type, model + gate_offset + (uint64_t)e * gate_expert_bytes, gate_blocks * mid_dim);
        fill_random_row(gate_type, model + up_offset + (uint64_t)e * gate_expert_bytes, gate_blocks * mid_dim);
        fill_random_row(down_type, model + down_offset + (uint64_t)e * down_expert_bytes, down_blocks * out_dim);
    }

    float *x = (float *)malloc((size_t)n_tokens * in_dim * sizeof(float));
    for (uint32_t i = 0; i < n_tokens * in_dim; i++) x[i] = rng_unit();

    int32_t *selected = (int32_t *)malloc((size_t)n_tokens * n_expert * sizeof(int32_t));
    float *weights = (float *)malloc((size_t)n_tokens * n_expert * sizeof(float));
    for (uint32_t t = 0; t < n_tokens; t++) {
        float wsum = 0.0f;
        for (uint32_t s = 0; s < n_expert; s++) {
            selected[t * n_expert + s] = (int32_t)(rng_next() % n_total_expert);
            float w = 0.1f + (float)(rng_next() % 100) / 100.0f;
            weights[t * n_expert + s] = w;
            wsum += w;
        }
        for (uint32_t s = 0; s < n_expert; s++) weights[t * n_expert + s] /= wsum;
    }

    /* CPU reference */
    float *ref_out = (float *)calloc((size_t)n_tokens * out_dim, sizeof(float));
    float *gate_row = (float *)malloc(mid_dim * sizeof(float));
    float *up_row = (float *)malloc(mid_dim * sizeof(float));
    float *mid_row = (float *)malloc(mid_dim * sizeof(float));
    float *down_row = (float *)malloc(out_dim * sizeof(float));
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t s = 0; s < n_expert; s++) {
            int32_t e = selected[t * n_expert + s];
            const uint8_t *gw = model + gate_offset + (uint64_t)e * gate_expert_bytes;
            const uint8_t *uw = model + up_offset + (uint64_t)e * gate_expert_bytes;
            const uint8_t *dw = model + down_offset + (uint64_t)e * down_expert_bytes;
            ref_matvec(gate_type, gate_row, gw, in_dim, mid_dim, x + (uint64_t)t * in_dim);
            ref_matvec(gate_type, up_row, uw, in_dim, mid_dim, x + (uint64_t)t * in_dim);
            float w = weights[t * n_expert + s];
            for (uint32_t i = 0; i < mid_dim; i++) {
                float g = gate_row[i];
                mid_row[i] = (g / (1.0f + expf(-g))) * up_row[i] * w;
            }
            ref_matvec(down_type, down_row, dw, mid_dim, out_dim, mid_row);
            for (uint32_t r = 0; r < out_dim; r++) ref_out[t * out_dim + r] += down_row[r];
        }
    }

    /* GPU path */
    if (!ds4_gpu_init()) {
        fprintf(stderr, "ds4_gpu_init failed (no CUDA device?) -- skipping\n");
        free(model); free(x); free(selected); free(weights); free(ref_out);
        free(gate_row); free(up_row); free(mid_row); free(down_row);
        return 77; /* signal "skip" to the caller */
    }
    if (!ds4_gpu_set_model_map(model, model_size)) {
        fprintf(stderr, "FAIL: ds4_gpu_set_model_map\n");
        g_fail = 1;
        return 1;
    }

    ds4_gpu_tensor *t_out = ds4_gpu_tensor_alloc((uint64_t)n_tokens * out_dim * sizeof(float));
    ds4_gpu_tensor *t_gate = ds4_gpu_tensor_alloc((uint64_t)n_tokens * n_expert * mid_dim * sizeof(float));
    ds4_gpu_tensor *t_up = ds4_gpu_tensor_alloc((uint64_t)n_tokens * n_expert * mid_dim * sizeof(float));
    ds4_gpu_tensor *t_mid = ds4_gpu_tensor_alloc((uint64_t)n_tokens * n_expert * mid_dim * sizeof(float));
    ds4_gpu_tensor *t_down = ds4_gpu_tensor_alloc((uint64_t)n_tokens * n_expert * out_dim * sizeof(float));
    ds4_gpu_tensor *t_x = ds4_gpu_tensor_alloc((uint64_t)n_tokens * in_dim * sizeof(float));
    ds4_gpu_tensor *t_selected = ds4_gpu_tensor_alloc((uint64_t)n_tokens * n_expert * sizeof(int32_t));
    ds4_gpu_tensor *t_weights = ds4_gpu_tensor_alloc((uint64_t)n_tokens * n_expert * sizeof(float));

    if (!t_out || !t_gate || !t_up || !t_mid || !t_down || !t_x || !t_selected || !t_weights) {
        fprintf(stderr, "FAIL: tensor alloc\n");
        g_fail = 1;
        return 1;
    }

    ds4_gpu_tensor_write(t_x, 0, x, (uint64_t)n_tokens * in_dim * sizeof(float));
    ds4_gpu_tensor_write(t_selected, 0, selected, (uint64_t)n_tokens * n_expert * sizeof(int32_t));
    ds4_gpu_tensor_write(t_weights, 0, weights, (uint64_t)n_tokens * n_expert * sizeof(float));

    bool mid_is_f16 = false;
    int ok = ds4_gpu_routed_moe_batch_tensor(
            t_out, t_gate, t_up, t_mid, t_down,
            model, model_size,
            gate_offset, up_offset, down_offset,
            gate_type, down_type,
            gate_expert_bytes, gate_row_bytes,
            down_expert_bytes, down_row_bytes,
            in_dim, mid_dim, out_dim,
            t_selected, t_weights,
            n_total_expert, n_expert,
            0.0f /* clamp */, t_x,
            0 /* layer_index */, n_tokens,
            &mid_is_f16, false);

    if (!ok) {
        fprintf(stderr, "FAIL: ds4_gpu_routed_moe_batch_tensor returned 0\n");
        g_fail = 1;
        return 1;
    }

    float *gpu_out = (float *)malloc((size_t)n_tokens * out_dim * sizeof(float));
    ds4_gpu_tensor_read(t_out, 0, gpu_out, (uint64_t)n_tokens * out_dim * sizeof(float));

    for (uint32_t i = 0; i < n_tokens * out_dim; i++) {
        char label[64];
        snprintf(label, sizeof(label), "out[%u]", i);
        /* f16 GEMM intermediate -> looser tolerance than a pure f32 path. */
        check_close(label, gpu_out[i], ref_out[i], 0.02f);
    }

    /* P3a: when n_tokens==1 and both gate/up and down types are in the
     * fused decode kernels' supported set (MXFP4=39, Q3_K=11 -- NOT
     * Q5_K=13, which stays on the generic path even at decode per the
     * ticket's own scope), the call above already took the new fused
     * decode dispatch by default. Re-run the identical inputs with
     * DS4_CUDA_DISABLE_FUSED_FP4_DECODE=1 to force the pre-existing
     * generic dequant+GEMM path, and require the two GPU paths to agree
     * -- the fused-vs-generic cross-check the P3a ticket asked for, now
     * exercised across the mixed MXFP4/Q3_K and Q3_K/Q3_K combinations
     * this file covers (not just the MXFP4/MXFP4 case test_mxfp4_moe.c
     * checks). */
    const int fused_supported = (gate_type == TYPE_MXFP4 || gate_type == TYPE_Q3_K) &&
                                 (down_type == TYPE_MXFP4 || down_type == TYPE_Q3_K);
    if (n_tokens == 1u && fused_supported) {
        setenv("DS4_CUDA_DISABLE_FUSED_FP4_DECODE", "1", 1);
        ds4_gpu_tensor *t_out2 = ds4_gpu_tensor_alloc((uint64_t)n_tokens * out_dim * sizeof(float));
        ds4_gpu_tensor *t_gate2 = ds4_gpu_tensor_alloc((uint64_t)n_tokens * n_expert * mid_dim * sizeof(float));
        ds4_gpu_tensor *t_up2 = ds4_gpu_tensor_alloc((uint64_t)n_tokens * n_expert * mid_dim * sizeof(float));
        ds4_gpu_tensor *t_mid2 = ds4_gpu_tensor_alloc((uint64_t)n_tokens * n_expert * mid_dim * sizeof(float));
        ds4_gpu_tensor *t_down2 = ds4_gpu_tensor_alloc((uint64_t)n_tokens * n_expert * out_dim * sizeof(float));
        bool mid_is_f16_2 = false;
        int ok2 = ds4_gpu_routed_moe_batch_tensor(
                t_out2, t_gate2, t_up2, t_mid2, t_down2,
                model, model_size,
                gate_offset, up_offset, down_offset,
                gate_type, down_type,
                gate_expert_bytes, gate_row_bytes,
                down_expert_bytes, down_row_bytes,
                in_dim, mid_dim, out_dim,
                t_selected, t_weights,
                n_total_expert, n_expert,
                0.0f /* clamp */, t_x,
                0 /* layer_index */, n_tokens,
                &mid_is_f16_2, false);
        unsetenv("DS4_CUDA_DISABLE_FUSED_FP4_DECODE");
        if (!ok2) {
            fprintf(stderr, "FAIL: generic-path (fused-disabled) dispatch returned 0\n");
            g_fail = 1;
        } else {
            float *gpu_out_generic = (float *)malloc((size_t)n_tokens * out_dim * sizeof(float));
            ds4_gpu_tensor_read(t_out2, 0, gpu_out_generic, (uint64_t)n_tokens * out_dim * sizeof(float));
            for (uint32_t i = 0; i < n_tokens * out_dim; i++) {
                char label[80];
                snprintf(label, sizeof(label), "fused-vs-generic out[%u]", i);
                check_close(label, gpu_out[i], gpu_out_generic[i], 0.02f);
            }
            free(gpu_out_generic);
        }
        ds4_gpu_tensor_free(t_out2);
        ds4_gpu_tensor_free(t_gate2);
        ds4_gpu_tensor_free(t_up2);
        ds4_gpu_tensor_free(t_mid2);
        ds4_gpu_tensor_free(t_down2);
    }

    ds4_gpu_tensor_free(t_out);
    ds4_gpu_tensor_free(t_gate);
    ds4_gpu_tensor_free(t_up);
    ds4_gpu_tensor_free(t_mid);
    ds4_gpu_tensor_free(t_down);
    ds4_gpu_tensor_free(t_x);
    ds4_gpu_tensor_free(t_selected);
    ds4_gpu_tensor_free(t_weights);
    free(model); free(x); free(selected); free(weights); free(ref_out); free(gpu_out);
    free(gate_row); free(up_row); free(mid_row); free(down_row);
    return 0;
}

int main(void) {
    /* routed_moe_launch's shared top-of-function validation (pre-existing,
     * unmodified by this port) hard-requires expert_in_dim/expert_mid_dim
     * % CUDA_QK_K == 0 (CUDA_QK_K == 256); all three fallback types'
     * block sizes (32 for MXFP4, 256 for Q3_K/Q5_K) divide dims chosen
     * below, so both constraints are satisfied simultaneously. */

    /* Mixed case 1: gate/up = MXFP4, down = Q3_K. Decode shape. */
    int rc1 = run_case(TYPE_MXFP4, TYPE_Q3_K, 1, 6, 16, 256, 256, 256);
    if (rc1 == 77) {
        fprintf(stderr, "SKIP: no CUDA device available in this environment\n");
        return 0;
    }
    /* Mixed case 2: gate/up = MXFP4, down = Q3_K. Prefill shape, different dims. */
    run_case(TYPE_MXFP4, TYPE_Q3_K, 5, 3, 16, 512, 256, 256);

    /* Mixed case 3: gate/up = MXFP4, down = Q5_K (the real artifact's
     * rarer combo -- 1 Q5_K down tensor alongside 98 MXFP4). Decode shape. */
    run_case(TYPE_MXFP4, TYPE_Q5_K, 1, 6, 16, 256, 256, 256);
    /* Mixed case 4: same, prefill shape. */
    run_case(TYPE_MXFP4, TYPE_Q5_K, 5, 3, 16, 512, 256, 256);

    /* Non-mixed sanity: both fallback types on their own, matching gate
     * and down, to make sure the generalization didn't regress the
     * plain single-type case either. */
    run_case(TYPE_Q3_K, TYPE_Q3_K, 1, 6, 16, 256, 256, 256);
    run_case(TYPE_Q5_K, TYPE_Q5_K, 1, 6, 16, 256, 256, 256);

    if (g_fail) {
        fprintf(stderr, "mixed routed-MoE test: FAILED\n");
        return 1;
    }
    fprintf(stderr, "mixed routed-MoE test: all cases passed\n");
    return 0;
}
