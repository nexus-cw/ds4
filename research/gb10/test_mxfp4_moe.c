/*
 * Correctness test for the MXFP4 (GGUF tensor type 39) routed-expert MoE
 * CUDA dispatch added to routed_moe_launch() in ds4_cuda.cu (the
 * routed_moe_mxfp4_dispatch / mxfp4_matmul_row_f16gemm branch).
 *
 * Builds synthetic MXFP4 gate/up/down expert weights in host memory
 * (which on this GB10 unified-memory box is directly usable as a
 * ds4_gpu_set_model_map() model_map, exactly like the mmapped GGUF file
 * path), runs them through the public ds4_gpu_routed_moe_batch_tensor()
 * entry point for both a decode-shaped call (n_tokens=1) and a
 * prefill-shaped call (n_tokens=5), and compares the result against an
 * independent CPU reference: dequantize_row_mxfp4 (byte-for-byte port of
 * the one added to ds4.c, from llama.cpp ggml-quants.c @5f55650, MIT) +
 * a plain float dot product + the same SiLU(gate)*up*routing_weight
 * combine used by the CUDA path.
 *
 * This is deliberately independent of ds4.c/ds4_cuda.cu internals beyond
 * the public ds4_gpu.h surface, so it exercises exactly what a real
 * inference call would exercise (weight resolution, dequant kernel,
 * cuBLAS GEMM, SiLU combine kernel, and the shared moe_sum kernel),
 * without needing a full 150GB GGUF on disk.
 *
 * Build (from ds4/ repo root, after `make cuda-spark` has produced
 * ds4_cuda.o):
 *   /usr/local/cuda/bin/nvcc -O2 -o research/gb10/test_mxfp4_moe \
 *       research/gb10/test_mxfp4_moe.c ds4_cuda.o -lcudart -lcublas -lm \
 *       -L/usr/local/cuda/targets/sbsa-linux/lib -L/usr/local/cuda/lib64
 * Run:
 *   ./research/gb10/test_mxfp4_moe
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "../../ds4_gpu.h"

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

static void fill_random_mxfp4_row(block_mxfp4 *row, uint32_t blocks) {
    for (uint32_t b = 0; b < blocks; b++) {
        /* Keep E8M0 scale bytes in a modest range so the reference
         * math stays in normal float range; nibbles fully random. */
        row[b].e = (uint8_t)(120 + (rng_next() % 16));
        for (uint32_t j = 0; j < QK_MXFP4 / 2; j++) {
            row[b].qs[j] = (uint8_t)(rng_next() & 0xFFu);
        }
    }
}

static float dot_f32(const float *a, const float *b, uint32_t n) {
    float acc = 0.0f;
    for (uint32_t i = 0; i < n; i++) acc += a[i] * b[i];
    return acc;
}

/* Reference row-major dequant matvec: weight is [out_dim, in_dim] packed
 * as out_dim rows of (in_dim/32) blocks, x is [in_dim]. */
static void ref_matvec_mxfp4(
        float *out, const uint8_t *w_bytes, uint32_t in_dim, uint32_t out_dim,
        const float *x) {
    const uint32_t blocks = in_dim / QK_MXFP4;
    const uint64_t row_bytes = (uint64_t)blocks * sizeof(block_mxfp4);
    float *row_f = (float *)malloc((size_t)in_dim * sizeof(float));
    for (uint32_t r = 0; r < out_dim; r++) {
        const block_mxfp4 *row = (const block_mxfp4 *)(w_bytes + (uint64_t)r * row_bytes);
        dequantize_row_mxfp4(row, row_f, in_dim);
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

static int run_case(uint32_t n_tokens, uint32_t n_expert, uint32_t n_total_expert,
                     uint32_t in_dim, uint32_t mid_dim, uint32_t out_dim) {
    fprintf(stderr, "-- case n_tokens=%u n_expert=%u n_total_expert=%u in=%u mid=%u out=%u --\n",
            n_tokens, n_expert, n_total_expert, in_dim, mid_dim, out_dim);

    const uint32_t gate_blocks = in_dim / QK_MXFP4;
    const uint32_t down_blocks = mid_dim / QK_MXFP4;
    const uint64_t gate_row_bytes = (uint64_t)gate_blocks * sizeof(block_mxfp4);
    const uint64_t down_row_bytes = (uint64_t)down_blocks * sizeof(block_mxfp4);
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
        fill_random_mxfp4_row((block_mxfp4 *)(model + gate_offset + (uint64_t)e * gate_expert_bytes), gate_blocks * mid_dim);
        fill_random_mxfp4_row((block_mxfp4 *)(model + up_offset + (uint64_t)e * gate_expert_bytes), gate_blocks * mid_dim);
        fill_random_mxfp4_row((block_mxfp4 *)(model + down_offset + (uint64_t)e * down_expert_bytes), down_blocks * out_dim);
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
            ref_matvec_mxfp4(gate_row, gw, in_dim, mid_dim, x + (uint64_t)t * in_dim);
            ref_matvec_mxfp4(up_row, uw, in_dim, mid_dim, x + (uint64_t)t * in_dim);
            float w = weights[t * n_expert + s];
            for (uint32_t i = 0; i < mid_dim; i++) {
                float g = gate_row[i];
                mid_row[i] = (g / (1.0f + expf(-g))) * up_row[i] * w;
            }
            ref_matvec_mxfp4(down_row, dw, mid_dim, out_dim, mid_row);
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
            39u /* MXFP4 */, 39u /* MXFP4 */,
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

    /* P3a: decode-shaped (n_tokens==1) MXFP4/MXFP4 calls now take the new
     * fused decode kernel path (routed_moe_fused_fp4q3k_decode_dispatch)
     * by default. Re-run the identical call with
     * DS4_CUDA_DISABLE_FUSED_FP4_DECODE=1 to force the pre-existing
     * generic dequant+GEMM path (routed_moe_dequant_gemm_dispatch) on the
     * exact same inputs, and require the two GPU paths to agree with each
     * other (not just each independently against the CPU reference
     * above) -- this is the direct fused-vs-generic cross-check the P3a
     * ticket asked for. */
    if (n_tokens == 1u) {
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
                39u /* MXFP4 */, 39u /* MXFP4 */,
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
     * unmodified by the MXFP4 change) hard-requires
     * expert_in_dim/expert_mid_dim % CUDA_QK_K == 0 (CUDA_QK_K == 256)
     * for every routed-expert type, not just the legacy Q4_K/IQ2_XXS
     * ones -- MXFP4's own block size is 32, but it inherits this coarser
     * gate. Real model hidden dims (DeepSeek-class, 2048/7168/...) are
     * always multiples of 256 in practice, so dims below are chosen to
     * respect it too. */

    /* Decode shape: n_tokens=1, n_expert=6 (mirrors typical DeepSeek top-k). */
    int rc1 = run_case(1, 6, 16, 256, 256, 256);
    if (rc1 == 77) {
        fprintf(stderr, "SKIP: no CUDA device available in this environment\n");
        return 0;
    }
    /* Prefill shape: n_tokens=5, n_expert=3, different in/mid/out dims. */
    run_case(5, 3, 16, 512, 256, 256);

    if (g_fail) {
        fprintf(stderr, "MXFP4 MoE test: FAILED\n");
        return 1;
    }
    fprintf(stderr, "MXFP4 MoE test: all cases passed\n");
    return 0;
}
