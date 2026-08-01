/*
 * STATUS (2026-08-01): CURRENTLY FAILING on every case. Kept in-repo as
 * the reproducer for the known dsv4_mxfp4_mma_gemm_kernel numeric bug --
 * see FP4_PORT_SCOPE.md's P3c-1 take 2 section and test_mxfp4_mma_diag.c
 * for the isolating diagnostic. The MMA prefill path itself is gated
 * off by default (DS4_CUDA_ENABLE_MMA_PREFILL_EXPERIMENTAL opt-in only)
 * specifically because of this.
 * Standalone isolation test for dsv4_mxfp4_mma_gemm_kernel (P3c-1 take 2),
 * bypassing the routed-MoE grouping machinery entirely via the TEMPORARY
 * ds4_debug_mxfp4_mma_gemm() entry point in ds4_cuda.cu.
 *
 * CPU reference quantizes the activation to E2M1+E8M0 exactly the way the
 * tensor core consumes it (round-to-nearest E2M1, per-32-element E8M0
 * scale) before dotting against the dequantized (exact) MXFP4 weight row,
 * so this checks bit-for-bit agreement with the *quantized* math the
 * hardware actually does, not just "close to the full-precision answer".
 *
 * Build:
 *   /usr/local/cuda/bin/nvcc -O2 -o /tmp/test_mxfp4_mma_gemm \
 *       test_mxfp4_mma_gemm.c ds4_cuda.o -lcudart -lcublas -lm \
 *       -L/usr/local/cuda/targets/sbsa-linux/lib -L/usr/local/cuda/lib64
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

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
    if (x < 2) bits = 0x00200000u << x;
    else bits = (uint32_t)(x - 1) << 23;
    float result; memcpy(&result, &bits, sizeof(float)); return result;
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

/* Raw (non-halved) e8m0 -> fp32, matching hardware ue8m0 semantics. */
static float e8m0_to_fp32_raw(uint8_t x) {
    uint32_t bits = (x == 0) ? 0x00400000u : ((uint32_t)x << 23);
    float r; memcpy(&r, &bits, sizeof(float)); return r;
}

static uint8_t compute_e8m0_scale(float amax) {
    if (!(amax > 0.0f)) return 0;
    const float e = log2f(amax);
    const int e_int = (int)lrintf(e);
    int shared_exp = e_int - 2;
    int biased = shared_exp + 127;
    if (biased < 0) biased = 0;
    if (biased > 254) biased = 254;
    return (uint8_t)biased;
}

static float quantize_dequantize_e2m1(float x, float inv_scale, float scale) {
    static const float pos_lut[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    const float sign = (x < 0.0f) ? -1.0f : 1.0f;
    const float ax = fabsf(x) * inv_scale;
    int best_i = 0; float best_err = fabsf(ax - pos_lut[0]);
    for (int i = 1; i < 8; i++) {
        float err = fabsf(ax - pos_lut[i]);
        if (err < best_err) { best_err = err; best_i = i; }
    }
    return sign * pos_lut[best_i] * scale;
}

/* Reference: dequant weight exactly, quantize-then-dequant activation
 * (per 32-elem sub-block E8M0), dot product. */
static void ref_row(float *out_row, const uint8_t *w_bytes, uint32_t in_dim,
                     uint32_t out_dim, const float *x_group, uint32_t group_size,
                     uint32_t out_dim_stride) {
    const uint32_t blocks = in_dim / QK_MXFP4;
    const uint64_t row_bytes = (uint64_t)blocks * sizeof(block_mxfp4);
    float *wf = (float *)malloc((size_t)in_dim * sizeof(float));
    float *xq = (float *)malloc((size_t)in_dim * sizeof(float));

    for (uint32_t t = 0; t < group_size; t++) {
        const float *x = x_group + (uint64_t)t * in_dim;
        for (uint32_t sb = 0; sb < in_dim / 32u; sb++) {
            float amax = 0.0f;
            for (uint32_t i = 0; i < 32u; i++) {
                float a = fabsf(x[sb * 32u + i]);
                if (a > amax) amax = a;
            }
            uint8_t e = compute_e8m0_scale(amax);
            float scale = e8m0_to_fp32_raw(e);
            float inv_scale = (amax == 0.0f) ? 0.0f : 1.0f / scale;
            for (uint32_t i = 0; i < 32u; i++) {
                xq[sb * 32u + i] = quantize_dequantize_e2m1(x[sb * 32u + i], inv_scale, scale);
            }
        }
        for (uint32_t r = 0; r < out_dim; r++) {
            const block_mxfp4 *row = (const block_mxfp4 *)(w_bytes + (uint64_t)r * row_bytes);
            dequantize_row_mxfp4(row, wf, in_dim);
            float acc = 0.0f;
            for (uint32_t i = 0; i < in_dim; i++) acc += wf[i] * xq[i];
            out_row[t * out_dim_stride + r] = acc;
        }
    }
    free(wf); free(xq);
}

extern int ds4_debug_mxfp4_mma_gemm(
        float *out_group_host, const unsigned char *w_host, uint64_t row_bytes,
        const float *x_host, uint32_t in_dim, uint32_t out_dim, uint32_t group_size);

static uint32_t g_rng = 0xC0FFEEu;
static uint32_t rng_next(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5;
    return g_rng;
}
static float rng_unit(void) { return (float)(rng_next() % 20001) / 10000.0f - 1.0f; }

static void fill_random_mxfp4_row(block_mxfp4 *row, uint32_t blocks) {
    for (uint32_t b = 0; b < blocks; b++) {
        row[b].e = (uint8_t)(120 + (rng_next() % 16));
        for (uint32_t j = 0; j < QK_MXFP4 / 2; j++) row[b].qs[j] = (uint8_t)(rng_next() & 0xFFu);
    }
}

static int run_case(uint32_t in_dim, uint32_t out_dim, uint32_t group_size) {
    fprintf(stderr, "-- case in_dim=%u out_dim=%u group_size=%u --\n", in_dim, out_dim, group_size);
    const uint32_t blocks = in_dim / QK_MXFP4;
    const uint64_t row_bytes = (uint64_t)blocks * sizeof(block_mxfp4);
    unsigned char *w = (unsigned char *)malloc((size_t)out_dim * row_bytes);
    for (uint32_t r = 0; r < out_dim; r++) {
        fill_random_mxfp4_row((block_mxfp4 *)(w + (uint64_t)r * row_bytes), blocks);
    }
    float *x = (float *)malloc((size_t)group_size * in_dim * sizeof(float));
    for (uint64_t i = 0; i < (uint64_t)group_size * in_dim; i++) x[i] = rng_unit();

    float *ref_out = (float *)calloc((size_t)group_size * out_dim, sizeof(float));
    ref_row(ref_out, w, in_dim, out_dim, x, group_size, out_dim);

    float *gpu_out = (float *)calloc((size_t)group_size * out_dim, sizeof(float));
    int ok = ds4_debug_mxfp4_mma_gemm(gpu_out, w, row_bytes, x, in_dim, out_dim, group_size);
    if (!ok) { fprintf(stderr, "kernel launch reported failure\n"); return 1; }

    int fail = 0;
    int nfail = 0;
    for (uint32_t t = 0; t < group_size; t++) {
        for (uint32_t r = 0; r < out_dim; r++) {
            float got = gpu_out[t * out_dim + r];
            float want = ref_out[t * out_dim + r];
            float diff = fabsf(got - want);
            float scale = fmaxf(1.0f, fabsf(want));
            /* Loose tol: block-scaled tensor core E2M1xE2M1 f32 accumulate
             * should match this reference to within fp round-off, not
             * exactly since accumulation order differs. */
            if (diff > 1e-2f * scale) {
                if (nfail < 20) {
                    fprintf(stderr, "FAIL tok=%u row=%u: got %.6f want %.6f (diff %.6f)\n",
                            t, r, got, want, diff);
                }
                nfail++;
                fail = 1;
            }
        }
    }
    fprintf(stderr, "case in=%u out=%u group=%u: %s (%d/%u mismatches)\n",
            in_dim, out_dim, group_size, fail ? "FAILED" : "passed", nfail, group_size * out_dim);
    free(w); free(x); free(ref_out); free(gpu_out);
    return fail;
}

int main(void) {
    int fail = 0;
    fail |= run_case(64, 16, 1);
    fail |= run_case(64, 16, 3);
    fail |= run_case(128, 32, 5);
    fail |= run_case(512, 256, 4);
    fail |= run_case(512, 256, 5);
    fail |= run_case(4096, 2048, 3);
    fail |= run_case(2048, 4096, 7);
    if (fail) { fprintf(stderr, "MMA GEMM isolation test: FAILED\n"); return 1; }
    fprintf(stderr, "MMA GEMM isolation test: all cases passed\n");
    return 0;
}
