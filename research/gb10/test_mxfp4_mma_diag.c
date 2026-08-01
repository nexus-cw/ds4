/*
 * Diagnostic isolation test for dsv4_mxfp4_mma_gemm_kernel (P3c-1 take 2).
 *
 * STATUS (2026-08-01): diagnostic reproducer for a known numeric bug (see
 * FP4_PORT_SCOPE.md P3c-1 take 2). One-hot weight rows (row r has a single
 * nonzero element at k=r) + distinct-per-k activation values; expected
 * output row r ~= quantized(x[r]), actual: only 2 of 16 output rows (a
 * fixed, lane-correlated pair) come out correct, the rest are silently
 * zero -- narrows the bug to the A-operand (weight) ldmatrix addressing
 * or the tile_C write-back lane mapping (both ported from llama.cpp's
 * mma.cuh, so either a transcription error, or a subtlety in how those
 * formulas compose outside the donor's full templated MMQ scheduling
 * context).
 *
 * Build (after `make cuda-spark` has produced ds4_cuda.o):
 *   /usr/local/cuda/bin/nvcc -O2 -o /tmp/test_mxfp4_mma_diag \
 *       research/gb10/test_mxfp4_mma_diag.c ds4_cuda.o -lcudart -lcublas -lm \
 *       -L/usr/local/cuda/targets/sbsa-linux/lib -L/usr/local/cuda/lib64
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define QK_MXFP4 32
typedef struct { uint8_t e; uint8_t qs[QK_MXFP4/2]; } block_mxfp4;

extern int ds4_debug_mxfp4_mma_gemm(
        float *out_group_host, const unsigned char *w_host, uint64_t row_bytes,
        const float *x_host, uint32_t in_dim, uint32_t out_dim, uint32_t group_size);

/* nibble index -> value (E2M1 doubled table, matches ds4/llama.cpp) */
static const int kvalues[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};

int main(void) {
    const uint32_t in_dim = 64, out_dim = 16, group_size = 1;
    const uint32_t blocks = in_dim/QK_MXFP4; /* 2 */
    const uint64_t row_bytes = (uint64_t)blocks*sizeof(block_mxfp4);
    unsigned char *w = calloc(out_dim, row_bytes);

    /* Row r: one-hot at k-position r (r in 0..15, so only within block0),
     * value nibble=2 (=1.0 with scale 127=2^0), all else nibble=0/scale=127. */
    for (uint32_t r = 0; r < out_dim; r++) {
        block_mxfp4 *rowblk = (block_mxfp4*)(w + (uint64_t)r*row_bytes);
        rowblk[0].e = 127; rowblk[1].e = 127;
        /* nibble j in block0 covers elements j (low) and j+16 (high) */
        uint32_t j = r % 16;
        uint8_t nib_lo = (r < 16) ? 2 : 0;
        (void)nib_lo;
        rowblk[0].qs[j] = 2; /* low nibble = element j = value 2*0.5=1.0 */
    }

    float x[64];
    for (uint32_t k = 0; k < 64; k++) x[k] = (float)(k+1); /* distinct per k */

    float out[16] = {0};
    int ok = ds4_debug_mxfp4_mma_gemm(out, w, row_bytes, x, in_dim, out_dim, group_size);
    printf("launch ok=%d\n", ok);
    for (uint32_t r = 0; r < out_dim; r++) {
        /* expected: element r should be selected with weight 1.0 (nibble2,
         * scale 2^0) -- so out[r] should approx equal quantized(x[r]).
         * But x[r] here is subject to E2M1 quantization per its own
         * 32-sub-block amax; compute expected via same quantize logic. */
        printf("row %2u: got=%10.4f  (weight one-hot at k=%u, x[k]=%.1f)\n", r, out[r], r, x[r]);
    }
    return 0;
}
