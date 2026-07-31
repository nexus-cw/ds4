/*
 * Standalone correctness test for ds4's MXFP4 (GGUF tensor type 39) CPU
 * dequant, ported from llama.cpp (MIT) ggml-common.h / ggml-quants.c /
 * ggml-impl.h @5f55650.
 *
 * Two independent implementations are cross-checked for float-exact
 * equality across all 16 E2M1 nibbles x all 256 E8M0 scale bytes
 * (4096 combinations):
 *
 *   1. ds4_dequantize_row_mxfp4() -- byte-for-byte copy of the function
 *      added to ds4.c (block_mxfp4 struct, kvalues_mxfp4 table, the
 *      "_half" E8M0 conversion, and the per-block unpack loop).
 *   2. donor_dequant_mxfp4_value() -- an independently hand-derived
 *      reference: llama.cpp's ggml_e8m0_to_fp32_half(x) bit-trick reduces
 *      algebraically to the closed form 2^(x-128) for every x in
 *      [0,255] (denormals cover x<2, normals cover x>=2 -- verified by
 *      hand, see comment below), so the reference recomputes the scale
 *      via ldexpf(1.0f, (int)x - 128) instead of ds4's bit-shift
 *      construction, and looks the E2M1 magnitude up independently from
 *      first principles (0,0.5,1,1.5,2,3,4,6, negated by the nibble's
 *      high bit) rather than reusing kvalues_mxfp4[].
 *
 * Build: cc -O2 -Wall -Wextra -std=c99 -o test_mxfp4_dequant test_mxfp4_dequant.c -lm
 * Run:   ./test_mxfp4_dequant
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ---- block layout + tables, copied verbatim from ds4.c ---- */

#define QK_MXFP4 32
typedef struct {
    uint8_t e;                /* E8M0 exponent-only scale */
    uint8_t qs[QK_MXFP4 / 2];  /* packed 4-bit E2M1 values */
} block_mxfp4;

static const int8_t kvalues_mxfp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

static inline float mxfp4_e8m0_to_fp32_half(uint8_t x) {
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

static void ds4_dequantize_row_mxfp4(const block_mxfp4 *x, float *y, int64_t k) {
    const int64_t qk = QK_MXFP4;
    const int64_t nb = k / qk;
    for (int64_t i = 0; i < nb; i++) {
        const float d = mxfp4_e8m0_to_fp32_half(x[i].e);
        for (int64_t j = 0; j < qk / 2; ++j) {
            const int8_t x0 = kvalues_mxfp4[x[i].qs[j] & 0x0F];
            const int8_t x1 = kvalues_mxfp4[x[i].qs[j] >> 4];
            y[i * qk + j + 0]      = x0 * d;
            y[i * qk + j + qk / 2] = x1 * d;
        }
    }
}

/* ---- independent reference, re-derived from donor semantics by hand ---- */

/*
 * llama.cpp's ggml_e8m0_to_fp32_half(x):
 *   x < 2:  bits = 0x00200000 << x         (IEEE-754 denormal)
 *   x >= 2: bits = (x - 1) << 23           (IEEE-754 normal, mantissa 0)
 *
 * For x < 2 the mantissa fraction is 0x00200000<<x / 0x00800000 = 2^(x-2),
 * and denormals evaluate to fraction * 2^(-126), so the value is
 * 2^(x-2-126) = 2^(x-128).
 * For x >= 2, exponent field = x-1, unbiased exponent = (x-1)-127 = x-128,
 * implicit mantissa 1.0, so the value is 1.0 * 2^(x-128) = 2^(x-128).
 * Both branches reduce to the same closed form: 2^(x-128), for all x in
 * [0,255] (ldexpf handles both the denormal and normal float32 range).
 */
static float donor_e8m0_half_closed_form(uint8_t x) {
    return ldexpf(1.0f, (int)x - 128);
}

/* E2M1 magnitude from first principles (mantissa bit m, exponent bits e,
 * per OCP MX spec): 0b0mmm -> {0, 0.5, 1, 1.5} for exp=0 (subnormal),
 * 0bemmm for exp>0 -> (1 + m/2) * 2^(e-1). Doubled (x2) to match
 * kvalues_mxfp4's pre-doubled convention, and sign taken from nibble
 * bit 3.
 */
static int donor_e2m1_doubled_magnitude(int mag3) {
    /* E2M1 = 2 exponent bits + 1 mantissa bit: mag3 = (exp<<1)|mantissa.
     * exp=0 (subnormal): value = mantissa * 0.5      -> 0, 0.5
     * exp>0 (normal):    value = (1+mantissa*0.5) * 2^(exp-1)
     *   exp=1 -> 1, 1.5 ; exp=2 -> 2, 3 ; exp=3 -> 4, 6
     */
    int e = mag3 >> 1;
    int m = mag3 & 1;
    float val;
    if (e == 0) {
        val = m * 0.5f;
    } else {
        val = (1.0f + m * 0.5f) * (float)(1 << (e - 1));
    }
    return (int)(val * 2.0f + 0.5f); /* doubled, rounded to nearest int */
}

static float donor_dequant_mxfp4_value(uint8_t nibble, uint8_t scale_byte) {
    const int sign = (nibble & 0x8) ? -1 : 1;
    const int mag3 = nibble & 0x7;
    const int doubled = sign * donor_e2m1_doubled_magnitude(mag3);
    const float d = donor_e8m0_half_closed_form(scale_byte);
    return (float)doubled * d;
}

/* ---- driver ---- */

int main(void) {
    int failures = 0;
    long checks = 0;

    /* Sanity check the two E8M0->fp32_half implementations agree first,
     * since both dequant paths depend on it. */
    for (int x = 0; x <= 255; x++) {
        float a = mxfp4_e8m0_to_fp32_half((uint8_t)x);
        float b = donor_e8m0_half_closed_form((uint8_t)x);
        if (a != b) {
            fprintf(stderr, "E8M0 half-scale mismatch at x=%d: ds4=%.9g closed_form=%.9g\n",
                    x, (double)a, (double)b);
            failures++;
        }
    }

    /* Full 16 nibbles x 256 scale bytes = 4096 combinations, driven
     * through the block-based dequant entry point (not just the scalar
     * helper) so the block/loop plumbing in dequantize_row_mxfp4 is
     * exercised too. */
    block_mxfp4 blk;
    float y[QK_MXFP4];
    for (int scale = 0; scale <= 255; scale++) {
        blk.e = (uint8_t)scale;
        for (int nibble = 0; nibble < 16; nibble++) {
            /* Fill every packed byte with this nibble in both halves so
             * every output slot in the block decodes to the same value,
             * then check all 32 outputs against the independent ref. */
            for (size_t j = 0; j < sizeof(blk.qs); j++) {
                blk.qs[j] = (uint8_t)(nibble | (nibble << 4));
            }
            memset(y, 0, sizeof(y));
            ds4_dequantize_row_mxfp4(&blk, y, QK_MXFP4);

            float expected = donor_dequant_mxfp4_value((uint8_t)nibble, (uint8_t)scale);
            for (int k = 0; k < QK_MXFP4; k++) {
                checks++;
                if (y[k] != expected) {
                    if (failures < 20) {
                        fprintf(stderr,
                                "mismatch: nibble=%d scale=%d k=%d ds4=%.9g expected=%.9g\n",
                                nibble, scale, k, (double)y[k], (double)expected);
                    }
                    failures++;
                }
            }
        }
    }

    /* Mixed-nibble block: qs[j] low/high nibbles differ, and scan a
     * handful of non-trivial scale bytes, to exercise the &0x0F / >>4
     * split itself (the loop above used the same nibble both halves). */
    {
        uint8_t scales_to_check[] = {0, 1, 2, 3, 100, 126, 127, 128, 200, 254, 255};
        for (size_t si = 0; si < sizeof(scales_to_check); si++) {
            blk.e = scales_to_check[si];
            for (size_t j = 0; j < sizeof(blk.qs); j++) {
                int lo = (int)(j % 16);
                int hi = (int)((j * 7 + 3) % 16);
                blk.qs[j] = (uint8_t)(lo | (hi << 4));
            }
            memset(y, 0, sizeof(y));
            ds4_dequantize_row_mxfp4(&blk, y, QK_MXFP4);
            for (size_t j = 0; j < sizeof(blk.qs); j++) {
                int lo = (int)(j % 16);
                int hi = (int)((j * 7 + 3) % 16);
                float exp_lo = donor_dequant_mxfp4_value((uint8_t)lo, blk.e);
                float exp_hi = donor_dequant_mxfp4_value((uint8_t)hi, blk.e);
                checks += 2;
                if (y[j] != exp_lo) {
                    fprintf(stderr, "mixed-block mismatch (lo) scale=%u j=%zu ds4=%.9g expected=%.9g\n",
                            blk.e, j, (double)y[j], (double)exp_lo);
                    failures++;
                }
                if (y[j + 16] != exp_hi) {
                    fprintf(stderr, "mixed-block mismatch (hi) scale=%u j=%zu ds4=%.9g expected=%.9g\n",
                            blk.e, j, (double)y[j + 16], (double)exp_hi);
                    failures++;
                }
            }
        }
    }

    if (failures) {
        fprintf(stderr, "FAIL: %d mismatches out of %ld checks\n", failures, checks);
        return 1;
    }
    printf("PASS: %ld checks, 0 mismatches (256 E8M0 scale bytes x 16 E2M1 nibbles, "
           "plus mixed-nibble block sweep)\n", checks);
    return 0;
}
