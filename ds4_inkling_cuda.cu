/* ds4_inkling_cuda.cu -- CUDA v1 forward path for the standalone "inkling"
 * engine.
 *
 * Design: the target machine (GB10 Grace-Blackwell) has HW-coherent
 * pageable (unified) host memory access, so kernels dereference the host
 * mmap'd GGUF pointers and host malloc'd activation buffers *directly* --
 * no cudaMemcpy of weights, no cudaHostRegister of the (multi-GB) model
 * file.  This is intentionally the slow-but-correct v1 design (see
 * PORT_NOTES.md); ink_cuda_init() refuses to run on a GPU/driver that
 * lacks cudaDevAttrPageableMemoryAccess rather than silently falling back
 * to something incorrect.
 *
 * The row dequantizers below are a line-by-line CUDA port of the CPU
 * reference in ds4_inkling.c (ink_dq_q8_0/q4_K/q5_K/q6_K/iq2_xxs/iq2_s/
 * iq3_xxs/iq4_xs), which are themselves adapted from llama.cpp
 * ggml-quants.c (MIT license, https://github.com/ggml-org/llama.cpp).
 * ds4_inkling.c is the correctness oracle: `--selftest LAYER` in this
 * binary runs every weight tensor of one block through both the CPU
 * ink_matvec() and the GPU ink_cuda_matvec() and diffs them. */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_inkling.h"
#include "ds4_inkling_tables.inc"

#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "ds4-inkling-cuda: CUDA error: %s (%s:%d)\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

/* ============================ IQ table upload ========================= */

typedef struct {
    const uint64_t *iq2xxs_grid;
    const uint64_t *iq2s_grid;
    const uint32_t *iq3xxs_grid;
    const uint8_t  *ksigns_iq2xs;
    const uint8_t  *kmask_iq2xs;
    const int8_t   *kvalues_iq4nl;
} ink_tables;

static ink_tables g_tables;
static int g_tables_ready = 0;

/* ============================ M7: single-stream execution ================
 * All kernel launches / async memcpys in this file go on g_stream.  Per-
 * call wrappers no longer cudaDeviceSynchronize(); ordering between
 * dependent GPU ops is guaranteed by stream program order alone.  The host
 * only synchronizes at the few points it actually reads GPU-written data:
 * MoE routing (needs `logits` for host qsort) and the final output-logits
 * readback in ink_forward_gpu().  cudaGetLastError() after every launch
 * still catches *launch-configuration* errors immediately (it does not
 * require a sync); it does not catch asynchronous kernel-body errors,
 * which will surface at the next ink_cuda_sync() or CUDA_CHECK that does
 * sync (matches the tradeoff of any async-stream design). */
static cudaStream_t g_stream = 0;

extern "C" void ink_cuda_sync(void) {
    CUDA_CHECK(cudaStreamSynchronize(g_stream));
}

/* M11: runtime A/B lever between the exact float-dequant path (our
 * correctness oracle, unconditionally used by --selftest regardless of the
 * env var -- see ink_run_selftest) and the fast int8-dp4a path for
 * IQ2_XXS/IQ3_XXS group-matvec (the measured decode bottleneck).
 * INK_EXACT_DEQUANT=1 forces the exact path everywhere; default (unset/0)
 * is the fast path. */
static int g_exact_dequant = 0;

/* M12 bisect support: enable the dp4a fast path for a LAYER RANGE only, so
 * the FAST-vs-EXACT divergence can be swept layer by layer.
 *   INK_FAST_LAYERS=none (default) | all | N | LO-HI
 * INK_FAST_DEQUANT=1 is equivalent to INK_FAST_LAYERS=all.  g_exact_dequant
 * is re-derived per layer inside ink_forward_gpu(); selftest/bench paths
 * set it directly and are unaffected (they pass layer -1). */
static int g_fast_lo = -1, g_fast_hi = -2;   /* empty range */

static void ink_fast_layers_parse(void) {
    const char *v = getenv("INK_FAST_LAYERS");
    if (!v || !v[0]) return;
    if (!strcmp(v, "none")) { g_fast_lo = -1; g_fast_hi = -2; return; }
    if (!strcmp(v, "all"))  { g_fast_lo = 0;  g_fast_hi = 1 << 30; return; }
    int lo = 0, hi = 0;
    if (sscanf(v, "%d-%d", &lo, &hi) == 2)      { g_fast_lo = lo; g_fast_hi = hi; }
    else if (sscanf(v, "%d", &lo) == 1)         { g_fast_lo = lo; g_fast_hi = lo; }
    else ink_die("INK_FAST_LAYERS: expected none|all|N|LO-HI");
}

static bool ink_layer_is_fast(int il) {
    return il >= g_fast_lo && il <= g_fast_hi;
}

/* M12: restrict the fast path to one quant TYPE, which for this artifact
 * separates the two activation distributions feeding it:
 *   iq2 -> gate_exps/up_exps, whose input is the post-rmsnorm hidden
 *          state (well-conditioned, roughly gaussian)
 *   iq3 -> down_exps, whose input is the post-SiLU expert hidden vector
 *          (heavy-tailed: one outlier sets the per-32-block int8 scale and
 *          crushes the resolution of the other 31 values)
 * INK_FAST_TYPES=both (default) | iq2 | iq3 */
static bool g_fast_iq2 = true, g_fast_iq3 = true;

static void ink_fast_types_parse(void) {
    const char *v = getenv("INK_FAST_TYPES");
    if (!v || !v[0] || !strcmp(v, "both")) return;
    if (!strcmp(v, "iq2")) { g_fast_iq2 = true;  g_fast_iq3 = false; return; }
    if (!strcmp(v, "iq3")) { g_fast_iq2 = false; g_fast_iq3 = true;  return; }
    ink_die("INK_FAST_TYPES: expected both|iq2|iq3");
}

static void *ink_cuda_upload(const void *src, size_t bytes) {
    void *dst = NULL;
    CUDA_CHECK(cudaMalloc(&dst, bytes));
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
    return dst;
}

extern "C" void ink_cuda_init(void) {
    int val = 0;
    cudaError_t e = cudaDeviceGetAttribute(&val, cudaDevAttrPageableMemoryAccess, 0);
    if (e != cudaSuccess || val != 1) {
        ink_die("ink_cuda_init: this GPU/driver does not report "
                "cudaDevAttrPageableMemoryAccess==1 (coherent pageable/unified "
                "host memory access from device code).  The v1 CUDA inkling "
                "path dereferences the mmap'd GGUF and host activation "
                "buffers directly from kernels and requires this (e.g. GB10 "
                "Grace-Blackwell with NVLink-C2C).  Refusing to run rather "
                "than silently doing something slow or wrong.");
    }

    g_tables.iq2xxs_grid  = (const uint64_t *)ink_cuda_upload(iq2xxs_grid, sizeof(iq2xxs_grid));
    g_tables.iq2s_grid    = (const uint64_t *)ink_cuda_upload(iq2s_grid, sizeof(iq2s_grid));
    g_tables.iq3xxs_grid  = (const uint32_t *)ink_cuda_upload(iq3xxs_grid, sizeof(iq3xxs_grid));
    g_tables.ksigns_iq2xs = (const uint8_t *)ink_cuda_upload(ksigns_iq2xs, sizeof(ksigns_iq2xs));
    g_tables.kmask_iq2xs  = (const uint8_t *)ink_cuda_upload(kmask_iq2xs, sizeof(kmask_iq2xs));
    g_tables.kvalues_iq4nl = (const int8_t *)ink_cuda_upload(kvalues_iq4nl, sizeof(kvalues_iq4nl));
    g_tables_ready = 1;

    CUDA_CHECK(cudaStreamCreate(&g_stream));

    /* M11 SAFETY: the int8/dp4a fast path DIVERGES from the exact path on
     * real prompts (PORT_NOTES.md M11: prompt 2 chose a different token 0
     * with a 2.3-logit gap -- systematic distortion, not int8 rounding),
     * so the EXACT float path is the default and the fast path is opt-in
     * via INK_FAST_DEQUANT=1 while it is under investigation. */
    const char *fa = getenv("INK_FAST_DEQUANT");
    bool want_fast = (fa && fa[0] && strcmp(fa, "0") != 0);
    if (want_fast) { g_fast_lo = 0; g_fast_hi = 1 << 30; }
    ink_fast_layers_parse();
    ink_fast_types_parse();
    g_exact_dequant = 1;   /* per-layer value is set in ink_forward_gpu */
}

/* ========================= device dequantizers =========================
 * v2: per-SUBGROUP (32-element) dequantizers.  Each dq32_<type>() function
 * dequantizes one 32-wide subgroup of a row into a register array `w[32]`,
 * hoisting every per-block/per-subgroup-invariant computation (scale/min
 * lookups, grid-table byte fetches, sign bytes, aux32 words, ...) OUTSIDE
 * the 32-wide inner loop instead of recomputing it per element the way the
 * v1 ink_dq_elem() scalar path did.  The arithmetic for each output
 * element is kept bit-for-bit identical to ink_dq_elem() / the CPU
 * ink_dq_* reference in ds4_inkling.c (same operand order, so no new FP
 * reassociation is introduced) -- only the *placement* of loop-invariant
 * work changes.  Adapted from llama.cpp ggml-quants.c dequant/vecdot
 * patterns (MIT license, https://github.com/ggml-org/llama.cpp).
 *
 * Subgroup indexing: for QK_K=256-element block types (everything except
 * F32 and Q8_0) each block holds 8 32-wide subgroups; global subgroup
 * index `sg` decomposes as block index i = sg / 8, in-block subgroup
 * local_sg = sg % 8 (this local_sg is exactly the "sg" variable the old
 * per-element code derived from p = idx % QK_K, sg = p / 32).  Q8_0 blocks
 * are exactly 32 elements (one subgroup per block, i = sg).  F32 has no
 * block structure at all. */

__device__ __forceinline__ size_t ink_dev_block_elems(uint32_t t) {
    switch (t) {
    case INK_T_F32: return 1;
    case INK_T_Q8_0: return QK8_0;
    default: return QK_K;
    }
}

__device__ __forceinline__ size_t ink_dev_block_bytes(uint32_t t) {
    switch (t) {
    case INK_T_F32: return 4;
    case INK_T_Q8_0: return sizeof(ink_block_q8_0);
    case INK_T_Q4_K: return sizeof(ink_block_q4_K);
    case INK_T_Q5_K: return sizeof(ink_block_q5_K);
    case INK_T_Q6_K: return sizeof(ink_block_q6_K);
    case INK_T_IQ2_XXS: return sizeof(ink_block_iq2_xxs);
    case INK_T_IQ3_XXS: return sizeof(ink_block_iq3_xxs);
    case INK_T_IQ2_S: return sizeof(ink_block_iq2_s);
    case INK_T_IQ4_XS: return sizeof(ink_block_iq4_xs);
    default: return 0;
    }
}

__device__ __forceinline__ void ink_dev_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

#ifdef INK_DEBUG_SCALAR_DQ
/* Debug-only fallback: the original v1 per-ELEMENT scalar dequantizer,
 * kept for bisection against the v2 per-subgroup path if a regression
 * ever needs isolating.  Not compiled/used by default. */
__device__ __forceinline__ float ink_dq_elem(uint32_t type, const uint8_t *rowp, uint64_t idx,
                                              ink_tables tb) {
    switch (type) {
    case INK_T_F32:
        return ((const float *)rowp)[idx];
    default:
        return 0.0f;
    }
}
#endif

/* M8: small load/arithmetic helpers shared by the dequantizers below.
 *
 * ink_load_u32: alignment-agnostic wide (32-bit) byte-stream read.  Block
 * structs are not always 4-byte aligned relative to a row's start (e.g.
 * ink_block_iq2_xxs is 66 bytes, so consecutive blocks' interesting
 * sub-regions alternate between 2- and 4-byte alignment) -- casting to
 * `const uint32_t *` and dereferencing would be undefined behavior on a
 * misaligned pointer, and vector loads (uint2/uint4) actively fault on
 * some paths if misaligned.  memcpy of a compile-time-constant size into a
 * register is the portable, well-defined way to get the same single wide
 * load nvcc would emit for an aligned pointer, without the UB.  Assumes
 * little-endian byte order (true of every CUDA-capable GPU and of the
 * GGUF format itself), matching the manual byte-shift reconstruction this
 * file already did before M8 (bp[0] | bp[1]<<8 | ...) -- same assumption,
 * just made once here instead of by construction at each call site.
 *
 * ink_signed: flips the sign bit of `mag` when `neg` is true.  IEEE-754
 * multiplication by exactly -1.0f is defined to do nothing but flip the
 * sign bit (no rounding), so this is bitwise identical to
 * `neg ? -mag : mag` / `mag * (neg ? -1.f : 1.f)` for every finite value
 * (and for +-0/+-inf too) -- it just replaces a float compare+select with
 * an integer XOR, avoiding the per-lane branch. */
__device__ __forceinline__ uint32_t ink_load_u32(const uint8_t *p) {
    uint32_t v;
    memcpy(&v, p, 4);
    return v;
}

__device__ __forceinline__ float ink_signed(float mag, bool neg) {
    uint32_t bits = __float_as_uint(mag);
    bits ^= neg ? 0x80000000u : 0u;
    return __uint_as_float(bits);
}

__device__ __forceinline__ void dq32_f32(const uint8_t *rowp, uint32_t sg, float w[32]) {
    const float *p = (const float *)rowp + (size_t)sg * 32;
#pragma unroll
    for (int l = 0; l < 32; l++) w[l] = p[l];
}

__device__ __forceinline__ void dq32_q8_0(const uint8_t *rowp, uint32_t sg, float w[32]) {
    const ink_block_q8_0 *x = (const ink_block_q8_0 *)rowp;
    const ink_block_q8_0 b = x[sg]; /* sg == block index: QK8_0 == 32 */
    float d = __half2float(__ushort_as_half(b.d));
#pragma unroll
    for (int l = 0; l < 32; l++) w[l] = (float)b.qs[l] * d;
}

__device__ __forceinline__ void dq32_q4_K(const uint8_t *rowp, uint32_t sg, float w[32]) {
    const ink_block_q4_K *x = (const ink_block_q4_K *)rowp;
    uint32_t i = sg / 8, local_sg = sg % 8;
    const ink_block_q4_K *b = &x[i];
    float d = __half2float(__ushort_as_half(b->d));
    float mn = __half2float(__ushort_as_half(b->dmin));
    uint8_t sc, m;
    ink_dev_scale_min_k4((int)local_sg, b->scales, &sc, &m);
    const uint8_t *q = b->qs + 32 * (local_sg / 2);
    bool hi = (local_sg & 1) != 0;
    float dsc = d * sc, dmnm = mn * m;
#pragma unroll
    for (int l = 0; l < 32; l++) {
        uint8_t nib = hi ? (q[l] >> 4) : (q[l] & 0xF);
        w[l] = dsc * nib - dmnm;
    }
}

__device__ __forceinline__ void dq32_q5_K(const uint8_t *rowp, uint32_t sg, float w[32]) {
    const ink_block_q5_K *x = (const ink_block_q5_K *)rowp;
    uint32_t i = sg / 8, local_sg = sg % 8;
    const ink_block_q5_K *b = &x[i];
    float d = __half2float(__ushort_as_half(b->d));
    float mn = __half2float(__ushort_as_half(b->dmin));
    uint8_t sc, m;
    ink_dev_scale_min_k4((int)local_sg, b->scales, &sc, &m);
    const uint8_t *ql = b->qs + 32 * (local_sg / 2);
    uint32_t shift = 2 * (local_sg / 2);
    uint8_t u1 = (uint8_t)(1u << shift), u2 = (uint8_t)(2u << shift);
    bool hi = (local_sg & 1) != 0;
    /* hi selects which of u1/u2 applies for the WHOLE subgroup (loop-
     * invariant, same algebraic hoist as dq32_q6_K's qh_shift) --
     * hbit = hi ? (qh[l]&u2) : (qh[l]&u1) becomes (qh[l] & umask). */
    uint8_t umask = hi ? u2 : u1;
    float dsc = d * sc, dmnm = mn * m;
    /* M13 item 2: wide (memcpy-safe) 32-bit reads, same technique as
     * dq32_q6_K -- 8 groups of 4 bytes instead of 32 separate byte loads
     * for both ql and b->qh; extracting byte `sub` back out gives the
     * exact same value ql[g*4+sub]/qh[g*4+sub] would have. */
#pragma unroll
    for (uint32_t g = 0; g < 8; g++) {
        uint32_t qlw = ink_load_u32(ql + g * 4);
        uint32_t qhw = ink_load_u32(b->qh + g * 4);
#pragma unroll
        for (uint32_t sub = 0; sub < 4; sub++) {
            uint32_t l = g * 4 + sub;
            uint8_t qlb = (uint8_t)(qlw >> (8 * sub));
            uint8_t qhb = (uint8_t)(qhw >> (8 * sub));
            uint8_t nib = hi ? (qlb >> 4) : (qlb & 0xF);
            uint8_t hbit = (qhb & umask) ? 16 : 0;
            w[l] = dsc * (nib + hbit) - dmnm;
        }
    }
}

__device__ __forceinline__ void dq32_q6_K(const uint8_t *rowp, uint32_t sg, float w[32]) {
    const ink_block_q6_K *x = (const ink_block_q6_K *)rowp;
    uint32_t i = sg / 8, local_sg = sg % 8;
    const ink_block_q6_K *b = &x[i];
    uint32_t half = local_sg / 4, qsel = local_sg % 4;
    float d = __half2float(__ushort_as_half(b->d));
    const int8_t *sc = b->scales + 8 * half;
    /* is = l/16 in {0,1}; hoist the two possible scale*d values, matching
     * scv = sc[is + {0,2,4,6}] for qsel {0,1,2,3} exactly. */
    int scoff = (int)qsel * 2;
    float dsc0 = d * (float)sc[scoff + 0];
    float dsc1 = d * (float)sc[scoff + 1];
    /* qsel is loop-invariant (fixed for this whole subgroup), so fold the
     * ql[l] vs ql[l+32] / nibble-half selection into one base pointer +
     * shift instead of re-branching per element (nvcc would likely hoist
     * this anyway via LICM, but making it explicit removes any doubt and
     * sets up the grouped 32-bit reads below cleanly). */
    const uint8_t *qlbase = b->ql + 64 * half + ((qsel == 1 || qsel == 3) ? 32 : 0);
    const uint8_t *qh = b->qh + 32 * half;
    bool hi_nib = (qsel == 2 || qsel == 3);
    uint32_t qh_shift = 2 * qsel;
    /* wide (memcpy-safe) 32-bit reads: 8 groups of 4 bytes cover all 32
     * ql/qh bytes touched by this subgroup instead of 32 separate byte
     * loads each; extracting byte `sub` back out of the loaded word gives
     * the exact same value ql[g*4+sub]/qh[g*4+sub] would have (little-
     * endian, see ink_load_u32). */
#pragma unroll
    for (uint32_t g = 0; g < 8; g++) {
        uint32_t qlw = ink_load_u32(qlbase + g * 4);
        uint32_t qhw = ink_load_u32(qh + g * 4);
#pragma unroll
        for (uint32_t sub = 0; sub < 4; sub++) {
            uint32_t l = g * 4 + sub;
            uint8_t qlb = (uint8_t)(qlw >> (8 * sub));
            uint8_t qhb = (uint8_t)(qhw >> (8 * sub));
            int val = (hi_nib ? (qlb >> 4) : (qlb & 0xF)) | (((qhb >> qh_shift) & 3) << 4);
            val -= 32;
            w[l] = (l < 16 ? dsc0 : dsc1) * (float)val;
        }
    }
}

__device__ __forceinline__ void dq32_iq2_xxs(const uint8_t *rowp, uint32_t sg, float w[32], ink_tables tb) {
    const ink_block_iq2_xxs *x = (const ink_block_iq2_xxs *)rowp;
    uint32_t i = sg / 8, local_sg = sg % 8;
    const ink_block_iq2_xxs *b = &x[i];
    float d = __half2float(__ushort_as_half(b->d));
    const uint8_t *bp = (const uint8_t *)b->qs + 8 * local_sg;
    /* one 8-byte region -> two wide 32-bit loads instead of 8 byte loads
     * (block stride is 66 bytes, so `bp` isn't reliably 4-byte aligned --
     * ink_load_u32 handles that safely, see its comment above). */
    uint32_t a0 = ink_load_u32(bp);
    uint32_t a1 = ink_load_u32(bp + 4);
    float db = d * (0.5f + (float)(a1 >> 28)) * 0.25f;
#pragma unroll
    for (uint32_t lg = 0; lg < 4; lg++) {
        uint8_t byte_l = (uint8_t)((a0 >> (8 * lg)) & 0xFF);
        uint64_t gridv = tb.iq2xxs_grid[byte_l];
        uint8_t signs = tb.ksigns_iq2xs[(a1 >> (7 * lg)) & 127];
#pragma unroll
        for (uint32_t j = 0; j < 8; j++) {
            uint8_t gbyte = (uint8_t)((gridv >> (8 * j)) & 0xFF);
            /* sign-bit XOR instead of a compare+select multiply by -1.f --
             * bitwise identical (see ink_signed's comment). */
            w[lg * 8 + j] = ink_signed(db * (float)gbyte, (signs & tb.kmask_iq2xs[j]) != 0);
        }
    }
}

__device__ __forceinline__ void dq32_iq2_s(const uint8_t *rowp, uint32_t sg, float w[32], ink_tables tb) {
    const ink_block_iq2_s *x = (const ink_block_iq2_s *)rowp;
    uint32_t i = sg / 8, local_sg = sg % 8;
    const ink_block_iq2_s *b = &x[i];
    float d = __half2float(__ushort_as_half(b->d));
    const uint8_t *qs_ptr = b->qs + 4 * local_sg;
    const uint8_t *signs_ptr = b->qs + QK_K / 8 + 4 * local_sg;
    uint32_t qh_val = b->qh[local_sg];
    uint8_t scale_byte = b->scales[local_sg];
    float db0 = d * (0.5f + (float)(scale_byte & 0xf)) * 0.25f;
    float db1 = d * (0.5f + (float)(scale_byte >> 4)) * 0.25f;
#pragma unroll
    for (uint32_t l4 = 0; l4 < 4; l4++) {
        float dl = (l4 < 2) ? db0 : db1;
        uint32_t grid_idx = qs_ptr[l4] | ((qh_val << (8 - 2 * l4)) & 0x300);
        uint64_t gridv = tb.iq2s_grid[grid_idx];
        uint8_t sign_byte = signs_ptr[l4];
#pragma unroll
        for (uint32_t j = 0; j < 8; j++) {
            uint8_t gbyte = (uint8_t)((gridv >> (8 * j)) & 0xFF);
            /* M13 item 3 (bonus): same sign-bit XOR already used by
             * dq32_iq2_xxs/dq32_iq3_xxs since M8 -- bitwise identical to
             * the compare+select*-1.f this replaces, not previously
             * applied here. */
            w[l4 * 8 + j] = ink_signed(dl * (float)gbyte, (sign_byte & tb.kmask_iq2xs[j]) != 0);
        }
    }
}

__device__ __forceinline__ void dq32_iq3_xxs(const uint8_t *rowp, uint32_t sg, float w[32], ink_tables tb) {
    const ink_block_iq3_xxs *x = (const ink_block_iq3_xxs *)rowp;
    uint32_t i = sg / 8, local_sg = sg % 8;
    const ink_block_iq3_xxs *b = &x[i];
    float d = __half2float(__ushort_as_half(b->d));
    const uint8_t *sas = b->qs + QK_K / 4 + 4 * local_sg;
    /* one wide 32-bit load instead of 4 byte loads (alignment-agnostic,
     * see ink_load_u32). */
    uint32_t aux32 = ink_load_u32(sas);
    float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
    const uint8_t *qsp = b->qs + 8 * local_sg;
#pragma unroll
    for (uint32_t l4 = 0; l4 < 4; l4++) {
        uint8_t signs = tb.ksigns_iq2xs[(aux32 >> (7 * l4)) & 127];
#pragma unroll
        for (uint32_t subsel = 0; subsel < 2; subsel++) {
            uint32_t gv = tb.iq3xxs_grid[qsp[2 * l4 + subsel]];
            uint32_t maskbase = subsel == 0 ? 0 : 4;
#pragma unroll
            for (uint32_t j = 0; j < 4; j++) {
                uint8_t gbyte = (uint8_t)((gv >> (8 * j)) & 0xFF);
                /* sign-bit XOR instead of compare+select*-1.f (bitwise
                 * identical, see ink_signed's comment). */
                w[l4 * 8 + subsel * 4 + j] = ink_signed(db * (float)gbyte,
                                                         (signs & tb.kmask_iq2xs[maskbase + j]) != 0);
            }
        }
    }
}

/* ===================== M11: int8/dp4a fast dot products =================
 * Port of llama.cpp's IQ2_XXS/IQ3_XXS vec_dot technique (MIT license,
 * ggml/src/ggml-cuda/vecdotq.cuh, vec_dot_iq2_xxs_q8_1 / vec_dot_iq3_xxs_q8_1
 * -- https://github.com/ggml-org/llama.cpp), adapted to this file's
 * per-subgroup dequant structure and using FLOAT scaling throughout
 * (deliberately NOT llama.cpp's `sumi*ls/8` integer-truncating scale
 * approximation -- there is no reason to accept that extra error here, so
 * the weight-subgroup scale `db` is computed exactly as the existing
 * float-path dq32_iq2_xxs/dq32_iq3_xxs do, and multiplied in as a float
 * after the integer dot product).
 *
 * The one piece actually ported byte-for-byte is the grid-byte layout and
 * the __dp4a accumulation: instead of extracting each of the 32 weight
 * bytes into a float register and doing 32 scalar FMAs against the
 * activation vector, the activation is pre-quantized to int8 (see
 * ink_kernel_quantize_act*, "reuse the same block-of-32 activation
 * quantization the weight side already groups by"), and each 4-byte grid
 * chunk is dot-producted against the matching 4 activation bytes with one
 * __dp4a() call (8 calls total per 32-wide subgroup, vs 32 float FMAs).
 *
 * Sign application: rather than port __vcmpne4/__vsub4 (SIMD-video
 * intrinsics whose exact semantics I could not verify without a compiler
 * here), ink_pack_signed4 does the equivalent per-byte conditional negate
 * with plain integer ops from the SAME already-verified sign table
 * (tb.ksigns_iq2xs / tb.kmask_iq2xs) the exact float path uses -- lower
 * risk, same result, and a much smaller diff to reason about by hand. */

/* Conditionally negate each of the 4 packed bytes in `grid4` (byte i =
 * bits 8*i, little-endian, matching ink_load_u32) based on bit
 * (maskbase+i) of `signs` -- same semantics as the float path's
 * `ink_signed(db*gbyte, (signs & kmask_iq2xs[maskbase+i]) != 0)`, just
 * producing a packed int32 of signed int8 lanes for __dp4a instead of 4
 * separate floats. */
__device__ __forceinline__ int32_t ink_pack_signed4(uint32_t grid4, uint8_t signs, uint8_t maskbase) {
    uint32_t out = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        uint8_t b = (uint8_t)(grid4 >> (8 * i));
        bool neg = (signs & (1u << (maskbase + i))) != 0;
        int8_t sb = neg ? (int8_t)(-(int)b) : (int8_t)b;
        out |= ((uint32_t)(uint8_t)sb) << (8 * i);
    }
    return (int32_t)out;
}

/* Returns sumi * db for this 32-wide subgroup (caller still multiplies by
 * the activation block's own float scale `dx`, same as `db*dx*sumi` in the
 * task's spec) -- NOT yet the final per-subgroup contribution.  `aq` points
 * at this subgroup's 32 already-quantized int8 activation values. */
__device__ __forceinline__ float ink_dp4a_dot32_iq2_xxs(const uint8_t *rowp, uint32_t sg,
                                                          const int8_t *aq, ink_tables tb) {
    const ink_block_iq2_xxs *x = (const ink_block_iq2_xxs *)rowp;
    uint32_t i = sg / 8, local_sg = sg % 8;
    const ink_block_iq2_xxs *b = &x[i];
    float d = __half2float(__ushort_as_half(b->d));
    const uint8_t *bp = (const uint8_t *)b->qs + 8 * local_sg;
    uint32_t a0 = ink_load_u32(bp);
    uint32_t a1 = ink_load_u32(bp + 4);
    float db = d * (0.5f + (float)(a1 >> 28)) * 0.25f;

    int sumi = 0;
#pragma unroll
    for (uint32_t lg = 0; lg < 4; lg++) {
        uint8_t byte_l = (uint8_t)(a0 >> (8 * lg));
        uint64_t gridv = tb.iq2xxs_grid[byte_l];
        uint8_t signs = tb.ksigns_iq2xs[(a1 >> (7 * lg)) & 127];
        int32_t sglo = ink_pack_signed4((uint32_t)gridv, signs, 0);
        int32_t sghi = ink_pack_signed4((uint32_t)(gridv >> 32), signs, 4);
        const uint8_t *aqp = (const uint8_t *)(aq + lg * 8);
        int32_t u0 = (int32_t)ink_load_u32(aqp);
        int32_t u1 = (int32_t)ink_load_u32(aqp + 4);
        sumi = __dp4a(sglo, u0, sumi);
        sumi = __dp4a(sghi, u1, sumi);
    }
    return (float)sumi * db;
}

__device__ __forceinline__ float ink_dp4a_dot32_iq3_xxs(const uint8_t *rowp, uint32_t sg,
                                                          const int8_t *aq, ink_tables tb) {
    const ink_block_iq3_xxs *x = (const ink_block_iq3_xxs *)rowp;
    uint32_t i = sg / 8, local_sg = sg % 8;
    const ink_block_iq3_xxs *b = &x[i];
    float d = __half2float(__ushort_as_half(b->d));
    const uint8_t *sas = b->qs + QK_K / 4 + 4 * local_sg;
    uint32_t aux32 = ink_load_u32(sas);
    float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
    const uint8_t *qsp = b->qs + 8 * local_sg;

    int sumi = 0;
#pragma unroll
    for (uint32_t l4 = 0; l4 < 4; l4++) {
        uint8_t signs = tb.ksigns_iq2xs[(aux32 >> (7 * l4)) & 127];
        uint32_t gv0 = tb.iq3xxs_grid[qsp[2 * l4 + 0]];
        uint32_t gv1 = tb.iq3xxs_grid[qsp[2 * l4 + 1]];
        int32_t sg0 = ink_pack_signed4(gv0, signs, 0);
        int32_t sg1 = ink_pack_signed4(gv1, signs, 4);
        const uint8_t *aqp = (const uint8_t *)(aq + l4 * 8);
        int32_t u0 = (int32_t)ink_load_u32(aqp);
        int32_t u1 = (int32_t)ink_load_u32(aqp + 4);
        sumi = __dp4a(sg0, u0, sumi);
        sumi = __dp4a(sg1, u1, sumi);
    }
    return (float)sumi * db;
}

__device__ __forceinline__ void dq32_iq4_xs(const uint8_t *rowp, uint32_t sg, float w[32], ink_tables tb) {
    const ink_block_iq4_xs *x = (const ink_block_iq4_xs *)rowp;
    uint32_t i = sg / 8, local_sg = sg % 8;
    const ink_block_iq4_xs *b = &x[i];
    float d = __half2float(__ushort_as_half(b->d));
    uint8_t scales_l_byte = b->scales_l[local_sg / 2];
    uint32_t ls = ((scales_l_byte >> (4 * (local_sg % 2))) & 0xf) |
                  (((b->scales_h >> (2 * local_sg)) & 3) << 4);
    float dl = d * ((float)ls - 32.0f);
    const uint8_t *qsp = b->qs + 16 * local_sg;
#pragma unroll
    for (uint32_t half = 0; half < 2; half++) {
#pragma unroll
        for (uint32_t j = 0; j < 16; j++) {
            uint8_t nib = half == 0 ? (qsp[j] & 0xf) : (qsp[j] >> 4);
            w[half * 16 + j] = dl * (float)tb.kvalues_iq4nl[nib];
        }
    }
}

/* Dispatch: dequantize subgroup `sg` of the row starting at `rowp` into
 * w[0..31].  See per-type comment above each dq32_* for the exact layout
 * (ported 1:1 from ds4_inkling.c's ink_dq_*() CPU reference). */
__device__ __forceinline__ void ink_dq_load32(uint32_t type, const uint8_t *rowp, uint32_t sg,
                                                float w[32], ink_tables tb) {
    switch (type) {
    case INK_T_F32:     dq32_f32(rowp, sg, w); return;
    case INK_T_Q8_0:    dq32_q8_0(rowp, sg, w); return;
    case INK_T_Q4_K:    dq32_q4_K(rowp, sg, w); return;
    case INK_T_Q5_K:    dq32_q5_K(rowp, sg, w); return;
    case INK_T_Q6_K:    dq32_q6_K(rowp, sg, w); return;
    case INK_T_IQ2_XXS: dq32_iq2_xxs(rowp, sg, w, tb); return;
    case INK_T_IQ2_S:   dq32_iq2_s(rowp, sg, w, tb); return;
    case INK_T_IQ3_XXS: dq32_iq3_xxs(rowp, sg, w, tb); return;
    case INK_T_IQ4_XS:  dq32_iq4_xs(rowp, sg, w, tb); return;
    default:
#pragma unroll
        for (int l = 0; l < 32; l++) w[l] = 0.0f; /* unreachable: host validates type before launch */
        return;
    }
}

/* ============================ bench instrumentation ======================
 * Per-stage cudaEvent timing, accumulated across the whole run, gated by
 * --bench / INK_BENCH=1.  Cheap when disabled (single int check, no event
 * create/record/sync, no stream sync -- the whole point of M7 is that
 * ordinary (non-bench) runs never block the host on a per-kernel basis).
 *
 * M7: split into two phases -- PREFILL (n_tok>1) and DECODE (n_tok==1).
 * ink_forward_gpu() sets g_bench_phase once per call based on n_tok; every
 * wrapper below records into g_bench_*[g_bench_phase].  NOTE: turning
 * --bench on reintroduces a host sync after every timed launch (events are
 * recorded on g_stream but read back via cudaEventSynchronize), because
 * that is the only way to attribute ms to an individual kernel -- so
 * --bench numbers include real kernel time only, NOT the async-pipeline
 * benefit M7 gives normal (non-bench) runs; a plain run's wall-clock
 * decode t/s is the number that reflects the sync-elision win. */

#define INK_BENCH_NTYPES 9
#define INK_BENCH_NPHASE 2
#define INK_BENCH_PREFILL 0
#define INK_BENCH_DECODE 1

static int g_bench = 0;
static int g_bench_phase = INK_BENCH_DECODE;
static cudaEvent_t g_bench_ev0, g_bench_ev1;
static int g_bench_ev_ready = 0;

typedef struct { double ms; double bytes; uint64_t calls; } ink_bench_stat;
static ink_bench_stat g_bench_matvec[INK_BENCH_NPHASE][INK_BENCH_NTYPES];
static ink_bench_stat g_bench_attn[INK_BENCH_NPHASE];
static ink_bench_stat g_bench_sconv[INK_BENCH_NPHASE];
static ink_bench_stat g_bench_group[INK_BENCH_NPHASE];

static const char *ink_bench_phase_name(int p) { return p == INK_BENCH_PREFILL ? "prefill" : "decode"; }

static int ink_bench_type_idx(uint32_t type) {
    switch (type) {
    case INK_T_F32: return 0;
    case INK_T_Q8_0: return 1;
    case INK_T_Q4_K: return 2;
    case INK_T_Q5_K: return 3;
    case INK_T_Q6_K: return 4;
    case INK_T_IQ2_XXS: return 5;
    case INK_T_IQ2_S: return 6;
    case INK_T_IQ3_XXS: return 7;
    case INK_T_IQ4_XS: return 8;
    default: return -1;
    }
}

static const char *ink_bench_type_name(int idx) {
    static const char *names[INK_BENCH_NTYPES] = {
        "F32", "Q8_0", "Q4_K", "Q5_K", "Q6_K", "IQ2_XXS", "IQ2_S", "IQ3_XXS", "IQ4_XS"
    };
    return (idx >= 0 && idx < INK_BENCH_NTYPES) ? names[idx] : "?";
}

static void ink_bench_ensure_events(void) {
    if (g_bench_ev_ready) return;
    CUDA_CHECK(cudaEventCreate(&g_bench_ev0));
    CUDA_CHECK(cudaEventCreate(&g_bench_ev1));
    g_bench_ev_ready = 1;
}

static inline void ink_bench_begin(void) {
    if (!g_bench) return;
    ink_bench_ensure_events();
    CUDA_CHECK(cudaEventRecord(g_bench_ev0, g_stream));
}

static inline void ink_bench_end(ink_bench_stat *st, double bytes) {
    if (!g_bench) return;
    CUDA_CHECK(cudaEventRecord(g_bench_ev1, g_stream));
    CUDA_CHECK(cudaEventSynchronize(g_bench_ev1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, g_bench_ev0, g_bench_ev1));
    st->ms += ms;
    st->bytes += bytes;
    st->calls += 1;
}

static void ink_bench_report_phase(int p) {
    fprintf(stderr, "--- phase=%s (ms includes launch overhead only where --bench forces syncs) ---\n",
            ink_bench_phase_name(p));
    for (int i = 0; i < INK_BENCH_NTYPES; i++) {
        ink_bench_stat *s = &g_bench_matvec[p][i];
        if (s->calls == 0) continue;
        double gbps = s->ms > 0 ? (s->bytes / 1.0e9) / (s->ms / 1000.0) : 0.0;
        fprintf(stderr, "matvec[%-8s] calls=%-8llu total_ms=%-10.1f bytes=%.3fGiB  %.2f GB/s\n",
                ink_bench_type_name(i), (unsigned long long)s->calls, s->ms,
                s->bytes / 1073741824.0, gbps);
    }
    {
        ink_bench_stat *s = &g_bench_attn[p];
        double gbps = s->ms > 0 ? (s->bytes / 1.0e9) / (s->ms / 1000.0) : 0.0;
        fprintf(stderr, "attention          calls=%-8llu total_ms=%-10.1f bytes=%.3fGiB  %.2f GB/s\n",
                (unsigned long long)s->calls, s->ms, s->bytes / 1073741824.0, gbps);
    }
    {
        ink_bench_stat *s = &g_bench_sconv[p];
        double gbps = s->ms > 0 ? (s->bytes / 1.0e9) / (s->ms / 1000.0) : 0.0;
        fprintf(stderr, "sconv               calls=%-8llu total_ms=%-10.1f bytes=%.3fGiB  %.2f GB/s\n",
                (unsigned long long)s->calls, s->ms, s->bytes / 1073741824.0, gbps);
    }
    {
        ink_bench_stat *s = &g_bench_group[p];
        double gbps = s->ms > 0 ? (s->bytes / 1.0e9) / (s->ms / 1000.0) : 0.0;
        fprintf(stderr, "group-matvec        calls=%-8llu total_ms=%-10.1f bytes=%.3fGiB  %.2f GB/s\n",
                (unsigned long long)s->calls, s->ms, s->bytes / 1073741824.0, gbps);
    }
}

static void ink_bench_report(void) {
    if (!g_bench) return;
    fprintf(stderr, "\n=== ds4-inkling-cuda --bench report ===\n");
    ink_bench_report_phase(INK_BENCH_PREFILL);
    ink_bench_report_phase(INK_BENCH_DECODE);
}

/* ======================= warp-per-row matvec/matmat kernels =============
 * One WARP (32 lanes) per output row, blockDim.x == 256 (8 warps/block),
 * grid-stride over rows.  Lane L handles subgroups L, L+32, L+64, ...
 * (whole 32-element quant subgroups, not individual elements): per
 * subgroup it dequantizes 32 weights into registers (ink_dq_load32, see
 * above) and forms a partial dot product, then the warp reduces via
 * __shfl_down_sync and lane 0 writes the row's output. */

#define INK_CUDA_MAX_TOK 128
#define INK_MATVEC_SHARED_MAX 4096   /* floats; larger `in` reads X from global */
#define INK_MATMAT_UNROLL 8          /* token accumulators per pass */

/* X may be `sx` (block-shared, cooperatively preloaded -- used by
 * ink_kernel_matvec/ink_kernel_matvec_group when in <= shared cap) or the
 * raw global/unified pointer (sx == NULL, `in` too large to cache). */
__device__ __forceinline__ void ink_matvec_row_warp(uint32_t type, const uint8_t *rowbase,
                                                      uint64_t in, uint64_t out,
                                                      const float *X, float *Y,
                                                      ink_tables tb, const float *sx) {
    uint32_t tid = threadIdx.x;
    uint32_t warp_id = tid / 32, lane = tid % 32;
    uint32_t nwarp_total = (blockDim.x / 32) * gridDim.x;
    uint64_t warp_global = (uint64_t)blockIdx.x * (blockDim.x / 32) + warp_id;

    size_t be = ink_dev_block_elems(type);
    size_t bb = ink_dev_block_bytes(type);
    uint64_t rowstride_blocks = in / be;
    uint32_t nsub = (uint32_t)(in / 32);
    const float *xsrc = sx ? sx : X;

    for (uint64_t row = warp_global; row < out; row += nwarp_total) {
        const uint8_t *rowp = rowbase + (size_t)row * rowstride_blocks * bb;
        /* M13 item 1: 2-way ILP.  The original loop was a single
         * accumulator with a dependent dequant->dot->acc+= chain per
         * subgroup, which serializes issue of the NEXT subgroup's dequant
         * ALU behind THIS subgroup's `acc +=` even though they don't
         * actually depend on each other.  Splitting the lane's own
         * stride-32 subgroup sequence into two independent accumulators
         * (acc0 takes subgroups sg, sg+64, sg+128, ...; acc1 takes
         * sg+32, sg+96, sg+160, ...) lets the compiler/scheduler overlap
         * subgroup B's dequant with subgroup A's FMA chain, since acc0 and
         * acc1 have no dependency on each other until the final merge.
         *
         * SUMMATION ORDER, stated explicitly: the per-lane partial sum is
         * no longer a single left-to-right accumulation over ALL of the
         * lane's subgroups in ascending sg order.  It is now two
         * independent left-to-right accumulations -- one over the
         * even-position subgroups of the lane's sequence, one over the
         * odd-position ones -- added together in one fixed final step
         * (acc0 + acc1) before the (unchanged) warp-shuffle reduction
         * tree.  This is a bounded, deterministic reassociation of the
         * SAME set of terms (same 32-wide dequant per subgroup, same
         * float4 loads, same per-element products) -- not a new
         * approximation and not run-to-run nondeterministic (no atomics,
         * no data-dependent branching in the split) -- of the same kind
         * already accepted for the warp-level reduction and the float4
         * grouping within one subgroup's 32-wide dot. */
        float acc0 = 0.0f, acc1 = 0.0f;
        uint32_t sg = lane;
        for (; sg + 32 < nsub; sg += 64) {
            float w0[32], w1[32];
            ink_dq_load32(type, rowp, sg, w0, tb);
            ink_dq_load32(type, rowp, sg + 32, w1, tb);
            /* see the (removed) single-chain comment above for the
             * alignment argument -- unchanged, applies identically to
             * both offsets since sg and sg+32 are both multiples of 32. */
            const float4 *xp4a = reinterpret_cast<const float4 *>(xsrc + (size_t)sg * 32);
            const float4 *xp4b = reinterpret_cast<const float4 *>(xsrc + (size_t)(sg + 32) * 32);
#pragma unroll
            for (int l4 = 0; l4 < 8; l4++) {
                float4 xva = xp4a[l4];
                float4 xvb = xp4b[l4];
                acc0 += w0[l4 * 4 + 0] * xva.x + w0[l4 * 4 + 1] * xva.y
                      + w0[l4 * 4 + 2] * xva.z + w0[l4 * 4 + 3] * xva.w;
                acc1 += w1[l4 * 4 + 0] * xvb.x + w1[l4 * 4 + 1] * xvb.y
                      + w1[l4 * 4 + 2] * xvb.z + w1[l4 * 4 + 3] * xvb.w;
            }
        }
        /* leftover: at most one more subgroup (nsub/32 parity for this
         * lane) -- folded into acc0 by fixed convention. */
        for (; sg < nsub; sg += 32) {
            float w[32];
            ink_dq_load32(type, rowp, sg, w, tb);
            const float4 *xp4 = reinterpret_cast<const float4 *>(xsrc + (size_t)sg * 32);
#pragma unroll
            for (int l4 = 0; l4 < 8; l4++) {
                float4 xv = xp4[l4];
                acc0 += w[l4 * 4 + 0] * xv.x + w[l4 * 4 + 1] * xv.y
                      + w[l4 * 4 + 2] * xv.z + w[l4 * 4 + 3] * xv.w;
            }
        }
        float acc = acc0 + acc1;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) Y[row] = acc;
    }
}

__global__ void ink_kernel_matvec(uint32_t type, const uint8_t *rowbase, uint64_t in, uint64_t out,
                                   const float *X, float *Y, ink_tables tb) {
    __shared__ __align__(16) float sx[INK_MATVEC_SHARED_MAX];
    const float *sxp = NULL;
    if (in <= INK_MATVEC_SHARED_MAX) {
        for (uint64_t i = threadIdx.x; i < in; i += blockDim.x) sx[i] = X[i];
        __syncthreads();
        sxp = sx;
    }
    ink_matvec_row_warp(type, rowbase, in, out, X, Y, tb, sxp);
}

/* Batched prefill matmat: X is [n_tok][in] (token-major), read straight
 * from global/unified memory via __ldg (no per-block shared cache -- n_tok
 * rows of `in` floats would typically blow the 4096-float cap).  Each lane
 * dequantizes a subgroup ONCE per pass of up to INK_MATMAT_UNROLL tokens
 * and reuses those registers for all tokens in that pass, bounding
 * register pressure while still amortizing dequant cost across up to 8
 * tokens; passes repeat for n_tok > INK_MATMAT_UNROLL. */
__global__ void ink_kernel_matmat(uint32_t type, const uint8_t *rowbase, uint64_t in, uint64_t out,
                                   const float *X, float *Y, uint32_t n_tok, ink_tables tb) {
    uint32_t tid = threadIdx.x;
    uint32_t warp_id = tid / 32, lane = tid % 32;
    uint32_t nwarp_total = (blockDim.x / 32) * gridDim.x;
    uint64_t warp_global = (uint64_t)blockIdx.x * (blockDim.x / 32) + warp_id;

    size_t be = ink_dev_block_elems(type);
    size_t bb = ink_dev_block_bytes(type);
    uint64_t rowstride_blocks = in / be;
    uint32_t nsub = (uint32_t)(in / 32);

    for (uint64_t row = warp_global; row < out; row += nwarp_total) {
        const uint8_t *rowp = rowbase + (size_t)row * rowstride_blocks * bb;
        for (uint32_t tbase = 0; tbase < n_tok; tbase += INK_MATMAT_UNROLL) {
            uint32_t tn = n_tok - tbase < INK_MATMAT_UNROLL ? n_tok - tbase : INK_MATMAT_UNROLL;
            float acc[INK_MATMAT_UNROLL];
#pragma unroll
            for (uint32_t k = 0; k < INK_MATMAT_UNROLL; k++) acc[k] = 0.0f;

            for (uint32_t sg = lane; sg < nsub; sg += 32) {
                float w[32];
                ink_dq_load32(type, rowp, sg, w, tb);
                for (uint32_t k = 0; k < tn; k++) {
                    const float *xp = X + (size_t)(tbase + k) * in + (size_t)sg * 32;
                    float dot = 0.0f;
#pragma unroll
                    for (int l = 0; l < 32; l++) dot += w[l] * __ldg(&xp[l]);
                    acc[k] += dot;
                }
            }
            for (uint32_t k = 0; k < tn; k++) {
                float v = acc[k];
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
                if (lane == 0) Y[(size_t)(tbase + k) * out + row] = v;
            }
        }
    }
}

/* ---- MoE grouped matvec: one launch runs n_group experts' gate/up/down
 * matvecs (same tensor type/shape, different row base and possibly
 * different x per group -- gate_exps/up_exps share one x per token, but
 * down_exps needs each group's OWN silu(gate)*up input, so the kernel
 * takes a per-group X array rather than a single shared x; callers that
 * do want to share one x across all groups just pass the same pointer
 * n_group times, which costs nothing extra).  blockIdx.y selects the
 * group, blockIdx.x/gridDim.x grid-strides over rows within it, exactly
 * like ink_kernel_matvec. */
#define INK_GROUP_MAX 8
typedef struct { const uint8_t *p[INK_GROUP_MAX]; } ink_ptr8;
typedef struct { const float *p[INK_GROUP_MAX]; } ink_fptr8;
typedef struct { float *p[INK_GROUP_MAX]; } ink_fptr8_mut;

__global__ void ink_kernel_matvec_group(uint32_t type, ink_ptr8 bases, uint64_t in, uint64_t out,
                                         ink_fptr8 xs, ink_fptr8_mut ys, uint32_t n_group, ink_tables tb) {
    uint32_t g = blockIdx.y;
    if (g >= n_group) return;
    const uint8_t *rowbase = bases.p[g];
    const float *X = xs.p[g];
    float *Y = ys.p[g];

    __shared__ __align__(16) float sx[INK_MATVEC_SHARED_MAX];
    const float *sxp = NULL;
    if (in <= INK_MATVEC_SHARED_MAX) {
        for (uint64_t i = threadIdx.x; i < in; i += blockDim.x) sx[i] = X[i];
        __syncthreads();
        sxp = sx;
    }
    ink_matvec_row_warp(type, rowbase, in, out, X, Y, tb, sxp);
}

/* ===================== M11: activation int8 quantizer ====================
 * Quantizes an activation vector into blocks of 32 (int8 qs + fp32 scale
 * d = max|x|/127 per block), matching the weight side's 32-wide subgroup
 * granularity exactly so ink_dp4a_dot32_* can dot a weight subgroup
 * against the matching activation block with one scale multiply.  Grouped
 * variant quantizes up to INK_GROUP_MAX activation vectors (one per
 * blockIdx.y) in a single launch -- called ONCE per ink_cuda_matvec_group()
 * call, not once per row, and the result is reused by every row/warp of
 * every group in that launch (this is the "quantize once, reuse across all
 * 6 expert groups / all rows" the fast path depends on for its win). */
__global__ void ink_kernel_quantize_act_group(ink_fptr8 xs, uint32_t nsub, int8_t *qs_out, float *d_out,
                                               uint32_t n_group) {
    uint32_t g = blockIdx.y;
    if (g >= n_group) return;
    const float *x = xs.p[g];
    int8_t *qsg = qs_out + (size_t)g * nsub * 32;
    float *dg = d_out + (size_t)g * nsub;
    for (uint32_t sg = threadIdx.x; sg < nsub; sg += blockDim.x) {
        const float *xp = x + (size_t)sg * 32;
        float amax = 0.0f;
#pragma unroll
        for (int l = 0; l < 32; l++) amax = fmaxf(amax, fabsf(xp[l]));
        float scale = amax * (1.0f / 127.0f);
        float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
        int8_t *qp = qsg + (size_t)sg * 32;
#pragma unroll
        for (int l = 0; l < 32; l++) qp[l] = (int8_t)lrintf(xp[l] * inv);
        dg[sg] = scale;
    }
}

static int8_t *g_actq_scratch = NULL;
static float *g_actd_scratch = NULL;
static size_t g_actq_cap = 0, g_actd_cap = 0;

static void ink_cuda_ensure_actq_scratch(size_t qs_need, size_t d_need) {
    if (qs_need > g_actq_cap) {
        if (g_actq_scratch) CUDA_CHECK(cudaFree(g_actq_scratch));
        CUDA_CHECK(cudaMalloc(&g_actq_scratch, qs_need));
        g_actq_cap = qs_need;
    }
    if (d_need > g_actd_cap) {
        if (g_actd_scratch) CUDA_CHECK(cudaFree(g_actd_scratch));
        CUDA_CHECK(cudaMalloc(&g_actd_scratch, d_need * sizeof(float)));
        g_actd_cap = d_need;
    }
}

/* ---- fast (dp4a) grouped matvec: same shape/contract as
 * ink_kernel_matvec_group, but for IQ2_XXS/IQ3_XXS rows dots against the
 * pre-quantized int8 activation via __dp4a instead of the float path.
 * Non-eligible types (Q4_K/Q5_K/Q6_K/IQ2_S/IQ4_XS/Q8_0/F32) fall back to
 * the exact float dq32_* path unchanged within the SAME launch/kernel --
 * this kernel is only ever selected when the whole group's tensor type is
 * dp4a-eligible (see ink_cuda_matvec_group), so in practice the fallback
 * branch is dead code for IQ2_XXS/IQ3_XXS launches, but keeping it here
 * (rather than a separate kernel) means one code path handles "is this
 * type eligible" instead of duplicating the whole row/warp loop. */
__device__ __forceinline__ void ink_matvec_row_warp_fast(uint32_t type, const uint8_t *rowbase,
                                                           uint64_t in, uint64_t out,
                                                           const float *X, float *Y, ink_tables tb,
                                                           const float *sx,
                                                           const int8_t *aq, const float *ad) {
    uint32_t tid = threadIdx.x;
    uint32_t warp_id = tid / 32, lane = tid % 32;
    uint32_t nwarp_total = (blockDim.x / 32) * gridDim.x;
    uint64_t warp_global = (uint64_t)blockIdx.x * (blockDim.x / 32) + warp_id;

    size_t be = ink_dev_block_elems(type);
    size_t bb = ink_dev_block_bytes(type);
    uint64_t rowstride_blocks = in / be;
    uint32_t nsub = (uint32_t)(in / 32);
    const float *xsrc = sx ? sx : X;
    bool use_dp4a = (type == INK_T_IQ2_XXS || type == INK_T_IQ3_XXS);

    for (uint64_t row = warp_global; row < out; row += nwarp_total) {
        const uint8_t *rowp = rowbase + (size_t)row * rowstride_blocks * bb;
        float acc = 0.0f;
        for (uint32_t sg = lane; sg < nsub; sg += 32) {
            if (use_dp4a) {
                const int8_t *aqp = aq + (size_t)sg * 32;
                float dx = ad[sg];
                float part = (type == INK_T_IQ2_XXS)
                    ? ink_dp4a_dot32_iq2_xxs(rowp, sg, aqp, tb)
                    : ink_dp4a_dot32_iq3_xxs(rowp, sg, aqp, tb);
                acc += part * dx;
            } else {
                float w[32];
                ink_dq_load32(type, rowp, sg, w, tb);
                const float4 *xp4 = reinterpret_cast<const float4 *>(xsrc + (size_t)sg * 32);
#pragma unroll
                for (int l4 = 0; l4 < 8; l4++) {
                    float4 xv = xp4[l4];
                    acc += w[l4 * 4 + 0] * xv.x + w[l4 * 4 + 1] * xv.y
                         + w[l4 * 4 + 2] * xv.z + w[l4 * 4 + 3] * xv.w;
                }
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) Y[row] = acc;
    }
}

__global__ void ink_kernel_matvec_group_fast(uint32_t type, ink_ptr8 bases, uint64_t in, uint64_t out,
                                              ink_fptr8 xs, ink_fptr8_mut ys, uint32_t n_group,
                                              const int8_t *actq, const float *actd, uint32_t nsub_act,
                                              ink_tables tb) {
    uint32_t g = blockIdx.y;
    if (g >= n_group) return;
    const uint8_t *rowbase = bases.p[g];
    const float *X = xs.p[g];
    float *Y = ys.p[g];
    const int8_t *gqs = actq + (size_t)g * nsub_act * 32;
    const float *gd = actd + (size_t)g * nsub_act;

    /* M11 item 3: stage the codebooks this kernel actually dot-products
     * against in shared memory -- iq2xxs_grid (2KB) + iq3xxs_grid (1KB) +
     * ksigns_iq2xs (128B) + kmask_iq2xs (8B) = ~3.1KB, cheap next to the
     * 16KB X cache below.  Per-lane grid/sign indices are data-dependent
     * (divergent), so these would otherwise be serialized/uncoalesced
     * global loads; shared memory tolerates that pattern far better. */
    __shared__ uint64_t s_iq2xxs_grid[256];
    __shared__ uint32_t s_iq3xxs_grid[256];
    __shared__ uint8_t s_ksigns[128];
    __shared__ uint8_t s_kmask[8];
    ink_tables tbl = tb;
    bool use_dp4a = (type == INK_T_IQ2_XXS || type == INK_T_IQ3_XXS);
    if (use_dp4a) {
        for (uint32_t idx = threadIdx.x; idx < 256; idx += blockDim.x) {
            s_iq2xxs_grid[idx] = tb.iq2xxs_grid[idx];
            s_iq3xxs_grid[idx] = tb.iq3xxs_grid[idx];
        }
        for (uint32_t idx = threadIdx.x; idx < 128; idx += blockDim.x) s_ksigns[idx] = tb.ksigns_iq2xs[idx];
        if (threadIdx.x < 8) s_kmask[threadIdx.x] = tb.kmask_iq2xs[threadIdx.x];
        __syncthreads();
        tbl.iq2xxs_grid = s_iq2xxs_grid;
        tbl.iq3xxs_grid = s_iq3xxs_grid;
        tbl.ksigns_iq2xs = s_ksigns;
        tbl.kmask_iq2xs = s_kmask;
    }

    __shared__ __align__(16) float sx[INK_MATVEC_SHARED_MAX];
    const float *sxp = NULL;
    if (!use_dp4a && in <= INK_MATVEC_SHARED_MAX) {
        for (uint64_t idx = threadIdx.x; idx < in; idx += blockDim.x) sx[idx] = X[idx];
        __syncthreads();
        sxp = sx;
    }
    ink_matvec_row_warp_fast(type, rowbase, in, out, X, Y, tbl, sxp, gqs, gd);
}

static uint32_t ink_cuda_row_blocks(uint64_t out) {
    uint64_t blocks = (out + 7) / 8; /* 8 warps/block -> 8 rows/block/grid-pass */
    if (blocks < 1) blocks = 1;
    if (blocks > 4096) blocks = 4096;
    return (uint32_t)blocks;
}

static double ink_matvec_bytes(uint32_t type, uint64_t in, uint64_t out) {
    size_t be = ink_type_block_elems(type), bb = ink_type_block_bytes(type);
    return (double)out * ((double)in / (double)be) * (double)bb;
}

extern "C" void ink_cuda_matvec(const ink_tensor *t, const uint8_t *base,
                                 uint64_t in, uint64_t out, const float *x, float *y) {
    if (!g_tables_ready) ink_die("ink_cuda: ink_cuda_init() was not called");
    uint32_t blocks = ink_cuda_row_blocks(out);
    ink_bench_begin();
    ink_kernel_matvec<<<blocks, 256, 0, g_stream>>>(t->type, base, in, out, x, y, g_tables);
    CUDA_CHECK(cudaGetLastError());
    int bi = ink_bench_type_idx(t->type);
    if (bi >= 0) ink_bench_end(&g_bench_matvec[g_bench_phase][bi], ink_matvec_bytes(t->type, in, out));
}

extern "C" void ink_cuda_matmat(const ink_tensor *t, const uint8_t *base,
                                 uint64_t in, uint64_t out, uint32_t n_tok,
                                 const float *X, float *Y) {
    if (!g_tables_ready) ink_die("ink_cuda: ink_cuda_init() was not called");
    if (n_tok == 0) return;
    if (n_tok == 1) { ink_cuda_matvec(t, base, in, out, X, Y); return; }
    if (n_tok > INK_CUDA_MAX_TOK) ink_die("ink_cuda: n_tok exceeds GPU kernel's fixed max (128)");
    uint32_t blocks = ink_cuda_row_blocks(out);
    ink_bench_begin();
    ink_kernel_matmat<<<blocks, 256, 0, g_stream>>>(t->type, base, in, out, X, Y, n_tok, g_tables);
    CUDA_CHECK(cudaGetLastError());
    int bi = ink_bench_type_idx(t->type);
    double bytes = ink_matvec_bytes(t->type, in, out) * (((n_tok + INK_MATMAT_UNROLL - 1) / INK_MATMAT_UNROLL));
    if (bi >= 0) ink_bench_end(&g_bench_matvec[g_bench_phase][bi], bytes);
}

/* Host-side selection (top-k routing, qsort over the expert logits) stays
 * on the host in ink_forward_gpu -- it operates on <= a few hundred floats
 * per token and costs microseconds; only the (much larger) weight matvecs
 * themselves are worth moving/grouping on the GPU. */
extern "C" void ink_cuda_matvec_group(const ink_tensor *t, const uint8_t *const *bases,
                                       uint64_t in, uint64_t out,
                                       const float *const *xs, float *const *ys, uint32_t n_group) {
    if (!g_tables_ready) ink_die("ink_cuda: ink_cuda_init() was not called");
    if (n_group == 0) return;
    uint32_t blocks = ink_cuda_row_blocks(out);
    double total_bytes = 0.0;
    /* M11: IQ2_XXS/IQ3_XXS (decode's measured bottleneck) get the int8/dp4a
     * fast path by default; INK_EXACT_DEQUANT=1 forces the exact float
     * path everywhere (this is also what --selftest always uses,
     * regardless of the env var -- see ink_run_selftest). */
    bool fast = !g_exact_dequant &&
                ((t->type == INK_T_IQ2_XXS && g_fast_iq2) ||
                 (t->type == INK_T_IQ3_XXS && g_fast_iq3));
    uint32_t nsub = (uint32_t)(in / 32);
    ink_bench_begin();
    for (uint32_t g0 = 0; g0 < n_group; g0 += INK_GROUP_MAX) {
        uint32_t ng = n_group - g0 < INK_GROUP_MAX ? n_group - g0 : INK_GROUP_MAX;
        ink_ptr8 pb; ink_fptr8 px; ink_fptr8_mut py;
        for (uint32_t i = 0; i < ng; i++) { pb.p[i] = bases[g0 + i]; px.p[i] = xs[g0 + i]; py.p[i] = ys[g0 + i]; }
        for (uint32_t i = ng; i < INK_GROUP_MAX; i++) { pb.p[i] = NULL; px.p[i] = NULL; py.p[i] = NULL; }
        dim3 grid(blocks, ng);
        if (fast) {
            /* Quantize each of this chunk's (<=8) activation vectors ONCE,
             * reused by every row/warp of every group below -- not once
             * per row.  (Groups that happen to share the same x pointer,
             * e.g. gate_exps/up_exps's shared token vector, get requantized
             * redundantly per group here rather than deduplicated by
             * pointer identity; that redundant work is a few KB of int8
             * quantization, negligible next to the matvec itself, and
             * keeping the quantize step uniform per-group is simpler and
             * safer than pointer-aliasing detection.) */
            ink_cuda_ensure_actq_scratch((size_t)INK_GROUP_MAX * nsub * 32, (size_t)INK_GROUP_MAX * nsub);
            ink_kernel_quantize_act_group<<<dim3(1, ng), 256, 0, g_stream>>>(px, nsub, g_actq_scratch, g_actd_scratch, ng);
            CUDA_CHECK(cudaGetLastError());
            ink_kernel_matvec_group_fast<<<grid, 256, 0, g_stream>>>(t->type, pb, in, out, px, py, ng,
                                                                      g_actq_scratch, g_actd_scratch, nsub, g_tables);
        } else {
            ink_kernel_matvec_group<<<grid, 256, 0, g_stream>>>(t->type, pb, in, out, px, py, ng, g_tables);
        }
        CUDA_CHECK(cudaGetLastError());
        total_bytes += ink_matvec_bytes(t->type, in, out) * ng;
    }
    ink_bench_end(&g_bench_group[g_bench_phase], total_bytes);
}

/* ================================ attention =============================
 * One block per head, 128 threads (== head_dim).  Scores for the causal
 * window [j0, pos] are written into a persistent global scratch buffer
 * (grown on demand -- the window can be much larger than one block's
 * shared memory), then max/sum are found via shared-memory tree
 * reductions, and finally each thread (indexed by head-dim lane) walks
 * the window once more to accumulate its weighted-V output element.
 * This mirrors ink_forward_batch's per-token attention loop exactly
 * (same scores, same softmax, same weighted sum), just reordered for the
 * GPU; it is the simple v1 form, not a fused/flash kernel. */

__global__ void ink_kernel_attention(const float *q, const float *kc, const float *vc, const float *rel,
                                      uint32_t hd, uint32_t kvw_max, uint32_t pos, uint32_t j0,
                                      uint32_t rel_extent, uint32_t gqa, float inv_hd, uint32_t len,
                                      float *scratch, float *attn_out) {
    uint32_t h = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t hkv = h / gqa;
    const float *qh = q + (size_t)h * hd;
    const float *relh = rel + (size_t)h * rel_extent;
    float *sc = scratch + (size_t)h * len;

    __shared__ float sred[128];

    for (uint32_t jj = tid; jj < len; jj += 128) {
        uint32_t j = j0 + jj;
        const float *kj = kc + (size_t)j * kvw_max + (size_t)hkv * hd;
        float s = 0.0f;
        for (uint32_t i = 0; i < hd; i++) s += qh[i] * kj[i];
        s *= inv_hd;
        uint32_t d = pos - j;
        if (d < rel_extent) s += relh[d];
        sc[jj] = s;
    }
    __syncthreads();

    float lmax = -1e30f;
    for (uint32_t jj = tid; jj < len; jj += 128) lmax = fmaxf(lmax, sc[jj]);
    sred[tid] = lmax;
    __syncthreads();
    for (uint32_t stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) sred[tid] = fmaxf(sred[tid], sred[tid + stride]);
        __syncthreads();
    }
    float maxs = sred[0];
    __syncthreads();

    float lsum = 0.0f;
    for (uint32_t jj = tid; jj < len; jj += 128) {
        float ev = expf(sc[jj] - maxs);
        sc[jj] = ev;
        lsum += ev;
    }
    sred[tid] = lsum;
    __syncthreads();
    for (uint32_t stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) sred[tid] += sred[tid + stride];
        __syncthreads();
    }
    float inv_sum = 1.0f / sred[0];
    __syncthreads();

    /* tid doubles as the head-dim output lane (blockDim.x == hd == 128) */
    float acc = 0.0f;
    for (uint32_t jj = 0; jj < len; jj++) {
        uint32_t j = j0 + jj;
        const float *vj = vc + (size_t)j * kvw_max + (size_t)hkv * hd;
        acc += sc[jj] * inv_sum * vj[tid];
    }
    attn_out[(size_t)h * hd + tid] = acc;
}

static float *g_attn_scratch = NULL;
static size_t g_attn_scratch_cap = 0;

static void ink_cuda_ensure_scratch(size_t need_floats) {
    if (need_floats <= g_attn_scratch_cap) return;
    if (g_attn_scratch) CUDA_CHECK(cudaFree(g_attn_scratch));
    CUDA_CHECK(cudaMalloc(&g_attn_scratch, need_floats * sizeof(float)));
    g_attn_scratch_cap = need_floats;
}

/* q, kc, vc, rel, attn_out are all host pointers (pageable/unified access).
 * q: this token's [n_head][hd] query.  kc/vc: this layer's full kcache/
 * vcache base ([n_ctx][kvw_max]).  rel: precomputed [n_head][rel_extent]
 * bias for this token (computed on host, same as ink_forward_batch).
 * attn_out: this token's [n_head][hd] output. */
extern "C" void ink_cuda_attention(const float *q, const float *kc, const float *vc, const float *rel,
                                    uint32_t n_head, uint32_t n_head_kv, uint32_t hd, uint32_t kvw_max,
                                    uint32_t pos, uint32_t j0, uint32_t rel_extent, float *attn_out) {
    if (!g_tables_ready) ink_die("ink_cuda: ink_cuda_init() was not called");
    if (hd != 128) ink_die("ink_cuda_attention: this v1 kernel assumes head_dim == 128");
    uint32_t gqa = n_head / n_head_kv;
    uint32_t len = pos - j0 + 1;
    ink_cuda_ensure_scratch((size_t)n_head * len);
    float inv_hd = 1.0f / (float)hd;
    ink_bench_begin();
    ink_kernel_attention<<<n_head, 128, 0, g_stream>>>(q, kc, vc, rel, hd, kvw_max, pos, j0, rel_extent, gqa,
                                           inv_hd, len, g_attn_scratch, attn_out);
    CUDA_CHECK(cudaGetLastError());
    /* bytes touched: this token's K/V window across all kv-heads, read once */
    double bytes = (double)len * (double)n_head_kv * (double)hd * 2.0 * sizeof(float);
    ink_bench_end(&g_bench_attn[g_bench_phase], bytes);
}

/* ================================ shortconv ==============================
 * One thread per channel.  Exact port of ink_sconv(). */

__global__ void ink_kernel_sconv(const float *w, float *state, uint32_t C, uint32_t K, float *x) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    uint32_t d = K - 1;
    float acc = 0.0f;
    for (uint32_t j = 0; j < d; j++) acc += state[j * C + c] * w[c * K + j];
    acc += x[c] * w[c * K + d];
    for (uint32_t j = 0; j + 1 < d; j++) state[j * C + c] = state[(j + 1) * C + c];
    if (d > 0) state[(d - 1) * C + c] = x[c];
    x[c] = x[c] + acc;
}

extern "C" void ink_cuda_sconv(const ink_tensor *kernel, float *state, uint32_t C, uint32_t K, float *x) {
    if (!g_tables_ready) ink_die("ink_cuda: ink_cuda_init() was not called");
    const float *w = ink_f32(kernel); /* dies if not F32, same as CPU ink_sconv */
    uint32_t threads = 256;
    uint32_t blocks = (C + threads - 1) / threads;
    ink_bench_begin();
    ink_kernel_sconv<<<blocks, threads, 0, g_stream>>>(w, state, C, K, x);
    CUDA_CHECK(cudaGetLastError());
    double bytes = (double)C * (double)K * sizeof(float) * 2.0; /* weights + state r/w */
    ink_bench_end(&g_bench_sconv[g_bench_phase], bytes);
}

/* ============================ M7: fused small kernels =====================
 * rmsnorm/silu-mul/residual-add/scale/rel-projection/MoE-accumulate, moved
 * onto the GPU so a whole layer of ink_forward_gpu() can run without a host
 * round-trip between GPU ops (see ink_cuda_sync() note above).  Every
 * kernel here keeps the SAME formula and the SAME per-output-element
 * accumulation order as its ds4_inkling.c host counterpart (ink_rmsnorm,
 * ink_silu, the rel_proj loop, the MoE weighted-sum loop): elementwise
 * kernels (silu-mul, add, scale) are trivially order-independent (one
 * output per thread, no reduction).  ink_kernel_moe_accumulate and
 * ink_kernel_relproj both reduce over an OUTER loop (expert index / d_rel)
 * ascending for a fixed output element, bit-for-bit the same order as the
 * host loops they replace.  The one deliberate exception is the rmsnorm
 * sum-of-squares, which uses a block-wide tree reduction instead of the
 * host's sequential accumulation (parallelizing a scalar reduction cannot
 * preserve sequential FP order) -- this is the same class of reordering
 * the existing matvec/matmat warp-reduction kernels already introduce
 * relative to ink_matvec()'s per-element CPU sum, which is exactly what
 * ink_test_tensor()'s "4x reordering allowance" tolerance already exists
 * to absorb; flagged here for the record. */

static uint32_t ink_flat_blocks(uint64_t n, uint32_t threads) {
    uint64_t b = (n + threads - 1) / threads;
    if (b < 1) b = 1;
    if (b > 65535) b = 65535; /* grid-stride loop covers the remainder */
    return (uint32_t)b;
}

/* One block per vector; dst may alias src (in-place norm: all of src is
 * reduced into `local` before any element of dst is written). */
__global__ void ink_kernel_rmsnorm(float *dst, const float *src, const float *w, uint32_t n, float eps) {
    uint32_t v = blockIdx.x;
    const float *s = src + (size_t)v * n;
    float *d = dst + (size_t)v * n;
    __shared__ float sred[256];
    float local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) local += s[i] * s[i];
    sred[threadIdx.x] = local;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sred[threadIdx.x] += sred[threadIdx.x + stride];
        __syncthreads();
    }
    float scale = 1.0f / sqrtf(sred[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) d[i] = s[i] * scale * w[i];
}

extern "C" void ink_cuda_rmsnorm(float *dst, const float *src, const float *w, uint32_t n_vec, uint32_t n, float eps) {
    if (n_vec == 0) return;
    ink_kernel_rmsnorm<<<n_vec, 256, 0, g_stream>>>(dst, src, w, n, eps);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void ink_kernel_silu_mul(float *hg, const float *hu, uint64_t n) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        float g = hg[i];
        hg[i] = (g / (1.0f + expf(-g))) * hu[i];
    }
}

extern "C" void ink_cuda_silu_mul(float *hg, const float *hu, uint64_t n) {
    if (n == 0) return;
    ink_kernel_silu_mul<<<ink_flat_blocks(n, 256), 256, 0, g_stream>>>(hg, hu, n);
    CUDA_CHECK(cudaGetLastError());
}

/* hg[row][i] = silu(hg[row][i]) * hu[row][i] * gamma[row], where gamma[row]
 * is read from a strided array (gamma_base[row*gamma_stride+gamma_offset])
 * rather than assumed contiguous -- lets the caller point straight at a
 * slice of the per-token wv_all[] weight table (gamma_stride = nu+ns,
 * gamma_offset = nu+sx for the shexp path) without an extra host-side
 * gather loop. */
__global__ void ink_kernel_silu_mul_gamma(float *hg, const float *hu, const float *gamma_base,
                                           uint32_t gamma_stride, uint32_t gamma_offset,
                                           uint32_t row_len, uint64_t n_total) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n_total; i += (uint64_t)gridDim.x * blockDim.x) {
        uint32_t row = (uint32_t)(i / row_len);
        float gamma = gamma_base[(size_t)row * gamma_stride + gamma_offset];
        float g = hg[i];
        hg[i] = (g / (1.0f + expf(-g))) * hu[i] * gamma;
    }
}

extern "C" void ink_cuda_silu_mul_gamma(float *hg, const float *hu, const float *gamma_base,
                                         uint32_t gamma_stride, uint32_t gamma_offset,
                                         uint32_t row_len, uint32_t n_rows) {
    uint64_t n = (uint64_t)row_len * n_rows;
    if (n == 0) return;
    ink_kernel_silu_mul_gamma<<<ink_flat_blocks(n, 256), 256, 0, g_stream>>>(
        hg, hu, gamma_base, gamma_stride, gamma_offset, row_len, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void ink_kernel_add(float *dst, const float *src, uint64_t n) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x)
        dst[i] += src[i];
}

extern "C" void ink_cuda_add(float *dst, const float *src, uint64_t n) {
    if (n == 0) return;
    ink_kernel_add<<<ink_flat_blocks(n, 256), 256, 0, g_stream>>>(dst, src, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void ink_kernel_scale(float *dst, uint64_t n, float alpha) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x)
        dst[i] *= alpha;
}

extern "C" void ink_cuda_scale(float *dst, uint64_t n, float alpha) {
    if (n == 0) return;
    ink_kernel_scale<<<ink_flat_blocks(n, 256), 256, 0, g_stream>>>(dst, n, alpha);
    CUDA_CHECK(cudaGetLastError());
}

/* rel[h][e] = tau_t * sum_{d=0}^{d_rel-1} pw[d*rel_extent+e] * r[h*d_rel+d]
 * -- one block per head, d ascending for fixed (h,e), same order as the
 * host loop it replaces. */
__global__ void ink_kernel_relproj(const float *pw, const float *r, float *rel,
                                    uint32_t d_rel, uint32_t rel_extent, float tau_t) {
    uint32_t h = blockIdx.x;
    const float *rh = r + (size_t)h * d_rel;
    float *relh = rel + (size_t)h * rel_extent;
    for (uint32_t e = threadIdx.x; e < rel_extent; e += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t d = 0; d < d_rel; d++) acc += pw[(size_t)d * rel_extent + e] * rh[d];
        relh[e] = acc * tau_t;
    }
}

extern "C" void ink_cuda_relproj(const float *pw, const float *r, float *rel,
                                  uint32_t n_head, uint32_t d_rel, uint32_t rel_extent, float tau_t) {
    if (n_head == 0) return;
    uint32_t threads = rel_extent < 256 ? (rel_extent < 32 ? 32 : rel_extent) : 256;
    ink_kernel_relproj<<<n_head, threads, 0, g_stream>>>(pw, r, rel, d_rel, rel_extent, tau_t);
    CUDA_CHECK(cudaGetLastError());
}

/* ff_acc[e] = sum_{i=0}^{nu-1} weights[i] * proj_g[i*n_embd+e] -- i
 * ascending for fixed e, same order as the host "for i: for e: acc[e] +=
 * w*proj_i[e]" loop it replaces. */
__global__ void ink_kernel_moe_accumulate(float *ff_acc, const float *proj_g, const float *weights,
                                           uint32_t n_embd, uint32_t nu) {
    for (uint32_t e = blockIdx.x * blockDim.x + threadIdx.x; e < n_embd; e += gridDim.x * blockDim.x) {
        float acc = 0.0f;
        for (uint32_t i = 0; i < nu; i++) acc += weights[i] * proj_g[(size_t)i * n_embd + e];
        ff_acc[e] = acc;
    }
}

extern "C" void ink_cuda_moe_accumulate(float *ff_acc, const float *proj_g, const float *weights,
                                         uint32_t n_embd, uint32_t nu) {
    if (n_embd == 0 || nu == 0) return;
    ink_kernel_moe_accumulate<<<ink_flat_blocks(n_embd, 256), 256, 0, g_stream>>>(ff_acc, proj_g, weights, n_embd, nu);
    CUDA_CHECK(cudaGetLastError());
}

/* On-stream copy of this token's k/v short-conv output into the rolling
 * kv cache at `pos`.  Previously a host memcpy() -- replaced with an async
 * copy queued on g_stream so it stays correctly ordered against the sconv
 * kernel that just wrote kt/vt and the attention kernel that will read
 * kc/vc, without a host sync in between. cudaMemcpyDefault relies on
 * unified virtual addressing (both src and dst are coherently-mapped host
 * pointers here, same as everywhere else in this file). */
__global__ void ink_kernel_copy(float *dst, const float *src, uint64_t n) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x)
        dst[i] = src[i];
}

/* Stream-ordered copy KERNEL, not cudaMemcpyAsync: an async memcpy between
 * two PAGEABLE host pointers is executed synchronously by the calling
 * thread, i.e. potentially BEFORE the queued kernels that produce `src`
 * have run -- which is exactly the stale-KV corruption this replaced
 * memcpy caused (M7 parity break, ' Paris' -> 'es'). A kernel launch is
 * always ordered on g_stream. */
extern "C" void ink_cuda_kv_copy(float *dst, const float *src, uint64_t n_floats) {
    if (n_floats == 0) return;
    ink_kernel_copy<<<ink_flat_blocks(n_floats, 256), 256, 0, g_stream>>>(dst, src, n_floats);
    CUDA_CHECK(cudaGetLastError());
}

/* ============================== GPU forward ==============================
 * Mirrors ink_forward_batch() in ds4_inkling.c line for line: same control
 * flow, same math.  rmsnorm, the per-token rel-bias projection, and MoE
 * expert selection stay on the host (small / branchy); the big matmuls go
 * through ink_cuda_matmat/matvec, shortconvs through ink_cuda_sconv, and
 * the attention inner loop through ink_cuda_attention. */

/* Stack-array cap for the host-side pointer arrays ink_forward_gpu builds
 * per token before calling ink_cuda_matvec_group() (n_expert_used is
 * normally single digits; this is a generous ceiling, independent of the
 * device kernel's own INK_GROUP_MAX chunking, which ink_cuda_matvec_group
 * handles internally). */
#define INK_GROUP_HOST_MAX 64

typedef struct { float score; int idx; } ink_scored_g;

static int ink_scored_cmp_g(const void *a, const void *b) {
    float d = ((const ink_scored_g *)b)->score - ((const ink_scored_g *)a)->score;
    return d > 0 ? 1 : d < 0 ? -1 : 0;
}

/* M7: whole-forward launch/sync accounting (see report) --  every ink_cuda_*
 * call below is an async launch on g_stream; ink_cuda_sync() (host
 * blocking) appears at exactly two kinds of point: once per MoE layer
 * (host needs `logits` for qsort routing) and once at the very end if the
 * caller wants out_logits back.  Dense-only layers issue zero syncs. */
extern "C" void ink_forward_gpu(ink_model *m, const int *tokens, uint32_t n_tok,
                                 uint32_t pos0, float *out_logits) {
    const uint32_t n_embd = m->n_embd;
    const uint32_t n_head = m->n_head;
    const uint32_t hd = m->head_dim;
    const uint32_t K = m->conv_k;
    const uint32_t kvw_max = m->kvw_max;
    const uint32_t nT = n_tok;
    if (nT == 0) return;
    if (nT > INK_CUDA_MAX_TOK) ink_die("ink_forward_gpu: batch exceeds GPU kernel's fixed max (128 tokens)");

    g_bench_phase = nT > 1 ? INK_BENCH_PREFILL : INK_BENCH_DECODE;

    const size_t conv_per_layer = (size_t)(K - 1) * (2u * kvw_max + 2u * n_embd);
    const size_t off_conv_k = 0;
    const size_t off_conv_v = (size_t)(K - 1) * kvw_max;
    const size_t off_conv_attn = (size_t)(K - 1) * 2 * kvw_max;
    const size_t off_conv_mlp = off_conv_attn + (size_t)(K - 1) * n_embd;

    float *x = (float *)ink_malloc((size_t)nT * n_embd * sizeof(float));
    float *xn = (float *)ink_malloc((size_t)nT * n_embd * sizeof(float));
    float *q = (float *)ink_malloc((size_t)nT * n_head * hd * sizeof(float));
    float *kf = (float *)ink_malloc((size_t)nT * kvw_max * sizeof(float));
    float *vf = (float *)ink_malloc((size_t)nT * kvw_max * sizeof(float));
    float *r = (float *)ink_malloc((size_t)nT * n_head * m->d_rel * sizeof(float));
    float *rel = (float *)ink_malloc((size_t)n_head * m->rel_extent * sizeof(float));
    float *attn_out = (float *)ink_malloc((size_t)nT * n_head * hd * sizeof(float));
    float *proj_out = (float *)ink_malloc((size_t)nT * n_embd * sizeof(float));
    uint32_t big_ff = m->n_ff_dense > m->n_ff_exp ? m->n_ff_dense : m->n_ff_exp;
    float *hg = (float *)ink_malloc((size_t)nT * big_ff * sizeof(float));
    float *hu = (float *)ink_malloc((size_t)nT * big_ff * sizeof(float));
    float *ff_out = (float *)ink_malloc((size_t)nT * n_embd * sizeof(float));
    float *taus = (float *)ink_malloc(nT * sizeof(float));
    /* MoE per-layer temporaries, hoisted out of the layer loop so they can
     * be freed only after the final stream sync (async kernels read them). */
    const uint32_t nE_h = m->n_expert, nu_h = m->n_expert_used, ns_h = m->n_shexp;
    const uint32_t nf_h = m->n_ff_exp;
    float *logits = (float *)ink_malloc((size_t)nT * (nE_h + ns_h) * sizeof(float));
    float *wv_all = (float *)ink_malloc((size_t)nT * (nu_h + ns_h) * sizeof(float));
    int *sel_all = (int *)ink_malloc((size_t)nT * nu_h * sizeof(int));
    float *hg_g = (float *)ink_malloc((size_t)nu_h * nf_h * sizeof(float));
    float *hu_g = (float *)ink_malloc((size_t)nu_h * nf_h * sizeof(float));
    float *proj_g = (float *)ink_malloc((size_t)nu_h * n_embd * sizeof(float));

    /* Embedding row fetch stays host (table lookup against the mmap'd
     * GGUF, once per token); tau is pure host arithmetic.  tok_norm is a
     * single whole-buffer GPU rmsnorm covering all nT tokens at once (x is
     * allocated exactly nT*n_embd, so the nT row-vectors are contiguous). */
    for (uint32_t t = 0; t < nT; t++) {
        float *xt = x + (size_t)t * n_embd;
        ink_row_f32(m->tok_embd, m->tok_embd->data, (uint64_t)tokens[t], n_embd, xt);
        taus[t] = (m->log_n_floor > 0)
            ? 1.0f + m->log_alpha * logf(fmaxf((float)(pos0 + t + 1) / (float)m->log_n_floor, 1.0f))
            : 1.0f;
    }
    ink_cuda_rmsnorm(x, x, ink_f32(m->tok_norm), nT, n_embd, m->rms_eps);

    for (uint32_t il = 0; il < m->n_layer; il++) {
        /* M12: the fast path is enabled per layer (see ink_fast_layers_parse) */
        g_exact_dequant = ink_layer_is_fast((int)il) ? 0 : 1;
        ink_layer *l = &m->layers[il];
        const uint32_t n_head_kv = l->n_head_kv;
        const uint32_t kvw = n_head_kv * hd;
        const uint32_t rel_extent = l->is_swa ? m->rel_extent_swa : m->rel_extent;
        float *conv_l = m->conv + (size_t)il * conv_per_layer;
        float *kc = m->kcache + ((size_t)il * m->n_ctx) * kvw_max;
        float *vc = m->vcache + ((size_t)il * m->n_ctx) * kvw_max;

        /* ---- attention block ---- */
        ink_cuda_rmsnorm(xn, x, ink_f32(l->attn_norm), nT, n_embd, m->rms_eps);

        ink_cuda_matmat(l->wq, l->wq->data, n_embd, (uint64_t)n_head * hd, nT, xn, q);
        ink_cuda_matmat(l->wk, l->wk->data, n_embd, kvw, nT, xn, kf);
        ink_cuda_matmat(l->wv, l->wv->data, n_embd, kvw, nT, xn, vf);
        ink_cuda_matmat(l->wr, l->wr->data, n_embd, (uint64_t)n_head * m->d_rel, nT, xn, r);

        /* q is allocated exactly nT*n_head*hd (no per-token padding), so
         * every (t,h) head-vector is contiguous across the WHOLE batch --
         * one rmsnorm launch for all of it, instead of nT*n_head. kf/vf are
         * allocated at kvw_max stride per token (pre-existing convention,
         * unrelated to M7) so k_norm stays per-t below. */
        ink_cuda_rmsnorm(q, q, ink_f32(l->q_norm), nT * n_head, hd, m->rms_eps);

        for (uint32_t t = 0; t < nT; t++) {
            const uint32_t pos = pos0 + t;
            float *kt = kf + (size_t)t * kvw_max;
            float *vt = vf + (size_t)t * kvw_max;
            float *qt = q + (size_t)t * n_head * hd;

            ink_cuda_sconv(l->sc_k, conv_l + off_conv_k, kvw, K, kt);
            ink_cuda_sconv(l->sc_v, conv_l + off_conv_v, kvw, K, vt);

            ink_cuda_rmsnorm(kt, kt, ink_f32(l->k_norm), n_head_kv, hd, m->rms_eps);

            if (!l->is_swa && taus[t] != 1.0f) ink_cuda_scale(qt, (uint64_t)n_head * hd, taus[t]);

            ink_cuda_kv_copy(kc + (size_t)pos * kvw_max, kt, kvw);
            ink_cuda_kv_copy(vc + (size_t)pos * kvw_max, vt, kvw);
        }

        const float *pw = ink_f32(l->rel_proj);

        for (uint32_t t = 0; t < nT; t++) {
            const uint32_t pos = pos0 + t;
            const float *rt = r + (size_t)t * n_head * m->d_rel;
            const float tau_t = (!l->is_swa) ? taus[t] : 1.0f;

            ink_cuda_relproj(pw, rt, rel, n_head, m->d_rel, rel_extent, tau_t);

            uint32_t j0 = 0;
            if (l->is_swa && pos + 1 > m->n_swa) j0 = pos + 1 - m->n_swa;

            ink_cuda_attention(q + (size_t)t * n_head * hd, kc, vc, rel,
                                n_head, n_head_kv, hd, kvw_max, pos, j0, rel_extent,
                                attn_out + (size_t)t * n_head * hd);
        }

        ink_cuda_matmat(l->wo, l->wo->data, (uint64_t)n_head * hd, n_embd, nT, attn_out, proj_out);

        /* sc_attn carries state across t (must stay a sequential per-t
         * loop); the residual add doesn't, so it happens once, after, as a
         * single flat kernel over the whole [nT][n_embd] buffer -- correct
         * because it's issued strictly after all nT sconv launches in
         * program order on the same stream. */
        for (uint32_t t = 0; t < nT; t++) {
            float *pt = proj_out + (size_t)t * n_embd;
            ink_cuda_sconv(l->sc_attn, conv_l + off_conv_attn, n_embd, K, pt);
        }
        ink_cuda_add(x, proj_out, (uint64_t)nT * n_embd);

        /* ---- ffn block ---- */
        ink_cuda_rmsnorm(xn, x, ink_f32(l->ffn_norm), nT, n_embd, m->rms_eps);
        const float gscale = ink_f32(l->gscale)[0];

        if (il < m->n_dense) {
            uint32_t nf = m->n_ff_dense;
            ink_cuda_matmat(l->ffn_gate, l->ffn_gate->data, n_embd, nf, nT, xn, hg);
            ink_cuda_matmat(l->ffn_up, l->ffn_up->data, n_embd, nf, nT, xn, hu);
            ink_cuda_silu_mul(hg, hu, (uint64_t)nT * nf);
            ink_cuda_matmat(l->ffn_down, l->ffn_down->data, nf, n_embd, nT, hg, ff_out);
            ink_cuda_scale(ff_out, (uint64_t)nT * n_embd, gscale);
        } else {
            const uint32_t nE = m->n_expert, nu = m->n_expert_used, ns = m->n_shexp;
            const uint32_t nf = m->n_ff_exp;
            ink_cuda_matmat(l->gate_inp, l->gate_inp->data, n_embd, nE + ns, nT, xn, logits);
            const float *bias = ink_f32(l->probs_b);

            size_t g_rb = (n_embd / ink_type_block_elems(l->gate_exps->type)) * ink_type_block_bytes(l->gate_exps->type);
            size_t u_rb = (n_embd / ink_type_block_elems(l->up_exps->type)) * ink_type_block_bytes(l->up_exps->type);
            size_t d_rb = (nf / ink_type_block_elems(l->down_exps->type)) * ink_type_block_bytes(l->down_exps->type);
            size_t sg_rb = (n_embd / ink_type_block_elems(l->gate_shexp->type)) * ink_type_block_bytes(l->gate_shexp->type);
            size_t su_rb = (n_embd / ink_type_block_elems(l->up_shexp->type)) * ink_type_block_bytes(l->up_shexp->type);
            size_t sd_rb = (nf / ink_type_block_elems(l->down_shexp->type)) * ink_type_block_bytes(l->down_shexp->type);


            /* ---- SYNC (1 of 2 per-call sync points): host needs `logits`
             * for the top-k routing qsort below. ---- */
            ink_cuda_sync();

            for (uint32_t t = 0; t < nT; t++) {
                const float *lg = logits + (size_t)t * (nE + ns);
                ink_scored_g *ranked = (ink_scored_g *)ink_malloc(nE * sizeof(ink_scored_g));
                for (uint32_t e = 0; e < nE; e++) {
                    ranked[e].score = 1.0f / (1.0f + expf(-lg[e])) + bias[e];
                    ranked[e].idx = (int)e;
                }
                qsort(ranked, nE, sizeof(ink_scored_g), ink_scored_cmp_g);
                float *wv = wv_all + (size_t)t * (nu + ns);
                float wmax = -1e30f;
                for (uint32_t i = 0; i < nu; i++) {
                    sel_all[(size_t)t * nu + i] = ranked[i].idx;
                    wv[i] = ink_logsigmoid(lg[ranked[i].idx]);
                    if (wv[i] > wmax) wmax = wv[i];
                }
                for (uint32_t sx = 0; sx < ns; sx++) {
                    wv[nu + sx] = ink_logsigmoid(lg[nE + sx]);
                    if (wv[nu + sx] > wmax) wmax = wv[nu + sx];
                }
                float wsum = 0.0f;
                for (uint32_t i = 0; i < nu + ns; i++) { wv[i] = expf(wv[i] - wmax); wsum += wv[i]; }
                const float wscale = m->expert_weights_scale * gscale / wsum;
                for (uint32_t i = 0; i < nu + ns; i++) wv[i] *= wscale;
                free(ranked);
            }

            /* Grouped launch: gate_exps/up_exps/down_exps for all nu selected
             * experts of one token go out as ONE kernel launch each
             * (blockIdx.y = expert group) instead of nu sequential
             * ink_cuda_matvec() calls.  silu-mul and the weighted MoE
             * accumulate are single GPU kernels too (ink_cuda_silu_mul,
             * ink_cuda_moe_accumulate) -- 4 launches per token total
             * instead of 3*nu matvecs + 2 host loops. gate/up share the
             * token's xt; down needs each expert's own silu(gate)*up
             * hidden vector, so it gets a per-group X array too
             * (ink_cuda_matvec_group takes xs[group] rather than a single
             * shared x for exactly this reason). */
            if (nu > INK_GROUP_HOST_MAX)
                ink_die("ink_forward_gpu: n_expert_used exceeds host group-batch array cap");
            const uint8_t *gate_bases[INK_GROUP_HOST_MAX];
            const uint8_t *up_bases[INK_GROUP_HOST_MAX];
            const uint8_t *down_bases[INK_GROUP_HOST_MAX];
            const float *shared_x[INK_GROUP_HOST_MAX];
            const float *down_x[INK_GROUP_HOST_MAX];
            float *gate_ys[INK_GROUP_HOST_MAX];
            float *up_ys[INK_GROUP_HOST_MAX];
            float *down_ys[INK_GROUP_HOST_MAX];

            for (uint32_t t = 0; t < nT; t++) {
                const float *xt = xn + (size_t)t * n_embd;
                float *ot = ff_out + (size_t)t * n_embd;

                for (uint32_t i = 0; i < nu; i++) {
                    const uint32_t e = (uint32_t)sel_all[(size_t)t * nu + i];
                    gate_bases[i] = l->gate_exps->data + (size_t)e * nf * g_rb;
                    up_bases[i]   = l->up_exps->data + (size_t)e * nf * u_rb;
                    down_bases[i] = l->down_exps->data + (size_t)e * n_embd * d_rb;
                    shared_x[i] = xt;
                    gate_ys[i] = hg_g + (size_t)i * nf;
                    up_ys[i]   = hu_g + (size_t)i * nf;
                    down_ys[i] = proj_g + (size_t)i * n_embd;
                }
                ink_cuda_matvec_group(l->gate_exps, gate_bases, n_embd, nf, shared_x, gate_ys, nu);
                ink_cuda_matvec_group(l->up_exps,   up_bases,   n_embd, nf, shared_x, up_ys,   nu);
                ink_cuda_silu_mul(hg_g, hu_g, (uint64_t)nu * nf);
                for (uint32_t i = 0; i < nu; i++) down_x[i] = hg_g + (size_t)i * nf;
                ink_cuda_matvec_group(l->down_exps, down_bases, nf, n_embd, down_x, down_ys, nu);

                ink_cuda_moe_accumulate(ot, proj_g, wv_all + (size_t)t * (nu + ns), n_embd, nu);
            }

            for (uint32_t sx = 0; sx < ns; sx++) {
                ink_cuda_matmat(l->gate_shexp, l->gate_shexp->data + (size_t)sx * nf * sg_rb, n_embd, nf, nT, xn, hg);
                ink_cuda_matmat(l->up_shexp, l->up_shexp->data + (size_t)sx * nf * su_rb, n_embd, nf, nT, xn, hu);
                ink_cuda_silu_mul_gamma(hg, hu, wv_all, nu + ns, nu + sx, nf, nT);
                ink_cuda_matmat(l->down_shexp, l->down_shexp->data + (size_t)sx * n_embd * sd_rb, nf, n_embd, nT, hg, proj_out);
                ink_cuda_add(ff_out, proj_out, (uint64_t)nT * n_embd);
            }
            /* NOTE: logits/wv_all/sel_all/hg_g/hu_g/proj_g are allocated
             * once per forward call and freed after the final sync --
             * freeing here would race the still-queued shexp/down/
             * accumulate kernels that read them (this exact bug produced
             * heap corruption; see M7 notes). */
        }

        for (uint32_t t = 0; t < nT; t++) {
            float *ft = ff_out + (size_t)t * n_embd;
            ink_cuda_sconv(l->sc_mlp, conv_l + off_conv_mlp, n_embd, K, ft);
        }
        ink_cuda_add(x, ff_out, (uint64_t)nT * n_embd);
    }

    if (out_logits) {
        float *xl = x + (size_t)(nT - 1) * n_embd;
        ink_cuda_rmsnorm(xl, xl, ink_f32(m->out_norm), 1, n_embd, m->rms_eps);
        ink_cuda_scale(xl, n_embd, m->logit_scale);
        ink_cuda_matvec(m->output, m->output->data, n_embd, m->n_vocab, xl, out_logits);
        /* ---- SYNC (2 of 2): caller reads out_logits right after this
         * call returns (argmax / ink_logits_guard). ---- */
        ink_cuda_sync();
        if (m->n_vocab_unpadded > 0) {
            for (uint32_t i = m->n_vocab_unpadded; i < m->n_vocab; i++) out_logits[i] = -INFINITY;
        }
    }

    /* MANDATORY sync (not one of the "two per-call sync points" above --
     * this one is a correctness requirement, not a perf tradeoff): every
     * ink_cuda_* call in this function is an async launch against these
     * host-malloc'd temporaries (x, xn, q, kf, vf, r, rel, attn_out,
     * proj_out, hg, hu, ff_out), including the very last layer's residual
     * add and kv-cache copies.  free()'ing them while g_stream still has
     * in-flight work reading/writing them would be a use-after-free race
     * the moment the allocator hands the same address back out on the
     * next ink_forward_gpu() call.  When out_logits was requested the sync
     * above already covers this; when it's NULL (mid-prompt prefill
     * chunks) this is the only sync in the whole call. */
    ink_cuda_sync();

    free(logits); free(wv_all); free(sel_all);
    free(hg_g); free(hu_g); free(proj_g);

    free(x); free(xn); free(q); free(kf); free(vf); free(r); free(rel);
    free(attn_out); free(proj_out); free(hg); free(hu);
    free(ff_out); free(taus);
}

/* ================================ CLI ==================================== */

static uint32_t g_lcg;

static float ink_lcg_rand(void) {
    g_lcg = g_lcg * 1664525u + 1013904223u;
    return ((float)(g_lcg >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;
}

static void ink_fill_rand(float *x, uint64_t n, uint32_t seed) {
    g_lcg = seed;
    for (uint64_t i = 0; i < n; i++) x[i] = ink_lcg_rand();
}

static bool ink_test_tensor(const char *name, const ink_tensor *t, const uint8_t *base,
                             uint64_t in, uint64_t out) {
    float *x = (float *)ink_malloc(in * sizeof(float));
    ink_fill_rand(x, in, 0xC0FFEEu);
    float *ycpu = (float *)ink_malloc(out * sizeof(float));
    float *ygpu = (float *)ink_malloc(out * sizeof(float));

    ink_matvec(t, base, in, out, x, ycpu);
    ink_cuda_matvec(t, base, in, out, x, ygpu);
    ink_cuda_sync();   /* async wrappers: sync before host reads/frees */
    ink_cuda_sync(); /* M7: wrappers no longer sync internally -- read ygpu only after this */

    /* double-accumulation reference from the SAME dequantized rows: both
     * the fp32 CPU path and the GPU path are compared against it, so
     * accumulation-order noise is measured instead of guessed. */
    float *row = (float *)ink_malloc(in * sizeof(float));
    double err_cpu = 0.0, err_gpu = 0.0, maxabs = 0.0;
    for (uint64_t i = 0; i < out; i++) {
        ink_row_f32(t, base, i, in, row);
        double ref = 0.0;
        for (uint64_t k = 0; k < in; k++) ref += (double)row[k] * (double)x[k];
        double ec = fabs((double)ycpu[i] - ref);
        double eg = fabs((double)ygpu[i] - ref);
        double a = fabs((double)ygpu[i] - (double)ycpu[i]);
        if (ec > err_cpu) err_cpu = ec;
        if (eg > err_gpu) err_gpu = eg;
        if (a > maxabs) maxabs = a;
    }
    free(row);
    /* pass when the GPU is no farther from the exact dot than the fp32
     * CPU path is, modulo a 4x reordering allowance */
    bool pass = err_gpu <= fmax(4.0 * err_cpu, 1e-5);
    printf("%-24s type=%2u in=%-6llu out=%-6llu gpu_vs_cpu=%.3g cpu_vs_ref=%.3g gpu_vs_ref=%.3g %s\n",
           name, t->type, (unsigned long long)in, (unsigned long long)out,
           maxabs, err_cpu, err_gpu, pass ? "PASS" : "FAIL");

    free(x); free(ycpu); free(ygpu);
    return pass;
}

/* M11: fast-path (int8/dp4a) INFO check.  Runs the SAME 6-group probe as
 * the exact-path group-matvec check above, but with g_exact_dequant forced
 * to 0 (fast), and reports error vs the CPU f32 reference against a loose
 * bound (rel 5e-3) WITHOUT failing the selftest -- int8 activation
 * quantization is expected to move the answer by more than the tight
 * exact-path bound tolerates; this check exists to print the actual
 * numbers, not to gate on them.  No-op for tensor types the fast path
 * doesn't touch. */
static void ink_selftest_fast_check(const char *name, const ink_tensor *t, uint64_t in, uint64_t out) {
    if (t->type != INK_T_IQ2_XXS && t->type != INK_T_IQ3_XXS) return;
    const uint32_t NG = 6;
    size_t row_bytes = (in / ink_type_block_elems(t->type)) * ink_type_block_bytes(t->type);
    const uint8_t *bases[NG];
    const float *xs[NG];
    float *ys_g[NG];
    float *x1 = (float *)ink_malloc(in * sizeof(float));
    ink_fill_rand(x1, in, 0xFACE03u);
    float *yc = (float *)ink_malloc(out * sizeof(float));
    for (uint32_t g2 = 0; g2 < NG; g2++) {
        bases[g2] = t->data + (size_t)(g2 * 37 + 1) * out * row_bytes;
        xs[g2] = x1;
        ys_g[g2] = (float *)ink_malloc(out * sizeof(float));
    }

    int prev = g_exact_dequant;
    g_exact_dequant = 0;
    ink_cuda_matvec_group(t, bases, in, out, xs, ys_g, NG);
    ink_cuda_sync();
    g_exact_dequant = prev;

    double maxabs = 0.0, maxrel = 0.0, sumabs = 0.0, sumsq = 0.0, ref_absmax = 0.0;
    uint64_t n = 0;
    for (uint32_t g2 = 0; g2 < NG; g2++) {
        ink_matvec(t, bases[g2], in, out, x1, yc); /* CPU f32 exact reference */
        for (uint64_t i = 0; i < out; i++) {
            double ref = (double)yc[i], got = (double)ys_g[g2][i];
            double a = fabs(got - ref);
            double rel = a / fmax(fabs(ref), 1e-6);
            if (a > maxabs) maxabs = a;
            if (rel > maxrel) maxrel = rel;
            if (fabs(ref) > ref_absmax) ref_absmax = fabs(ref);
            sumabs += a; sumsq += ref * ref; n++;
        }
        free(ys_g[g2]);
    }
    double ref_rms = n ? sqrt(sumsq / (double)n) : 0.0;
    bool loose_ok = maxrel <= 5e-3;
    printf("group-matvec(%s,6,FAST-dp4a) ref_rms=%.4g ref_absmax=%.4g "
           "err/ref_rms=%.3g\n", name, ref_rms, ref_absmax,
           ref_rms > 0 ? (sumabs / (double)(n ? n : 1)) / ref_rms : 0.0);
    printf("group-matvec(%s,6,FAST-dp4a) maxabsdiff=%.3g maxreldiff=%.3g meanabsdiff=%.3g "
           "%s (INFO only -- loose bound 5e-3 rel, does not affect PASS/FAIL)\n",
           name, maxabs, maxrel, n ? sumabs / (double)n : 0.0, loose_ok ? "within-loose-bound" : "OUTSIDE-loose-bound");
    free(x1); free(yc);
}

static int ink_run_selftest(const char *model_path, int layer) {
    ink_model m;
    ink_model_open(&m, model_path, 8);
    if (layer < 0 || (uint32_t)layer >= m.n_layer) ink_die("--selftest LAYER out of range");
    ink_layer *l = &m.layers[layer];

    /* Every exact-path check below (ink_test_tensor, matmat/group-matvec
     * comparisons) must keep passing at its original tight bound
     * regardless of the fast path's env-var default -- selftest is the
     * correctness oracle, so force the exact float path for its duration
     * and restore whatever was configured on the way out. */
    int saved_exact = g_exact_dequant;
    g_exact_dequant = 1;

    bool all_pass = true;
    uint64_t n_embd = m.n_embd;
    uint64_t hn = (uint64_t)m.n_head * m.head_dim;
    uint64_t kvw = (uint64_t)l->n_head_kv * m.head_dim;

    all_pass &= ink_test_tensor("wq", l->wq, l->wq->data, n_embd, hn);
    all_pass &= ink_test_tensor("wk", l->wk, l->wk->data, n_embd, kvw);
    all_pass &= ink_test_tensor("wv", l->wv, l->wv->data, n_embd, kvw);
    all_pass &= ink_test_tensor("wr", l->wr, l->wr->data, n_embd, (uint64_t)m.n_head * m.d_rel);
    all_pass &= ink_test_tensor("wo", l->wo, l->wo->data, hn, n_embd);

    if ((uint32_t)layer < m.n_dense) {
        all_pass &= ink_test_tensor("ffn_gate", l->ffn_gate, l->ffn_gate->data, n_embd, m.n_ff_dense);
        all_pass &= ink_test_tensor("ffn_up", l->ffn_up, l->ffn_up->data, n_embd, m.n_ff_dense);
        all_pass &= ink_test_tensor("ffn_down", l->ffn_down, l->ffn_down->data, m.n_ff_dense, n_embd);
    } else {
        uint64_t nE = m.n_expert, nf = m.n_ff_exp;
        all_pass &= ink_test_tensor("gate_inp", l->gate_inp, l->gate_inp->data, n_embd, nE + m.n_shexp);

        size_t g_rb = (n_embd / ink_type_block_elems(l->gate_exps->type)) * ink_type_block_bytes(l->gate_exps->type);
        size_t u_rb = (n_embd / ink_type_block_elems(l->up_exps->type)) * ink_type_block_bytes(l->up_exps->type);
        size_t d_rb = (nf / ink_type_block_elems(l->down_exps->type)) * ink_type_block_bytes(l->down_exps->type);

        uint64_t probe_experts[2] = { 0, 7 };
        for (int pi = 0; pi < 2; pi++) {
            uint64_t e = probe_experts[pi];
            if (e >= nE) continue;
            char nm[64];
            snprintf(nm, sizeof(nm), "gate_exps[%llu]", (unsigned long long)e);
            all_pass &= ink_test_tensor(nm, l->gate_exps, l->gate_exps->data + (size_t)e * nf * g_rb, n_embd, nf);
            snprintf(nm, sizeof(nm), "up_exps[%llu]", (unsigned long long)e);
            all_pass &= ink_test_tensor(nm, l->up_exps, l->up_exps->data + (size_t)e * nf * u_rb, n_embd, nf);
            snprintf(nm, sizeof(nm), "down_exps[%llu]", (unsigned long long)e);
            all_pass &= ink_test_tensor(nm, l->down_exps, l->down_exps->data + (size_t)e * n_embd * d_rb, nf, n_embd);
        }

        size_t sg_rb = (n_embd / ink_type_block_elems(l->gate_shexp->type)) * ink_type_block_bytes(l->gate_shexp->type);
        size_t su_rb = (n_embd / ink_type_block_elems(l->up_shexp->type)) * ink_type_block_bytes(l->up_shexp->type);
        size_t sd_rb = (nf / ink_type_block_elems(l->down_shexp->type)) * ink_type_block_bytes(l->down_shexp->type);
        for (uint32_t sx = 0; sx < m.n_shexp; sx++) {
            char nm[64];
            snprintf(nm, sizeof(nm), "gate_shexp[%u]", sx);
            all_pass &= ink_test_tensor(nm, l->gate_shexp, l->gate_shexp->data + (size_t)sx * nf * sg_rb, n_embd, nf);
            snprintf(nm, sizeof(nm), "up_shexp[%u]", sx);
            all_pass &= ink_test_tensor(nm, l->up_shexp, l->up_shexp->data + (size_t)sx * nf * su_rb, n_embd, nf);
            snprintf(nm, sizeof(nm), "down_shexp[%u]", sx);
            all_pass &= ink_test_tensor(nm, l->down_shexp, l->down_shexp->data + (size_t)sx * n_embd * sd_rb, nf, n_embd);
        }
    }

    /* token_embd / output: huge vocab-sized `out`, cap rows to bound time */
    uint64_t out_te = m.tok_embd->dims[1] < 4096 ? m.tok_embd->dims[1] : 4096;
    all_pass &= ink_test_tensor("token_embd", m.tok_embd, m.tok_embd->data, n_embd, out_te);
    uint64_t out_o = m.output->dims[1] < 4096 ? m.output->dims[1] : 4096;
    all_pass &= ink_test_tensor("output", m.output, m.output->data, n_embd, out_o);

    /* batch (matmat) and grouped-matvec checks against the CPU engine */
    {
        const uint32_t NT = 5;
        uint64_t in = n_embd, outn = hn;
        float *X = (float *)ink_malloc((size_t)NT * in * sizeof(float));
        ink_fill_rand(X, (size_t)NT * in, 0xBEEF01u);
        float *Yc = (float *)ink_malloc((size_t)NT * outn * sizeof(float));
        float *Yg = (float *)ink_malloc((size_t)NT * outn * sizeof(float));
        ink_matmat(l->wq, l->wq->data, in, outn, NT, X, Yc);
        ink_cuda_matmat(l->wq, l->wq->data, in, outn, NT, X, Yg);
        ink_cuda_sync();
        double mx = 0.0;
        for (uint64_t i = 0; i < (uint64_t)NT * outn; i++) {
            double a = fabs((double)Yg[i] - (double)Yc[i]);
            if (a > mx) mx = a;
        }
        bool bp = mx <= 5e-4;
        printf("matmat(wq,n_tok=5)       maxabsdiff=%.3g %s\n", mx, bp ? "PASS" : "FAIL");
        all_pass &= bp;
        free(X); free(Yc); free(Yg);
    }
    if ((uint32_t)layer >= m.n_dense) {
        const uint32_t NG = 6;
        uint64_t in = n_embd, outn = m.n_ff_exp;
        size_t rb = (in / ink_type_block_elems(l->gate_exps->type)) * ink_type_block_bytes(l->gate_exps->type);
        const uint8_t *bases[NG];
        const float *xs[NG];
        float *ys_g[NG];
        float *x1 = (float *)ink_malloc(in * sizeof(float));
        ink_fill_rand(x1, in, 0xFACE02u);
        float *yc = (float *)ink_malloc(outn * sizeof(float));
        double mx = 0.0;
        for (uint32_t g2 = 0; g2 < NG; g2++) {
            bases[g2] = l->gate_exps->data + (size_t)(g2 * 37 + 1) * outn * rb;
            xs[g2] = x1;
            ys_g[g2] = (float *)ink_malloc(outn * sizeof(float));
        }
        ink_cuda_matvec_group(l->gate_exps, bases, in, outn, xs, ys_g, NG);
        ink_cuda_sync();
        for (uint32_t g2 = 0; g2 < NG; g2++) {
            ink_matvec(l->gate_exps, bases[g2], in, outn, x1, yc);
            for (uint64_t i = 0; i < outn; i++) {
                double a = fabs((double)ys_g[g2][i] - (double)yc[i]);
                if (a > mx) mx = a;
            }
            free(ys_g[g2]);
        }
        bool gp = mx <= 5e-4;
        printf("group-matvec(gate_exps,6) maxabsdiff=%.3g %s\n", mx, gp ? "PASS" : "FAIL");
        all_pass &= gp;
        free(x1); free(yc);

        /* M11 fast-path (int8/dp4a) INFO checks -- see ink_selftest_fast_check.
         * No-ops for tensors whose type isn't IQ2_XXS/IQ3_XXS. */
        ink_selftest_fast_check("gate_exps", l->gate_exps, n_embd, m.n_ff_exp);
        ink_selftest_fast_check("up_exps",   l->up_exps,   n_embd, m.n_ff_exp);
        ink_selftest_fast_check("down_exps", l->down_exps, m.n_ff_exp, n_embd);
    }
    g_exact_dequant = saved_exact;
    printf(all_pass ? "SELFTEST PASS (layer %d)\n" : "SELFTEST FAIL (layer %d)\n", layer);
    return all_pass ? 0 : 1;
}

/* --resident: copy tensors out of the file-backed mmap into managed
 * memory owned by us.  Root cause: GPU pageable access over file-backed
 * mmap returned bad data under page-cache saturation/reclaim (see
 * PORT_NOTES.md, resident window 2026-08-08); owned memory sidesteps
 * file-backed residency entirely. */
static void *ink_cuda_managed_alloc(size_t n) {
    void *p = NULL;
    cudaError_t err = cudaMallocManaged(&p, n, cudaMemAttachGlobal);
    if (err != cudaSuccess) return NULL;
    return p;
}

static uint64_t ink_sys_available_bytes(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    uint64_t kb = 0;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "MemAvailable: %llu kB", (unsigned long long *)&kb) == 1) break;
    }
    fclose(f);
    return kb * 1024;
}

/* M8: cudaMalloc weight arena.  Everything a GPU KERNEL dereferences
 * (matvec/matmat/group-matvec/attention/sconv weights) is fine living in
 * plain device memory -- device memory is faster than managed/pageable
 * (see --bench-membw: cudaMalloc ~225-237 GB/s vs managed ~162 GB/s on
 * this box) and none of those tensors are ever read by HOST code.  The
 * exceptions are the handful of tensors HOST code actually dereferences
 * directly:
 *   - token_embd.weight: ink_row_f32() in ink_forward_gpu's embedding
 *     fetch loop is a host CPU function (memcpy/dequant into `xt`).
 *   - every F32 tensor: most are only ever handed to a kernel as an
 *     opaque device-dereferenced pointer (attn_norm, q_norm, k_norm,
 *     sc_k/v/attn/mlp, rel_proj, ...) and would be fine on-device too, but
 *     TWO of them are read element-wise on the HOST in ink_forward_gpu:
 *     `ink_f32(l->gscale)[0]` and `bias[e]` from `ink_f32(l->probs_b)`
 *     during MoE routing.  Rather than special-case just those two (and
 *     risk missing a future host read of some other F32 tensor), the
 *     simplest SAFE rule is: all F32 tensors stay host-resident (malloc).
 *     They are tiny (norm vectors, scalars, small conv kernels) relative
 *     to the quantized weight matrices, so this costs effectively nothing.
 * Kernels can dereference either arena identically (device pointers are
 * device pointers; host pointers work too under this file's coherent-
 * pageable-access design) -- the split is purely about which arena is
 * fastest for who actually reads it. */
static const ink_tensor *g_resident_tok_embd = NULL;
static uint64_t g_resident_device_bytes = 0;
static uint64_t g_resident_host_bytes = 0;

static bool ink_resident_needs_host(const ink_tensor *t) {
    return t->type == INK_T_F32 || t == g_resident_tok_embd;
}

static void *ink_cuda_resident_alloc_ex(size_t n, const ink_tensor *t) {
    if (ink_resident_needs_host(t)) {
        void *p = malloc(n ? n : 1);
        if (!p) ink_die("resident arena: host malloc failed");
        g_resident_host_bytes += n;
        return p;
    }
    void *p = NULL;
    cudaError_t err = cudaMalloc(&p, n ? n : 1);
    if (err != cudaSuccess) ink_die("resident arena: cudaMalloc failed (device out of memory?)");
    g_resident_device_bytes += n;
    return p;
}

/* cudaMemcpyDefault: works for host<-host and device<-host alike under
 * unified virtual addressing (this file already requires
 * cudaDevAttrPageableMemoryAccess==1, so UVA is guaranteed present). */
static void ink_cuda_resident_copy(void *dst, const void *src, size_t n) {
    CUDA_CHECK(cudaMemcpy(dst, src, n, cudaMemcpyDefault));
}

/* Exported for the server (which links this TU without its main()). */
extern "C" void ink_cuda_make_resident(ink_model *m, uint64_t budget_bytes);

static void ink_make_resident(ink_model *m, uint64_t budget_bytes) {
    uint64_t total = 0, host_est = 0;
    for (uint64_t i = 0; i < m->gg.n_tensors; i++) {
        const ink_tensor *t = &m->gg.tensors[i];
        uint64_t nb = ink_tensor_bytes(t);
        total += nb;
        if (t->type == INK_T_F32 || t == m->tok_embd) host_est += nb;
    }
    uint64_t want = budget_bytes ? (budget_bytes < total ? budget_bytes : total) : total;
    /* Host-RAM check only covers the F32+token_embd slice that actually
     * lands in host malloc() -- the rest goes to cudaMalloc device memory,
     * which doesn't compete with host RAM.  host_est is computed over ALL
     * tensors regardless of --resident-budget (a conservative upper bound:
     * budget only ever shrinks what's copied, in file order, never grows
     * it), so this can only over-estimate host pressure, never under. */
    uint64_t avail = ink_sys_available_bytes();
    const uint64_t margin = 4ull << 30;
    if (avail && host_est + margin > avail) {
        fprintf(stderr, "ds4-inkling-cuda: FATAL --resident host slice does not fit: "
                "need %.1f GiB (F32 + token_embd) + %.1f GiB margin, MemAvailable %.1f GiB\n",
                host_est / 1073741824.0, margin / 1073741824.0, avail / 1073741824.0);
        exit(2);
    }
    /* Best-effort device free-memory check for the rest (quantized weight
     * matrices); soft (warn, don't exit) since cudaMemGetInfo's notion of
     * "free" can be conservative on unified-memory systems and the
     * per-tensor cudaMalloc in ink_cuda_resident_alloc_ex will die loudly
     * with a clear message if it actually fails. */
    size_t dev_free = 0, dev_total = 0;
    if (cudaMemGetInfo(&dev_free, &dev_total) == cudaSuccess) {
        uint64_t device_est = want > host_est ? want - host_est : 0;
        if (device_est + margin > dev_free) {
            fprintf(stderr, "ds4-inkling-cuda: WARNING --resident device slice may not fit: "
                    "want ~%.1f GiB, device free %.1f GiB (will fail loudly if it doesn't)\n",
                    device_est / 1073741824.0, dev_free / 1073741824.0);
        }
    }

    g_resident_tok_embd = m->tok_embd;
    g_resident_device_bytes = 0;
    g_resident_host_bytes = 0;
    double t0 = ink_now_sec();
    uint64_t n_res = 0;
    uint64_t copied = ink_model_make_resident_ex(m, budget_bytes, ink_cuda_resident_alloc_ex,
                                                  ink_cuda_resident_copy, &n_res);
    fprintf(stderr, "ds4-inkling-cuda: resident %.1f GiB in %llu/%llu tensors "
            "(%.1f GiB device / %.1f GiB host) in %.1fs%s\n",
            copied / 1073741824.0, (unsigned long long)n_res,
            (unsigned long long)m->gg.n_tensors,
            g_resident_device_bytes / 1073741824.0, g_resident_host_bytes / 1073741824.0,
            ink_now_sec() - t0,
            budget_bytes && copied < total ? " [PARTIAL: rest stays mmap-paged]" : "");
}

extern "C" void ink_cuda_make_resident(ink_model *m, uint64_t budget_bytes) {
    ink_make_resident(m, budget_bytes);
}

/* --bench-layers L: measure raw kernel throughput independent of disk.
 * Copies layer L's weight tensors into cudaMallocManaged buffers (so
 * timing isn't polluted by page faults against the file-backed mmap),
 * fills a random x, runs 20 timed iterations (5 warmup) per tensor, and
 * prints name/type/bytes/ms-per-iter/GB/s.  Mirrors the tensor set probed
 * by --selftest (probes experts 0 and 7 for MoE layers) but does not
 * compare against the CPU reference -- this mode is about speed, not
 * correctness. */
static void ink_bench_one_tensor(const char *name, const ink_tensor *t, const uint8_t *base,
                                  uint64_t in, uint64_t out) {
    size_t be = ink_type_block_elems(t->type), bb = ink_type_block_bytes(t->type);
    size_t bytes = (size_t)out * (in / be) * bb;

    uint8_t *mbase = (uint8_t *)ink_cuda_managed_alloc(bytes);
    float *mx = (float *)ink_cuda_managed_alloc(in * sizeof(float));
    float *my = (float *)ink_cuda_managed_alloc(out * sizeof(float));
    if (!mbase || !mx || !my) { fprintf(stderr, "  (skip %s: cudaMallocManaged failed)\n", name); return; }
    /* base may be a cudaMalloc device pointer when --resident placed this
     * tensor in the device arena -- host memcpy would fault; UVA copy
     * handles host/device/managed sources alike. */
    CUDA_CHECK(cudaMemcpy(mbase, base, bytes, cudaMemcpyDefault));
    ink_fill_rand(mx, in, 0xBEEFu);

    for (int i = 0; i < 5; i++) ink_cuda_matvec(t, mbase, in, out, mx, my);
    ink_cuda_sync(); /* M7: wrappers no longer sync internally */

    double total_ms = 0.0;
    for (int i = 0; i < 20; i++) {
        cudaEvent_t s, e;
        CUDA_CHECK(cudaEventCreate(&s));
        CUDA_CHECK(cudaEventCreate(&e));
        CUDA_CHECK(cudaEventRecord(s, g_stream)); /* must share g_stream with the kernel to bracket it correctly */
        ink_cuda_matvec(t, mbase, in, out, mx, my);
        CUDA_CHECK(cudaEventRecord(e, g_stream));
        CUDA_CHECK(cudaEventSynchronize(e));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
        total_ms += ms;
        cudaEventDestroy(s);
        cudaEventDestroy(e);
    }
    double ms_per = total_ms / 20.0;
    double gbps = ms_per > 0 ? (bytes / 1.0e9) / (ms_per / 1000.0) : 0.0;
    /* NOTE: single-tensor numbers here are L2-RESIDENT (one ~2-19 MB slice
     * hammered 20x) and therefore optimistic -- real decode streams six
     * experts chosen fresh per layer out of a 76 GiB arena with no reuse.
     * They also move the WRONG way under M13's 2-way ILP, which trades
     * registers for latency hiding that an in-cache benchmark has no
     * latency to hide. Read the rotating grouped table below for
     * decode-representative rates. */
    printf("%-24s type=%2u bytes=%-12zu ms/iter=%.4f GB/s=%.2f\n",
           name, t->type, bytes, ms_per, gbps);

    cudaFree(mbase); cudaFree(mx); cudaFree(my);
}

/* M8: time the grouped kernel itself, at the real decode shape (6 routed
 * experts, blockIdx.y grouping) instead of only single-tensor matvecs --
 * this is the number the M8 report is actually about (group-matvec is the
 * measured decode bottleneck).  When `resident_active` is false (default),
 * behaves like ink_bench_one_tensor: copies the 6 experts' contiguous byte
 * range into one cudaMallocManaged arena so timing is independent of disk/
 * mmap page faults.  When true (--resident was also requested), benches
 * directly against the tensor's already-resident data pointer (whatever
 * arena ink_make_resident put it in -- device cudaMalloc for a non-F32
 * tensor like gate_exps/down_exps under the M8 arena-split rule) instead
 * of making a redundant managed copy, so the number reflects the arena
 * that's actually in play. */
/* M11: `force_exact` selects which path this bench run measures (the
 * label printed is "FAST" when g_exact_dequant ends up 0 -- i.e. always,
 * unless the tensor type isn't dp4a-eligible, in which case FAST and EXACT
 * are the same kernel/numbers and that's expected).  Restores whatever
 * g_exact_dequant was set to on the way out. */
static void ink_bench_group6(const char *name, const ink_tensor *t, const uint8_t *live_base,
                              uint64_t in, uint64_t out, size_t row_bytes, bool resident_active,
                              bool force_exact) {
    int saved_exact = g_exact_dequant;
    g_exact_dequant = force_exact ? 1 : 0;
    const uint32_t NG = 6;
    /* CACHE-DEFEAT: benching the same 6 experts 20x measures L2 residency,
     * not memory bandwidth (a 2 MB expert slice lives in L2 all run, which
     * is why single-tensor numbers here read 70-175 GB/s while real decode
     * -- which walks 6 experts chosen fresh per layer out of an 82 GB model
     * -- never sees that). Stage INK_BENCH_SETS distinct expert sets and
     * rotate through them so each timed iteration touches memory the
     * previous ones did not. */
#define INK_BENCH_SETS 12
    size_t span = (size_t)NG * INK_BENCH_SETS * out * row_bytes;

    uint8_t *mbase = NULL;
    const uint8_t *base_for_bench = live_base;
    if (!resident_active) {
        mbase = (uint8_t *)ink_cuda_managed_alloc(span);
        if (!mbase) { fprintf(stderr, "  (skip %s-group6: cudaMallocManaged failed)\n", name); g_exact_dequant = saved_exact; return; }
        CUDA_CHECK(cudaMemcpy(mbase, live_base, span, cudaMemcpyDefault)); /* see note above */
        base_for_bench = mbase;
    }

    float *mx = (float *)ink_cuda_managed_alloc(in * sizeof(float));
    if (!mx) {
        fprintf(stderr, "  (skip %s-group6: cudaMallocManaged failed)\n", name);
        if (mbase) cudaFree(mbase);
        g_exact_dequant = saved_exact;
        return;
    }
    ink_fill_rand(mx, in, 0xBEEF03u);

    const uint8_t *bases[6];
    const float *xs[6];
    float *ys[6];
    for (uint32_t g = 0; g < NG; g++) {
        bases[g] = base_for_bench + (size_t)g * out * row_bytes;
        xs[g] = mx;
        ys[g] = (float *)ink_cuda_managed_alloc(out * sizeof(float));
        if (!ys[g]) { fprintf(stderr, "  (skip %s-group6: cudaMallocManaged failed)\n", name); g_exact_dequant = saved_exact; return; }
    }

    /* rotate over the staged sets: iteration i uses set (i % INK_BENCH_SETS) */
    const uint8_t *set_bases[6];
    for (uint32_t g = 0; g < NG; g++) set_bases[g] = bases[g];
    for (int i = 0; i < 5; i++) ink_cuda_matvec_group(t, set_bases, in, out, xs, ys, NG);
    ink_cuda_sync();

    double total_ms = 0.0;
    for (int i = 0; i < 20; i++) {
        cudaEvent_t s, e;
        CUDA_CHECK(cudaEventCreate(&s));
        CUDA_CHECK(cudaEventCreate(&e));
        for (uint32_t g = 0; g < NG; g++) {
            set_bases[g] = base_for_bench +
                ((size_t)(i % INK_BENCH_SETS) * NG + g) * out * row_bytes;
        }
        CUDA_CHECK(cudaEventRecord(s, g_stream));
        ink_cuda_matvec_group(t, set_bases, in, out, xs, ys, NG);
        CUDA_CHECK(cudaEventRecord(e, g_stream));
        CUDA_CHECK(cudaEventSynchronize(e));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
        total_ms += ms;
        cudaEventDestroy(s);
        cudaEventDestroy(e);
    }
    double ms_per = total_ms / 20.0;
    size_t be = ink_type_block_elems(t->type), bb = ink_type_block_bytes(t->type);
    size_t bytes = (size_t)NG * out * (in / be) * bb;
    double gbps = ms_per > 0 ? (bytes / 1.0e9) / (ms_per / 1000.0) : 0.0;
    bool eligible = (t->type == INK_T_IQ2_XXS || t->type == INK_T_IQ3_XXS);
    printf("%-24s type=%2u path=%-5s bytes=%-12zu ms/iter=%.4f GB/s=%.2f arena=%s\n",
           name, t->type, eligible ? (force_exact ? "EXACT" : "FAST") : "n/a",
           bytes, ms_per, gbps, resident_active ? "resident" : "managed");

    for (uint32_t g = 0; g < NG; g++) cudaFree(ys[g]);
    cudaFree(mx);
    if (mbase) cudaFree(mbase);
    g_exact_dequant = saved_exact;
}

static int ink_run_bench_layers(const char *model_path, int layer, bool resident, uint64_t resident_budget) {
    ink_model m;
    ink_model_open(&m, model_path, 8);
    if (layer < 0 || (uint32_t)layer >= m.n_layer) ink_die("--bench-layers LAYER out of range");
    if (resident) ink_make_resident(&m, resident_budget);
    ink_layer *l = &m.layers[layer];

    uint64_t n_embd = m.n_embd;
    uint64_t hn = (uint64_t)m.n_head * m.head_dim;
    uint64_t kvw = (uint64_t)l->n_head_kv * m.head_dim;

    ink_bench_one_tensor("wq", l->wq, l->wq->data, n_embd, hn);
    ink_bench_one_tensor("wk", l->wk, l->wk->data, n_embd, kvw);
    ink_bench_one_tensor("wv", l->wv, l->wv->data, n_embd, kvw);
    ink_bench_one_tensor("wr", l->wr, l->wr->data, n_embd, (uint64_t)m.n_head * m.d_rel);
    ink_bench_one_tensor("wo", l->wo, l->wo->data, hn, n_embd);

    if ((uint32_t)layer < m.n_dense) {
        ink_bench_one_tensor("ffn_gate", l->ffn_gate, l->ffn_gate->data, n_embd, m.n_ff_dense);
        ink_bench_one_tensor("ffn_up", l->ffn_up, l->ffn_up->data, n_embd, m.n_ff_dense);
        ink_bench_one_tensor("ffn_down", l->ffn_down, l->ffn_down->data, m.n_ff_dense, n_embd);
    } else {
        uint64_t nE = m.n_expert, nf = m.n_ff_exp;
        ink_bench_one_tensor("gate_inp", l->gate_inp, l->gate_inp->data, n_embd, nE + m.n_shexp);

        size_t g_rb = (n_embd / ink_type_block_elems(l->gate_exps->type)) * ink_type_block_bytes(l->gate_exps->type);
        size_t u_rb = (n_embd / ink_type_block_elems(l->up_exps->type)) * ink_type_block_bytes(l->up_exps->type);
        size_t d_rb = (nf / ink_type_block_elems(l->down_exps->type)) * ink_type_block_bytes(l->down_exps->type);

        uint64_t probe_experts[2] = { 0, 7 };
        for (int pi = 0; pi < 2; pi++) {
            uint64_t e = probe_experts[pi];
            if (e >= nE) continue;
            char nm[64];
            snprintf(nm, sizeof(nm), "gate_exps[%llu]", (unsigned long long)e);
            ink_bench_one_tensor(nm, l->gate_exps, l->gate_exps->data + (size_t)e * nf * g_rb, n_embd, nf);
            snprintf(nm, sizeof(nm), "up_exps[%llu]", (unsigned long long)e);
            ink_bench_one_tensor(nm, l->up_exps, l->up_exps->data + (size_t)e * nf * u_rb, n_embd, nf);
            snprintf(nm, sizeof(nm), "down_exps[%llu]", (unsigned long long)e);
            ink_bench_one_tensor(nm, l->down_exps, l->down_exps->data + (size_t)e * n_embd * d_rb, nf, n_embd);
        }

        if (nE >= 6) {
            /* M11: both paths in one run so the before/after table is
             * self-generating -- FAST (default int8/dp4a for IQ2_XXS/
             * IQ3_XXS, identical kernel to EXACT for everything else) and
             * EXACT (INK_EXACT_DEQUANT=1 forced, the M6-M8 float path). */
            printf("--- grouped (6-expert, real decode shape), FAST path ---\n");
            ink_bench_group6("gate_exps", l->gate_exps, l->gate_exps->data, n_embd, nf, g_rb, resident, false);
            ink_bench_group6("up_exps",   l->up_exps,   l->up_exps->data,   n_embd, nf, u_rb, resident, false);
            ink_bench_group6("down_exps", l->down_exps, l->down_exps->data, nf, n_embd, d_rb, resident, false);
            printf("--- grouped (6-expert, real decode shape), EXACT path ---\n");
            ink_bench_group6("gate_exps", l->gate_exps, l->gate_exps->data, n_embd, nf, g_rb, resident, true);
            ink_bench_group6("up_exps",   l->up_exps,   l->up_exps->data,   n_embd, nf, u_rb, resident, true);
            ink_bench_group6("down_exps", l->down_exps, l->down_exps->data, nf, n_embd, d_rb, resident, true);
        }
    }
    return 0;
}

/* --bench-membw: decide whether the ~15-30 GB/s effective bandwidth seen in
 * matvec/group-matvec bench numbers is a real managed/pageable-memory
 * streaming ceiling on this GB10, or just async-launch overhead (which M7
 * should have mostly eliminated for non-bench runs, but --bench itself
 * re-adds a sync per launch, see the bench-report header).  Allocates 2 GiB
 * with cudaMallocManaged (touched from the host first, matching
 * ink_make_resident's pattern) and 2 GiB with cudaMalloc+cudaMemcpy from a
 * host buffer, then runs a plain grid-stride sum-reduction kernel over
 * each, 10x timed (5 warmup), and reports GB/s for both. */
__global__ void ink_kernel_membw_sum(const float *p, uint64_t n, float *out) {
    float local = 0.0f;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x)
        local += p[i];
    __shared__ float sred[256];
    sred[threadIdx.x] = local;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sred[threadIdx.x] += sred[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, sred[0]);
}

static double ink_membw_probe(const char *label, float *p, uint64_t n_floats) {
    float *dout = NULL;
    CUDA_CHECK(cudaMalloc(&dout, sizeof(float)));
    uint32_t blocks = ink_flat_blocks(n_floats, 256);

    for (int i = 0; i < 5; i++) {
        CUDA_CHECK(cudaMemsetAsync(dout, 0, sizeof(float), g_stream));
        ink_kernel_membw_sum<<<blocks, 256, 0, g_stream>>>(p, n_floats, dout);
    }
    CUDA_CHECK(cudaStreamSynchronize(g_stream));

    double total_ms = 0.0;
    for (int i = 0; i < 10; i++) {
        cudaEvent_t s, e;
        CUDA_CHECK(cudaEventCreate(&s));
        CUDA_CHECK(cudaEventCreate(&e));
        CUDA_CHECK(cudaMemsetAsync(dout, 0, sizeof(float), g_stream));
        CUDA_CHECK(cudaEventRecord(s, g_stream));
        ink_kernel_membw_sum<<<blocks, 256, 0, g_stream>>>(p, n_floats, dout);
        CUDA_CHECK(cudaEventRecord(e, g_stream));
        CUDA_CHECK(cudaEventSynchronize(e));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
        total_ms += ms;
        cudaEventDestroy(s);
        cudaEventDestroy(e);
    }
    double ms_per = total_ms / 10.0;
    double bytes = (double)n_floats * sizeof(float);
    double gbps = ms_per > 0 ? (bytes / 1.0e9) / (ms_per / 1000.0) : 0.0;
    printf("%-28s bytes=%.3fGiB ms/iter=%.4f GB/s=%.2f\n", label, bytes / 1073741824.0, ms_per, gbps);
    cudaFree(dout);
    return gbps;
}

static void ink_run_bench_membw(void) {
    const uint64_t bytes = 2ull << 30; /* 2 GiB */
    const uint64_t n_floats = bytes / sizeof(float);

    printf("=== ds4-inkling-cuda --bench-membw (2 GiB each) ===\n");

    /* managed: touch from host first, same pattern as ink_make_resident's
     * managed-memory residency path (first-touch establishes host-side
     * physical backing before the device streams it). */
    float *managed = (float *)ink_cuda_managed_alloc(bytes);
    if (!managed) { fprintf(stderr, "cudaMallocManaged(2 GiB) failed\n"); } else {
        for (uint64_t i = 0; i < n_floats; i++) managed[i] = 1.0f;
        ink_membw_probe("managed (host-touched)", managed, n_floats);
        cudaFree(managed);
    }

    /* cudaMalloc + explicit cudaMemcpy from a host buffer -- the
     * conventional "not coherent, but a plain device allocation" ceiling. */
    float *hostbuf = (float *)ink_malloc(bytes);
    for (uint64_t i = 0; i < n_floats; i++) hostbuf[i] = 1.0f;
    float *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc(2 GiB) failed: %s\n", cudaGetErrorString(err));
    } else {
        CUDA_CHECK(cudaMemcpy(dev, hostbuf, bytes, cudaMemcpyHostToDevice));
        ink_membw_probe("cudaMalloc+memcpy", dev, n_floats);
        cudaFree(dev);
    }
    free(hostbuf);
}

#ifndef DS4_INKLING_NO_MAIN
int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *prompt = NULL;
    int n_predict = 5;
    uint32_t n_ctx = 512;
    bool dump_tokens = false;
    const char *logits_out = NULL;
    int selftest_layer = -1;
    bool resident = false;
    uint64_t resident_budget = 0;
    int bench_layers_layer = -1;
    bool bench_membw = false;

    const char *env_bench = getenv("INK_BENCH");
    if (env_bench && strcmp(env_bench, "0") != 0 && env_bench[0] != '\0') g_bench = 1;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--prompt-file") && i + 1 < argc) {
            FILE *pf = fopen(argv[++i], "rb");
            if (!pf) ink_die("cannot open --prompt-file");
            fseek(pf, 0, SEEK_END);
            long pl = ftell(pf);
            fseek(pf, 0, SEEK_SET);
            char *pb = (char *)ink_malloc((size_t)pl + 1);
            if (fread(pb, 1, (size_t)pl, pf) != (size_t)pl) ink_die("short read on --prompt-file");
            pb[pl] = 0;
            fclose(pf);
            prompt = pb;
        }
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) n_predict = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) n_ctx = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--logits-out") && i + 1 < argc) logits_out = argv[++i];
        else if (!strcmp(argv[i], "--dump-tokens")) dump_tokens = true;
        else if (!strcmp(argv[i], "--selftest") && i + 1 < argc) selftest_layer = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--resident")) resident = true;
        else if (!strcmp(argv[i], "--resident-budget") && i + 1 < argc) {
            resident = true;
            resident_budget = (uint64_t)(atof(argv[++i]) * 1073741824.0);
        }
        else if (!strcmp(argv[i], "--bench")) g_bench = 1;
        else if (!strcmp(argv[i], "--bench-layers") && i + 1 < argc) bench_layers_layer = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bench-membw")) bench_membw = true;
        else {
            fprintf(stderr, "usage: ds4-inkling-cuda -m model.gguf -p prompt [-n N] [-c CTX] [--resident] [--resident-budget GiB] [--dump-tokens] [--bench]\n");
            fprintf(stderr, "       ds4-inkling-cuda -m model.gguf --selftest LAYER\n");
            fprintf(stderr, "       ds4-inkling-cuda -m model.gguf --bench-layers LAYER\n");
            fprintf(stderr, "       ds4-inkling-cuda --bench-membw\n");
            return 1;
        }
    }

    if (bench_membw) {
        ink_cuda_init();
        ink_run_bench_membw();
        return 0;
    }

    if (!model_path) {
        fprintf(stderr, "usage: ds4-inkling-cuda -m model.gguf -p prompt [-n N] [-c CTX] [--resident] [--resident-budget GiB] [--dump-tokens] [--bench]\n");
        fprintf(stderr, "       ds4-inkling-cuda -m model.gguf --selftest LAYER\n");
        fprintf(stderr, "       ds4-inkling-cuda -m model.gguf --bench-layers LAYER\n");
        fprintf(stderr, "       ds4-inkling-cuda --bench-membw\n");
        return 1;
    }

    ink_cuda_init();

    if (selftest_layer >= 0) {
        return ink_run_selftest(model_path, selftest_layer);
    }

    if (bench_layers_layer >= 0) {
        return ink_run_bench_layers(model_path, bench_layers_layer, resident, resident_budget);
    }

    if (!prompt) {
        fprintf(stderr, "usage: ds4-inkling-cuda -m model.gguf -p prompt [-n N] [-c CTX] [--resident] [--resident-budget GiB] [--dump-tokens] [--bench]\n");
        return 1;
    }

    ink_model m;
    double t0 = ink_now_sec();
    ink_model_open(&m, model_path, n_ctx);
    fprintf(stderr, "ds4-inkling-cuda: loaded %s (%u layers, %u dense, vocab %u) in %.1fs\n",
            model_path, m.n_layer, m.n_dense, m.n_vocab, ink_now_sec() - t0);
    if (resident) ink_make_resident(&m, resident_budget);

    ink_ids ids = {0};
    ink_tokenize(&m.tk, prompt, &ids);
    if (dump_tokens) {
        fprintf(stderr, "tokens (%d):", ids.len);
        for (int i = 0; i < ids.len; i++) fprintf(stderr, " %d", ids.ids[i]);
        fprintf(stderr, "\n");
        char buf[512];
        for (int i = 0; i < ids.len; i++) {
            ink_detokenize(&m.tk, ids.ids[i], buf, sizeof(buf));
            fprintf(stderr, "  %d -> '%s'\n", ids.ids[i], buf);
        }
    }
    if (ids.len == 0) ink_die("empty prompt tokenization");
    if ((uint32_t)(ids.len + n_predict) > n_ctx) ink_die("prompt + n_predict exceeds context");

    float *logits = (float *)ink_malloc((size_t)m.n_vocab * sizeof(float));

    {
        const int chunk = 128;
        for (int i = 0; i < ids.len; i += chunk) {
            double ts = ink_now_sec();
            int n = ids.len - i < chunk ? ids.len - i : chunk;
            bool last = i + n == ids.len;
            ink_forward_gpu(&m, ids.ids + i, (uint32_t)n, (uint32_t)i, last ? logits : NULL);
            fprintf(stderr, "prefill %d..%d/%d: %.2fs\n", i, i + n, ids.len, ink_now_sec() - ts);
        }
    }

    int pos = ids.len;
    char buf[512];
    for (int t = 0; t < n_predict; t++) {
        ink_logits_guard(logits, m.n_vocab, m.n_vocab_unpadded, "gpu decode");
        int best = 0;
        float bestv = -INFINITY;
        for (uint32_t i = 0; i < m.n_vocab; i++) {
            if (logits[i] > bestv) { bestv = logits[i]; best = (int)i; }
        }
        ink_detokenize(&m.tk, best, buf, sizeof(buf));
        if (logits_out) {
            FILE *lf = fopen(logits_out, t == 0 ? "wb" : "ab");
            if (lf) {
                fwrite(&best, sizeof(int), 1, lf);
                fwrite(logits, sizeof(float), m.n_vocab, lf);
                fclose(lf);
            }
        }
        printf("GEN %d: id=%d logit=%.6f text='%s'\n", t, best, bestv, buf);
        fflush(stdout);
        if (best == m.tk.eos) break;
        if (t + 1 == n_predict) break;
        double ts = ink_now_sec();
        ink_forward_gpu(&m, &best, 1, (uint32_t)pos++, logits);
        fprintf(stderr, "decode %d: %.3fs\n", t + 1, ink_now_sec() - ts);
    }
    ink_bench_report();
    return 0;
}
#endif /* DS4_INKLING_NO_MAIN */
