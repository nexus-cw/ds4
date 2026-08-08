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
    float dsc = d * sc, dmnm = mn * m;
#pragma unroll
    for (int l = 0; l < 32; l++) {
        uint8_t nib = hi ? (ql[l] >> 4) : (ql[l] & 0xF);
        uint8_t hbit = hi ? ((b->qh[l] & u2) ? 16 : 0) : ((b->qh[l] & u1) ? 16 : 0);
        w[l] = dsc * (nib + hbit) - dmnm;
    }
}

__device__ __forceinline__ void dq32_q6_K(const uint8_t *rowp, uint32_t sg, float w[32]) {
    const ink_block_q6_K *x = (const ink_block_q6_K *)rowp;
    uint32_t i = sg / 8, local_sg = sg % 8;
    const ink_block_q6_K *b = &x[i];
    uint32_t half = local_sg / 4, qsel = local_sg % 4;
    float d = __half2float(__ushort_as_half(b->d));
    const uint8_t *ql = b->ql + 64 * half;
    const uint8_t *qh = b->qh + 32 * half;
    const int8_t *sc = b->scales + 8 * half;
    /* is = l/16 in {0,1}; hoist the two possible scale*d values, matching
     * scv = sc[is + {0,2,4,6}] for qsel {0,1,2,3} exactly. */
    int scoff = (int)qsel * 2;
    float dsc0 = d * (float)sc[scoff + 0];
    float dsc1 = d * (float)sc[scoff + 1];
#pragma unroll
    for (int l = 0; l < 32; l++) {
        int val;
        if (qsel == 0)      val = (ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4);
        else if (qsel == 1) val = (ql[l + 32]  & 0xF) | (((qh[l] >> 2) & 3) << 4);
        else if (qsel == 2) val = (ql[l]       >> 4) | (((qh[l] >> 4) & 3) << 4);
        else                val = (ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4);
        val -= 32;
        w[l] = (l < 16 ? dsc0 : dsc1) * (float)val;
    }
}

__device__ __forceinline__ void dq32_iq2_xxs(const uint8_t *rowp, uint32_t sg, float w[32], ink_tables tb) {
    const ink_block_iq2_xxs *x = (const ink_block_iq2_xxs *)rowp;
    uint32_t i = sg / 8, local_sg = sg % 8;
    const ink_block_iq2_xxs *b = &x[i];
    float d = __half2float(__ushort_as_half(b->d));
    const uint8_t *bp = (const uint8_t *)b->qs + 8 * local_sg;
    uint32_t a0 = (uint32_t)bp[0] | ((uint32_t)bp[1] << 8) | ((uint32_t)bp[2] << 16) | ((uint32_t)bp[3] << 24);
    uint32_t a1 = (uint32_t)bp[4] | ((uint32_t)bp[5] << 8) | ((uint32_t)bp[6] << 16) | ((uint32_t)bp[7] << 24);
    float db = d * (0.5f + (float)(a1 >> 28)) * 0.25f;
#pragma unroll
    for (uint32_t lg = 0; lg < 4; lg++) {
        uint8_t byte_l = (uint8_t)((a0 >> (8 * lg)) & 0xFF);
        uint64_t gridv = tb.iq2xxs_grid[byte_l];
        uint8_t signs = tb.ksigns_iq2xs[(a1 >> (7 * lg)) & 127];
#pragma unroll
        for (uint32_t j = 0; j < 8; j++) {
            uint8_t gbyte = (uint8_t)((gridv >> (8 * j)) & 0xFF);
            w[lg * 8 + j] = db * (float)gbyte * ((signs & tb.kmask_iq2xs[j]) ? -1.f : 1.f);
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
            w[l4 * 8 + j] = dl * (float)gbyte * ((sign_byte & tb.kmask_iq2xs[j]) ? -1.f : 1.f);
        }
    }
}

__device__ __forceinline__ void dq32_iq3_xxs(const uint8_t *rowp, uint32_t sg, float w[32], ink_tables tb) {
    const ink_block_iq3_xxs *x = (const ink_block_iq3_xxs *)rowp;
    uint32_t i = sg / 8, local_sg = sg % 8;
    const ink_block_iq3_xxs *b = &x[i];
    float d = __half2float(__ushort_as_half(b->d));
    const uint8_t *sas = b->qs + QK_K / 4 + 4 * local_sg;
    uint32_t aux32 = (uint32_t)sas[0] | ((uint32_t)sas[1] << 8) | ((uint32_t)sas[2] << 16) | ((uint32_t)sas[3] << 24);
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
                w[l4 * 8 + subsel * 4 + j] = db * (float)gbyte *
                    ((signs & tb.kmask_iq2xs[maskbase + j]) ? -1.f : 1.f);
            }
        }
    }
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
 * create/record/sync). */

#define INK_BENCH_NTYPES 9

static int g_bench = 0;
static cudaEvent_t g_bench_ev0, g_bench_ev1;
static int g_bench_ev_ready = 0;

typedef struct { double ms; double bytes; uint64_t calls; } ink_bench_stat;
static ink_bench_stat g_bench_matvec[INK_BENCH_NTYPES];
static ink_bench_stat g_bench_attn;
static ink_bench_stat g_bench_sconv;
static ink_bench_stat g_bench_group;

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
    CUDA_CHECK(cudaEventRecord(g_bench_ev0));
}

static inline void ink_bench_end(ink_bench_stat *st, double bytes) {
    if (!g_bench) return;
    CUDA_CHECK(cudaEventRecord(g_bench_ev1));
    CUDA_CHECK(cudaEventSynchronize(g_bench_ev1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, g_bench_ev0, g_bench_ev1));
    st->ms += ms;
    st->bytes += bytes;
    st->calls += 1;
}

static void ink_bench_report(void) {
    if (!g_bench) return;
    fprintf(stderr, "\n=== ds4-inkling-cuda --bench report ===\n");
    for (int i = 0; i < INK_BENCH_NTYPES; i++) {
        ink_bench_stat *s = &g_bench_matvec[i];
        if (s->calls == 0) continue;
        double gbps = s->ms > 0 ? (s->bytes / 1.0e9) / (s->ms / 1000.0) : 0.0;
        fprintf(stderr, "matvec[%-8s] calls=%-8llu total_ms=%-10.1f bytes=%.3fGiB  %.2f GB/s\n",
                ink_bench_type_name(i), (unsigned long long)s->calls, s->ms,
                s->bytes / 1073741824.0, gbps);
    }
    {
        ink_bench_stat *s = &g_bench_attn;
        double gbps = s->ms > 0 ? (s->bytes / 1.0e9) / (s->ms / 1000.0) : 0.0;
        fprintf(stderr, "attention          calls=%-8llu total_ms=%-10.1f bytes=%.3fGiB  %.2f GB/s\n",
                (unsigned long long)s->calls, s->ms, s->bytes / 1073741824.0, gbps);
    }
    {
        ink_bench_stat *s = &g_bench_sconv;
        double gbps = s->ms > 0 ? (s->bytes / 1.0e9) / (s->ms / 1000.0) : 0.0;
        fprintf(stderr, "sconv               calls=%-8llu total_ms=%-10.1f bytes=%.3fGiB  %.2f GB/s\n",
                (unsigned long long)s->calls, s->ms, s->bytes / 1073741824.0, gbps);
    }
    {
        ink_bench_stat *s = &g_bench_group;
        double gbps = s->ms > 0 ? (s->bytes / 1.0e9) / (s->ms / 1000.0) : 0.0;
        fprintf(stderr, "group-matvec        calls=%-8llu total_ms=%-10.1f bytes=%.3fGiB  %.2f GB/s\n",
                (unsigned long long)s->calls, s->ms, s->bytes / 1073741824.0, gbps);
    }
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
        float acc = 0.0f;
        for (uint32_t sg = lane; sg < nsub; sg += 32) {
            float w[32];
            ink_dq_load32(type, rowp, sg, w, tb);
            const float *xp = xsrc + (size_t)sg * 32;
#pragma unroll
            for (int l = 0; l < 32; l++) acc += w[l] * xp[l];
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) Y[row] = acc;
    }
}

__global__ void ink_kernel_matvec(uint32_t type, const uint8_t *rowbase, uint64_t in, uint64_t out,
                                   const float *X, float *Y, ink_tables tb) {
    __shared__ float sx[INK_MATVEC_SHARED_MAX];
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

    __shared__ float sx[INK_MATVEC_SHARED_MAX];
    const float *sxp = NULL;
    if (in <= INK_MATVEC_SHARED_MAX) {
        for (uint64_t i = threadIdx.x; i < in; i += blockDim.x) sx[i] = X[i];
        __syncthreads();
        sxp = sx;
    }
    ink_matvec_row_warp(type, rowbase, in, out, X, Y, tb, sxp);
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
    ink_kernel_matvec<<<blocks, 256>>>(t->type, base, in, out, x, y, g_tables);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int bi = ink_bench_type_idx(t->type);
    if (bi >= 0) ink_bench_end(&g_bench_matvec[bi], ink_matvec_bytes(t->type, in, out));
    else if (g_bench) { CUDA_CHECK(cudaEventRecord(g_bench_ev1)); }
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
    ink_kernel_matmat<<<blocks, 256>>>(t->type, base, in, out, X, Y, n_tok, g_tables);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    int bi = ink_bench_type_idx(t->type);
    double bytes = ink_matvec_bytes(t->type, in, out) * (((n_tok + INK_MATMAT_UNROLL - 1) / INK_MATMAT_UNROLL));
    if (bi >= 0) ink_bench_end(&g_bench_matvec[bi], bytes);
    else if (g_bench) { CUDA_CHECK(cudaEventRecord(g_bench_ev1)); }
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
    ink_bench_begin();
    for (uint32_t g0 = 0; g0 < n_group; g0 += INK_GROUP_MAX) {
        uint32_t ng = n_group - g0 < INK_GROUP_MAX ? n_group - g0 : INK_GROUP_MAX;
        ink_ptr8 pb; ink_fptr8 px; ink_fptr8_mut py;
        for (uint32_t i = 0; i < ng; i++) { pb.p[i] = bases[g0 + i]; px.p[i] = xs[g0 + i]; py.p[i] = ys[g0 + i]; }
        for (uint32_t i = ng; i < INK_GROUP_MAX; i++) { pb.p[i] = NULL; px.p[i] = NULL; py.p[i] = NULL; }
        dim3 grid(blocks, ng);
        ink_kernel_matvec_group<<<grid, 256>>>(t->type, pb, in, out, px, py, ng, g_tables);
        CUDA_CHECK(cudaGetLastError());
        total_bytes += ink_matvec_bytes(t->type, in, out) * ng;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    ink_bench_end(&g_bench_group, total_bytes);
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
    ink_kernel_attention<<<n_head, 128>>>(q, kc, vc, rel, hd, kvw_max, pos, j0, rel_extent, gqa,
                                           inv_hd, len, g_attn_scratch, attn_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    /* bytes touched: this token's K/V window across all kv-heads, read once */
    double bytes = (double)len * (double)n_head_kv * (double)hd * 2.0 * sizeof(float);
    ink_bench_end(&g_bench_attn, bytes);
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
    ink_kernel_sconv<<<blocks, threads>>>(w, state, C, K, x);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    double bytes = (double)C * (double)K * sizeof(float) * 2.0; /* weights + state r/w */
    ink_bench_end(&g_bench_sconv, bytes);
}

/* ============================== GPU forward ==============================
 * Mirrors ink_forward_batch() in ds4_inkling.c line for line: same control
 * flow, same math.  rmsnorm, the per-token rel-bias projection, and MoE
 * expert selection stay on the host (small / branchy); the big matmuls go
 * through ink_cuda_matmat/matvec, shortconvs through ink_cuda_sconv, and
 * the attention inner loop through ink_cuda_attention. */

static float ink_silu_h(float x) { return x / (1.0f + expf(-x)); }

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
    float *ff_acc = (float *)ink_malloc(n_embd * sizeof(float));
    float *taus = (float *)ink_malloc(nT * sizeof(float));

    for (uint32_t t = 0; t < nT; t++) {
        float *xt = x + (size_t)t * n_embd;
        ink_row_f32(m->tok_embd, m->tok_embd->data, (uint64_t)tokens[t], n_embd, xt);
        ink_rmsnorm(xt, ink_f32(m->tok_norm), n_embd, m->rms_eps);
        taus[t] = (m->log_n_floor > 0)
            ? 1.0f + m->log_alpha * logf(fmaxf((float)(pos0 + t + 1) / (float)m->log_n_floor, 1.0f))
            : 1.0f;
    }

    for (uint32_t il = 0; il < m->n_layer; il++) {
        ink_layer *l = &m->layers[il];
        const uint32_t n_head_kv = l->n_head_kv;
        const uint32_t kvw = n_head_kv * hd;
        const uint32_t rel_extent = l->is_swa ? m->rel_extent_swa : m->rel_extent;
        float *conv_l = m->conv + (size_t)il * conv_per_layer;
        float *kc = m->kcache + ((size_t)il * m->n_ctx) * kvw_max;
        float *vc = m->vcache + ((size_t)il * m->n_ctx) * kvw_max;

        /* ---- attention block ---- */
        for (uint32_t t = 0; t < nT; t++) {
            memcpy(xn + (size_t)t * n_embd, x + (size_t)t * n_embd, n_embd * sizeof(float));
            ink_rmsnorm(xn + (size_t)t * n_embd, ink_f32(l->attn_norm), n_embd, m->rms_eps);
        }

        ink_cuda_matmat(l->wq, l->wq->data, n_embd, (uint64_t)n_head * hd, nT, xn, q);
        ink_cuda_matmat(l->wk, l->wk->data, n_embd, kvw, nT, xn, kf);
        ink_cuda_matmat(l->wv, l->wv->data, n_embd, kvw, nT, xn, vf);
        ink_cuda_matmat(l->wr, l->wr->data, n_embd, (uint64_t)n_head * m->d_rel, nT, xn, r);

        for (uint32_t t = 0; t < nT; t++) {
            const uint32_t pos = pos0 + t;
            float *kt = kf + (size_t)t * kvw_max;
            float *vt = vf + (size_t)t * kvw_max;
            float *qt = q + (size_t)t * n_head * hd;

            ink_cuda_sconv(l->sc_k, conv_l + off_conv_k, kvw, K, kt);
            ink_cuda_sconv(l->sc_v, conv_l + off_conv_v, kvw, K, vt);

            for (uint32_t h = 0; h < n_head; h++)
                ink_rmsnorm(qt + (size_t)h * hd, ink_f32(l->q_norm), hd, m->rms_eps);
            for (uint32_t h = 0; h < n_head_kv; h++)
                ink_rmsnorm(kt + (size_t)h * hd, ink_f32(l->k_norm), hd, m->rms_eps);

            if (!l->is_swa && taus[t] != 1.0f) {
                for (uint32_t i = 0; i < (uint32_t)n_head * hd; i++) qt[i] *= taus[t];
            }

            memcpy(kc + (size_t)pos * kvw_max, kt, kvw * sizeof(float));
            memcpy(vc + (size_t)pos * kvw_max, vt, kvw * sizeof(float));
        }

        const float *pw = ink_f32(l->rel_proj);

        for (uint32_t t = 0; t < nT; t++) {
            const uint32_t pos = pos0 + t;
            const float *rt = r + (size_t)t * n_head * m->d_rel;
            const float tau_t = (!l->is_swa) ? taus[t] : 1.0f;

            for (uint32_t h = 0; h < n_head; h++) {
                const float *rh = rt + (size_t)h * m->d_rel;
                float *relh = rel + (size_t)h * rel_extent;
                for (uint32_t e = 0; e < rel_extent; e++) {
                    float acc = 0.0f;
                    for (uint32_t d = 0; d < m->d_rel; d++)
                        acc += pw[(size_t)d * rel_extent + e] * rh[d];
                    relh[e] = acc * tau_t;
                }
            }

            uint32_t j0 = 0;
            if (l->is_swa && pos + 1 > m->n_swa) j0 = pos + 1 - m->n_swa;

            ink_cuda_attention(q + (size_t)t * n_head * hd, kc, vc, rel,
                                n_head, n_head_kv, hd, kvw_max, pos, j0, rel_extent,
                                attn_out + (size_t)t * n_head * hd);
        }

        ink_cuda_matmat(l->wo, l->wo->data, (uint64_t)n_head * hd, n_embd, nT, attn_out, proj_out);

        for (uint32_t t = 0; t < nT; t++) {
            float *pt = proj_out + (size_t)t * n_embd;
            ink_cuda_sconv(l->sc_attn, conv_l + off_conv_attn, n_embd, K, pt);
            float *xt = x + (size_t)t * n_embd;
            for (uint32_t i = 0; i < n_embd; i++) xt[i] += pt[i];
        }

        /* ---- ffn block ---- */
        for (uint32_t t = 0; t < nT; t++) {
            memcpy(xn + (size_t)t * n_embd, x + (size_t)t * n_embd, n_embd * sizeof(float));
            ink_rmsnorm(xn + (size_t)t * n_embd, ink_f32(l->ffn_norm), n_embd, m->rms_eps);
        }
        const float gscale = ink_f32(l->gscale)[0];

        if (il < m->n_dense) {
            uint32_t nf = m->n_ff_dense;
            ink_cuda_matmat(l->ffn_gate, l->ffn_gate->data, n_embd, nf, nT, xn, hg);
            ink_cuda_matmat(l->ffn_up, l->ffn_up->data, n_embd, nf, nT, xn, hu);
            for (uint32_t i = 0; i < nT * nf; i++) hg[i] = ink_silu_h(hg[i]) * hu[i];
            ink_cuda_matmat(l->ffn_down, l->ffn_down->data, nf, n_embd, nT, hg, ff_out);
            for (uint32_t i = 0; i < nT * n_embd; i++) ff_out[i] *= gscale;
        } else {
            const uint32_t nE = m->n_expert, nu = m->n_expert_used, ns = m->n_shexp;
            const uint32_t nf = m->n_ff_exp;
            float *logits = (float *)ink_malloc((size_t)nT * (nE + ns) * sizeof(float));
            ink_cuda_matmat(l->gate_inp, l->gate_inp->data, n_embd, nE + ns, nT, xn, logits);
            const float *bias = ink_f32(l->probs_b);

            size_t g_rb = (n_embd / ink_type_block_elems(l->gate_exps->type)) * ink_type_block_bytes(l->gate_exps->type);
            size_t u_rb = (n_embd / ink_type_block_elems(l->up_exps->type)) * ink_type_block_bytes(l->up_exps->type);
            size_t d_rb = (nf / ink_type_block_elems(l->down_exps->type)) * ink_type_block_bytes(l->down_exps->type);
            size_t sg_rb = (n_embd / ink_type_block_elems(l->gate_shexp->type)) * ink_type_block_bytes(l->gate_shexp->type);
            size_t su_rb = (n_embd / ink_type_block_elems(l->up_shexp->type)) * ink_type_block_bytes(l->up_shexp->type);
            size_t sd_rb = (nf / ink_type_block_elems(l->down_shexp->type)) * ink_type_block_bytes(l->down_shexp->type);

            float *wv_all = (float *)ink_malloc((size_t)nT * (nu + ns) * sizeof(float));
            int *sel_all = (int *)ink_malloc((size_t)nT * nu * sizeof(int));

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
             * ink_cuda_matvec() calls -- 3 launches per token instead of
             * 3*nu.  gate/up share the token's xt; down needs each expert's
             * own silu(gate)*up hidden vector, so it gets a per-group X
             * array too (ink_cuda_matvec_group takes xs[group] rather than
             * a single shared x for exactly this reason). */
            if (nu > INK_GROUP_HOST_MAX)
                ink_die("ink_forward_gpu: n_expert_used exceeds host group-batch array cap");
            float *hg_g = (float *)ink_malloc((size_t)nu * nf * sizeof(float));
            float *hu_g = (float *)ink_malloc((size_t)nu * nf * sizeof(float));
            float *proj_g = (float *)ink_malloc((size_t)nu * n_embd * sizeof(float));
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
                memset(ff_acc, 0, n_embd * sizeof(float));

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
                for (uint32_t i = 0; i < nu; i++) {
                    float *hgi = hg_g + (size_t)i * nf;
                    const float *hui = hu_g + (size_t)i * nf;
                    for (uint32_t v2 = 0; v2 < nf; v2++) hgi[v2] = ink_silu_h(hgi[v2]) * hui[v2];
                    down_x[i] = hgi;
                }
                ink_cuda_matvec_group(l->down_exps, down_bases, nf, n_embd, down_x, down_ys, nu);

                for (uint32_t i = 0; i < nu; i++) {
                    const float w = wv_all[(size_t)t * (nu + ns) + i];
                    const float *proj_i = proj_g + (size_t)i * n_embd;
                    for (uint32_t v2 = 0; v2 < n_embd; v2++) ff_acc[v2] += w * proj_i[v2];
                }
                memcpy(ot, ff_acc, n_embd * sizeof(float));
            }

            for (uint32_t sx = 0; sx < ns; sx++) {
                ink_cuda_matmat(l->gate_shexp, l->gate_shexp->data + (size_t)sx * nf * sg_rb, n_embd, nf, nT, xn, hg);
                ink_cuda_matmat(l->up_shexp, l->up_shexp->data + (size_t)sx * nf * su_rb, n_embd, nf, nT, xn, hu);
                for (uint32_t t = 0; t < nT; t++) {
                    const float gamma = wv_all[(size_t)t * (nu + ns) + nu + sx];
                    float *hgt = hg + (size_t)t * nf;
                    const float *hut = hu + (size_t)t * nf;
                    for (uint32_t v2 = 0; v2 < nf; v2++) hgt[v2] = ink_silu_h(hgt[v2]) * hut[v2] * gamma;
                }
                ink_cuda_matmat(l->down_shexp, l->down_shexp->data + (size_t)sx * n_embd * sd_rb, nf, n_embd, nT, hg, proj_out);
                for (uint32_t i = 0; i < nT * n_embd; i++) ff_out[i] += proj_out[i];
            }
            free(logits); free(wv_all); free(sel_all);
            free(hg_g); free(hu_g); free(proj_g);
        }

        for (uint32_t t = 0; t < nT; t++) {
            float *ft = ff_out + (size_t)t * n_embd;
            ink_cuda_sconv(l->sc_mlp, conv_l + off_conv_mlp, n_embd, K, ft);
            float *xt = x + (size_t)t * n_embd;
            for (uint32_t i = 0; i < n_embd; i++) xt[i] += ft[i];
        }
    }

    if (out_logits) {
        float *xl = x + (size_t)(nT - 1) * n_embd;
        ink_rmsnorm(xl, ink_f32(m->out_norm), n_embd, m->rms_eps);
        for (uint32_t i = 0; i < n_embd; i++) xl[i] *= m->logit_scale;
        ink_cuda_matvec(m->output, m->output->data, n_embd, m->n_vocab, xl, out_logits);
        if (m->n_vocab_unpadded > 0) {
            for (uint32_t i = m->n_vocab_unpadded; i < m->n_vocab; i++) out_logits[i] = -INFINITY;
        }
    }

    free(x); free(xn); free(q); free(kf); free(vf); free(r); free(rel);
    free(attn_out); free(proj_out); free(hg); free(hu);
    free(ff_out); free(ff_acc); free(taus);
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

static int ink_run_selftest(const char *model_path, int layer) {
    ink_model m;
    ink_model_open(&m, model_path, 8);
    if (layer < 0 || (uint32_t)layer >= m.n_layer) ink_die("--selftest LAYER out of range");
    ink_layer *l = &m.layers[layer];

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

static void ink_make_resident(ink_model *m, uint64_t budget_bytes) {
    uint64_t total = 0;
    for (uint64_t i = 0; i < m->gg.n_tensors; i++) total += ink_tensor_bytes(&m->gg.tensors[i]);
    uint64_t want = budget_bytes ? (budget_bytes < total ? budget_bytes : total) : total;
    uint64_t avail = ink_sys_available_bytes();
    const uint64_t margin = 4ull << 30;
    if (avail && want + margin > avail) {
        fprintf(stderr, "ds4-inkling-cuda: FATAL --resident budget does not fit: "
                "need %.1f GiB + %.1f GiB margin, MemAvailable %.1f GiB\n",
                want / 1073741824.0, margin / 1073741824.0, avail / 1073741824.0);
        exit(2);
    }
    double t0 = ink_now_sec();
    uint64_t n_res = 0;
    uint64_t copied = ink_model_make_resident(m, budget_bytes, ink_cuda_managed_alloc, &n_res);
    fprintf(stderr, "ds4-inkling-cuda: resident %.1f GiB in %llu/%llu tensors "
            "(managed memory) in %.1fs%s\n",
            copied / 1073741824.0, (unsigned long long)n_res,
            (unsigned long long)m->gg.n_tensors, ink_now_sec() - t0,
            budget_bytes && copied < total ? " [PARTIAL: rest stays mmap-paged]" : "");
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
    memcpy(mbase, base, bytes);
    ink_fill_rand(mx, in, 0xBEEFu);

    for (int i = 0; i < 5; i++) ink_cuda_matvec(t, mbase, in, out, mx, my);
    CUDA_CHECK(cudaDeviceSynchronize());

    double total_ms = 0.0;
    for (int i = 0; i < 20; i++) {
        cudaEvent_t s, e;
        CUDA_CHECK(cudaEventCreate(&s));
        CUDA_CHECK(cudaEventCreate(&e));
        CUDA_CHECK(cudaEventRecord(s));
        ink_cuda_matvec(t, mbase, in, out, mx, my);
        CUDA_CHECK(cudaEventRecord(e));
        CUDA_CHECK(cudaEventSynchronize(e));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
        total_ms += ms;
        cudaEventDestroy(s);
        cudaEventDestroy(e);
    }
    double ms_per = total_ms / 20.0;
    double gbps = ms_per > 0 ? (bytes / 1.0e9) / (ms_per / 1000.0) : 0.0;
    printf("%-24s type=%2u bytes=%-12zu ms/iter=%.4f GB/s=%.2f\n",
           name, t->type, bytes, ms_per, gbps);

    cudaFree(mbase); cudaFree(mx); cudaFree(my);
}

static int ink_run_bench_layers(const char *model_path, int layer) {
    ink_model m;
    ink_model_open(&m, model_path, 8);
    if (layer < 0 || (uint32_t)layer >= m.n_layer) ink_die("--bench-layers LAYER out of range");
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
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *prompt = NULL;
    int n_predict = 5;
    uint32_t n_ctx = 512;
    bool dump_tokens = false;
    int selftest_layer = -1;
    bool resident = false;
    uint64_t resident_budget = 0;
    int bench_layers_layer = -1;

    const char *env_bench = getenv("INK_BENCH");
    if (env_bench && strcmp(env_bench, "0") != 0 && env_bench[0] != '\0') g_bench = 1;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) n_predict = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) n_ctx = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dump-tokens")) dump_tokens = true;
        else if (!strcmp(argv[i], "--selftest") && i + 1 < argc) selftest_layer = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--resident")) resident = true;
        else if (!strcmp(argv[i], "--resident-budget") && i + 1 < argc) {
            resident = true;
            resident_budget = (uint64_t)(atof(argv[++i]) * 1073741824.0);
        }
        else if (!strcmp(argv[i], "--bench")) g_bench = 1;
        else if (!strcmp(argv[i], "--bench-layers") && i + 1 < argc) bench_layers_layer = atoi(argv[++i]);
        else {
            fprintf(stderr, "usage: ds4-inkling-cuda -m model.gguf -p prompt [-n N] [-c CTX] [--resident] [--resident-budget GiB] [--dump-tokens] [--bench]\n");
            fprintf(stderr, "       ds4-inkling-cuda -m model.gguf --selftest LAYER\n");
            fprintf(stderr, "       ds4-inkling-cuda -m model.gguf --bench-layers LAYER\n");
            return 1;
        }
    }
    if (!model_path) {
        fprintf(stderr, "usage: ds4-inkling-cuda -m model.gguf -p prompt [-n N] [-c CTX] [--resident] [--resident-budget GiB] [--dump-tokens] [--bench]\n");
        fprintf(stderr, "       ds4-inkling-cuda -m model.gguf --selftest LAYER\n");
        fprintf(stderr, "       ds4-inkling-cuda -m model.gguf --bench-layers LAYER\n");
        return 1;
    }

    ink_cuda_init();

    if (selftest_layer >= 0) {
        return ink_run_selftest(model_path, selftest_layer);
    }

    if (bench_layers_layer >= 0) {
        return ink_run_bench_layers(model_path, bench_layers_layer);
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
            fprintf(stderr, "prefill %d..%d/%d: %.1fs\n", i, i + n, ids.len, ink_now_sec() - ts);
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
        printf("GEN %d: id=%d logit=%.6f text='%s'\n", t, best, bestv, buf);
        fflush(stdout);
        if (best == m.tk.eos) break;
        if (t + 1 == n_predict) break;
        double ts = ink_now_sec();
        ink_forward_gpu(&m, &best, 1, (uint32_t)pos++, logits);
        fprintf(stderr, "decode %d: %.1fs\n", t + 1, ink_now_sec() - ts);
    }
    ink_bench_report();
    return 0;
}
