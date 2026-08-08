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
 * Exact per-element port of the CPU row dequantizers in ds4_inkling.c.
 * Each function takes the *row start* pointer (as computed the same way
 * ink_row_f32() computes it) and a flat element index 0..row_len-1, and
 * returns the single dequantized float at that index.  Correctness, not
 * speed, is the point of v1: we recompute the per-block scale/min lookup
 * for every element rather than caching it across a 32-wide subgroup. */

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

__device__ __forceinline__ float ink_dq_elem(uint32_t type, const uint8_t *rowp, uint64_t idx,
                                              ink_tables tb) {
    switch (type) {
    case INK_T_F32: {
        return ((const float *)rowp)[idx];
    }
    case INK_T_Q8_0: {
        const ink_block_q8_0 *x = (const ink_block_q8_0 *)rowp;
        uint64_t i = idx / QK8_0; uint32_t l = (uint32_t)(idx % QK8_0);
        float d = __half2float(__ushort_as_half(x[i].d));
        return (float)x[i].qs[l] * d;
    }
    case INK_T_Q4_K: {
        const ink_block_q4_K *x = (const ink_block_q4_K *)rowp;
        uint64_t i = idx / QK_K; uint32_t p = (uint32_t)(idx % QK_K);
        uint32_t sg = p / 32, l = p % 32;
        float d = __half2float(__ushort_as_half(x[i].d));
        float mn = __half2float(__ushort_as_half(x[i].dmin));
        uint8_t sc, m;
        ink_dev_scale_min_k4((int)sg, x[i].scales, &sc, &m);
        const uint8_t *q = x[i].qs + 32 * (sg / 2);
        uint8_t nib = (sg & 1) == 0 ? (q[l] & 0xF) : (q[l] >> 4);
        return d * sc * nib - mn * m;
    }
    case INK_T_Q5_K: {
        const ink_block_q5_K *x = (const ink_block_q5_K *)rowp;
        uint64_t i = idx / QK_K; uint32_t p = (uint32_t)(idx % QK_K);
        uint32_t sg = p / 32, l = p % 32;
        float d = __half2float(__ushort_as_half(x[i].d));
        float mn = __half2float(__ushort_as_half(x[i].dmin));
        uint8_t sc, m;
        ink_dev_scale_min_k4((int)sg, x[i].scales, &sc, &m);
        const uint8_t *ql = x[i].qs + 32 * (sg / 2);
        uint32_t shift = 2 * (sg / 2);
        uint8_t u1 = (uint8_t)(1u << shift), u2 = (uint8_t)(2u << shift);
        uint8_t nib = (sg & 1) == 0 ? (ql[l] & 0xF) : (ql[l] >> 4);
        uint8_t hbit = (sg & 1) == 0 ? ((x[i].qh[l] & u1) ? 16 : 0)
                                     : ((x[i].qh[l] & u2) ? 16 : 0);
        return d * sc * (nib + hbit) - mn * m;
    }
    case INK_T_Q6_K: {
        const ink_block_q6_K *x = (const ink_block_q6_K *)rowp;
        uint64_t i = idx / QK_K; uint32_t p = (uint32_t)(idx % QK_K);
        uint32_t sg = p / 32, l = p % 32;
        uint32_t half = sg / 4, qsel = sg % 4;
        float d = __half2float(__ushort_as_half(x[i].d));
        const uint8_t *ql = x[i].ql + 64 * half;
        const uint8_t *qh = x[i].qh + 32 * half;
        const int8_t *sc = x[i].scales + 8 * half;
        uint32_t is = l / 16;
        int val, scv;
        if (qsel == 0)      { val = (ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4); scv = sc[is + 0]; }
        else if (qsel == 1) { val = (ql[l + 32]  & 0xF) | (((qh[l] >> 2) & 3) << 4); scv = sc[is + 2]; }
        else if (qsel == 2) { val = (ql[l]       >> 4) | (((qh[l] >> 4) & 3) << 4); scv = sc[is + 4]; }
        else                { val = (ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4); scv = sc[is + 6]; }
        val -= 32;
        return d * (float)scv * (float)val;
    }
    case INK_T_IQ2_XXS: {
        const ink_block_iq2_xxs *x = (const ink_block_iq2_xxs *)rowp;
        uint64_t i = idx / QK_K; uint32_t p = (uint32_t)(idx % QK_K);
        uint32_t sg = p / 32, lane = p % 32;
        uint32_t l = lane / 8, j = lane % 8;
        float d = __half2float(__ushort_as_half(x[i].d));
        const uint8_t *bp = (const uint8_t *)x[i].qs + 8 * sg;
        uint32_t a0 = (uint32_t)bp[0] | ((uint32_t)bp[1] << 8) | ((uint32_t)bp[2] << 16) | ((uint32_t)bp[3] << 24);
        uint32_t a1 = (uint32_t)bp[4] | ((uint32_t)bp[5] << 8) | ((uint32_t)bp[6] << 16) | ((uint32_t)bp[7] << 24);
        float db = d * (0.5f + (float)(a1 >> 28)) * 0.25f;
        uint8_t byte_l = (uint8_t)((a0 >> (8 * l)) & 0xFF);
        uint64_t gridv = tb.iq2xxs_grid[byte_l];
        uint8_t signs = tb.ksigns_iq2xs[(a1 >> (7 * l)) & 127];
        uint8_t gbyte = (uint8_t)((gridv >> (8 * j)) & 0xFF);
        return db * (float)gbyte * ((signs & tb.kmask_iq2xs[j]) ? -1.f : 1.f);
    }
    case INK_T_IQ2_S: {
        const ink_block_iq2_s *x = (const ink_block_iq2_s *)rowp;
        uint64_t i = idx / QK_K; uint32_t p = (uint32_t)(idx % QK_K);
        uint32_t sg = p / 32, lane = p % 32;
        uint32_t l = lane / 8, j = lane % 8;
        float d = __half2float(__ushort_as_half(x[i].d));
        const uint8_t *qs_ptr = x[i].qs + 4 * sg;
        const uint8_t *signs_ptr = x[i].qs + QK_K / 8 + 4 * sg;
        uint32_t qh_val = x[i].qh[sg];
        uint8_t scale_byte = x[i].scales[sg];
        float db0 = d * (0.5f + (float)(scale_byte & 0xf)) * 0.25f;
        float db1 = d * (0.5f + (float)(scale_byte >> 4)) * 0.25f;
        float dl = (l < 2) ? db0 : db1;
        uint32_t grid_idx = qs_ptr[l] | ((qh_val << (8 - 2 * l)) & 0x300);
        uint64_t gridv = tb.iq2s_grid[grid_idx];
        uint8_t gbyte = (uint8_t)((gridv >> (8 * j)) & 0xFF);
        uint8_t sign_byte = signs_ptr[l];
        return dl * (float)gbyte * ((sign_byte & tb.kmask_iq2xs[j]) ? -1.f : 1.f);
    }
    case INK_T_IQ3_XXS: {
        const ink_block_iq3_xxs *x = (const ink_block_iq3_xxs *)rowp;
        uint64_t i = idx / QK_K; uint32_t p = (uint32_t)(idx % QK_K);
        uint32_t sg = p / 32, lane = p % 32;
        uint32_t l = lane / 8, rem = lane % 8;
        uint32_t subsel = rem / 4, j = rem % 4;
        float d = __half2float(__ushort_as_half(x[i].d));
        const uint8_t *sas = x[i].qs + QK_K / 4 + 4 * sg;
        uint32_t aux32 = (uint32_t)sas[0] | ((uint32_t)sas[1] << 8) | ((uint32_t)sas[2] << 16) | ((uint32_t)sas[3] << 24);
        float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
        const uint8_t *qsp = x[i].qs + 8 * sg;
        uint8_t signs = tb.ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
        uint32_t gv;
        uint32_t maskidx;
        if (subsel == 0) { gv = tb.iq3xxs_grid[qsp[2 * l + 0]]; maskidx = j; }
        else             { gv = tb.iq3xxs_grid[qsp[2 * l + 1]]; maskidx = j + 4; }
        uint8_t gbyte = (uint8_t)((gv >> (8 * j)) & 0xFF);
        return db * (float)gbyte * ((signs & tb.kmask_iq2xs[maskidx]) ? -1.f : 1.f);
    }
    case INK_T_IQ4_XS: {
        const ink_block_iq4_xs *x = (const ink_block_iq4_xs *)rowp;
        uint64_t i = idx / QK_K; uint32_t p = (uint32_t)(idx % QK_K);
        uint32_t sg = p / 32, lane = p % 32;
        uint32_t j = lane % 16, half = lane / 16;
        float d = __half2float(__ushort_as_half(x[i].d));
        uint8_t scales_l_byte = x[i].scales_l[sg / 2];
        uint32_t ls = ((scales_l_byte >> (4 * (sg % 2))) & 0xf) |
                      (((x[i].scales_h >> (2 * sg)) & 3) << 4);
        float dl = d * ((float)ls - 32.0f);
        const uint8_t *qsp = x[i].qs + 16 * sg;
        uint8_t nib = half == 0 ? (qsp[j] & 0xf) : (qsp[j] >> 4);
        return dl * (float)tb.kvalues_iq4nl[nib];
    }
    default:
        return 0.0f; /* unreachable: host validates type before launch */
    }
}

/* ======================= generic matvec/matmat kernel ==================
 * One block per output row `o`, 256 threads.  Each thread walks a strided
 * set of 32-element subgroups of the row, dequantizes them, and forms a
 * partial dot product against every token column (n_tok <= 32).  A
 * per-token shared-memory tree reduction across the block then produces
 * Y[t][o]. */

#define INK_CUDA_MAX_TOK 32

__global__ void ink_kernel_matmul(uint32_t type, const uint8_t *rowbase, uint64_t in, uint64_t out,
                                   const float *X, float *Y, uint32_t n_tok, ink_tables tb) {
    uint64_t o = blockIdx.x;
    if (o >= out) return;
    uint32_t tid = threadIdx.x;

    size_t be = ink_dev_block_elems(type);
    size_t bb = ink_dev_block_bytes(type);
    const uint8_t *rowp = rowbase + (size_t)o * (in / be) * bb;
    uint32_t nsub = (uint32_t)(in / 32);

    float acc[INK_CUDA_MAX_TOK];
    for (uint32_t t = 0; t < n_tok; t++) acc[t] = 0.0f;

    for (uint32_t sg = tid; sg < nsub; sg += 256) {
        for (uint32_t l = 0; l < 32; l++) {
            uint64_t idx = (uint64_t)sg * 32 + l;
            float w = ink_dq_elem(type, rowp, idx, tb);
            for (uint32_t t = 0; t < n_tok; t++) {
                acc[t] += w * X[(size_t)t * in + idx];
            }
        }
    }

    __shared__ float sdata[256];
    for (uint32_t t = 0; t < n_tok; t++) {
        sdata[tid] = acc[t];
        __syncthreads();
        for (uint32_t stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) sdata[tid] += sdata[tid + stride];
            __syncthreads();
        }
        if (tid == 0) Y[(size_t)t * out + o] = sdata[0];
        __syncthreads();
    }
}

static void ink_cuda_matmul_launch(uint32_t type, const uint8_t *base, uint64_t in, uint64_t out,
                                    const float *X, float *Y, uint32_t n_tok) {
    if (!g_tables_ready) ink_die("ink_cuda: ink_cuda_init() was not called");
    if (n_tok == 0) return;
    if (n_tok > INK_CUDA_MAX_TOK) ink_die("ink_cuda: n_tok exceeds GPU kernel's fixed max (32)");
    dim3 grid((unsigned int)out);
    ink_kernel_matmul<<<grid, 256>>>(type, base, in, out, X, Y, n_tok, g_tables);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void ink_cuda_matvec(const ink_tensor *t, const uint8_t *base,
                                 uint64_t in, uint64_t out, const float *x, float *y) {
    ink_cuda_matmul_launch(t->type, base, in, out, x, y, 1);
}

extern "C" void ink_cuda_matmat(const ink_tensor *t, const uint8_t *base,
                                 uint64_t in, uint64_t out, uint32_t n_tok,
                                 const float *X, float *Y) {
    ink_cuda_matmul_launch(t->type, base, in, out, X, Y, n_tok);
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
    ink_kernel_attention<<<n_head, 128>>>(q, kc, vc, rel, hd, kvw_max, pos, j0, rel_extent, gqa,
                                           inv_hd, len, g_attn_scratch, attn_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
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
    ink_kernel_sconv<<<blocks, threads>>>(w, state, C, K, x);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

/* ============================== GPU forward ==============================
 * Mirrors ink_forward_batch() in ds4_inkling.c line for line: same control
 * flow, same math.  rmsnorm, the per-token rel-bias projection, and MoE
 * expert selection stay on the host (small / branchy); the big matmuls go
 * through ink_cuda_matmat/matvec, shortconvs through ink_cuda_sconv, and
 * the attention inner loop through ink_cuda_attention. */

static float ink_silu_h(float x) { return x / (1.0f + expf(-x)); }

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
    if (nT > INK_CUDA_MAX_TOK) ink_die("ink_forward_gpu: batch exceeds GPU kernel's fixed max (32 tokens)");

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

            for (uint32_t t = 0; t < nT; t++) {
                const float *xt = xn + (size_t)t * n_embd;
                float *ot = ff_out + (size_t)t * n_embd;
                memset(ff_acc, 0, n_embd * sizeof(float));
                for (uint32_t i = 0; i < nu; i++) {
                    const uint32_t e = (uint32_t)sel_all[(size_t)t * nu + i];
                    ink_cuda_matvec(l->gate_exps, l->gate_exps->data + (size_t)e * nf * g_rb, n_embd, nf, xt, hg);
                    ink_cuda_matvec(l->up_exps, l->up_exps->data + (size_t)e * nf * u_rb, n_embd, nf, xt, hu);
                    for (uint32_t v2 = 0; v2 < nf; v2++) hg[v2] = ink_silu_h(hg[v2]) * hu[v2];
                    ink_cuda_matvec(l->down_exps, l->down_exps->data + (size_t)e * n_embd * d_rb, nf, n_embd, hg, proj_out);
                    const float w = wv_all[(size_t)t * (nu + ns) + i];
                    for (uint32_t v2 = 0; v2 < n_embd; v2++) ff_acc[v2] += w * proj_out[v2];
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

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *prompt = NULL;
    int n_predict = 5;
    uint32_t n_ctx = 512;
    bool dump_tokens = false;
    int selftest_layer = -1;
    bool resident = false;
    uint64_t resident_budget = 0;

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
        else {
            fprintf(stderr, "usage: ds4-inkling-cuda -m model.gguf -p prompt [-n N] [-c CTX] [--resident] [--resident-budget GiB] [--dump-tokens]\n");
            fprintf(stderr, "       ds4-inkling-cuda -m model.gguf --selftest LAYER\n");
            return 1;
        }
    }
    if (!model_path) {
        fprintf(stderr, "usage: ds4-inkling-cuda -m model.gguf -p prompt [-n N] [-c CTX] [--resident] [--resident-budget GiB] [--dump-tokens]\n");
        fprintf(stderr, "       ds4-inkling-cuda -m model.gguf --selftest LAYER\n");
        return 1;
    }

    ink_cuda_init();

    if (selftest_layer >= 0) {
        return ink_run_selftest(model_path, selftest_layer);
    }

    if (!prompt) {
        fprintf(stderr, "usage: ds4-inkling-cuda -m model.gguf -p prompt [-n N] [-c CTX] [--resident] [--resident-budget GiB] [--dump-tokens]\n");
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
        const int chunk = 32;
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
    return 0;
}
