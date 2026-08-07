/* ds4_inkling.c -- self-contained CPU inference for the "inkling" GGUF
 * architecture (Thinking Machines Inkling-Small, 276B/A12B).
 *
 * This is an additive module: it does not touch the deepseek4/GLM engine
 * paths.  It mmaps a canonical llama.cpp-layout inkling GGUF and runs a
 * scalar (OpenMP-parallel over matrix rows) forward pass with greedy
 * sampling.  Built as the `ds4-inkling` CLI.
 *
 * Architecture semantics were ported from the llama.cpp implementation
 * (MIT license, https://github.com/ggml-org/llama.cpp, PR #25731,
 * src/models/inkling.cpp + llama-kv-cache.cpp + llama-vocab.cpp).  The
 * IQ/K-quant row dequantizers are adapted from llama.cpp ggml-quants.c
 * (MIT).  See PORT_NOTES.md for the survey and the semantics summary.
 *
 * Model shape (verified against the real GGUF header):
 *   42 blocks (2 dense FFN + 40 MoE), n_embd 4096, 32 q heads / 8 kv
 *   heads, head_dim 128, GQA KV cache, no RoPE: relative attention bias
 *   (attn_r [4096,512] projected through attn_rel_proj [E,16]) plus
 *   log-N tau scaling on global layers; hybrid SWA (sliding_window 512,
 *   per-layer pattern); per-block 4-tap short convolutions on the flat
 *   k/v projections and on the attn/mlp block outputs; MoE: 256 routed
 *   experts top-6 + 2 score-weighted shared experts, sigmoid gating with
 *   selection bias (exp_probs_b), weights = softmax(logsigmoid(raw
 *   logits)) * expert_weights_scale * ffn_gscale; gpt2 BPE with the
 *   "inkling" (o200k-family) pre-tokenizer; logits scaled by
 *   1/logit_scale_denom and padded vocab rows masked to -inf. */

#include <assert.h>
#include <fcntl.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "ds4_inkling_tables.inc"

/* ========================== small utilities ========================== */

static void ink_die(const char *msg) {
    fprintf(stderr, "ds4-inkling: %s\n", msg);
    exit(1);
}

static void ink_dief(const char *fmt, const char *a) {
    fprintf(stderr, "ds4-inkling: ");
    fprintf(stderr, fmt, a);
    fprintf(stderr, "\n");
    exit(1);
}

static void *ink_malloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) ink_die("out of memory");
    return p;
}

static void *ink_calloc(size_t n, size_t sz) {
    void *p = calloc(n ? n : 1, sz);
    if (!p) ink_die("out of memory");
    return p;
}

static float fp16_to_fp32(uint16_t h) {
    const float sign = (h & 0x8000) ? -1.0f : 1.0f;
    const uint32_t exp = (h >> 10) & 0x1f;
    const uint32_t mant = h & 0x3ff;
    if (exp == 0) return sign * ldexpf((float)mant, -24);
    if (exp == 31) return mant ? NAN : sign * INFINITY;
    return sign * ldexpf((float)(mant | 0x400), (int)exp - 25);
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ============================ GGUF reader ============================ */

/* GGML tensor type ids present in inkling artifacts. */
#define INK_T_F32     0
#define INK_T_F16     1
#define INK_T_Q8_0    8
#define INK_T_Q4_K   12
#define INK_T_Q5_K   13
#define INK_T_Q6_K   14
#define INK_T_IQ2_XXS 16
#define INK_T_IQ3_XXS 18
#define INK_T_IQ2_S  22
#define INK_T_IQ4_XS 23
#define INK_T_BF16   30

#define QK_K 256
#define QK8_0 32

typedef struct { uint16_t d; int8_t qs[QK8_0]; } ink_block_q8_0;
typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[QK_K/2]; } ink_block_q4_K;
typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qh[QK_K/8]; uint8_t qs[QK_K/2]; } ink_block_q5_K;
typedef struct { uint8_t ql[QK_K/2]; uint8_t qh[QK_K/4]; int8_t scales[QK_K/16]; uint16_t d; } ink_block_q6_K;
typedef struct { uint16_t d; uint16_t qs[QK_K/8]; } ink_block_iq2_xxs;
typedef struct { uint16_t d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32]; uint8_t scales[QK_K/32]; } ink_block_iq2_s;
typedef struct { uint16_t d; uint8_t qs[3*QK_K/8]; } ink_block_iq3_xxs;
typedef struct { uint16_t d; uint16_t scales_h; uint8_t scales_l[QK_K/64]; uint8_t qs[QK_K/2]; } ink_block_iq4_xs;

static size_t ink_type_block_elems(uint32_t t) {
    switch (t) {
    case INK_T_F32: case INK_T_F16: case INK_T_BF16: return 1;
    case INK_T_Q8_0: return QK8_0;
    default: return QK_K;
    }
}

static size_t ink_type_block_bytes(uint32_t t) {
    switch (t) {
    case INK_T_F32: return 4;
    case INK_T_F16: case INK_T_BF16: return 2;
    case INK_T_Q8_0: return sizeof(ink_block_q8_0);
    case INK_T_Q4_K: return sizeof(ink_block_q4_K);
    case INK_T_Q5_K: return sizeof(ink_block_q5_K);
    case INK_T_Q6_K: return sizeof(ink_block_q6_K);
    case INK_T_IQ2_XXS: return sizeof(ink_block_iq2_xxs);
    case INK_T_IQ3_XXS: return sizeof(ink_block_iq3_xxs);
    case INK_T_IQ2_S: return sizeof(ink_block_iq2_s);
    case INK_T_IQ4_XS: return sizeof(ink_block_iq4_xs);
    default: ink_die("unsupported tensor type"); return 0;
    }
}

typedef struct {
    char name[128];
    uint32_t type;
    uint32_t ndim;
    uint64_t dims[4];
    const uint8_t *data;    /* mapped pointer */
} ink_tensor;

typedef struct { const char *ptr; uint64_t len; } ink_str;

typedef struct {
    char key[96];
    uint32_t type;          /* gguf value type */
    uint32_t arr_type;      /* element type when type==9 */
    uint64_t arr_len;
    const uint8_t *val;     /* pointer into the mapped header at the value */
} ink_kv;

typedef struct {
    int fd;
    const uint8_t *map;
    size_t map_len;
    ink_kv *kv;
    uint64_t n_kv;
    ink_tensor *tensors;
    uint64_t n_tensors;
} ink_gguf;

typedef struct { const uint8_t *p; const uint8_t *end; } ink_cur;

static void cur_need(ink_cur *c, size_t n) {
    if ((size_t)(c->end - c->p) < n) ink_die("truncated GGUF header");
}

static uint64_t cur_u64(ink_cur *c) { cur_need(c, 8); uint64_t v; memcpy(&v, c->p, 8); c->p += 8; return v; }
static uint32_t cur_u32(ink_cur *c) { cur_need(c, 4); uint32_t v; memcpy(&v, c->p, 4); c->p += 4; return v; }
static ink_str cur_str(ink_cur *c) {
    ink_str s;
    s.len = cur_u64(c);
    cur_need(c, s.len);
    s.ptr = (const char *)c->p;
    c->p += s.len;
    return s;
}

static size_t gguf_scalar_size(uint32_t t) {
    switch (t) {
    case 0: case 1: case 7: return 1;       /* u8 i8 bool */
    case 2: case 3: return 2;               /* u16 i16 */
    case 4: case 5: case 6: return 4;       /* u32 i32 f32 */
    case 10: case 11: case 12: return 8;    /* u64 i64 f64 */
    default: return 0;
    }
}

/* Skip (or record) one value.  Strings and arrays are variable length. */
static void cur_skip_value(ink_cur *c, uint32_t t) {
    if (t == 8) { cur_str(c); return; }
    if (t == 9) {
        uint32_t et = cur_u32(c);
        uint64_t n = cur_u64(c);
        if (et == 8) {
            for (uint64_t i = 0; i < n; i++) cur_str(c);
        } else if (et == 9) {
            for (uint64_t i = 0; i < n; i++) cur_skip_value(c, 9);
        } else {
            size_t sz = gguf_scalar_size(et);
            if (!sz) ink_die("bad gguf array elem type");
            cur_need(c, sz * n);
            c->p += sz * n;
        }
        return;
    }
    size_t sz = gguf_scalar_size(t);
    if (!sz) ink_die("bad gguf value type");
    cur_need(c, sz);
    c->p += sz;
}

static void ink_gguf_open(ink_gguf *g, const char *path) {
    memset(g, 0, sizeof(*g));
    g->fd = open(path, O_RDONLY);
    if (g->fd < 0) ink_dief("cannot open %s", path);
    struct stat st;
    if (fstat(g->fd, &st) != 0) ink_die("fstat failed");
    g->map_len = (size_t)st.st_size;
    g->map = mmap(NULL, g->map_len, PROT_READ, MAP_SHARED, g->fd, 0);
    if (g->map == MAP_FAILED) ink_die("mmap failed");

    ink_cur c = { g->map, g->map + g->map_len };
    if (cur_u32(&c) != 0x46554747u) ink_die("not a GGUF file");
    uint32_t ver = cur_u32(&c);
    if (ver < 2) ink_die("GGUF version too old");
    g->n_tensors = cur_u64(&c);
    g->n_kv = cur_u64(&c);
    g->kv = ink_calloc(g->n_kv, sizeof(ink_kv));
    for (uint64_t i = 0; i < g->n_kv; i++) {
        ink_str k = cur_str(&c);
        uint32_t t = cur_u32(&c);
        ink_kv *kv = &g->kv[i];
        snprintf(kv->key, sizeof(kv->key), "%.*s", (int)(k.len < 95 ? k.len : 95), k.ptr);
        kv->type = t;
        if (t == 9) {
            kv->arr_type = cur_u32(&c);
            kv->arr_len = cur_u64(&c);
            kv->val = c.p;
            /* skip the payload */
            if (kv->arr_type == 8) {
                for (uint64_t j = 0; j < kv->arr_len; j++) cur_str(&c);
            } else {
                size_t sz = gguf_scalar_size(kv->arr_type);
                if (!sz) ink_die("bad gguf array elem type");
                cur_need(&c, sz * kv->arr_len);
                c.p += sz * kv->arr_len;
            }
        } else {
            kv->val = c.p;
            cur_skip_value(&c, t);
        }
    }
    g->tensors = ink_calloc(g->n_tensors, sizeof(ink_tensor));
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        ink_tensor *t = &g->tensors[i];
        ink_str nm = cur_str(&c);
        snprintf(t->name, sizeof(t->name), "%.*s", (int)(nm.len < 127 ? nm.len : 127), nm.ptr);
        t->ndim = cur_u32(&c);
        if (t->ndim > 4) ink_die("tensor ndim > 4");
        for (uint32_t d = 0; d < t->ndim; d++) t->dims[d] = cur_u64(&c);
        t->type = cur_u32(&c);
        uint64_t off = cur_u64(&c);
        t->data = (const uint8_t *)(uintptr_t)off;   /* fixed up below */
    }
    /* data section starts at the header end aligned to general.alignment */
    uint64_t align = 32;
    for (uint64_t i = 0; i < g->n_kv; i++) {
        if (strcmp(g->kv[i].key, "general.alignment") == 0 && g->kv[i].type == 4) {
            memcpy(&align, g->kv[i].val, 4);
        }
    }
    size_t hdr = (size_t)(c.p - g->map);
    size_t base = (hdr + align - 1) / align * align;
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        g->tensors[i].data = g->map + base + (uintptr_t)g->tensors[i].data;
    }
}

static const ink_kv *ink_kv_find(const ink_gguf *g, const char *key) {
    for (uint64_t i = 0; i < g->n_kv; i++) {
        if (strcmp(g->kv[i].key, key) == 0) return &g->kv[i];
    }
    return NULL;
}

static bool ink_get_u32(const ink_gguf *g, const char *key, uint32_t *out) {
    const ink_kv *kv = ink_kv_find(g, key);
    if (!kv) return false;
    if (kv->type == 4 || kv->type == 5) { memcpy(out, kv->val, 4); return true; }
    if (kv->type == 10 || kv->type == 11) { uint64_t v; memcpy(&v, kv->val, 8); *out = (uint32_t)v; return true; }
    return false;
}

static bool ink_get_f32(const ink_gguf *g, const char *key, float *out) {
    const ink_kv *kv = ink_kv_find(g, key);
    if (!kv || kv->type != 6) return false;
    memcpy(out, kv->val, 4);
    return true;
}

static bool ink_get_str(const ink_gguf *g, const char *key, ink_str *out) {
    const ink_kv *kv = ink_kv_find(g, key);
    if (!kv || kv->type != 8) return false;
    ink_cur c = { kv->val, g->map + g->map_len };
    *out = cur_str(&c);
    return true;
}

static const ink_tensor *ink_tensor_find(const ink_gguf *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        if (strcmp(g->tensors[i].name, name) == 0) return &g->tensors[i];
    }
    return NULL;
}

static const ink_tensor *ink_tensor_get(const ink_gguf *g, const char *fmt, int blk) {
    char name[128];
    snprintf(name, sizeof(name), fmt, blk);
    const ink_tensor *t = ink_tensor_find(g, name);
    if (!t) ink_dief("missing tensor %s", name);
    return t;
}

/* ========================= row dequantization ========================
 * Adapted from llama.cpp ggml-quants.c (MIT).  Each function expands one
 * quantized row segment of k values into f32. */

static void ink_dq_q8_0(const void *vx, float *y, int64_t k) {
    const ink_block_q8_0 *x = vx;
    const int nb = (int)(k / QK8_0);
    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < QK8_0; ++j) y[i*QK8_0 + j] = x[i].qs[j]*d;
    }
}

static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

static void ink_dq_q4_K(const void *vx, float *y, int64_t k) {
    const ink_block_q4_K *x = vx;
    const int nb = (int)(k / QK_K);
    for (int i = 0; i < nb; i++) {
        const uint8_t *q = x[i].qs;
        const float d = fp16_to_fp32(x[i].d);
        const float min = fp16_to_fp32(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }
    }
}

static void ink_dq_q5_K(const void *vx, float *y, int64_t k) {
    const ink_block_q5_K *x = vx;
    const int nb = (int)(k / QK_K);
    for (int i = 0; i < nb; i++) {
        const uint8_t *ql = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const float d = fp16_to_fp32(x[i].d);
        const float min = fp16_to_fp32(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l]  >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

static void ink_dq_q6_K(const void *vx, float *y, int64_t k) {
    const ink_block_q6_K *x = vx;
    const int nb = (int)(k / QK_K);
    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t *sc = x[i].scales;
        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128; ql += 64; qh += 32; sc += 8;
        }
    }
}

static void ink_dq_iq2_xxs(const void *vx, float *y, int64_t k) {
    const ink_block_iq2_xxs *x = vx;
    const int nb = (int)(k / QK_K);
    uint32_t aux32[2];
    const uint8_t *aux8 = (const uint8_t *)aux32;
    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(aux32, x[i].qs + 4*ib32, 2*sizeof(uint32_t));
            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t *grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    }
}

static void ink_dq_iq2_s(const void *vx, float *y, int64_t k) {
    const ink_block_iq2_s *x = vx;
    const int nb = (int)(k / QK_K);
    float db[2];
    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const uint8_t *qs = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const uint8_t *signs = qs + QK_K/8;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            db[0] = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (x[i].scales[ib32] >>  4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const float dl = db[l/2];
                const uint8_t *grid = (const uint8_t *)(iq2s_grid + (qs[l] | ((qh[ib32] << (8-2*l)) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    y[j] = dl * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 4;
            signs += 4;
        }
    }
}

static void ink_dq_iq3_xxs(const void *vx, float *y, int64_t k) {
    const ink_block_iq3_xxs *x = vx;
    const int nb = (int)(k / QK_K);
    uint32_t aux32;
    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const uint8_t *qs = x[i].qs;
        const uint8_t *scales_and_signs = qs + QK_K/4;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(&aux32, scales_and_signs + 4*ib32, sizeof(uint32_t));
            const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                const uint8_t *grid1 = (const uint8_t *)(iq3xxs_grid + qs[2*l+0]);
                const uint8_t *grid2 = (const uint8_t *)(iq3xxs_grid + qs[2*l+1]);
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = db * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    y[j+4] = db * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
        }
    }
}

static void ink_dq_iq4_xs(const void *vx, float *y, int64_t k) {
    const ink_block_iq4_xs *x = vx;
    const int nb = (int)(k / QK_K);
    for (int i = 0; i < nb; i++) {
        const uint8_t *qs = x[i].qs;
        const float d = fp16_to_fp32(x[i].d);
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const int ls = ((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl * kvalues_iq4nl[qs[j] & 0xf];
                y[j+16] = dl * kvalues_iq4nl[qs[j] >>  4];
            }
            y  += 32;
            qs += 16;
        }
    }
}

static void ink_dq_f16(const void *vx, float *y, int64_t k) {
    const uint16_t *x = vx;
    for (int64_t i = 0; i < k; i++) y[i] = fp16_to_fp32(x[i]);
}

static void ink_dq_bf16(const void *vx, float *y, int64_t k) {
    const uint16_t *x = vx;
    for (int64_t i = 0; i < k; i++) {
        uint32_t bits = (uint32_t)x[i] << 16;
        memcpy(&y[i], &bits, 4);
    }
}

typedef void (*ink_dq_fn)(const void *, float *, int64_t);

static ink_dq_fn ink_dq_for(uint32_t type) {
    switch (type) {
    case INK_T_F16: return ink_dq_f16;
    case INK_T_BF16: return ink_dq_bf16;
    case INK_T_Q8_0: return ink_dq_q8_0;
    case INK_T_Q4_K: return ink_dq_q4_K;
    case INK_T_Q5_K: return ink_dq_q5_K;
    case INK_T_Q6_K: return ink_dq_q6_K;
    case INK_T_IQ2_XXS: return ink_dq_iq2_xxs;
    case INK_T_IQ3_XXS: return ink_dq_iq3_xxs;
    case INK_T_IQ2_S: return ink_dq_iq2_s;
    case INK_T_IQ4_XS: return ink_dq_iq4_xs;
    default: ink_die("no dequant for tensor type"); return NULL;
    }
}

/* Dequantize row `r` (dims[0] values) of a 2D/3D tensor slice. */
static void ink_row_f32(const ink_tensor *t, const uint8_t *base,
                        uint64_t row, uint64_t row_len, float *out) {
    if (t->type == INK_T_F32) {
        memcpy(out, base + row * row_len * 4, row_len * 4);
        return;
    }
    size_t be = ink_type_block_elems(t->type);
    size_t bb = ink_type_block_bytes(t->type);
    if (row_len % be) ink_die("row not block aligned");
    const uint8_t *src = base + row * (row_len / be) * bb;
    ink_dq_for(t->type)(src, out, (int64_t)row_len);
}

/* ============================ linear algebra ========================= */

static float ink_dot(const float *a, const float *b, uint64_t n) {
    float s = 0.0f;
    for (uint64_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

/* y[o] = sum_i W[o][i] * x[i]; W is a GGUF 2D tensor ne={in,out} (dim0
 * contiguous).  base allows 3D expert slices.  Parallel over rows. */
static void ink_matvec(const ink_tensor *t, const uint8_t *base,
                       uint64_t in, uint64_t out, const float *x, float *y) {
    #pragma omp parallel
    {
        float *buf = ink_malloc(in * sizeof(float));
        #pragma omp for schedule(static)
        for (uint64_t o = 0; o < out; o++) {
            ink_row_f32(t, base, o, in, buf);
            y[o] = ink_dot(buf, x, in);
        }
        free(buf);
    }
}

static void ink_rmsnorm(float *x, const float *w, uint64_t n, float eps) {
    float ss = 0.0f;
    for (uint64_t i = 0; i < n; i++) ss += x[i] * x[i];
    float scale = 1.0f / sqrtf(ss / (float)n + eps);
    for (uint64_t i = 0; i < n; i++) x[i] = x[i] * scale * w[i];
}

static float ink_silu(float x) { return x / (1.0f + expf(-x)); }

static float ink_logsigmoid(float x) {
    /* logsigmoid(x) = -softplus(-x), numerically stable */
    if (x >= 0.0f) return -log1pf(expf(-x));
    return x - log1pf(expf(x));
}

/* ============================== tokenizer ============================ */

/* gpt2 byte <-> unicode mapping.  byte_to_cp[b] is the codepoint the BPE
 * vocab uses to represent raw byte b. */
static uint32_t ink_byte_to_cp[256];
static int ink_cp_to_byte(uint32_t cp) {
    for (int b = 0; b < 256; b++) if (ink_byte_to_cp[b] == cp) return b;
    return -1;
}

static void ink_init_byte_map(void) {
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if ((b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)) {
            ink_byte_to_cp[b] = (uint32_t)b;
        } else {
            ink_byte_to_cp[b] = 256 + n++;
        }
    }
}

static int ink_cp_utf8(uint32_t cp, char *out) {
    if (cp < 0x80) { out[0] = (char)cp; return 1; }
    if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    out[0] = (char)(0xE0 | (cp >> 12));
    out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[2] = (char)(0x80 | (cp & 0x3F));
    return 3;
}

static uint32_t ink_utf8_cp(const char *s, int len, int *adv) {
    const uint8_t *u = (const uint8_t *)s;
    if (u[0] < 0x80) { *adv = 1; return u[0]; }
    if ((u[0] & 0xE0) == 0xC0 && len >= 2) { *adv = 2; return ((u[0] & 0x1F) << 6) | (u[1] & 0x3F); }
    if ((u[0] & 0xF0) == 0xE0 && len >= 3) { *adv = 3; return ((u[0] & 0x0F) << 12) | ((u[1] & 0x3F) << 6) | (u[2] & 0x3F); }
    if ((u[0] & 0xF8) == 0xF0 && len >= 4) {
        *adv = 4;
        return ((u[0] & 0x07) << 18) | ((u[1] & 0x3F) << 12) | ((u[2] & 0x3F) << 6) | (u[3] & 0x3F);
    }
    *adv = 1;
    return u[0];
}

/* FNV string hashmap: key -> int value. */
typedef struct { char *key; uint32_t len; int val; } ink_map_slot;
typedef struct { ink_map_slot *slots; uint64_t mask; } ink_map;

static uint64_t ink_hash(const char *s, uint32_t len) {
    uint64_t h = 1469598103934665603ULL;
    for (uint32_t i = 0; i < len; i++) { h ^= (uint8_t)s[i]; h *= 1099511628211ULL; }
    return h;
}

static void ink_map_init(ink_map *m, uint64_t cap_pow2) {
    m->slots = ink_calloc(cap_pow2, sizeof(ink_map_slot));
    m->mask = cap_pow2 - 1;
}

static void ink_map_put(ink_map *m, const char *key, uint32_t len, int val) {
    uint64_t i = ink_hash(key, len) & m->mask;
    while (m->slots[i].key) {
        if (m->slots[i].len == len && memcmp(m->slots[i].key, key, len) == 0) {
            return; /* keep first (merge ranks: earlier wins) */
        }
        i = (i + 1) & m->mask;
    }
    m->slots[i].key = ink_malloc(len + 1);
    memcpy(m->slots[i].key, key, len);
    m->slots[i].key[len] = 0;
    m->slots[i].len = len;
    m->slots[i].val = val;
}

static int ink_map_get(const ink_map *m, const char *key, uint32_t len) {
    uint64_t i = ink_hash(key, len) & m->mask;
    while (m->slots[i].key) {
        if (m->slots[i].len == len && memcmp(m->slots[i].key, key, len) == 0) {
            return m->slots[i].val;
        }
        i = (i + 1) & m->mask;
    }
    return -1;
}

typedef struct {
    ink_str *tokens;        /* byte-encoded vocab strings */
    uint32_t n_tokens;
    ink_map vocab;          /* token string -> id */
    ink_map merges;         /* "left right" -> rank */
    int bos, eos;
} ink_tokenizer;

static void ink_tokenizer_init(ink_tokenizer *tk, const ink_gguf *g) {
    memset(tk, 0, sizeof(*tk));
    ink_init_byte_map();
    const ink_kv *toks = ink_kv_find(g, "tokenizer.ggml.tokens");
    if (!toks || toks->type != 9 || toks->arr_type != 8) ink_die("tokenizer.ggml.tokens missing");
    tk->n_tokens = (uint32_t)toks->arr_len;
    tk->tokens = ink_calloc(tk->n_tokens, sizeof(ink_str));
    ink_map_init(&tk->vocab, 1u << 19);
    ink_cur c = { toks->val, g->map + g->map_len };
    for (uint32_t i = 0; i < tk->n_tokens; i++) {
        tk->tokens[i] = cur_str(&c);
        ink_map_put(&tk->vocab, tk->tokens[i].ptr, (uint32_t)tk->tokens[i].len, (int)i);
    }
    const ink_kv *mg = ink_kv_find(g, "tokenizer.ggml.merges");
    if (!mg || mg->type != 9 || mg->arr_type != 8) ink_die("tokenizer.ggml.merges missing");
    ink_map_init(&tk->merges, 1u << 21);
    ink_cur mc = { mg->val, g->map + g->map_len };
    for (uint64_t i = 0; i < mg->arr_len; i++) {
        ink_str s = cur_str(&mc);
        ink_map_put(&tk->merges, s.ptr, (uint32_t)s.len, (int)i);
    }
    uint32_t id;
    tk->bos = ink_get_u32(g, "tokenizer.ggml.bos_token_id", &id) ? (int)id : -1;
    tk->eos = ink_get_u32(g, "tokenizer.ggml.eos_token_id", &id) ? (int)id : -1;
}

/* --- pre-tokenizer: the "inkling" o200k-family regex, hand-compiled.
 *
 * "[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))*((?=[\p{L}\p{M}])([^A-Z]))+('(s|t|re|ve|m|ll|d) ci)?
 *  |[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))+((?=[\p{L}\p{M}])([^A-Z]))*('...)?
 *  |\p{N}{1,3}
 *  | ?[^\s\p{L}\p{N}]+[\r\n/]*
 *  |\s*[\r\n]+
 *  |\s+(?!\S)
 *  |\s+"
 *
 * Unicode class handling is exact for ASCII; codepoints >= 0x80 that are
 * not whitespace are treated as caseless letters (correct for the
 * overwhelmingly common cases; matches llama.cpp behavior for CJK and
 * accented text where the char is neither [a-z] nor [A-Z] ASCII). */

static bool cp_is_ws(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\f' || cp == 0x0b ||
           cp == 0x85 || cp == 0xA0 || (cp >= 0x2000 && cp <= 0x200A) ||
           cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000 || cp == 0x1680;
}
static bool cp_is_digit(uint32_t cp) { return cp >= '0' && cp <= '9'; }
static bool cp_is_letter(uint32_t cp) {
    if (cp < 0x80) return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z');
    return !cp_is_ws(cp);
}
static bool cp_is_lower_ascii(uint32_t cp) { return cp >= 'a' && cp <= 'z'; }
static bool cp_is_upper_ascii(uint32_t cp) { return cp >= 'A' && cp <= 'Z'; }

typedef void (*ink_piece_fn)(void *ud, const char *ptr, int len);

static int ink_match_contraction(const char *s, int len) {
    if (len < 2 || s[0] != '\'') return 0;
    char c1 = s[1] | 0x20;
    if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return 2;
    if (len >= 3) {
        char c2 = s[2] | 0x20;
        if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) return 3;
    }
    return 0;
}

static void ink_pretokenize(const char *text, int len, ink_piece_fn emit, void *ud) {
    int i = 0;
    while (i < len) {
        int adv;
        uint32_t cp = ink_utf8_cp(text + i, len - i, &adv);
        int start = i;

        /* rule 1/2: optional single non-[\r\n L N] prefix + letter run */
        {
            int j = i;
            uint32_t c0 = cp;
            int a0 = adv;
            bool prefix_ok = c0 != '\r' && c0 != '\n' && !cp_is_letter(c0) && !cp_is_digit(c0);
            int k = j;
            if (prefix_ok) k = j + a0;
            /* need a letter at k */
            if (k < len) {
                int a1;
                uint32_t c1 = ink_utf8_cp(text + k, len - k, &a1);
                if (cp_is_letter(c1)) {
                    int p = k;
                    /* U-part: letters that are not ASCII lowercase */
                    int m1 = 0;
                    while (p < len) {
                        int a; uint32_t c = ink_utf8_cp(text + p, len - p, &a);
                        if (!cp_is_letter(c) || cp_is_lower_ascii(c)) break;
                        p += a; m1++;
                    }
                    /* L-part: letters that are not ASCII uppercase */
                    int m2 = 0;
                    while (p < len) {
                        int a; uint32_t c = ink_utf8_cp(text + p, len - p, &a);
                        if (!cp_is_letter(c) || cp_is_upper_ascii(c)) break;
                        p += a; m2++;
                    }
                    if (m1 + m2 > 0) {
                        p += ink_match_contraction(text + p, len - p);
                        emit(ud, text + start, p - start);
                        i = p;
                        continue;
                    }
                }
            }
        }

        /* rule 3: 1-3 digits */
        if (cp_is_digit(cp)) {
            int p = i, nd = 0;
            while (p < len && nd < 3 && cp_is_digit((uint8_t)text[p])) { p++; nd++; }
            emit(ud, text + start, p - start);
            i = p;
            continue;
        }

        /* rule 4: ' ?[^\s L N]+[\r\n/]*' */
        {
            int j = i;
            if (cp == ' ') j = i + 1;
            if (j < len) {
                int a; uint32_t c = ink_utf8_cp(text + j, len - j, &a);
                if (!cp_is_ws(c) && !cp_is_letter(c) && !cp_is_digit(c)) {
                    int p = j;
                    while (p < len) {
                        int a2; uint32_t c2 = ink_utf8_cp(text + p, len - p, &a2);
                        if (cp_is_ws(c2) || cp_is_letter(c2) || cp_is_digit(c2)) break;
                        p += a2;
                    }
                    while (p < len && (text[p] == '\r' || text[p] == '\n' || text[p] == '/')) p++;
                    emit(ud, text + start, p - start);
                    i = p;
                    continue;
                }
            }
        }

        /* rules 5-7: whitespace */
        if (cp_is_ws(cp)) {
            int p = i;
            int last_nl_end = -1;
            while (p < len) {
                int a; uint32_t c = ink_utf8_cp(text + p, len - p, &a);
                if (!cp_is_ws(c)) break;
                p += a;
                if (c == '\r' || c == '\n') last_nl_end = p;
            }
            int end;
            if (last_nl_end > 0) {
                end = last_nl_end;                 /* \s*[\r\n]+ */
            } else if (p == len) {
                end = p;                           /* \s+(?!\S) at end */
            } else if (p - i > 1) {
                end = p - 1;                       /* \s+(?!\S): leave last ws */
            } else {
                end = p;                           /* single ws before non-ws: \s+ */
            }
            emit(ud, text + start, end - start);
            i = end;
            continue;
        }

        /* fallback: single char */
        emit(ud, text + start, adv);
        i = start + adv;
    }
}

/* BPE merge of one pre-token piece (raw bytes). */
typedef struct { int *ids; int len; int cap; } ink_ids;

static void ink_ids_push(ink_ids *v, int id) {
    if (v->len == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 64;
        v->ids = realloc(v->ids, v->cap * sizeof(int));
        if (!v->ids) ink_die("oom");
    }
    v->ids[v->len++] = id;
}

typedef struct {
    const ink_tokenizer *tk;
    ink_ids *out;
} ink_bpe_ctx;

static void ink_bpe_piece(void *ud, const char *ptr, int len) {
    ink_bpe_ctx *ctx = ud;
    const ink_tokenizer *tk = ctx->tk;
    if (len <= 0) return;

    /* byte-encode each raw byte to its unicode char, then iterative
     * lowest-rank adjacent merges over heap-string symbols */
    {
        char **s = ink_malloc(sizeof(char *) * len);
        int *sl = ink_malloc(sizeof(int) * len);
        int n = len;
        for (int i = 0; i < len; i++) {
            s[i] = ink_malloc(8);
            sl[i] = ink_cp_utf8(ink_byte_to_cp[(uint8_t)ptr[i]], s[i]);
        }
        while (n > 1) {
            int best_rank = INT32_MAX, best_i = -1;
            for (int i = 0; i + 1 < n; i++) {
                int l = sl[i] + 1 + sl[i+1];
                char *pair = ink_malloc(l);
                memcpy(pair, s[i], sl[i]);
                pair[sl[i]] = ' ';
                memcpy(pair + sl[i] + 1, s[i+1], sl[i+1]);
                int r = ink_map_get(&tk->merges, pair, (uint32_t)l);
                free(pair);
                if (r >= 0 && r < best_rank) { best_rank = r; best_i = i; }
            }
            if (best_i < 0) break;
            int nl = sl[best_i] + sl[best_i+1];
            char *m = ink_malloc(nl);
            memcpy(m, s[best_i], sl[best_i]);
            memcpy(m + sl[best_i], s[best_i+1], sl[best_i+1]);
            free(s[best_i]); free(s[best_i+1]);
            s[best_i] = m; sl[best_i] = nl;
            for (int i = best_i + 1; i + 1 < n; i++) { s[i] = s[i+1]; sl[i] = sl[i+1]; }
            n--;
        }
        for (int i = 0; i < n; i++) {
            int id = ink_map_get(&tk->vocab, s[i], (uint32_t)sl[i]);
            if (id < 0) {
                fprintf(stderr, "ds4-inkling: warning: unknown BPE symbol (%d bytes), skipped\n", sl[i]);
            } else {
                ink_ids_push(ctx->out, id);
            }
            free(s[i]);
        }
        free(s); free(sl);
    }
}

static void ink_tokenize(const ink_tokenizer *tk, const char *text, ink_ids *out) {
    ink_bpe_ctx ctx = { tk, out };
    ink_pretokenize(text, (int)strlen(text), ink_bpe_piece, &ctx);
}

/* Decode one token id to raw bytes (caller frees). */
static int ink_detokenize(const ink_tokenizer *tk, int id, char *out, int cap) {
    if (id < 0 || (uint32_t)id >= tk->n_tokens) return 0;
    ink_str s = tk->tokens[id];
    int n = 0;
    int i = 0;
    while (i < (int)s.len && n < cap - 1) {
        int adv;
        uint32_t cp = ink_utf8_cp(s.ptr + i, (int)s.len - i, &adv);
        int b = ink_cp_to_byte(cp);
        if (b >= 0) out[n++] = (char)b;
        else {
            /* not a byte-encoded char (special token text): copy through */
            for (int j = 0; j < adv && n < cap - 1; j++) out[n++] = s.ptr[i + j];
        }
        i += adv;
    }
    out[n] = 0;
    return n;
}

/* =========================== model + forward ========================= */

typedef struct {
    const ink_tensor *attn_norm, *wq, *wk, *wv, *wr, *wo;
    const ink_tensor *q_norm, *k_norm, *rel_proj;
    const ink_tensor *sc_k, *sc_v, *sc_attn, *sc_mlp;
    const ink_tensor *ffn_norm, *gscale;
    /* dense */
    const ink_tensor *ffn_gate, *ffn_up, *ffn_down;
    /* moe */
    const ink_tensor *gate_inp, *probs_b;
    const ink_tensor *gate_exps, *up_exps, *down_exps;
    const ink_tensor *gate_shexp, *up_shexp, *down_shexp;
    bool is_swa;
    uint32_t n_head_kv;
} ink_layer;

typedef struct {
    ink_gguf gg;
    ink_tokenizer tk;

    uint32_t n_layer, n_dense, n_embd, n_head, head_dim, n_ff_dense;
    uint32_t n_expert, n_expert_used, n_shexp, n_ff_exp;
    uint32_t n_swa, d_rel, rel_extent, rel_extent_swa, conv_k;
    uint32_t n_vocab, n_vocab_unpadded;
    uint32_t log_n_floor;
    float log_alpha, logit_scale, expert_weights_scale, rms_eps;

    const ink_tensor *tok_embd, *tok_norm, *out_norm, *output;
    ink_layer *layers;

    /* state */
    uint32_t n_ctx;
    float *kcache;   /* [n_layer][n_ctx][kvw] post-norm k */
    float *vcache;   /* [n_layer][n_ctx][kvw] post-sconv v */
    float *conv;     /* [n_layer][(K-1)*(kvw+kvw+2*n_embd)] rolling inputs */
} ink_model;

static const float *ink_f32(const ink_tensor *t) {
    if (t->type != INK_T_F32) ink_dief("expected f32 tensor %s", t->name);
    return (const float *)t->data;
}

static void ink_model_open(ink_model *m, const char *path, uint32_t n_ctx) {
    memset(m, 0, sizeof(*m));
    ink_gguf_open(&m->gg, path);
    const ink_gguf *g = &m->gg;

    ink_str arch;
    if (!ink_get_str(g, "general.architecture", &arch) ||
        arch.len != 7 || memcmp(arch.ptr, "inkling", 7) != 0) {
        ink_die("general.architecture is not 'inkling'");
    }

    if (!ink_get_u32(g, "inkling.block_count", &m->n_layer)) ink_die("missing block_count");
    if (!ink_get_u32(g, "inkling.dense_block_count", &m->n_dense)) ink_die("missing dense_block_count");
    if (!ink_get_u32(g, "inkling.embedding_length", &m->n_embd)) ink_die("missing embedding_length");
    if (!ink_get_u32(g, "inkling.attention.head_count", &m->n_head)) ink_die("missing head_count");
    if (!ink_get_u32(g, "inkling.attention.key_length", &m->head_dim)) m->head_dim = m->n_embd / m->n_head;
    if (!ink_get_u32(g, "inkling.feed_forward_length", &m->n_ff_dense)) ink_die("missing feed_forward_length");
    if (!ink_get_u32(g, "inkling.expert_count", &m->n_expert)) ink_die("missing expert_count");
    if (!ink_get_u32(g, "inkling.expert_used_count", &m->n_expert_used)) ink_die("missing expert_used_count");
    if (!ink_get_u32(g, "inkling.expert_shared_count", &m->n_shexp)) ink_die("missing expert_shared_count");
    if (!ink_get_u32(g, "inkling.expert_feed_forward_length", &m->n_ff_exp)) ink_die("missing expert_feed_forward_length");
    if (!ink_get_f32(g, "inkling.expert_weights_scale", &m->expert_weights_scale)) ink_die("missing expert_weights_scale");
    if (!ink_get_u32(g, "inkling.attention.sliding_window", &m->n_swa)) ink_die("missing sliding_window");
    if (!ink_get_u32(g, "inkling.d_rel", &m->d_rel)) ink_die("missing d_rel");
    if (!ink_get_u32(g, "inkling.rel_extent", &m->rel_extent)) ink_die("missing rel_extent");
    if (!ink_get_u32(g, "inkling.rel_extent_swa", &m->rel_extent_swa)) ink_die("missing rel_extent_swa");
    if (!ink_get_u32(g, "inkling.shortconv_kernel", &m->conv_k)) ink_die("missing shortconv_kernel");
    if (!ink_get_f32(g, "inkling.attention.layer_norm_rms_epsilon", &m->rms_eps)) ink_die("missing rms eps");
    float denom = 0.0f;
    if (!ink_get_f32(g, "inkling.logit_scale_denom", &denom) || denom == 0.0f) ink_die("missing logit_scale_denom");
    m->logit_scale = 1.0f / denom;
    if (!ink_get_u32(g, "inkling.log_scaling_n_floor", &m->log_n_floor)) m->log_n_floor = 0;
    if (!ink_get_f32(g, "inkling.log_scaling_alpha", &m->log_alpha)) m->log_alpha = 0.0f;
    if (!ink_get_u32(g, "inkling.unpadded_vocab_size", &m->n_vocab_unpadded)) m->n_vocab_unpadded = 0;

    ink_tokenizer_init(&m->tk, g);
    m->n_vocab = m->tk.n_tokens;

    m->tok_embd = ink_tensor_get(g, "token_embd.weight", 0);
    m->tok_norm = ink_tensor_get(g, "token_embd_norm.weight", 0);
    m->out_norm = ink_tensor_get(g, "output_norm.weight", 0);
    m->output   = ink_tensor_get(g, "output.weight", 0);

    /* per-layer kv-head counts + swa pattern (arrays or scalars) */
    const ink_kv *kvh = ink_kv_find(g, "inkling.attention.head_count_kv");
    const ink_kv *swp = ink_kv_find(g, "inkling.attention.sliding_window_pattern");
    if (!kvh) ink_die("missing head_count_kv");
    if (!swp) ink_die("missing sliding_window_pattern");

    m->layers = ink_calloc(m->n_layer, sizeof(ink_layer));
    for (uint32_t il = 0; il < m->n_layer; il++) {
        ink_layer *l = &m->layers[il];

        if (kvh->type == 9) {
            if (kvh->arr_len <= il) ink_die("head_count_kv array too short");
            memcpy(&l->n_head_kv, kvh->val + 4*il, 4);
        } else {
            memcpy(&l->n_head_kv, kvh->val, 4);
        }
        if (swp->type == 9) {
            if (swp->arr_len <= il) ink_die("sliding_window_pattern too short");
            if (swp->arr_type == 7 || swp->arr_type == 0 || swp->arr_type == 1) {
                l->is_swa = swp->val[il] != 0;
            } else {
                uint32_t v; memcpy(&v, swp->val + 4*il, 4);
                l->is_swa = v != 0;
            }
        } else {
            l->is_swa = swp->val[0] != 0;
        }

        l->attn_norm = ink_tensor_get(g, "blk.%d.attn_norm.weight", il);
        l->wq = ink_tensor_get(g, "blk.%d.attn_q.weight", il);
        l->wk = ink_tensor_get(g, "blk.%d.attn_k.weight", il);
        l->wv = ink_tensor_get(g, "blk.%d.attn_v.weight", il);
        l->wr = ink_tensor_get(g, "blk.%d.attn_r.weight", il);
        l->wo = ink_tensor_get(g, "blk.%d.attn_output.weight", il);
        l->q_norm = ink_tensor_get(g, "blk.%d.attn_q_norm.weight", il);
        l->k_norm = ink_tensor_get(g, "blk.%d.attn_k_norm.weight", il);
        l->rel_proj = ink_tensor_get(g, "blk.%d.attn_rel_proj.weight", il);
        l->sc_k = ink_tensor_get(g, "blk.%d.shortconv_k.weight", il);
        l->sc_v = ink_tensor_get(g, "blk.%d.shortconv_v.weight", il);
        l->sc_attn = ink_tensor_get(g, "blk.%d.shortconv_attn.weight", il);
        l->sc_mlp = ink_tensor_get(g, "blk.%d.shortconv_mlp.weight", il);
        l->ffn_norm = ink_tensor_get(g, "blk.%d.ffn_norm.weight", il);
        l->gscale = ink_tensor_get(g, "blk.%d.ffn_gscale.weight", il);
        if (il < m->n_dense) {
            l->ffn_gate = ink_tensor_get(g, "blk.%d.ffn_gate.weight", il);
            l->ffn_up = ink_tensor_get(g, "blk.%d.ffn_up.weight", il);
            l->ffn_down = ink_tensor_get(g, "blk.%d.ffn_down.weight", il);
        } else {
            l->gate_inp = ink_tensor_get(g, "blk.%d.ffn_gate_inp.weight", il);
            l->probs_b = ink_tensor_get(g, "blk.%d.exp_probs_b.bias", il);
            l->gate_exps = ink_tensor_get(g, "blk.%d.ffn_gate_exps.weight", il);
            l->up_exps = ink_tensor_get(g, "blk.%d.ffn_up_exps.weight", il);
            l->down_exps = ink_tensor_get(g, "blk.%d.ffn_down_exps.weight", il);
            l->gate_shexp = ink_tensor_get(g, "blk.%d.ffn_gate_shexp.weight", il);
            l->up_shexp = ink_tensor_get(g, "blk.%d.ffn_up_shexp.weight", il);
            l->down_shexp = ink_tensor_get(g, "blk.%d.ffn_down_shexp.weight", il);
        }
        /* shape sanity for the load-bearing tensors */
        uint32_t kvw = l->n_head_kv * m->head_dim;
        uint32_t rel_extent = l->is_swa ? m->rel_extent_swa : m->rel_extent;
        if (l->wq->dims[0] != m->n_embd || l->wq->dims[1] != m->n_head * m->head_dim) ink_dief("bad shape %s", l->wq->name);
        if (l->wk->dims[1] != kvw || l->wv->dims[1] != kvw) ink_dief("bad shape %s", l->wk->name);
        if (l->wr->dims[1] != (uint64_t)m->n_head * m->d_rel) ink_dief("bad shape %s", l->wr->name);
        if (l->rel_proj->dims[0] != rel_extent || l->rel_proj->dims[1] != m->d_rel) ink_dief("bad shape %s", l->rel_proj->name);
        if (l->sc_k->dims[0] != m->conv_k || l->sc_k->dims[1] != kvw) ink_dief("bad shape %s", l->sc_k->name);
        if (l->sc_attn->dims[0] != m->conv_k || l->sc_attn->dims[1] != m->n_embd) ink_dief("bad shape %s", l->sc_attn->name);
        if (l->gate_inp && l->gate_inp->dims[1] != m->n_expert + m->n_shexp) ink_dief("bad shape %s", l->gate_inp->name);
    }

    m->n_ctx = n_ctx;
    uint32_t kvw_max = 0;
    for (uint32_t il = 0; il < m->n_layer; il++) {
        uint32_t kvw = m->layers[il].n_head_kv * m->head_dim;
        if (kvw > kvw_max) kvw_max = kvw;
    }
    size_t kv_elem = (size_t)m->n_layer * n_ctx * kvw_max;
    m->kcache = ink_calloc(kv_elem, sizeof(float));
    m->vcache = ink_calloc(kv_elem, sizeof(float));
    size_t conv_per_layer = (size_t)(m->conv_k - 1) * (2u * kvw_max + 2u * m->n_embd);
    m->conv = ink_calloc((size_t)m->n_layer * conv_per_layer, sizeof(float));
}

/* Short convolution: y = x + depthwise causal conv over [state, x].
 * kernel gguf ne {K, C}: element (j, c) at c*K + j.  state holds the
 * previous K-1 inputs per channel, laid out [K-1][C] (time-major, oldest
 * first).  Updates state in place. */
static void ink_sconv(const ink_tensor *kernel, float *state, uint32_t C,
                      uint32_t K, float *x) {
    const float *w = ink_f32(kernel);
    uint32_t d = K - 1;
    for (uint32_t c = 0; c < C; c++) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < d; j++) {
            acc += state[j*C + c] * w[c*K + j];
        }
        acc += x[c] * w[c*K + d];
        /* shift state and append x BEFORE overwriting x[c] */
        for (uint32_t j = 0; j + 1 < d; j++) state[j*C + c] = state[(j+1)*C + c];
        state[(d-1)*C + c] = x[c];
        x[c] = x[c] + acc;
    }
}

typedef struct { float score; int idx; } ink_scored;

static int ink_scored_cmp(const void *a, const void *b) {
    float d = ((const ink_scored *)b)->score - ((const ink_scored *)a)->score;
    return d > 0 ? 1 : d < 0 ? -1 : 0;
}

/* One token forward pass; logits (n_vocab) written if out_logits. */
static void ink_forward(ink_model *m, int token, uint32_t pos, float *out_logits) {
    const uint32_t n_embd = m->n_embd;
    const uint32_t n_head = m->n_head;
    const uint32_t hd = m->head_dim;
    const uint32_t K = m->conv_k;

    uint32_t kvw_max = 0;
    for (uint32_t il = 0; il < m->n_layer; il++) {
        uint32_t kvw = m->layers[il].n_head_kv * hd;
        if (kvw > kvw_max) kvw_max = kvw;
    }
    const size_t conv_per_layer = (size_t)(K - 1) * (2u * kvw_max + 2u * n_embd);
    const size_t off_conv_k = 0;
    const size_t off_conv_v = (size_t)(K - 1) * kvw_max;
    const size_t off_conv_attn = (size_t)(K - 1) * 2 * kvw_max;
    const size_t off_conv_mlp = off_conv_attn + (size_t)(K - 1) * n_embd;

    float *x = ink_malloc(n_embd * sizeof(float));
    float *xn = ink_malloc(n_embd * sizeof(float));
    float *q = ink_malloc((size_t)n_head * hd * sizeof(float));
    float *kf = ink_malloc(kvw_max * sizeof(float));
    float *vf = ink_malloc(kvw_max * sizeof(float));
    float *r = ink_malloc((size_t)n_head * m->d_rel * sizeof(float));
    float *rel = ink_malloc((size_t)n_head * m->rel_extent * sizeof(float));
    float *attn_out = ink_malloc((size_t)n_head * hd * sizeof(float));
    float *proj_out = ink_malloc(n_embd * sizeof(float));
    float *scores = ink_malloc(m->n_ctx * sizeof(float));
    uint32_t big_ff = m->n_ff_dense > m->n_ff_exp ? m->n_ff_dense : m->n_ff_exp;
    float *hg = ink_malloc(big_ff * sizeof(float));
    float *hu = ink_malloc(big_ff * sizeof(float));
    float *ff_out = ink_malloc(n_embd * sizeof(float));
    float *ff_acc = ink_malloc(n_embd * sizeof(float));

    /* embedding + embd norm */
    ink_row_f32(m->tok_embd, m->tok_embd->data, (uint64_t)token, n_embd, x);
    ink_rmsnorm(x, ink_f32(m->tok_norm), n_embd, m->rms_eps);

    const float tau = (m->log_n_floor > 0)
        ? 1.0f + m->log_alpha * logf(fmaxf((float)(pos + 1) / (float)m->log_n_floor, 1.0f))
        : 1.0f;

    for (uint32_t il = 0; il < m->n_layer; il++) {
        ink_layer *l = &m->layers[il];
        const uint32_t n_head_kv = l->n_head_kv;
        const uint32_t kvw = n_head_kv * hd;
        const uint32_t rel_extent = l->is_swa ? m->rel_extent_swa : m->rel_extent;
        float *conv_l = m->conv + (size_t)il * conv_per_layer;
        float *kc = m->kcache + ((size_t)il * m->n_ctx) * kvw_max;
        float *vc = m->vcache + ((size_t)il * m->n_ctx) * kvw_max;

        /* ---- attention block ---- */
        memcpy(xn, x, n_embd * sizeof(float));
        ink_rmsnorm(xn, ink_f32(l->attn_norm), n_embd, m->rms_eps);

        ink_matvec(l->wq, l->wq->data, n_embd, (uint64_t)n_head * hd, xn, q);
        ink_matvec(l->wk, l->wk->data, n_embd, kvw, xn, kf);
        ink_matvec(l->wv, l->wv->data, n_embd, kvw, xn, vf);
        ink_matvec(l->wr, l->wr->data, n_embd, (uint64_t)n_head * m->d_rel, xn, r);

        /* k/v short convs on the flat projections */
        ink_sconv(l->sc_k, conv_l + off_conv_k, kvw, K, kf);
        ink_sconv(l->sc_v, conv_l + off_conv_v, kvw, K, vf);

        /* per-head q/k RMS norms */
        for (uint32_t h = 0; h < n_head; h++) {
            ink_rmsnorm(q + (size_t)h * hd, ink_f32(l->q_norm), hd, m->rms_eps);
        }
        for (uint32_t h = 0; h < n_head_kv; h++) {
            ink_rmsnorm(kf + (size_t)h * hd, ink_f32(l->k_norm), hd, m->rms_eps);
        }

        /* log-N tau on q, global layers only */
        if (!l->is_swa && tau != 1.0f) {
            for (uint32_t i = 0; i < (uint32_t)n_head * hd; i++) q[i] *= tau;
        }

        /* cache k (post-norm) and v (post-sconv) */
        memcpy(kc + (size_t)pos * kvw_max, kf, kvw * sizeof(float));
        memcpy(vc + (size_t)pos * kvw_max, vf, kvw * sizeof(float));

        /* rel bias per head: rel[h][e] = sum_d proj[e + d*E] * r[h*d_rel + d],
         * scaled by tau on global layers */
        {
            const ink_tensor *pt = l->rel_proj;
            float *pbuf = NULL;
            const float *pw;
            if (pt->type == INK_T_F32) {
                pw = (const float *)pt->data;
            } else {
                pbuf = ink_malloc((size_t)rel_extent * m->d_rel * sizeof(float));
                for (uint32_t d = 0; d < m->d_rel; d++) {
                    ink_row_f32(pt, pt->data, d, rel_extent, pbuf + (size_t)d * rel_extent);
                }
                pw = pbuf;
            }
            for (uint32_t h = 0; h < n_head; h++) {
                const float *rh = r + (size_t)h * m->d_rel;
                float *relh = rel + (size_t)h * rel_extent;
                for (uint32_t e = 0; e < rel_extent; e++) {
                    float acc = 0.0f;
                    for (uint32_t d = 0; d < m->d_rel; d++) acc += pw[(size_t)d * rel_extent + e] * rh[d];
                    relh[e] = (!l->is_swa && tau != 1.0f) ? acc * tau : acc;
                }
            }
            free(pbuf);
        }

        /* attention */
        uint32_t j0 = 0;
        if (l->is_swa && pos + 1 > m->n_swa) j0 = pos + 1 - m->n_swa;
        const uint32_t gqa = n_head / n_head_kv;
        const float inv_hd = 1.0f / (float)hd;
        #pragma omp parallel for schedule(static)
        for (uint32_t h = 0; h < n_head; h++) {
            const float *qh = q + (size_t)h * hd;
            const uint32_t hkv = h / gqa;
            const float *relh = rel + (size_t)h * rel_extent;
            float *sc = scores; /* shared scratch: give each thread its own */
            float local_scores[4096];
            sc = (pos + 1 - j0) <= 4096 ? local_scores : NULL;
            if (!sc) sc = ink_malloc((pos + 1 - j0) * sizeof(float));
            float maxs = -1e30f;
            for (uint32_t j = j0; j <= pos; j++) {
                const float *kj = kc + (size_t)j * kvw_max + (size_t)hkv * hd;
                float s = ink_dot(qh, kj, hd) * inv_hd;
                uint32_t d = pos - j;
                if (d < rel_extent) s += relh[d];
                sc[j - j0] = s;
                if (s > maxs) maxs = s;
            }
            float sum = 0.0f;
            for (uint32_t j = j0; j <= pos; j++) {
                sc[j - j0] = expf(sc[j - j0] - maxs);
                sum += sc[j - j0];
            }
            float inv_sum = 1.0f / sum;
            float *oh = attn_out + (size_t)h * hd;
            memset(oh, 0, hd * sizeof(float));
            for (uint32_t j = j0; j <= pos; j++) {
                const float *vj = vc + (size_t)j * kvw_max + (size_t)hkv * hd;
                float w = sc[j - j0] * inv_sum;
                for (uint32_t i = 0; i < hd; i++) oh[i] += w * vj[i];
            }
            if (sc != local_scores) free(sc);
        }

        ink_matvec(l->wo, l->wo->data, (uint64_t)n_head * hd, n_embd, attn_out, proj_out);

        /* attn output short conv + residual */
        ink_sconv(l->sc_attn, conv_l + off_conv_attn, n_embd, K, proj_out);
        for (uint32_t i = 0; i < n_embd; i++) x[i] += proj_out[i];

        /* ---- ffn block ---- */
        memcpy(xn, x, n_embd * sizeof(float));
        ink_rmsnorm(xn, ink_f32(l->ffn_norm), n_embd, m->rms_eps);
        const float gscale = ink_f32(l->gscale)[0];

        if (il < m->n_dense) {
            uint32_t nf = m->n_ff_dense;
            ink_matvec(l->ffn_gate, l->ffn_gate->data, n_embd, nf, xn, hg);
            ink_matvec(l->ffn_up, l->ffn_up->data, n_embd, nf, xn, hu);
            for (uint32_t i = 0; i < nf; i++) hg[i] = ink_silu(hg[i]) * hu[i];
            ink_matvec(l->ffn_down, l->ffn_down->data, nf, n_embd, hg, ff_out);
            for (uint32_t i = 0; i < n_embd; i++) ff_out[i] *= gscale;
        } else {
            const uint32_t nE = m->n_expert, nu = m->n_expert_used, ns = m->n_shexp;
            float *logits = ink_malloc((nE + ns) * sizeof(float));
            ink_matvec(l->gate_inp, l->gate_inp->data, n_embd, nE + ns, xn, logits);

            const float *bias = ink_f32(l->probs_b);
            ink_scored *ranked = ink_malloc(nE * sizeof(ink_scored));
            for (uint32_t e = 0; e < nE; e++) {
                ranked[e].score = 1.0f / (1.0f + expf(-logits[e])) + bias[e];
                ranked[e].idx = (int)e;
            }
            qsort(ranked, nE, sizeof(ink_scored), ink_scored_cmp);

            /* weights: softmax over logsigmoid of the raw top-k + shared
             * logits, * expert_weights_scale * gscale */
            float wv[64];
            float wmax = -1e30f;
            for (uint32_t i = 0; i < nu; i++) {
                wv[i] = ink_logsigmoid(logits[ranked[i].idx]);
                if (wv[i] > wmax) wmax = wv[i];
            }
            for (uint32_t s = 0; s < ns; s++) {
                wv[nu + s] = ink_logsigmoid(logits[nE + s]);
                if (wv[nu + s] > wmax) wmax = wv[nu + s];
            }
            float wsum = 0.0f;
            for (uint32_t i = 0; i < nu + ns; i++) { wv[i] = expf(wv[i] - wmax); wsum += wv[i]; }
            const float wscale = m->expert_weights_scale * gscale / wsum;
            for (uint32_t i = 0; i < nu + ns; i++) wv[i] *= wscale;

            memset(ff_acc, 0, n_embd * sizeof(float));
            const uint32_t nf = m->n_ff_exp;

            /* routed experts */
            for (uint32_t i = 0; i < nu; i++) {
                const uint32_t e = (uint32_t)ranked[i].idx;
                size_t grow = (size_t)e * nf;   /* expert base row in the 3D bank */
                const uint8_t *gbase = l->gate_exps->data;
                const uint8_t *ubase = l->up_exps->data;
                const uint8_t *dbase = l->down_exps->data;
                size_t g_rb = (n_embd / ink_type_block_elems(l->gate_exps->type)) * ink_type_block_bytes(l->gate_exps->type);
                size_t u_rb = (n_embd / ink_type_block_elems(l->up_exps->type)) * ink_type_block_bytes(l->up_exps->type);
                size_t d_rb = (nf / ink_type_block_elems(l->down_exps->type)) * ink_type_block_bytes(l->down_exps->type);
                ink_matvec(l->gate_exps, gbase + grow * g_rb, n_embd, nf, xn, hg);
                ink_matvec(l->up_exps, ubase + grow * u_rb, n_embd, nf, xn, hu);
                for (uint32_t t2 = 0; t2 < nf; t2++) hg[t2] = ink_silu(hg[t2]) * hu[t2];
                ink_matvec(l->down_exps, dbase + (size_t)e * n_embd * d_rb, nf, n_embd, hg, ff_out);
                const float w = wv[i];
                for (uint32_t t2 = 0; t2 < n_embd; t2++) ff_acc[t2] += w * ff_out[t2];
            }

            /* shared experts: gamma applied to hidden BEFORE down-proj */
            for (uint32_t s = 0; s < ns; s++) {
                size_t srow = (size_t)s * nf;
                size_t g_rb = (n_embd / ink_type_block_elems(l->gate_shexp->type)) * ink_type_block_bytes(l->gate_shexp->type);
                size_t u_rb = (n_embd / ink_type_block_elems(l->up_shexp->type)) * ink_type_block_bytes(l->up_shexp->type);
                size_t d_rb = (nf / ink_type_block_elems(l->down_shexp->type)) * ink_type_block_bytes(l->down_shexp->type);
                ink_matvec(l->gate_shexp, l->gate_shexp->data + srow * g_rb, n_embd, nf, xn, hg);
                ink_matvec(l->up_shexp, l->up_shexp->data + srow * u_rb, n_embd, nf, xn, hu);
                const float gamma = wv[nu + s];
                for (uint32_t t2 = 0; t2 < nf; t2++) hg[t2] = ink_silu(hg[t2]) * hu[t2] * gamma;
                ink_matvec(l->down_shexp, l->down_shexp->data + (size_t)s * n_embd * d_rb, nf, n_embd, hg, ff_out);
                for (uint32_t t2 = 0; t2 < n_embd; t2++) ff_acc[t2] += ff_out[t2];
            }
            memcpy(ff_out, ff_acc, n_embd * sizeof(float));
            free(logits);
            free(ranked);
        }

        /* mlp output short conv + residual */
        ink_sconv(l->sc_mlp, conv_l + off_conv_mlp, n_embd, K, ff_out);
        for (uint32_t i = 0; i < n_embd; i++) x[i] += ff_out[i];
    }

    if (out_logits) {
        ink_rmsnorm(x, ink_f32(m->out_norm), n_embd, m->rms_eps);
        for (uint32_t i = 0; i < n_embd; i++) x[i] *= m->logit_scale;
        ink_matvec(m->output, m->output->data, n_embd, m->n_vocab, x, out_logits);
        if (m->n_vocab_unpadded > 0) {
            for (uint32_t i = m->n_vocab_unpadded; i < m->n_vocab; i++) out_logits[i] = -INFINITY;
        }
    }

    free(x); free(xn); free(q); free(kf); free(vf); free(r); free(rel);
    free(attn_out); free(proj_out); free(scores); free(hg); free(hu);
    free(ff_out); free(ff_acc);
}

/* ================================ CLI ================================ */

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *prompt = NULL;
    const char *logits_path = NULL;
    int n_predict = 5;
    uint32_t n_ctx = 512;
    bool dump_tokens = false;
    bool tokens_only = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) n_predict = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) n_ctx = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--logits-out") && i + 1 < argc) logits_path = argv[++i];
        else if (!strcmp(argv[i], "--dump-tokens")) dump_tokens = true;
        else if (!strcmp(argv[i], "--tokens-only")) tokens_only = true;
        else {
            fprintf(stderr, "usage: ds4-inkling -m model.gguf -p prompt [-n N] [-c CTX] [--logits-out f] [--dump-tokens] [--tokens-only]\n");
            return 1;
        }
    }
    if (!model_path || !prompt) {
        fprintf(stderr, "usage: ds4-inkling -m model.gguf -p prompt [-n N] [-c CTX]\n");
        return 1;
    }

    ink_model m;
    double t0 = now_sec();
    ink_model_open(&m, model_path, n_ctx);
    fprintf(stderr, "ds4-inkling: loaded %s (%u layers, %u dense, vocab %u) in %.1fs\n",
            model_path, m.n_layer, m.n_dense, m.n_vocab, now_sec() - t0);

    ink_ids ids = {0};
    ink_tokenize(&m.tk, prompt, &ids);
    if (dump_tokens || tokens_only) {
        fprintf(stderr, "tokens (%d):", ids.len);
        for (int i = 0; i < ids.len; i++) fprintf(stderr, " %d", ids.ids[i]);
        fprintf(stderr, "\n");
        char buf[512];
        for (int i = 0; i < ids.len; i++) {
            ink_detokenize(&m.tk, ids.ids[i], buf, sizeof(buf));
            fprintf(stderr, "  %d -> '%s'\n", ids.ids[i], buf);
        }
        if (tokens_only) return 0;
    }
    if (ids.len == 0) ink_die("empty prompt tokenization");
    if ((uint32_t)(ids.len + n_predict) > n_ctx) ink_die("prompt + n_predict exceeds context");

    float *logits = ink_malloc((size_t)m.n_vocab * sizeof(float));
    FILE *lf = logits_path ? fopen(logits_path, "wb") : NULL;

    /* prefill (token by token; conv states advance every position) */
    for (int i = 0; i < ids.len; i++) {
        double ts = now_sec();
        bool last = i == ids.len - 1;
        ink_forward(&m, ids.ids[i], (uint32_t)i, last ? logits : NULL);
        fprintf(stderr, "prefill %d/%d token %d: %.1fs\n", i + 1, ids.len, ids.ids[i], now_sec() - ts);
    }

    int pos = ids.len;
    char buf[512];
    for (int t = 0; t < n_predict; t++) {
        /* greedy argmax */
        int best = 0;
        float bestv = -INFINITY;
        for (uint32_t i = 0; i < m.n_vocab; i++) {
            if (logits[i] > bestv) { bestv = logits[i]; best = (int)i; }
        }
        if (lf) {
            fwrite(&best, sizeof(int), 1, lf);
            fwrite(logits, sizeof(float), m.n_vocab, lf);
            fflush(lf);
        }
        ink_detokenize(&m.tk, best, buf, sizeof(buf));
        printf("GEN %d: id=%d logit=%.6f text='%s'\n", t, best, bestv, buf);
        fflush(stdout);
        if (best == m.tk.eos) break;
        if (t + 1 == n_predict) break;
        double ts = now_sec();
        ink_forward(&m, best, (uint32_t)pos++, logits);
        fprintf(stderr, "decode %d: %.1fs\n", t + 1, now_sec() - ts);
    }
    if (lf) fclose(lf);
    return 0;
}
