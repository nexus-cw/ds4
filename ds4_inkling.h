/* ds4_inkling.h -- shared surface of the self-contained inkling engine
 * (ds4_inkling.c).  Used by the CLI, the CUDA path (ds4_inkling_cuda.cu)
 * and the thin server (ds4_inkling_server.c).  See PORT_NOTES.md. */

#ifndef DS4_INKLING_H
#define DS4_INKLING_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

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

typedef struct {
    char name[128];
    uint32_t type;
    uint32_t ndim;
    uint64_t dims[4];
    const uint8_t *data;
} ink_tensor;

typedef struct { const char *ptr; uint64_t len; } ink_str;

typedef struct {
    char key[96];
    uint32_t type;
    uint32_t arr_type;
    uint64_t arr_len;
    const uint8_t *val;
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

typedef struct { char *key; uint32_t len; int val; } ink_map_slot;
typedef struct { ink_map_slot *slots; uint64_t mask; } ink_map;

typedef struct {
    ink_str *tokens;
    uint32_t n_tokens;
    ink_map vocab;
    ink_map merges;
    int bos, eos;
} ink_tokenizer;

typedef struct {
    const ink_tensor *attn_norm, *wq, *wk, *wv, *wr, *wo;
    const ink_tensor *q_norm, *k_norm, *rel_proj;
    const ink_tensor *sc_k, *sc_v, *sc_attn, *sc_mlp;
    const ink_tensor *ffn_norm, *gscale;
    const ink_tensor *ffn_gate, *ffn_up, *ffn_down;          /* dense */
    const ink_tensor *gate_inp, *probs_b;                    /* moe */
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

    uint32_t n_ctx;
    uint32_t kvw_max;
    float *kcache;   /* [n_layer][n_ctx][kvw_max] post-norm k */
    float *vcache;   /* [n_layer][n_ctx][kvw_max] post-sconv v */
    float *conv;     /* [n_layer][(K-1)*(2*kvw_max+2*n_embd)] */
} ink_model;

typedef struct { int *ids; int len; int cap; } ink_ids;

/* utils */
void ink_die(const char *msg);
void *ink_malloc(size_t n);
void *ink_calloc(size_t n, size_t sz);
float ink_fp16_to_fp32(uint16_t h);
double ink_now_sec(void);
float ink_logsigmoid(float x);

/* gguf */
bool ink_get_u32(const ink_gguf *g, const char *key, uint32_t *out);
bool ink_get_f32(const ink_gguf *g, const char *key, float *out);
bool ink_get_str(const ink_gguf *g, const char *key, ink_str *out);
const ink_kv *ink_kv_find(const ink_gguf *g, const char *key);

/* quant */
size_t ink_type_block_elems(uint32_t t);
size_t ink_type_block_bytes(uint32_t t);
void ink_row_f32(const ink_tensor *t, const uint8_t *base,
                 uint64_t row, uint64_t row_len, float *out);

/* math (CPU) */
void ink_matvec(const ink_tensor *t, const uint8_t *base,
                uint64_t in, uint64_t out, const float *x, float *y);
/* Y[t][o] = sum_i W[o][i] * X[t][i]; each weight row dequantized once. */
void ink_matmat(const ink_tensor *t, const uint8_t *base,
                uint64_t in, uint64_t out, uint32_t n_tok,
                const float *X, float *Y);
void ink_rmsnorm(float *x, const float *w, uint64_t n, float eps);
void ink_sconv(const ink_tensor *kernel, float *state, uint32_t C,
               uint32_t K, float *x);
const float *ink_f32(const ink_tensor *t);

/* model */
void ink_model_open(ink_model *m, const char *path, uint32_t n_ctx);
/* Copy tensor data out of the file-backed mmap into owned memory
 * allocated with alloc_fn (malloc, cudaMallocManaged, ...), repointing
 * tensor->data.  Tensors are made resident in file order until
 * budget_bytes is exhausted (0 = everything).  Returns bytes copied;
 * *n_resident_out (optional) gets the tensor count made resident.
 * Dies loudly if alloc_fn fails. */
uint64_t ink_model_make_resident(ink_model *m, uint64_t budget_bytes,
                                 void *(*alloc_fn)(size_t),
                                 uint64_t *n_resident_out);
/* Per-tensor variant: alloc_fn sees the tensor and may return NULL to
 * SKIP it (tensor stays mmap-backed); a failed allocation should die
 * inside alloc_fn.  copy_fn performs the transfer into the returned
 * arena (pass NULL for plain memcpy); needed when the arena is not
 * host-writable (e.g. cudaMalloc device memory). */
uint64_t ink_model_make_resident_ex(ink_model *m, uint64_t budget_bytes,
                                    void *(*alloc_fn)(size_t, const ink_tensor *),
                                    void (*copy_fn)(void *dst, const void *src, size_t n),
                                    uint64_t *n_resident_out);
/* Total bytes of one tensor's data. */
uint64_t ink_tensor_bytes(const ink_tensor *t);
/* Die with a diagnostic if the logits vector is NaN-poisoned or the
 * unpadded range is entirely -inf (both mean upstream data corruption:
 * never silently emit token 0). */
void ink_logits_guard(const float *logits, uint32_t n_vocab,
                      uint32_t n_unpadded, const char *where);
/* Process n_tok tokens starting at absolute position pos0 (KV/conv state
 * must already cover [0, pos0)).  If out_logits != NULL, logits of the
 * LAST token are written (n_vocab floats). */
void ink_forward_batch(ink_model *m, const int *tokens, uint32_t n_tok,
                       uint32_t pos0, float *out_logits);
void ink_forward(ink_model *m, int token, uint32_t pos, float *out_logits);
/* Reset KV + shortconv state to empty (position 0). */
void ink_state_reset(ink_model *m);
/* v1 snapshot: KV cache [0,pos) + the 4-tap shortconv rolling states +
 * pos, memcpy'd into caller storage.  Size via ink_state_bytes(). */
size_t ink_state_bytes(const ink_model *m);
void ink_state_save(const ink_model *m, uint32_t pos, void *buf);
uint32_t ink_state_load(ink_model *m, const void *buf);

/* tokenizer */
void ink_ids_push(ink_ids *v, int id);
void ink_tokenize(const ink_tokenizer *tk, const char *text, ink_ids *out);
int ink_detokenize(const ink_tokenizer *tk, int id, char *out, int cap);
int ink_token_lookup(const ink_tokenizer *tk, const char *text);

/* chat template (v1 hand-rendered form of the GGUF jinja template; see
 * PORT_NOTES.md).  Appends the special-token framing for one message. */
void ink_chat_append(const ink_model *m, ink_ids *out,
                     const char *role, const char *content);
/* Appends the assistant generation prefix (<|message_model|>). */
void ink_chat_append_model_prefix(const ink_model *m, ink_ids *out);

#ifdef __cplusplus
}
#endif

#endif
