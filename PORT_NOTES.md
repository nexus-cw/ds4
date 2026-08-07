# Inkling port notes (S3, task #35)

Target: text-only CPU inference for arch `inkling` (Thinking Machines
Inkling-Small, 276B/A12B, 42 blocks: 2 dense + 40 MoE).

## M0 survey: the ds4 arch seam

ds4 has no generic architecture seam. The engine is hard-coded to two
shapes that share the deepseek4 lineage:

- deepseek4 (Flash/Pro): CPU path `forward_token_raw_swa_cpu_decode_scratch`
  (ds4.c:14690) with MLA attention, plus the CUDA/Metal graphs.
- GLM (DSA variant): its own graph (`glm_graph_forward_token`, ds4.c:43748)
  selected via `ds4_session_is_glm` switches sprinkled through the session
  layer (~40 call sites).

Metadata loading is keyed by literal "deepseek4." strings (ds4.c:2788..),
the tokenizer is a hand-written pre-tokenizer per dialect
(`bpe_tokenize_text_glm4` ds4.c:37159, joyai-llm ds4.c:37279), and the
tensor table (`ds4_layer_weights`, ds4.c:4680) only has deepseek/GLM
members. Threading a third, structurally alien arch (no RoPE, relative
attention, shortconv recurrent state, hybrid SWA, sigmoid MoE with scored
shared experts) through those switches would touch hundreds of sites.

Port decision: a self-contained additive module, `ds4_inkling.c`, with its
own GGUF reader, tokenizer, dequant and CPU forward, plus a small CLI
(`ds4-inkling`). No change to existing engine paths, zero risk to the
serving builds. Engine/session integration (model picker, server) is
follow-up S3 work once numerics are proven.

## Reference semantics (llama.cpp PR #25731, MIT, code donor)

From `src/models/inkling.cpp` + `llama-kv-cache.cpp` + `llama-vocab.cpp`
on branch danielhanchen/llama.cpp:add-inkling (3fd7901):

- Block: h += sconv_attn(attn(rms(h))); h += sconv_mlp(ffn(rms(h))).
- sconv(x) = x + causal depthwise conv1d (kernel K=4, F32 [K, C]),
  rolling state = last K-1 inputs. k/v also get sconvs on the flat
  projections BEFORE head reshape and BEFORE q/k RMS norm.
- Attention: GQA 32q/8kv, head_dim 128, post-sconv per-head q/k RMS norm,
  cache stores post-norm k and post-sconv v. No RoPE. Score =
  dot(q,k)/128 + rel_bias[d], d = pos_q - pos_k, bias 0 for d outside
  [0, extent). rel = attn_rel_proj^T contraction of r = wr(x) per head:
  rel[e] = sum_d proj[e,d] * r_head[d] (proj stored gguf ne
  {extent, d_rel}: element (e,d) at d*E+e). extent 512 for SWA layers,
  1024 global. SWA visibility: pos_q - pos_k < 512 (includes self).
  Global layers only: q and rel both scaled by
  tau = 1 + 0.1*log(max((pos+1)/128000, 1)).
- MoE (40 blocks): router [4096, 258] = 256 routed + 2 shared columns.
  Selection: top-6 of sigmoid(routed_logits) + exp_probs_b bias.
  Weights: softmax over 8 values (6 raw top-k routed logits + 2 raw
  shared logits) of logsigmoid(x), scaled by expert_weights_scale (8.0)
  and by the per-block scalar ffn_gscale. Routed expert outputs get their
  weight after down-proj; shared expert hidden gets its gamma BEFORE
  down-proj. Dense blocks (0,1): plain swiglu ffn * ffn_gscale.
- Head: rms(output_norm), scale by 1/logit_scale_denom (1/16), output
  matmul, then ids >= unpadded_vocab (200058) masked to -inf.
- Embedding: token_embd row then token_embd_norm RMS.
- Tokenizer: gpt2 byte-level BPE, pre "inkling" = o200k regex with \p{M}
  added to the letter classes, clean_spaces false, no add_bos,
  bos=eos=200006, extra EOG token "<|content_model_end_sampling|>".

## Quant coverage needed (from the real IQ2_XXS artifact header)

F32, Q8_0, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ3_XXS, IQ2_S, IQ4_XS.
ds4 core has F32/Q8_0/Q4_K-family and IQ2_XXS; IQ3_XXS/IQ2_S/IQ4_XS CPU
dequant ported from llama.cpp ggml-quants.c (MIT, attributed) into the
module so it stays self-contained.

## M1 parity result (robo-dog, 2026-08-07)

Artifact: /data/gguf/inkling/Inkling-Small-UD-IQ2_XXS.gguf (82.3GB, SATA
mmap). Prompt "The capital of France is", temp 0, n_ctx 512, 5 tokens.

- Tokenizer: ds4-inkling and llama-completion (branch add-inkling,
  3fd7901, CPU build) both produce [976, 9029, 328, 10128, 382].
- Greedy generation, token by token:
    ds4-inkling: 12650 ' Paris' / 13 '.' / 12650 ' Paris' / 382 ' is' /
                 290 ' the'   (greedy logits 18.54 / 19.40 / 13.79 /
                 18.74 / 18.99)
    llama.cpp:   "The capital of France is Paris. Paris is the"
  => top-1 agreement 5/5. Full-logit max-abs-diff not captured:
  llama-completion has no logits dump; ds4-inkling wrote its logits to
  ink_logits.bin for a future eval-callback comparison.
- Speed: ds4-inkling ~120-190s/token (8 threads, scalar dequant,
  SATA-bound), llama.cpp ~118s/token eval — both dominated by the 82GB
  mmap stream from SATA.

## M2 CUDA path survey (write-up only; no CUDA work in this run)

What full-speed inkling on the GB10 CUDA path needs, in dependency order:

1. IQ dequant/matvec kernels: ds4_cuda.cu has IQ2_XXS (routed experts,
   ds4_iq2_tables_cuda.inc). Missing: IQ3_XXS (all routed ffn_down_exps),
   IQ2_S + IQ4_XS (one UD layer each), Q5_K/Q6_K matvec coverage for the
   dense/attention/shexp tensors if not already generic. The iq3xxs_grid
   and iq2s_grid tables port the same way as the existing IQ2 tables.
2. Attention kernel: no RoPE anywhere (remove that stage), GQA 32/8
   head_dim 128 with an additive per-(head,query) relative bias vector of
   length rel_extent indexed by pos_q - pos_k, plus SWA masking per the
   per-layer pattern and log-N tau scaling of q and bias on global
   layers. Closest existing donor is the deepseek4 SWA decode kernel;
   the bias lookup is the only new element (llama.cpp does it as a
   banded flash-attn extension, ggml_flash_attn_ext_banded).
3. Short convolutions: 4 small depthwise conv1d states per layer
   (k/v/attn/mlp). Trivial kernels, but they add a recurrent state
   object to the session (snapshot/restore, rewind: states must be
   recomputable or checkpointed per position -- rewind support needs
   per-position state history or replay, same problem class as GLM MTP
   rollback).
4. MoE routing: the sigmoid+bias top-k selection reuses the deepseek4
   noaux-tc machinery; new pieces are the logsigmoid-softmax weighting
   over topk+shared logits, the score-weighted shared experts (2-expert
   mul_mat_id bank), and per-block ffn_gscale.
5. Expert streaming (SSD path): the routed experts are IQ2_XXS/IQ3_XXS
   in a 3D bank layout identical to deepseek4's, so ds4_ssd.c expert
   extraction should apply with a new tensor-name map.
6. Output head: logit_scale + unpadded-vocab -inf mask in the sampler
   or logits kernel.

Estimated order: kernels (1) are mechanical; (2)+(3) are the real work;
(3) interacts with session snapshot/rewind semantics and needs a design
decision before implementation.
