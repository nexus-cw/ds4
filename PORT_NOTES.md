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

## M2 CUDA survey — see bottom of this file once M1 lands.
