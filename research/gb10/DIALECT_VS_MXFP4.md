# Dialect-compat set vs upstream ds4f-mxfp4 (analysis, 2026-08-03)

Upstream: antirez/ds4 branch `ds4f-mxfp4`, HEAD 4893e0c, 18 commits over
upstream/main (merge-base 54b36ed), ~2561 insertions across ds4.c, ds4_metal.m,
metal/*.metal, gguf-tools/deepseek4-quantize.c, plus MXFP4 tests. Method:
read the full upstream diff, then sequentially cherry-picked our six
dialect-compat commits onto a local scratch branch off upstream/ds4f-mxfp4
(deleted after) and built `make cpu` clean at the tip.

## What ds4f-mxfp4 actually adds

- `DS4_TENSOR_MXFP4 = 39` (block_mxfp4, 17 bytes/32 elems), gguf_types entry,
  scalar reference dot (`ds4_vec_dot_mxfp4_f32`), routed-expert type
  acceptance, Metal MXFP4 expert kernels (metal/moe.metal), streamed-expert
  cache seeding, and lossless MXFP4 repack in deepseek4-quantize.c
  (safetensors I8 codes + F8_E8M0 scales -> GGUF MXFP4, nibble-exact).
- Unrelated riders on the same branch: DSpark scheduling/verifier work
  (74e8f11, 4591cb1, a51e6ec), Metal prefill/decode tuning, imatrix fixes.
- **Zero dialect tolerance.** Metadata is still `required_u32(m,
  "deepseek4.vocab_size")`-style hard requirements; no key derivation, no
  tensor-name alias table, no load-time type conversion, no BF16 dense-tensor
  type (`DS4_TENSOR_BF16` does not exist upstream). The branch assumes
  ds4-native GGUFs produced by its own quantizer from original safetensors.

## Classification (cherry-pick trial onto ds4f-mxfp4, in order)

| Commit  | What it does | Verdict | Conflict detail |
|---------|--------------|---------|-----------------|
| 9c4b760 | 8-key derived-metadata compat (hc/lora/hash-layer keys) | SURVIVES-CLEAN | docs-only conflict (research/gb10/*.md absent upstream) |
| 99e7f1a | ~20 tensor-name aliases | SURVIVES-CLEAN | applied clean |
| 3106c3c | BF16/Q6_K dense -> F16 at load, mmap-grow | PARTIAL (one mechanical hunk) | both sides add `DS4_TENSOR_MXFP4 = 39` in the enum; keep ours' `BF16 = 30`, drop the duplicate. Rest applies clean. |
| 0528b32 | F32 -> F16 conversion + cuda_model_range fixes | SURVIVES-CLEAN | applied clean (incl. ds4_cuda.cu hunks; upstream branch never touches ds4_cuda.cu) |
| f7ec45f | vocab_size fallback from tokenizer.ggml.tokens length | SURVIVES-CLEAN | docs-only conflict |
| 6a4644e | Q6_K dequant per-block offset fix + decode-graph Q8_0-kernel type guard | SURVIVES-CLEAN | docs-only conflict; `dequantize_row_q6_K` is ours-only (upstream has vec-dot only, no row dequant) |

Nothing is SUBSUMED except the two enum lines in 3106c3c. No semantic
conflicts. Full stack builds clean (`make cpu`) on top of ds4f-mxfp4.

## Answers

**(a) Would the bullerwins DeepSeek-V4-Flash-0731-MXFP4_MOE-Q8_0 GGUF load on
stock ds4f-mxfp4?** No. It dies at the first missing `deepseek4.*` metadata
key (`required_u32` around the config loader, upstream ds4.c ~5613 —
vocab_size and the 8 hc/lora/hash-layer keys our 9c4b760/f7ec45f derive are
absent in the llama.cpp dialect). Even with keys stubbed, ~20 tensor-name
lookups fail (community naming vs ds4 canonical), and its BF16/F32/Q6_K dense
tensors are unloadable: type 30 (BF16) isn't even in upstream's enum/type
table, and no load-time conversion path exists. The only part upstream *does*
handle is the MXFP4 routed-expert payload itself.

**(b) Minimal PR against ds4f-mxfp4 to close the gap.** Exactly our six
commits' ds4.c side, squashed/regrouped; ~750 net lines, one file (ds4.c)
plus optionally the 0528b32 ds4_cuda.cu hunks (CUDA-only, separable):
1. metadata key derivation + vocab_size fallback (9c4b760 + f7ec45f, ~180 loc)
2. tensor-name alias table (99e7f1a, ~160 loc)
3. dense BF16/Q6_K/F32 -> F16 load-time conversion with the fixed
   `dequantize_row_q6_K` and the decode-graph type guard
   (3106c3c + 0528b32 + 6a4644e, ~420 loc), resolving the one enum overlap.
Suggested split: two PRs — (1)+(2) "dialect metadata/name compat", (3) "dense
tensor type conversion" — since (3) is where reviewer attention belongs.

**(c) Bugs revealed in ds4f-mxfp4's own code?** The 6a4644e per-block-offset
bug was in our own ported `dequantize_row_q6_K`; upstream has no such function
— no upstream bug there. However, the *second* fix in 6a4644e is a latent
upstream hazard that ds4f-mxfp4 still has: the single-token decode graph
dispatches the Q8_0-specific fused kernel for attn_q_a/attn_kv with no type
check (the batch/prefill graph guards via
`metal_graph_matmul_dense_quant_tensor`; the decode graph does not). Harmless
today only because native ds4 GGUFs always carry Q8_0 there; it silently
corrupts output for any other dense type. Worth including in any PR.

**(d) Rebase-risk for research/gb10 when ds4f-mxfp4 merges to main.**
- HIGH — DSpark: upstream 74e8f11 ("Commit partial DSpark verifier blocks
  without replay") rewrites `ds4_session_eval_dspark_speculative_argmax`, the
  same function our ae80554 greedy-identity fix rewrites, with the *opposite*
  philosophy (ours forces single-token replay always; theirs removes replay).
  Semantic re-derivation required, not a mechanical rebase. 05f0fde (dspark.*
  dialect detection) likely drifts too.
- LOW — CUDA streaming stack (ds4_cuda.cu, ds4_ssd.c, cuda_model_stage_read,
  DS4_STREAM_FADVISE, expert LRU): upstream branch touches none of these
  files. Context drift only.
- LOW/MECHANICAL — loader/dequant area of ds4.c: the enum/type-table overlap
  shown above; trivial.
- N/A — ds4_metal.m / metal/*.metal churn (largest part of the branch): our
  gb10 stack is CUDA-side and doesn't carry Metal patches.

## Recommended next action (operator approval)

Prepare the two-PR series in (b) against ds4f-mxfp4 (issue+PR pairing per
convention: issue = community-dialect MXFP4_MOE GGUFs unloadable; PRs = compat
layers), flagging the decode-graph type-guard hazard from (c) in the second
PR. Separately, plan a re-derivation (not rebase) of the DSpark
greedy-identity fix on top of upstream's replay-free verifier before
ds4f-mxfp4 reaches main. No filings made; analysis only.
