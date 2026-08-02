# FP4 port scope — llama.cpp (MIT) → ds4 (MIT)

Status: scoping complete, implementation not started.
Donor pinned at llama.cpp `5f55650` (2026-07-30), cloned at `~/src/llama.cpp` on robo-dog.
Rule of record: llama.cpp is MIT like ds4 — direct porting is sanctioned. colibri (Apache-2.0)
remains papers-only / never source.

## Why this is smaller than feared

Two structural facts collapse the work:

1. **llama.cpp's `MXFP4_MOE` ftype quantizes only MoE expert tensors to MXFP4; every other
   tensor becomes Q8_0** (`src/llama-quant.cpp:474-480`). ds4 already loads and runs Q8_0
   for attention/shared/dense tensors. So a `MXFP4_MOE` GGUF needs exactly one new ds4
   capability: MXFP4 **routed experts** — which are precisely the tensors ds4 streams.
2. **ds4 has a generic fallback matmul**: the "streaming dequant + f16 GEMM" path
   (`ds4_cuda.cu:12053-12190`) covers tensor types without native tile kernels. Phase 1
   needs a dequant kernel, not hand-written FP4 GEMM kernels.

Streaming is type-agnostic: `routed_expert_row_bytes()` is pure block math and `ds4_ssd.c`
reads bytes in 256MB chunks. Register the block size and streaming works unchanged.

## Donor inventory (llama.cpp, what to port)

| What | Where | Size of job |
|---|---|---|
| Block structs: `block_mxfp4` (1B E8M0 scale + 16B packed E2M1, QK=32), `block_nvfp4` (4×E4M3 sub-scales + 32B, QK=64/sub 16) | `ggml/src/ggml-common.h:218-227` | trivial |
| E2M1 value LUT `kvalues_mxfp4` + CPU dequant `dequantize_row_mxfp4/nvfp4` | `ggml/src/ggml-quants.c:569-608` | small |
| CPU quantize (for tooling/tests) `quantize_row_mxfp4_ref/nvfp4_ref` | `ggml-quants.c:350-411` | small |
| CUDA dequant | `ggml/src/ggml-cuda/convert.cu` | small |
| CUDA dot-product (decode/mmvq) | `ggml/src/ggml-cuda/vecdotq.cuh` | medium, phase 3 |
| CUDA tensor-core MMA (prefill), incl. sm_121 `GGML_CUDA_CC_DGX_SPARK=1210` gating | `mma.cuh`, `mmq.cuh`, `mmq-config-blackwell.cuh`, `template-instances/mmq-instance-{mxfp4,nvfp4}.cu` | large, phase 3 |

ds4 head start: E2M1 value table + nearest-value dequant already exist (activation-sim code,
`ds4.c:3231-3295`, `ds4_cuda.cu:6086-6111`) — same number system, reusable.

## ds4 integration points (what to touch)

1. `gguf_type_info` registry — add MXFP4 (GGUF type 39) with `block_elems=32`,
   block bytes 17; NVFP4 (type 40), `block_elems=64`, block bytes 36.
2. `tensor_is_routed_expert_type()` (ds4.c ~4367) — whitelist the new types.
3. `routed_expert_block_bytes()` (ds4.c:4384 switch) — add cases (kills the
   `ds4_die("unsupported routed expert tensor type")` for FP4).
4. CPU path — `dequantize_row_mxfp4` port next to existing dequant helpers.
5. CUDA path — `mxfp4_dequant_f16_kernel` mirroring `q8_0_dequant_f16_kernel`
   (`ds4_cuda.cu:12060`), wired into the streaming dequant+f16 GEMM dispatch.
6. Loader shape checks (`tensor_expect_layout`) — verify row alignment `dim[0] % 32 == 0`.

## Phases

- **P1 — MXFP4 correct**: items 1-5 above; validate logits vs llama.cpp dequant on the same
  tensor bytes (bit-exact dequant is testable in isolation); then ds4-eval + KLD vs the
  IQ2_XXS baseline. Fallback GEMM speed is acceptable for validation.

### P1 status (2026-07-31)

GPU routed-expert dispatch landed: `routed_moe_launch` (`ds4_cuda.cu`) now has an
`mxfp4_path` branch (`gate_type == down_type == 39`) alongside the pre-existing Q4_K /
IQ2_XXS+Q2_K branches, so it no longer `return 0`s (-> upstream `ds4_die`) for an
MXFP4_MOE GGUF. It is a naive per-(token,expert)-pair loop
(`routed_moe_mxfp4_dispatch` / `mxfp4_matmul_row_f16gemm`): one device->host readback of
the `selected` expert-index table per call (needed because cuBLAS wants the weight
pointer on the host, unlike the fused kernels' device-side indexing), then per pair:
dequant gate/up MXFP4 rows to f16 (`mxfp4_dequant_f16_kernel`, reused verbatim), a cuBLAS
f16 GEMM each, an elementwise SiLU(gate)*up*routing-weight combine kernel, a dequant+GEMM
down projection, and the existing `moe_sum_kernel` / `moe_sum_owned_kernel` for the
cross-expert sum -- same accumulation machinery every other type path uses. No new tile
kernels. Covers both decode (n_tokens=1) and prefill (n_tokens>1) through the same loop;
per-pair GEMMs are the acknowledged P3 speed gap.

Validated with `research/gb10/test_mxfp4_moe.c`: synthetic MXFP4 gate/up/down expert
tensors through the public `ds4_gpu_routed_moe_batch_tensor()` entry point, decode-shaped
(n_tokens=1, n_expert=6) and prefill-shaped (n_tokens=5, n_expert=3) cases, both matching
an independent CPU dequant+matvec reference within f16-GEMM tolerance. Note:
`routed_moe_launch`'s shared top-of-function validation (pre-existing, not touched here)
hard-requires `expert_in_dim`/`expert_mid_dim % CUDA_QK_K(=256) == 0` for every routed
type, inherited by MXFP4 even though its own block size is 32 -- fine for real MoE hidden
dims (always multiples of 256 in practice) but worth knowing if a future test wants
smaller synthetic dims.

CPU (non-CUDA) routed-expert path was scoped and explicitly **not** wired: it dispatches
per weight-type via ~17 separate hand-specialized worker functions in `ds4.c`
(`matvec_q2_k_expert*`, `matvec_q4_k_experts_*_prequant`, the `matvec_expert_pair_prequant`
/ `matvec_expert_down` tracing helpers, and batch variants around `ds4.c:11457`/`11553`),
each with its own `ds4_die("unsupported gate/up expert tensor type")` /
`ds4_die("unsupported down expert tensor type")` fallback -- not a single generic dispatch
point the way the CUDA side has `ds4_gpu_matmul_quant_tensor`. Wiring `dequantize_row_mxfp4`
(already ported, `ds4.c` ~3504) into all of them is real, scattered work; deferred and
reported per the ticket's own escape hatch rather than forced. Net effect: an MXFP4_MOE
GGUF run with `--ssd-streaming` or on a CPU-only build still `ds4_die`s the first time a
CPU-side routed-expert matvec sees type 39. GPU-resident (non-streaming-CPU-fallback)
execution is what's covered by this pass.
- **P2 — NVFP4**: same skeleton, E4M3 sub-scales (worth ~3.1 PPL per FP4 research); blocked
  upstream on quantize-tool emit (ftype 39 reserved in llama.h but no CLI entry at donor
  HEAD) unless we target an existing NVFP4 GGUF.
- **P3 — speed**: native decode dot-product (vecdotq) then sm_121 tensor-core MMA prefill
  path. Only after P1 quality is proven.

### P1 status update (2026-08-01): Q3_K port + generalized routed-MoE fallback

The real target artifact's routed experts are **not** uniformly MXFP4: a header-only scan
of `DeepSeek-V4-Flash-MXFP4_MOE.gguf` (see below) shows 98 `ffn_{gate,up}_exps`/`ffn_down_exps`
tensors as MXFP4 (type 39), 30 as Q3_K (type 11), and 1 as Q5_K (type 13), across 43 MoE
layers (43*3 = 129 routed-expert tensors total). ds4 had zero Q3_K support before this
pass (not in `gguf_type_info`'s enum constants/whitelist at all), so any layer with a Q3_K
tensor previously died at load (`tensor_expect_routed_expert` -> `ds4_die`) before MXFP4 was
ever exercised. Q5_K was already a supported routed-expert *type* in the whitelist, but had
no CPU dequant-to-float or CUDA dequant-to-f16 kernel anywhere in the codebase (only fused
`vec_dot_q5_K_*` dot-product helpers) -- so it worked wherever a fused/vec_dot path covered
it, but not through the new dequant+GEMM fallback family MXFP4 introduced in 3238a16.

**Q3_K port** (`ds4.c`): `block_q3_K` struct + `DS4_STATIC_ASSERT` (110 bytes), `DS4_TENSOR_Q3_K`
enum constant (11 -- the `gguf_type_info[11]` registry row already existed, just unused by
the enum/whitelist), `dequantize_row_q3_K` (CPU, ported line-for-line from llama.cpp
`ggml-quants.c` @5f55650, MIT), and whitelisting in `tensor_is_routed_expert_type` /
`routed_expert_block_bytes`. CUDA: `q3_k_dequant_f16_kernel` (`ds4_cuda.cu`), one thread per
QK_K=256-element super-block, same "phase 1 correctness not throughput" idiom as
`mxfp4_dequant_f16_kernel`.

**Q5_K dequant added** (had none): CPU `dequantize_row_q5_K` (`ds4.c`, ported from llama.cpp,
reuses the existing `q4_k_get_scale_min` helper -- Q5_K shares Q4_K's packed 6-bit scale/min
layout plus a high-bit mask) and CUDA `q5_k_dequant_f16_kernel` (`ds4_cuda.cu`, reuses the
existing `dev_q4_K_get_scale_min` device helper).

**Generalized dispatch** (`ds4_cuda.cu`): the MXFP4-only branch from 3238a16
(`mxfp4_path = gate_type == down_type == 39`, `routed_moe_mxfp4_dispatch`,
`mxfp4_matmul_row_f16gemm`) is now a generic dequant+GEMM fallback family:
`dequant_gemm_type_supported`/`_block_elems`/`_block_bytes`/`launch_dequant_f16` (a small
type -> {MXFP4, Q3_K, Q5_K} dequant-kernel table), `dequant_gemm_row_f16gemm` (generalized
`mxfp4_matmul_row_f16gemm`, takes a `type` param), and `routed_moe_dequant_gemm_dispatch`
(generalized `routed_moe_mxfp4_dispatch`, takes `gate_type`/`down_type` and looks each up in
the table independently). `routed_moe_launch`'s branch selection became `dequant_gemm_path =
!q4k_path && !iq2_q2k_path && dequant_gemm_type_supported(gate_type) &&
dequant_gemm_type_supported(down_type)` -- a strict fallback: the fused Q4_K / IQ2_XXS+Q2_K
tile-kernel branches still win whenever they match; this is only taken otherwise. Because
gate and down are looked up independently, a layer with MXFP4 gate/up and Q3_K (or Q5_K)
down experts dispatches through the same loop, each half using its own dequant kernel --
exactly the real artifact's shape.

**Scope note on gate vs. up type**: `routed_moe_launch` (and its whole call chain --
`ds4_gpu_routed_moe_one_tensor`/`_batch_tensor`/`_batch_owned_tensor`, ~14 call sites in
`ds4.c`) has only ever taken `gate_type` and `down_type`, never a separate `up_type` --
this predates the FP4 port. That's not an oversight: ds4 enforces gate_type == up_type as a
hard **load-time** invariant across every model family (`ds4.c`, e.g. "routed gate/up
experts use different quant types" -> `exit(1)`/`ds4_die`, ~8 call sites spanning DeepSeek,
GLM, MTP, DSpark loaders), so by the time any routed-MoE dispatcher runs, gate and up are
already guaranteed identical. A genuinely independent gate != up type (beyond down
differing) would require relaxing that invariant everywhere it's enforced -- a real
cross-cutting architecture change, not attempted here; not needed for the real artifact
either, confirmed by the header scan below (`gate_type == up_type` holds for all 43 layers).

**Header-only validation of the real artifact** (`research/gb10/gguf_header_check.py`, not
committed -- see below -- a throwaway metadata-prefix GGUF parser: magic/version/counts, KV
pairs, then the tensor-info table, stopping before the 150GB tensor-data blob, so it runs in
~0.1s without touching most of the file): confirms `general.architecture=deepseek4`,
1328 tensors total, 129 `*_exps` tensors (98 MXFP4 / 30 Q3_K / 1 Q5_K) across 43 layers,
`gate_type == up_type` for every layer, and down-type distribution `{Q3_K: 10, MXFP4: 32,
Q5_K: 1}` layers -- i.e. every type combination this pass added support for is exactly what
the file needs, and nothing else appears among routed-expert tensors. (This script was a
disposable validation aid, run from `/tmp` on robo-dog against the live file; not added to
the repo since it duplicates functionality a real `gguf-tools` inspector would better own if
this becomes a recurring need.)

**Tests**: `research/gb10/test_mixed_moe.c`, new -- extends the `test_mxfp4_moe.c` pattern
with standalone (non-ds4.c) ports of `dequantize_row_q3_K`/`dequantize_row_q5_K` and runs 6
cases through the public `ds4_gpu_routed_moe_batch_tensor()`: MXFP4 gate/up + Q3_K down
(decode- and prefill-shaped), MXFP4 gate/up + Q5_K down (decode- and prefill-shaped), and
Q3_K-only / Q5_K-only single-type sanity cases, all checked against an independent CPU
dequant+matvec reference. All 6 pass. The pre-existing `test_mxfp4_dequant.c` and
`test_mxfp4_moe.c` self-tests still pass unmodified after the generalization (confirms the
refactor didn't regress the original MXFP4-only path).

**No full-file load test was run**: ds4 has no `--info`/`--show`/header-only CLI mode, and a
real `./ds4 -m ...` load would mmap and (for CUDA-resident, non-`--ssd-streaming` runs)
upload the full 150GB to the unified-memory GPU pool -- not attempted here (out of scope for
a header/dequant-correctness pass, and the file is larger than the documented 121GB pool
per the "Target artifact" section below, so a full-residency load was never going to
succeed regardless of quant-type support; that's the `--ssd-streaming` P1 validation, not
yet run against this file).

### P1 status update (2026-07-31): deepseek4 GGUF dialect compat + first non-metadata blocker

First `--inspect` attempt against the real target artifact died immediately:
`required metadata key is missing: deepseek4.attention.output_lora_rank`. A header-only
scan (throwaway `gguf_dump.py`, same idiom as the prior pass's `gguf_header_check.py`, not
committed) plus a full read of `config_validate_deepseek4_model()`
(`ds4.c`) diffed against llama.cpp `@5f55650`'s own deepseek4 loader
(`src/models/deepseek4.cpp::load_arch_hparams`, which requires the exact same keys via
`ml.get_key(..., required=true)` with **no fallback of its own**) established the real
picture: this is not a key-renaming dialect difference antirez's converter would recognize
under another name -- the community conversion (produced by an earlier/lighter
`convert_hf_to_gguf.py` than the one at donor HEAD, whose `DeepseekV4Model.set_gguf_parameters`
writes these same keys unconditionally from `config.json`) simply **omits** eight required
keys outright: `attention.output_lora_rank`, `attention.output_group_count`,
`hash_layer_count`, `hyper_connection.count`, `hyper_connection.sinkhorn_iterations`,
`hyper_connection.epsilon`, `attention.compress_rope_freq_base`, and the
`attention.compress_ratios` array. Every other required deepseek4 key in the file (28 of
36) matches ds4's compiled-in `DS4_SHAPE_FLASH` profile exactly.

**Compat layer** (`ds4.c`, new block ahead of `config_validate_deepseek4_model`): for each
missing key, recovers the value -- in preference order -- from (1) tensor shapes/presence
already in the file, tying the value to exactly what llama.cpp's own tensor-creation code
derives it from, or (2), only where no tensor encodes the value, the exact constant ds4
already hardcodes and validates every *present* key against for the shape identified from
the file's other required keys. Nothing is guessed from outside these two sources, and a
one-line notice prints per compat fallback so a run is honest about it being active:

- `output_lora_rank` / `output_group_count`: llama.cpp creates `wo_a` (`attn_output_a`) =
  `{n_head*n_embd_head/o_groups, o_lora_rank*o_groups}` and `wo_b` (`attn_output_b`) =
  `{o_groups*o_lora_rank, n_embd}` -- both hparams are exactly recoverable from the two
  tensors' `dim0` given already-required `n_head`/`key_length`. Verified: derives 8 / 1024,
  matching `DS4_SHAPE_FLASH` exactly.
- `hyper_connection.count`: llama.cpp creates `hc_attn_fn` = `{hc_mult*n_embd,
  (2+hc_mult)*hc_mult}`, so `hc_mult = dim0(hc_attn_fn) / n_embd`. Verified: derives 4,
  cross-checked against `dim1` too (`(2+4)*4 = 24`, matches the file's tensor exactly).
- `hash_layer_count`: llama.cpp only creates `blk.<i>.ffn_gate_tid2eid` for
  `i < hash_layer_count`; counting the contiguous run of present tensors from layer 0
  recovers the exact count. Verified: derives 3 (layers 0-2 have the tensor, layer 3
  doesn't), matching `DS4_SHAPE_FLASH` exactly.
- `attention.compress_ratios` (array): `ds4_expected_layer_compress_ratio()` (pre-existing,
  `ds4.c`) already encodes, per shape, the exact per-layer pattern this array is normally
  *validated against* -- when the array itself is missing, that expected pattern is used
  directly rather than dying (it's not a guess, it's the ground truth the array is always
  required to equal anyway). Cross-checked independently against tensor presence/shape: the
  file's `blk.{40,42}.attn_compress_*` tensors imply ratio 4 and `blk.41.attn_compress_*`
  implies ratio 128 at those exact layers, matching the FLASH pattern's `il&1` alternation.
- `hyper_connection.sinkhorn_iterations`, `hyper_connection.epsilon`,
  `attention.compress_rope_freq_base`: none of these are tensor-derivable (pure algorithm
  scalars/RoPE base, not shape parameters), and llama.cpp's own loader has no fallback for
  them either -- so these are the one fallback tier below tensor-derivation: the exact value
  ds4 already hardcodes for the shape identified via `block_count` (unique across
  `DS4_SHAPE_FLASH`/`DS4_SHAPE_PRO`) and validates every other key against. Noted as the
  weakest-evidence tier in code comments; `compress_rope_freq_base`'s fallback value (160000)
  is at least corroborated by the file's own (otherwise-unused-by-ds4)
  `deepseek4.rope.freq_base_swa = 160000.0`.

**Result**: `--inspect` (`--cuda --ssd-streaming --ssd-streaming-cold
--ssd-streaming-cache-experts 8GB`) now gets past all metadata validation (prints all 8
compat notices, then the pre-existing shape/tensor-layout checks) and reaches the first
**non-metadata** blocker: `required tensor is missing: output_hc_base.weight`. This is a
genuine tensor **naming** mismatch, not a metadata gap -- confirmed by header scan: the
file's top-level (non-per-layer) hyper-connection head tensors are named `hc_head_base`,
`hc_head_fn`, `hc_head_scale` (no `.weight` suffix, GGML type F32/F16), while
`weights_bind_output()` (`ds4.c`) requires `output_hc_base.weight`, `output_hc_fn.weight`,
`output_hc_scale.weight`. Per-layer hc tensors (`blk.<i>.hc_attn_fn` etc.) and the ordinary
`output.weight`/`output_norm.weight`/`token_embd.weight` names already match between ds4 and
the file -- only this one top-level triplet differs. Per the ticket's own instruction, this
is reported rather than improvised: a tensor-name mapping (and a check for whether any other
top-level or per-layer tensor names differ further into the load path) is real follow-up
work but out of scope for this pass.

**Verification**: `make cuda-spark` -- clean build from `make clean`, no warnings. `make
test` / `./ds4_test` -- ran twice against the patched binary and twice against the
pre-patch binary (`git stash`) for comparison; failure counts fluctuated run-to-run on
*both* (5-6 on unpatched, 10-11 on patched) with different individual tests (`tool-call-quality`,
`think-tool-recovery`, `logprob-vectors`, `metal-kernels`, `metal-tensor-equivalence`)
flipping OK/ERR between runs of the *identical* unpatched binary -- i.e. this suite has
pre-existing run-to-run nondeterminism on this hardware unrelated to this change. Code
review also confirms the patch is a no-op whenever a key is present (the compat helpers all
try `model_get_u32`/`model_get_f32_compat` first and return immediately on success, byte
identical to the prior `required_u32`/`required_f32` behavior), which the existing/pre-patch
test fixture GGUF exercises throughout (no compat branch fires when running `make test`'s
own model). No regression attributable to this change.

(2026-08-03 follow-up, MEASUREMENTS.md "Determinism/identity probe" unit: a targeted 10-run sweep on the GA model at --ssd-streaming-cache-experts 65GB, two prompts, 400/800-token budgets, warm and cold cache regimes, found this specific --ssd-streaming decode nondeterminism does NOT reproduce -- all runs byte-identical. Left open whether it was fixed between the preview artifact and GA or is a low-frequency event this sweep did not hit; see that unit for the full protocol and negative-result caveats. A separate, root-caused, STRUCTURAL greedy-identity divergence was confirmed in the same unit for DSpark spec-decode specifically -- batch-verify GEMM numerics vs. single-token decode numerics for accepted tokens' compressed-KV frontier state -- see the same MEASUREMENTS.md entry, CASE B.)

### P1 status update (2026-07-31, cont'd): tensor-name dialect compat + BF16/Q6_K dense-tensor blocker

Follow-on pass to the deepseek4-metadata-compat work above, closing the "genuine tensor
NAME mismatch" it stopped at (`output_hc_base.weight` missing). Added a general
tensor-name-alias mechanism to the loader (`ds4.c`): `find_tensor_alias()` /
`required_tensor_alias()` (plain names, used for the three top-level hc-head tensors) and
their formatted, per-layer analogs `tensor_by_namef_alias()` / `required_tensorf_alias()`
-- each tries the canonical (ds4/llama.cpp-dialect) name first and only falls back to a
documented alias if that's absent, printing a one-line `dialect compat` notice per hit, same
idiom as the metadata-key compat layer.

Iterating `--inspect` through successive `required tensor is missing` dies (each fixed only
after checking the donor's own tensor-creation code -- `src/models/deepseek4.cpp` /
`src/llama-arch.cpp` @5f55650 -- to confirm the alias really is the same tensor, plus a
header-only scan of the real artifact to confirm dims/type match what ds4 binds at that
site) surfaced one consistent dialect across the whole file: this community conversion
drops the GGUF `.weight`/`.bias` suffix entirely on several tensor families, and separately
renames a few tensors outright. Aliases added (canonical ds4/llama.cpp name -> file's alias),
all shape/type-verified against the header before wiring:

- Top-level: `output_hc_base.weight` -> `hc_head_base` {4} F32; `output_hc_fn.weight` ->
  `hc_head_fn` {16384,4} F32 (== `hc_dim` x `DS4_N_HC`); `output_hc_scale.weight` ->
  `hc_head_scale` {1} F32.
- Per-layer, suffix-dropped (verified on layer 0 unless noted, applies uniformly): 
  `hc_attn_fn/scale/base.weight` -> `hc_attn_fn/scale/base` ({16384,24}/{3}/{24} F32);
  `hc_ffn_fn/scale/base.weight` -> `hc_ffn_fn/scale/base` (same shapes); `attn_sinks.weight`
  -> `attn_sinks` ({64} F32 == `DS4_N_HEAD`); `exp_probs_b.bias` -> `exp_probs_b` ({256} F32
  == `DS4_N_EXPERT`); `ffn_gate_tid2eid.weight` -> `ffn_gate_tid2eid` ({6,129280} I32 ==
  `DS4_N_EXPERT_USED` x `DS4_N_VOCAB`).
- Per-layer, renamed: `attn_kv.weight` -> `attn_kv_latent.weight` ({4096,512} BF16 == 
  `DS4_N_EMBD` x `DS4_N_HEAD_DIM`/key_length=512; donor: `LLM_TENSOR_ATTN_KV`,
  `models/deepseek4.cpp:95`).
- Per-layer, compress-ratio-gated ("compressor" -> "compress", plus the same
  suffix-drop-on-the-get_rows-tensor pattern as `ffn_gate_tid2eid`): 
  `attn_compressor_{ape,kv,gate,norm}.weight` -> `attn_compress_ape` (bare) /
  `attn_compress_{kv,gate,norm}.weight`; `indexer_compressor_{ape,kv,gate,norm}.weight` ->
  `indexer.compress_ape` (bare) / `indexer.compress_{kv,gate,norm}.weight` (donor:
  `LLM_TENSOR_ATTN_COMPRESSOR_*` / `LLM_TENSOR_INDEXER_COMPRESSOR_*`,
  `llama-arch.cpp:477-480,610-613`).

**Result**: with these aliases, `--inspect` (`--cuda --ssd-streaming --ssd-streaming-cold
--ssd-streaming-cache-experts 8GB`) now binds every required tensor across all 43 layers and
the top level -- zero `required tensor is missing` for the rest of the run -- and reaches a
new, structurally different blocker: `tensor token_embd.weight has type bf16, expected f16`.
This is **not** a naming problem (the name matches exactly) but a dtype-support gap: this
artifact stores several dense (non-routed-expert) tensor families as GGUF BF16 (type 30) or
Q6_K (type 12) -- `token_embd.weight`, `output.weight`, `attn_kv_latent.weight`,
`ffn_{gate,up,down}_shexp.weight`, `indexer.proj.weight`, `attn_compress_{kv,gate}.weight`,
`indexer.compress_{kv,gate}.weight` (all BF16), and `attn_q_a/q_b/output_a/output_b.weight`,
`indexer.attn_q_b.weight` (all Q6_K) -- and ds4 has **no BF16 or Q6_K support anywhere in its
dense-tensor path**: `tensor_type_is_dense_quant()` (`ds4.c`) accepts only Q8_0/Q4_K/Q4_0,
the plain-tensor type checks (`tensor_expect_layout` call sites for `token_embd`,
`attn_kv`/`indexer_proj`, `attn_compressor_kv`/`gate`) hard-require `DS4_TENSOR_F16`/
`DS4_TENSOR_F32` specifically, and there is no `DS4_TENSOR_BF16` enum constant, no
BF16-to-float/f16 dequant helper (CPU or CUDA), and no Q6_K dequant-to-float path either
(`gguf_type_info[30]` registers BF16's block size for the type-name table only, unused
anywhere else; Q6_K has fused `vec_dot_q6_K` dot-product support per the type-info registry
comments elsewhere in the file but, like the pre-Q3_K/Q5_K state described in the
2026-08-01 entry above, no dequant-to-float/f16 kernel of its own). This contradicts this
scope doc's original assumption ("every other tensor becomes Q8_0",
`llama-quant.cpp:474-480`) for *this specific* community conversion: whatever quantize
recipe produced it kept token embeddings, the output head, several MoE-adjacent dense
projections, and MLA/indexer attention weights at BF16 or Q6_K rather than Q8_0. Confirmed
this is the true next blocker and not another naming difference: the `--inspect` log shows
zero further `required tensor is missing` lines before it.

**Not attempted here**: BF16 dequant (trivial bit-shift widening to F16/F32, unlike MXFP4/
Q3_K/Q5_K's block math) and Q6_K dequant-to-float (structurally identical porting job to the
Q3_K/Q5_K work already done for routed experts, just for a dense/non-routed tensor path) are
both real, scoped, and very likely small ports from `ggml-quants.c`/`convert.cu`
(`dequantize_row_bf16`-equivalent is nearly free; `dequantize_row_q6_K` already has a direct
llama.cpp analog) -- but wiring either into ds4's dense-tensor type-check/read path (as
opposed to the routed-expert dequant+GEMM family this port has focused on) is a distinct unit
of work from tensor-name aliasing and is deliberately left unstarted, per this ticket's own
scope boundary between "fix names" and "hit a structural blocker, stop and document."

No full-model load was reached this pass (blocked as above); the micro smoke-generation step
was accordingly not run.

### P1 status update (2026-07-31, cont'd): BF16/Q6_K dense-tensor load-time conversion

Follow-on pass closing the BF16/Q6_K dense-tensor blocker the previous pass stopped at
(`tensor token_embd.weight has type bf16, expected f16`). Rather than teaching every
dense-tensor validation/matmul call site a third accepted dtype (a much larger, more
invasive change touching ~15 CUDA dispatch sites for a format that should never need a
dedicated GEMM path), this converts BF16 and Q6_K dense tensors to F16 **once, at load
time**, in `model_open()` -- so every downstream consumer (CPU and CUDA, validation and
matmul) sees an ordinary F16 tensor and needs no changes at all.

**Conversion helpers** (`ds4.c`): `DS4_TENSOR_BF16` (30) added to the tensor-type enum
(the `gguf_types[30]` registry row already existed, unused). `bf16_to_f32()` (BF16 is
exactly the high 16 bits of a binary32, so this is a zero-extend + bitcast, no rounding).
`f32_to_f16_clamped()` -- unlike the existing `f32_to_f16()` (which lets IEEE-754 overflow
silently become infinity), this clamps to +-65504 and counts clamps; weights should never
be large enough to hit it, so a nonzero count is a correctness red flag the load now
surfaces rather than hides. `dequantize_row_q6_K()` (CPU, ported line-for-line from
llama.cpp `ggml-quants.c` @5f55650, MIT, same idiom as the pre-existing Q3_K/Q5_K ports --
Q6_K had a fused `vec_dot` but no dequant-to-float helper of its own before this).
`convert_row_q6_K_to_f16()` wraps it with a transient per-tensor float scratch buffer (the
Q6_K dense families this feeds -- attn_q_a/q_b, attn_output_a/b, indexer.attn_q_b -- are
LoRA-rank sized, small relative to the model, so one tensor's worth of scratch is not a
full-model memory concern).

**Scope-safe selection, no name matching needed**: `model_convert_dense_bf16_q6k()`
converts every tensor with GGUF type BF16 or Q6_K **and `ndim <= 2`**. Every dense role ds4
binds (token embeddings, the output head, attention/FFN projections, LoRA factors) is
<=2D; routed MoE expert tensors are always 3D (`in, mid, n_expert`). This makes the
`ndim<=2` filter a purely structural way to skip routed experts -- critically, it also
means GLM's *genuine* Q6_K/Q5_K **routed** experts (which must stay quantized/streamed, not
blown up to F16 -- they're the reason Q6_K already had routed-expert dequant support) are
left untouched by construction, with no risk of a blanket-by-type pass accidentally
converting them.

**Mechanism (grow the same mapping, not a second one)**: every ds4_tensor's data is
addressed as `model->map + tensor->abs_offset` throughout the codebase (CPU reads and every
CUDA dispatch site alike), and there's no per-tensor override-pointer concept to hook a
second, independent buffer into that addressing scheme without touching every call site.
So instead of allocating converted tensors elsewhere, this **grows `model->map` itself**:
reserve an anonymous address range sized to the original file plus the total conversion
output (computed by one pre-scan of the tensor table for eligible BF16/Q6_K tensors),
remap the file at the front of that reservation (`MAP_FIXED` into space just reserved, so
it cannot collide with anything else in the process), then map a writable anonymous
extension immediately after it. Each eligible tensor is then converted into that extension
and its `ds4_tensor` mutated in place: `type = F16`, `abs_offset` = its new position in the
extension, `bytes` recomputed. Runs right after `parse_tensors()` inside `model_open()`,
before `weights_bind()`/`weights_validate_layout()` (or anything else) ever sees the
tensor table, so **no validation or matmul code needed any BF16/Q6_K-awareness at all** --
they only ever observe already-converted F16 tensors.

One subtlety caught by a first crash (SIGSEGV in `tensor_family_key`, via gdb backtrace):
the original mapping is deliberately **not** unmapped after the switch, because every
`ds4_tensor.name` (a `ds4_str`) is a raw pointer into it, captured by `parse_tensors()`
before the remap -- unmapping would leave every tensor name (and this pass's own
family-notice printing) dangling. Leaking the original mapping is a pure
virtual-address-space cost (no extra physical memory beyond the handful of header/name
pages already touched), accepted rather than adding a name-copying step.

**Compat notice, grouped by family not by layer**: `tensor_family_key()` folds a
`blk.<N>.` layer index out of a tensor name (e.g. `blk.7.attn_kv_latent.weight` ->
`blk.N.attn_kv_latent.weight`) so one notice line covers all layers of the same tensor
role, avoiding 43x spam. Verified against the real artifact: 14 distinct families printed
(`output.weight`, `token_embd.weight`, `blk.N.attn_kv_latent.weight` (bf16, 43 tensors),
`blk.N.attn_output_a/b.weight` and `blk.N.attn_q_a/b.weight` (q6_k, 43 tensors each),
`blk.N.ffn_{gate,up,down}_shexp.weight` (bf16, 43 each), `blk.N.attn_compress_{kv,gate}.weight`
(bf16, 41 -- the two non-compressed early layers excluded), `blk.N.indexer.attn_q_b.weight`
(q6_k, 21 -- only compress-ratio-4 layers have an indexer), `blk.N.indexer.compress_{kv,gate}.weight`
and `blk.N.indexer.proj.weight` (bf16, 21 each) -- exactly the tensor families the prior
pass identified as BF16/Q6_K, and zero clamps fired on any of them.

**Result**: `--inspect` (`--cuda --ssd-streaming --ssd-streaming-cold
--ssd-streaming-cache-experts 8GB`) now gets past the BF16/Q6_K dense-tensor class of
blocker entirely -- no further `has type bf16/q6_k, expected f16` anywhere in the run --
and reaches a new, **structurally different** blocker: `tensor hc_head_fn has type f32,
expected f16`. `hc_head_fn` (the top-level hyper-connection function tensor, bound via the
`output_hc_fn.weight` -> `hc_head_fn` alias from the earlier tensor-name-compat pass) is
stored as **F32** in this file, not BF16 or Q6_K -- a third, distinct dtype-mismatch case
on a tensor outside this pass's scope (top-level `hc_head_fn`/`hc_head_base`/`hc_head_scale`
were not part of the BF16/Q6_K family list the previous pass identified; `hc_head_base` and
`hc_head_scale` are F32-typed *and* F32-expected already, so only `hc_head_fn` mismatches).
Per this ticket's own scope boundary, this is reported rather than folded in as a "same
class" fix: F32->F16 narrowing for a hyper-connection routing tensor is a different
precision-loss profile than BF16/Q6_K->F16 (BF16 and Q6_K already carry less mantissa
precision than F16 in the directions that matter, so widening to F16 is loss-free/near-
loss-free by construction; truncating an actual F32 tensor is a real new precision
question the ticket didn't scope) and deserves its own explicit sign-off rather than being
silently swept into this pass. No full-model load was reached; the micro smoke-generation
step was accordingly not run this pass either.

**Verification**: `make cuda-spark` -- clean build, no warnings. `./ds4_test` -- 6
failures, all within the pre-existing flaky set from the prior pass's own two-run
comparison (`tool-call-quality`, `think-tool-recovery`, `logprob-vectors`, `metal-kernels`;
`metal-tensor-equivalence` passed this run). No new failing test names.

### P1 status update (2026-07-31, cont'd): F32->F16 sign-off, --inspect reaches summary, first real --ssd-streaming run

Operator sign-off granted on the F32 question above. Extended `model_convert_dense_bf16_q6k()`
(renamed conceptually, not literally, to also cover F32) to convert F32 tensors to F16
under a narrower, explicitly source-checked eligibility rule, factored into a shared
`tensor_is_dense_conversion_candidate(type, ndim)` predicate:

- BF16/Q6_K: unchanged, `ndim <= 2`.
- F32: only `ndim == 2` (never `ndim == 1` -- every 1D F32 tensor in every validate path
  checked, deepseek4 FLASH/PRO, GLM, MTP, and DSpark alike, is a norm/bias/scale vector
  required to *stay* F32) **and** only when compiled `DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_DEEPSEEK4`.
  Verified directly against source before writing the rule: in `weights_validate_layout()`
  (deepseek4, non-GLM -- what this build is) every 2D tensor is hardcoded
  `DS4_TENSOR_F16` or dense-quant/routed-expert, *never* F32, so this is exhaustively safe
  there. `weights_validate_glm_dsa_layout()` (GLM) is the opposite: `ffn_gate_inp` and
  `indexer_proj` are *required* as literal 2D F32 -- the family gate exists specifically to
  keep this pass a no-op for a GLM-family build, so it cannot break GLM's real requirement.
  `mtp_weights_validate_layout()` uses `tensor_expect_plain_layout()` (F16-or-F32 accepted)
  for its 2D hc tensors, so F32->F16 there is harmless either way.

**A dense-tensor validator gap this surfaced**: `tensor_type_is_dense_quant()` (gates
`attn_q_a/q_b`, `attn_kv`, `attn_output_a/b`, `ffn_*_shexp`, `output` -- the exact BF16/Q6_K
families from the prior pass) accepted only Q8_0/Q4_K/Q4_0 -- not F16. Harmless before this
port (a "normal" `MXFP4_MOE` GGUF puts Q8_0 there per llama.cpp's own quantize recipe), but
our BF16/Q6_K->F16 conversion made these tensors F16, and validation for that whole role
group is only ever reached *after* the top-level output-head checks earlier in the same
function -- so the previous pass's `--inspect` run never actually exercised it (it died on
`hc_head_fn` first, before the per-layer loop). Fixed by adding `DS4_TENSOR_F16` to
`tensor_type_is_dense_quant()`'s accepted set; confirmed harmless everywhere else it's used
(e.g. `metal_graph_matmul_plain_tensor()` already special-cases F16 *before* falling through
to this check, so F16 tensors there were always routed to the dedicated F16 matmul path
regardless).

**Result**: `--inspect` now reaches the model summary, zero clamps across all 22 converted
tensor families (16 BF16/Q6_K + 6 new F32 ones: top-level `hc_head_fn`, and per-layer
`blk.N.ffn_gate_inp.weight`, `blk.N.hc_attn_fn`, `blk.N.hc_ffn_fn`, `blk.N.attn_compress_ape`,
`blk.N.indexer.compress_ape`):

```
model: DeepSeek V4 Flash
arch:  deepseek4
gguf:  v3, 51 metadata keys, 1328 tensors
layers: 43
train context: 1048576
attention: heads=64 kv_heads=1 head_dim=512 swa=128
indexer: heads=64 head_dim=128 top_k=512
experts: count=256 used=6 groups=0 groups_used=0
file size: 153.52 GiB
tensor bytes described by GGUF: 144.90 GiB
logical parameters: 284.33 B
tensor types:
  f32        492 tensors, 0.00 GiB
  f16        704 tensors, 13.61 GiB
  q3_k        30 tensors, 25.78 GiB
  q5_k         1 tensors, 1.38 GiB
  i32          3 tensors, 0.01 GiB
  mxfp4       98 tensors, 104.12 GiB
```

**Two more blockers hit and fixed while running the actual smoke generation** (both in
`ds4_cuda.cu`, both a consequence of the same root cause: this build's SSD-streaming path
has several places that assume every host-mapped tensor byte is also reachable via a
direct read of the *real on-disk file* at the same offset -- an assumption our load-time
conversion (which grows the host mapping past the real file's end to hold converted bytes)
breaks for exactly the tensors this port converts, and only those):

1. `cuda_model_range_ptr_from_fd()`: the direct/O_DIRECT-file-read fast path used under
   `--ssd-streaming` read via `pread(g_model_fd, ..., offset)` at the tensor's *current*
   (post-conversion) offset -- valid inside the grown host mapping, but past
   `g_model_file_size` (an `fstat` of the same fd, captured once, reflecting the real
   on-disk length) for every converted tensor, so the disk read was doomed regardless of
   retry. Fixed by checking `offset >= g_model_file_size` up front and short-circuiting
   straight to the host-mapping pointer in that case (the bytes are already correctly
   resident there -- nothing to fetch from disk).
2. `ds4_gpu_cache_model_range()` (called while preparing the token-embedding span)
   additionally requires the resolved range to show up in `cuda_model_range_is_cached()`'s
   bookkeeping (`g_model_ranges`) before it reports success, which only the *other*
   resolution branches populated -- the (1) fast-path bypass didn't. Fixed by pushing the
   same `g_model_ranges` entry the other successful-resolution branches push, so a later
   "is this range ready" query also sees it as covered.

Both fixes are scoped to `offset >= g_model_file_size`, i.e. they only change behavior for
bytes that did not exist on disk before this port's conversion pass -- zero behavior change
for any tensor at its real on-disk offset.

**Smoke generation** (`--cuda --ssd-streaming --ssd-streaming-cold --ssd-streaming-cache-experts
40GB --nothink -p "Reply with exactly: ok"`, foreground, 600s timeout): gets through model
load, the (now-fixed) token-embedding span preparation, and into prefill, then hits a
**third, unrelated** blocker -- unrelated to this pass's BF16/Q6_K/F32 conversion work, in
the pre-existing generic MXFP4/Q3_K/Q5_K routed-expert dequant+GEMM dispatch path
(`routed_moe_dequant_gemm_dispatch` family, `ds4_cuda.cu`): under `--ssd-streaming`, that
path requires a populated per-layer `g_stream_selected_cache` ("which experts are selected,
and their host-resolved gate/up/down pointers") before it will run, and reports rather than
crashes when it isn't populated:

```
ds4: CUDA streaming selected experts are unavailable for layer 0
ds4: prompt processing failed: cuda prefill failed
```

This is the "known unwired streaming-expert" class of gap the ticket anticipated (a
different specific mechanism than a CPU-matvec `ds4_die`, but the same general shape: P1's
GPU dequant+GEMM MoE path -- landed and unit-tested in isolation per the 2026-07-31 P1
status entry above -- was never previously exercised against a real `--ssd-streaming` run
of the actual 150GB artifact, since no full-model load had reached this far before this
pass). Populating/wiring `g_stream_selected_cache` for the generic dequant+GEMM family under
cold SSD streaming is real, scoped follow-up work, but is routed-expert MoE streaming
infrastructure, not a dense-tensor dtype question -- outside this pass's scope, reported
per the ticket's own stop-and-document instruction rather than chased further here. No
tok/s measured (prefill never completed).

**Verification**: `make cuda-spark` -- clean build, no warnings, across both the `ds4.c`
and `ds4_cuda.cu` changes in this update. `./ds4_test` -- 6 failures, same names as every
prior run (`tool-call-quality`, `think-tool-recovery`, `logprob-vectors`, `metal-kernels`);
no new failing test names.

### P1 status update (2026-08-01): wire the generic dequant+GEMM routed-MoE path into
CUDA SSD streaming's selected-expert cache

Follow-on pass closing the streaming-expert-cache gap the previous pass stopped at
(`ds4: CUDA streaming selected experts are unavailable for layer 0` /
`prompt processing failed: cuda prefill failed`, first hit running the real 150GB
artifact end-to-end under `--ssd-streaming`).

**Root cause, traced (not guessed) by reading the fused paths' own protocol**:
`routed_moe_launch` (`ds4_cuda.cu`) gates *every* routed-MoE branch (fused Q4_K,
fused IQ2_XXS+Q2_K, and the generic dequant+GEMM family alike) on a single shared
check: under `--ssd-streaming`, it refuses to run unless the per-layer
`g_stream_selected_cache` ("which experts are selected this step, plus their
host-resolved gate/up/down device pointers") is already populated and matches the
current layer/model/offsets -- if not, it prints the "streaming selected experts are
unavailable" line and returns 0. This check itself was never type-specific; the
generic MXFP4/Q3_K/Q5_K path was already going through the *same* gate the fused
paths use, contrary to what the previous pass's blocker message suggested at first
read. The actual gap was one level up, in `ds4.c`, at the two places that *populate*
the cache before `routed_moe_launch` ever runs:

1. `metal_graph_decode_cuda_selected_slots_expected()` (`ds4.c`) is the single
   predicate both `metal_graph_decode_cuda_selected_load()` (per-token decode) and
   `metal_graph_cuda_stream_prefill_batch_selected_load()` (batched prefill) consult
   to decide whether populating the cache is even worth attempting for a layer. It
   only ever recognized two type combos -- Q4_K/Q4_K/Q4_K and IQ2_XXS/IQ2_XXS/Q2_K --
   returning `q4 || iq2`, so a layer using the generic dequant+GEMM family
   (MXFP4/Q3_K/Q5_K, from f8d6222) never reached either populate call, regardless of
   dequant-kernel support already existing for those types.
2. `metal_graph_decode_set_hash_selected_override()` (`ds4.c`) is the analogous
   populate step for *hash layers* (`ffn_gate_tid2eid` present, layers 0-2 in the real
   artifact per the earlier dialect-compat pass): hash-layer expert selection is a
   deterministic per-token formula computed on the CPU
   (`layer_hash_selected_experts()`), not a learned router read back from the GPU, so
   it has its own cache-populate call gated on the same `q4_selected || iq2_selected`
   pair of Metal-specific predicates -- same gap, same two types, independently
   missing the generic family.

**Fix, not a new mechanism**: added one small predicate,
`metal_graph_dequant_gemm_selected_slots_type()` (mirrors `ds4_cuda.cu`'s own
`dequant_gemm_type_supported()` -- MXFP4/Q3_K/Q5_K -- kept in lock-step by hand since
the two live in different translation units with no shared header for this constant
list), and used it in both places above: `metal_graph_decode_cuda_selected_slots_expected()`
now also returns true when gate/up/down are each independently in that set (gate ==
up is ds4's pre-existing load-time invariant; down may legitimately differ, exactly
mirroring how `routed_moe_launch` already looks gate_type/down_type up independently
in its own dispatch table), and `metal_graph_decode_set_hash_selected_override()`'s
early-exit condition gained the same third alternative. Both changes call the exact
same populate functions (`ds4_gpu_stream_expert_cache_begin_selected_load()` /
`ds4_gpu_stream_expert_cache_prepare_selected_batch()`, both in `ds4_cuda.cu`) the
Q4_K/IQ2_XXS paths already used -- those functions were always type-agnostic (byte-
table-driven via `graph_stream_expert_table_make()` / `routed_expert_row_bytes()`,
already generalized for these types), so no cache/prefetch/eviction code needed
touching at all, only the two decision points that were skipping the call for our
tensor types. New env escape hatch (matching the existing per-family opt-out idiom):
`DS4_CUDA_DISABLE_DEQUANT_GEMM_SELECTED_EXPERT_VIEWS`.

**Verification**: `make cuda-spark` -- clean build from `make clean`, no warnings,
both edits. Test ladder against the real 150GB artifact (`--cuda --ssd-streaming
--ssd-streaming-cache-experts 40GB --nothink`), foreground, 600s timeout, polled
rather than blind-waited:

- `-p "Reply with exactly: ok"`, **without** `--ssd-streaming-cold` (default
  popularity-preload warm path): no longer hits the streaming-cache error anywhere.
  Completes end to end: `ds4: prefill: 0.73 t/s, generation: 0.75 t/s`. Output is
  fluent-looking but **not** the requested text and not coherent (see correctness
  note below): `  _0.**\n\n#  *  * It app\n#  *  *  *  * & is rarely of//s ... 및\n\n#
  What;architecture。\n\n babysitter\n\n##  _ _ POSTR8a5 "DYZyowndego-  _aRz6\n\n#  _
  · ,pDlz2释然红娘ande\n\n### wheels weelaborition,`.
- Same **with** `--ssd-streaming-cold`: also completes, no streaming-cache error,
  similar speed (`0.72 t/s` prefill / `0.77 t/s` generation), different garbage
  (`  *  *! \n\n#  *:^pDlz4lz+3 about:诶!1lek stress删除了 macros asus.Ir Év2 Release`).
  Confirms the fix is unconditional on the cold/warm preload choice, as expected
  (cache population, not preload policy, was the gap).
- Longer prompt (`-p "What is the capital of France? Answer in one sentence."`):
  **hit a different, new failure** -- `ds4: prompt processing failed: cuda prefill
  failed` with no diagnostic line at all (not the streaming-cache message this pass
  fixed). Traced far enough to identify it as a distinct gap, not a regression of
  this fix: `metal_graph_streaming_decode_prefill_max_tokens()` (`ds4.c`) caps the
  decode-style (per-token) streaming prefill path this fix targets at 18 tokens
  unless layer 0 is Q4_K (64 tokens then) -- the real artifact's layer 0 is MXFP4, so
  18 is the effective cap, and the longer prompt's token count exceeds it, diverting
  prefill into `metal_graph_prefill_layer_major()`'s streaming
  page-in/readahead/pread/madvise branch (`ds4.c`, the `layer_prepare` block after the
  `split_commands` check), which fails silently (no `fprintf` on the failing path) --
  a separate unit of streaming-prefetch infrastructure this pass did not touch.
  Confirmed this is a genuinely different code path, not our fix's blind spot, by
  forcing the short-prompt decode-style path with
  `DS4_METAL_STREAMING_DECODE_PREFILL_MAX=4096`: the same longer prompt then
  completes (no crash) through the exact mechanism this pass fixed, producing more
  (still incoherent) garbage output. Per this ticket's own stop-and-document
  precedent, the layer-major page-in path's silent-failure gap is reported here but
  not chased further -- distinct unit of work, streaming-prefetch scheduling rather
  than the selected-expert-cache population this pass scoped.

**Correctness note (not chased further, per the ticket's own instruction)**: this is
the first real look at whether MXFP4 dequant+GEMM inference is numerically correct,
and the answer is *partially* -- `--dump-logits` on a 1-token prompt (`"Hi"`) shows
finite logits throughout (no NaN/Inf, 129280/129280 finite), a plausible magnitude
range (min -37.9/max 21.2, mean -0.63, stdev 4.53), comparable to the IQ2_XXS
baseline's own 1-token dump on the same prompt (finite throughout, min -42.6/max 28.0,
mean 1.96, stdev 4.34) -- so this is not a catastrophic blow-up (no obviously broken
scale/exponent handling). But the argmax token is wrong: MXFP4 picks token 223 (a
plain space, logit 21.16) as the top continuation of "Hi", where IQ2_XXS picks token
19923 ("Hello", logit 27.97) -- a semantically sensible continuation. Generated text
in all three ladder runs above is fluent-looking token salad, not coherent language,
consistent with a real dequant/layout bug somewhere in the MXFP4 (or the Q3_K/Q5_K
down-projection, or the surrounding BF16/Q6_K/F32 dense-tensor conversion, or the
hash-layer/hyper-connection/indexer plumbing) rather than a total numerical collapse.
Root-causing which of these is out of scope for this pass (ticket explicitly says
"do not chase deep" here); flagging as the next P1 priority once streaming itself is
confirmed stable.

**Tests**: `./ds4_test` -- ran to completion once (900s timeout; a first attempt at
the default 300s timeout truncated mid-suite mid-way through `metal-tensor-equivalence`
and is not counted). Result: `ds4 tests: 6 failure(s)` across four failing test
sections -- `tool-call-quality` (2 assertions), `think-tool-recovery` (1),
`logprob-vectors` (1), `metal-kernels` (2) -- all four within the pre-existing flaky
set documented by every prior pass on this hardware (`tool-call-quality`,
`think-tool-recovery`, `logprob-vectors`, `metal-kernels`, `metal-tensor-equivalence`);
`metal-tensor-equivalence` passed this run. No new failing test names.

## Target artifact

Chosen (checkpoint hunt 2026-07-31): **lovedheart/DeepSeek-V4-Flash-GGUF**
`DeepSeek-V4-Flash-MXFP4_MOE.gguf` — single file, 150GB, standard GGML MXFP4 experts +
Q8_0 everything else. Bigger than the 121GB pool → genuine streaming target.

Alternatives noted: nsparks/DeepSeek-V4-Flash-FP4-FP8-GGUF (156GB, native-precision but
nonstandard FP8-block dense type); GLM-5.2 via MaliAir MXFP4_MOE (411GB / 77 shards,
streaming-only, later). No NVFP4 GGUF exists for either arch (P2 blocked on ecosystem).

Prior art: twaggs88/DeepSeek-V4-Flash-REAP25-DSpark-ds4-GGUF — a modified-ds4-only mixed
MXFP4/MXFP8 REAP25 build running ~24 tok/s decode on DGX Spark; independent proof of
concept on this hardware class. Read their layout before inventing ours.


### P1 status update (2026-08-01, cont'd): regression bisect (clean) + independent llama.cpp
ground truth (coherent) -- confirms the correctness bug is in ds4's own MXFP4/Q3_K/Q5_K
GPU inference path, not the artifact or the metadata/tensor-name dialect-compat layers

Follow-on pass to the previous entry's "argmax token wrong" correctness flag
(`--dump-logits` on `"Hi"`: MXFP4 argmax token 223 " " vs IQ2_XXS baseline's correct
token 19923 "Hello"). Per the debug plan's own cost-ordering, ran the two cheapest
discriminators before touching any code.

**1. Regression bisect (verdict: CLEAN)**: ran the known-good IQ2_XXS baseline
(`gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`, same
`deepseek4`/FLASH architecture, no `--ssd-streaming` needed -- fits resident) through the
**current** binary (`b15cc29`), `--cuda`, `--nothink -p "Hi" --dump-logits`. Argmax:
token 19923 "Hello", logit 27.97 -- byte-identical to the pre-FP4-port expectation, and
the run log shows **zero** dialect-compat/conversion notices (`grep -i
'compat|dialect|clamp'` on the log: no matches). This proves two things at once: (a) the
shared loader/dense-conversion machinery this port added is a true no-op for a file that
doesn't need it (no regression to the working path), and (b) baseline never exercises
*any* of the new FP4-port code (metadata-key compat, tensor-name alias compat,
BF16/Q6_K/F32 dense conversion, MXFP4/Q3_K/Q5_K dequant, or the streaming
selected-expert-cache generalization) -- so whatever is wrong is confined to that new
code, exactly as the debug plan anticipated, with no need to bisect the commit chain.

**2. Ground truth via independent llama.cpp @5f55650 (verdict: ARTIFACT IS CORRECT --
COHERENT OUTPUT)**: vanilla llama.cpp cannot load this GGUF any more directly than ds4
originally could -- it hits the identical missing-metadata-key wall ds4's compat layer
was built to paper over (`key not found in model: deepseek4.attention.output_group_count`,
then, once patched, `deepseek4.expert_gating_func`, a ninth required key this port's own
compat layer never needed to touch because ds4 hardcodes sqrt-softplus router scoring
unconditionally for the deepseek4 family rather than reading it from metadata -- new
finding, noted below), and then the identical tensor-name dialect mismatch ds4's
tensor-alias compat layer was built to paper over (`missing tensor 'output_hc_fn.weight'`).
Rather than treat this as a dead end, built a disposable (not committed --
`/tmp/patch_gguf_full.py` on robo-dog, same throwaway-tooling precedent as
`gguf_header_check.py`/`gguf_dump.py` from earlier passes) GGUF metadata+tensor-name
patcher: it rewrites only the header (KV pairs + tensor-info table) in a new output file,
adding the 9 metadata keys ds4's own compat layer derives (byte-identical values -- 8,
1024, 3, 4, 20, 1e-6, 160000.0, the 43-entry compress-ratio array, and the one new key,
`expert_gating_func=4`/`LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS`) and renaming every
tensor from this file's alias dialect to llama.cpp's own canonical names (the exact
reverse of ds4's `find_tensor_alias()` table, cross-checked against
`src/llama-arch.cpp`'s `LLM_TENSOR_*` string table directly rather than guessed) --
then appends the original 150GB tensor-data blob **unchanged** (tensor offsets are
relative to the data section start, so no per-tensor byte editing is needed, only a
~1-4KB header-growth shift; verified via the delta printed each run, e.g. 640 bytes for
the metadata-only patch). No tensor bytes are touched at any point -- ground truth is
against the exact same weight bytes ds4 reads.

Loaded successfully with `llama-cli` (CPU-only build, `-DGGML_CUDA=OFF`, since this
donor-inventory box's driver/toolkit didn't need testing here -- CPU is sufficient for one
forward pass per the ticket's own allowance) at `--ctx-size 2048 --temp 0 --top-k 1`:

```
> Hi
Hello! How can I help you today?
[ Prompt: 2.1 t/s | Generation: 1.3 t/s ]
```

Coherent, on-topic, grammatical -- matching the IQ2_XXS baseline's own correct behavior on
the same prompt. **This conclusively rules out the artifact itself** (the checkpoint hunt
picked a good file) **and, more specifically, rules out the metadata-key and
tensor-name dialect-compat layers as the bug source**: an independently-written loader
(llama.cpp, sharing no code with ds4 beyond both being MIT ports of the same upstream
dequant/architecture logic) given the exact same derived metadata values and the exact
same tensor-to-role bindings produces correct output. The bug is therefore isolated to
ds4's own MXFP4/Q3_K/Q5_K GPU numerical/wiring path: the dequant kernels
(`mxfp4_dequant_f16_kernel`/`q3_k_dequant_f16_kernel`/`q5_k_dequant_f16_kernel`,
`ds4_cuda.cu`), the generic dequant+GEMM dispatch
(`routed_moe_dequant_gemm_dispatch`/`dequant_gemm_row_f16gemm`), the BF16/Q6_K/F32
dense-tensor load-time conversion (`model_convert_dense_bf16_q6k`, `ds4.c`), or the
SSD-streaming selected-expert-cache wiring landed in the immediately preceding commit
(`b15cc29`) -- not the file, and not the two dialect-compat layers from the two passes
before that.

**New finding not previously documented**: llama.cpp's deepseek4 loader *requires*
`deepseek4.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS` (4) and
throws otherwise (`src/models/deepseek4.cpp:51-53`, "DeepSeek-V4 loader currently expects
sqrtsoftplus MoE scoring"). This artifact's community conversion omits this key too (a
tenth metadata gap beyond the eight this port's metadata-compat layer already handles),
but ds4 was never affected because it never reads this key at all for the deepseek4
family -- it hardcodes `sqrt(softplus(logit))` router scoring unconditionally
(`ds4.c`, `softplus_stable()` / the "Router scores use sqrt(softplus(logit))" comment
ahead of the routing-probability computation), which happens to already be the correct
(and only, per llama.cpp) DeepSeek-V4 gating function. No ds4 change needed here; noted
for completeness since it came up investigating ground truth, and as a decision precedent
if a future artifact ever needs this key surfaced.

**Localization attempted, inconclusive by inspection (root cause NOT found this pass)**:
manually re-read every candidate area the debug plan's step 3 lists, cross-checking
against the exact `sizeof(block_*)`/`gguf_type_info` registry entries and the actual
llama.cpp-donor dequant algorithms line-by-line, and found no structural defect:

- MXFP4/Q3_K/Q5_K CUDA dequant kernels (`ds4_cuda.cu`): `kvalues_mxfp4_dev` table and
  `mxfp4_e8m0_to_fp32_half_dev` match the standard E2M1/E8M0 constants; block sizes
  (17/110/176 bytes, 32/256/256 elements) match `gguf_type_info[39]`/`sizeof(block_q3_K)`/
  `sizeof(block_q5_K)` exactly; the row-major `[out_dim][in_dim]` write layout in each
  kernel matches the `cublasGemmEx(CUBLAS_OP_T, CUBLAS_OP_N, ...)` call's expected weight
  orientation in `dequant_gemm_row_f16gemm`. (Independent CPU reference ports in
  `test_mxfp4_moe.c`/`test_mixed_moe.c` already validated the underlying math in
  isolation, per the 2026-08-01 P1 entry above; this pass additionally confirmed the CUDA
  kernels' byte-layout constants against the registry rather than re-deriving the math.)
- `routed_expert_row_bytes()`/`routed_expert_block_bytes()` (`ds4.c`): generic,
  registry-driven, no MXFP4/Q3_K/Q5_K-specific special-casing that could introduce an
  off-by-one; `streaming_layer_routed_expert_bytes()`/`streaming_layer_gate_down_expert_bytes()`
  compute gate/up/down per-expert byte strides independently per tensor's own type, which
  is required (and correctly implemented) for the real artifact's mixed MXFP4-gate/
  Q3_K-or-Q5_K-down layers.
- Tensor-name aliasing (`find_tensor_alias()` call sites, `ds4.c` ~6490-6650): every
  canonical/alias pair matches this pass's own independently-derived llama.cpp
  `LLM_TENSOR_*` name table exactly (cross-checked while building the ground-truth
  patcher's rename map) -- no swapped or mistargeted alias found.
- `model_convert_dense_bf16_q6k()` (`ds4.c`): `bf16_to_f32` (zero-extend + bitcast) and
  `dequantize_row_q6_K` (ported line-for-line, same idiom as Q3_K/Q5_K) look correct by
  inspection; the `tensor_is_dense_conversion_candidate()` ndim<=2 filter structurally
  cannot catch a 3D routed-expert tensor.
- `routed_moe_dequant_gemm_dispatch()`'s per-pair loop (`ds4_cuda.cu`): gate/up/down
  buffer assignments, the SiLU(gate)*up*routing-weight combine order, and the
  `moe_sum_kernel`/`moe_sum_owned_kernel` reuse are all consistent with the fused
  Q4_K/IQ2_XXS paths' own conventions.
- `b15cc29`'s streaming selected-expert-cache wiring
  (`metal_graph_dequant_gemm_selected_slots_type()` and its two call sites): small,
  mechanical, mirrors the pre-existing Q4_K/IQ2_XXS predicates exactly; no asymmetry
  found between the two populate paths (per-token decode vs batched prefill vs the
  hash-layer override).

None of this rules out a subtle bug in any of the above (inspection is not proof), but it
means the remaining root-causing work needs runtime evidence, not more reading: the debug
plan's own step 3a (dump ds4's post-load/post-dequant tensor bytes for one known routed
expert row and diff against an independent Python dequant of the same raw file bytes) is
the next concrete, cheap step and was not reached this pass due to time spent on the
ground-truth detour above (justified, since it eliminated three of the plan's five
localization candidate categories -- artifact, metadata compat, tensor-name compat -- in
one shot, collapsing the remaining search space from "five suspect areas" to "the GPU
MXFP4/Q3_K/Q5_K numerical/wiring path specifically").

**No code change made this pass** -- nothing to fix yet, root cause not isolated to a
specific line. Per the ticket's own stop-and-document instruction, reporting rather than
guessing at a patch. `make cuda-spark` / `./ds4_test` not re-run (no source changed).

**Artifacts left on robo-dog for follow-up** (not committed, disposable):
`/tmp/patch_gguf_metadata.py`, `/tmp/patch_gguf_full.py` (the ground-truth GGUF patcher,
reusable for future ds4-vs-llama.cpp comparisons on this or similar community
conversions), `~/src/llama.cpp/build` (CPU-only build @5f55650), and
`~/src/ds4/gguf/DeepSeek-V4-Flash-MXFP4_MOE.patched.gguf` (150GB, the fully-patched
ground-truth file -- safe to delete to reclaim disk if not needed for a follow-up run;
285GB free on `/` as of this pass).


### P1 status update (2026-07-31): priority-hypothesis check (DISPROVEN, evidence-backed) --
runtime instrumentation rules out slot/cache indexing, weight-fetch offset math, and
dequant-kernel correctness for the generic dequant+GEMM routed-MoE path

Follow-on pass to the "root cause not isolated to a specific line" entry above. This pass's
brief specifically prioritized one hypothesis fitting the wrong-argmax signature exactly:
expert-index-vs-cache-slot confusion in `routed_moe_dequant_gemm_dispatch`/
`cuda_stream_selected_cache_begin_load` (`ds4_cuda.cu`) under `--ssd-streaming` -- i.e. that
the generic dequant+GEMM path might index the streaming selected-expert cache by *global
expert id* while the cache is populated *by compact slot*, or pair routing weights with the
wrong expert's dequantized rows. Rather than re-reading (three prior passes' worth of static
inspection already covered this code with no defect found), this pass added small,
env-gated (`DS4_DEBUG_STREAM_CACHE_LAYER=<n>`) `fprintf` instrumentation at three points and
ran the real 150GB artifact under `--cuda --ssd-streaming --ssd-streaming-cache-experts
40GB --nothink -p "Hi" -n 1`, capturing ground truth at each stage:

1. **Slot-remap correctness (cache build, `cuda_stream_selected_cache_begin_load`)**:
   printed each layer-3 call's `(global_expert_id -> cache_slot)` table alongside the
   dispatch loop's own `(pair, expert_i-used-as-index)` readback. For every observed call,
   `expert_i` used as the cache-buffer index in `routed_moe_dequant_gemm_dispatch` exactly
   matched the slot the cache-build step had assigned that same global expert to -- no
   transposition, staleness, or off-by-one found. (Note: DeepSeek-V4's top-6 expert
   selection never repeats an expert within one token, so the compact slot table happened
   to equal the trivial identity mapping in every sample observed here; this still
   positively confirms the *mechanism*, since the printed global-expert-id inputs differ
   sample to sample while the index arithmetic checked out every time.)

2. **Weight-fetch offset arithmetic and raw bytes (byte-exact against the file)**: for
   layer 3 / cache-slot 0 / global-expert 143, printed `gate_src_off` (the computed absolute
   file offset: `gate_offset + expert * gate_expert_bytes`) and the first 32 raw bytes ds4
   actually staged into the device cache buffer from that offset. Independently read the
   same 32 bytes directly from `gguf/DeepSeek-V4-Flash-MXFP4_MOE.gguf` at that exact offset
   in Python: **byte-for-byte identical**. This rules out any offset-computation or
   streamed-copy corruption in the cache-population path.

3. **Dequant kernel correctness on real (not synthetic) data**: layer 3's `ffn_gate_exps`
   turned out to be Q3_K (ggml type 11), not MXFP4 -- a useful reminder that "gate_type ==
   up_type, down varies" (per `f8d6222`) does *not* mean gate is always MXFP4; 30 of the
   98+30+1 routed-expert tensors are Q3_K and can appear on gate/up too. Ported
   `q3_k_dequant_f16_kernel`'s exact algorithm to a standalone Python reference and ran it
   against the same 110-byte raw Q3_K block captured above, then printed the CUDA kernel's
   actual dequantized f16 output for the same block (`DS4_DEBUG_STREAM_CACHE_LAYER=3`,
   pair 0, first 32 of 256 elements). The two match to f16 rounding precision (e.g.
   `0.0325928` GPU vs `0.03260612...` independent Python double-precision reference,
   `<0.001` relative difference, consistent with `__float2half` truncation, not a bug).

**Verdict: priority hypothesis DISPROVEN.** The streaming selected-expert cache is
populated with byte-exact correct weight data at the byte-exact correct offset for the
byte-exact correct global expert, addressed by the byte-exact correct compact slot, and
the fallback dequant kernel (Q3_K case verified directly against a from-scratch reference;
MXFP4/Q5_K already covered by the synthetic unit tests plus the earlier passes' registry-
level constant audit) reproduces that data correctly. Every stage of "which expert's bytes
get fetched from the file and turned into a dequantized row" for the generic dequant+GEMM
path is now runtime-verified correct on the real artifact, not just by static inspection.

**Narrowed remaining search space**: the defect must be in one of the stages *after*
per-expert dequant (the per-pair cuBLAS GEMM orientation, the SiLU(gate)*up*weight combine
kernel, or the final `moe_sum`/`moe_sum_owned` reduction -- all re-audited by inspection
this pass with no defect found either, but not yet runtime-verified end-to-end the way
stages 1-3 above were) or, per the debug plan's own step 3b, somewhere entirely outside the
routed-MoE dispatch (attention, the hash-layer/hyper-connection plumbing visible in the
llama.cpp eval-callback trace -- `hc_head_mixes`/`hc_attn_scale` etc. -- or an interaction
between per-layer streaming and those mechanisms that a single-layer synthetic test
wouldn't catch). The next concrete, cheap step is debug plan step 3b: per-layer hidden-
state norm/sum comparison against the llama.cpp ground truth (already captured in this
pass's `/tmp/llamacpp_eval.log` on robo-dog -- `ffn_out-N`/`ffn_moe_out-N`/`attn_out-N` sums
for all 43 layers on the same `"Hi"` prompt, ready to diff against an equivalent ds4-side
per-layer dump once one exists) to find the first diverging layer, rather than further
inspection of the routed-MoE arithmetic this pass already exercised at the byte level.

**No code fix made this pass** -- the disproven hypothesis was the ticket's specified
priority check; per its own instruction, reporting the (negative, but evidence-backed)
result rather than continuing to guess. The three small debug hooks added
(`DS4_DEBUG_STREAM_CACHE_LAYER`, gated `fprintf`s in `ds4_cuda.cu`'s
`cuda_stream_selected_cache_begin_load`/`routed_moe_dequant_gemm_dispatch`/
`dequant_gemm_row_f16gemm`) are left in place, committed -- zero-cost when the env var is
unset, useful for the next pass's step 3b work, and follow the codebase's existing
env-gated debug/profile idiom (e.g. `DS4_CUDA_STREAMING_EXPERT_CACHE_PROFILE`).

**Verification**: `make cuda-spark` -- clean build from `make clean`, no warnings, all
debug-instrumented edits. Smoke ladder against the real artifact, foreground with
timeouts, polled: `-p "Reply with exactly: ok"` and the France prompt (with
`DS4_METAL_STREAMING_DECODE_PREFILL_MAX=4096`) both reproduce the exact prior-documented
symptom (finite, fluent-shaped, incoherent text; `~0.70-0.76 t/s` prefill/generation) --
confirms the debug instrumentation is a true no-op on the actual generation path.
`./ds4_test`: `ds4 tests: 10 failure(s)` across four sections -- `tool-call-quality`, `logprob-vectors`, `metal-kernels`, `metal-tensor-equivalence` -- all four within the pre-existing flaky set documented by every prior pass on this hardware; `think-tool-recovery` passed this run. No new failing test names.


### P1 status update (2026-08-01, cont'd): FIRST DIVERGENCE FOUND AND FIXED --
`dequantize_row_q6_K` missing per-block output offset zeroed nearly all Q6_K-dialect-
compat-converted dense attention weights; MXFP4 artifact now produces coherent output

Follow-on pass to the priority-hypothesis-disproven entry above. Per this pass's own
brief, ran debug plan step 3b: captured per-layer hidden-state stats for the real MXFP4
artifact under `--cuda --ssd-streaming --ssd-streaming-cache-experts 40GB --nothink`
and diffed against `/tmp/llamacpp_eval.log`'s ground truth.

**Step 1 (prompt/token confirmation)**: `/tmp/llamacpp_eval.log`'s header shows `number
of input tokens = 1`, token `23166`; independently confirmed via a throwaway GGUF-vocab
reader that token 23166 decodes to `"Hi"` -- llama.cpp's eval-callback ground truth is a
**raw, untemplated single-token completion**, not a chat-formatted prompt. ds4's default
`-p "Hi"` path applies its chat template (10 tokens with `--dump-logits` confirming
`prompt_tokens: 10`), so matching the SAME token sequence required ds4's undocumented
`--raw`/`--raw-prompt` flag (`ds4_cli.c`, `gen->raw_prompt`); `-p "Hi" --raw --nothink
--dump-tokens` confirms `[23166]`, byte-identical to llama.cpp's input.

**Step 2 (instrumentation)**: no new instrumentation was needed -- `ds4.c` already has a
generic, env-gated per-tensor graph dump (`metal_graph_debug_dump_tensor`, controlled by
`DS4_METAL_GRAPH_DUMP_PREFIX`/`_NAME`/`_LAYER`/`_POS`, already wired at every named point
`llama.cpp`'s eval-callback also logs: `attn_out-N`, `hc_attn_post-N`, `ffn_out-N`,
`hc_ffn_post-N` per layer) that this port's earlier passes had not yet exercised for this
purpose. Used it directly, `DS4_METAL_GRAPH_DUMP_LAYER=all`, against both the MXFP4
artifact and (as the sanity yardstick the ticket asked for) the known-good IQ2_XXS
baseline under the identical `--raw --nothink --ssd-streaming` flags.

**Step 2 result (yardstick + first divergence)**: IQ2_XXS's per-layer `attn_out`/`ffn_out`
sums track llama.cpp's ground truth closely at every layer checked (e.g. layer 1 `attn_out`
sum 22.07 ds4-IQ2XXS vs 23.05 llama.cpp; layer 3 `ffn_out` 8.09 vs 7.36) -- confirming
"benign difference" looks like agreement within roughly 10%, consistent with IQ2_XXS's own
lossy quantization versus llama.cpp's higher-precision reference. MXFP4, by contrast,
diverges catastrophically starting at **layer 0's `attn_out`**: every element is ~0
(`sum=-0.0000`, `absmax=0.0001`) instead of the expected magnitude-tens values, and this
`attn_out ~ 0` pattern repeats identically for **every one of the 43 layers** -- a uniform,
layer-0-onward failure, not a single-layer defect. `hc_attn_post`/`hc_ffn_post` stay
superficially plausible only because the hyper-connection residual mostly passes the prior
layer's state through unchanged when the attention contribution is ~0.

**Step 3 (drill-down, attention pipeline)**: dumped every intermediate attention tensor
for layer 0 and found the exact break point: `q_lora`/`Qraw`/`Qnorm`/`Qcur` (the Q-LoRA
projection and its descendants) have **only their first element populated; every other
element is exactly 0.0** (not garbage -- a clean, untouched-buffer zero), while the
sibling `KVraw`/`KVnorm`/`KVcur` computed from the same `attn_norm` input are fully
populated and numerically sane. This "row 0 real, every other row exactly 0" signature,
reproducible identically whether `attn_q_a`'s matmul goes through the codebase's
Q8_0-hardcoded fused pair kernel (`ds4_gpu_matmul_q8_0_pair_tensor`) or the type-correct
F16 cuBLAS path (`ds4_gpu_matmul_f16_tensor`), pointed away from the GEMM/kernel layer
entirely and at the **weight data itself**.

**Root cause, confirmed at the byte level**: `dequantize_row_q6_K()` (`ds4.c`), the CPU
routine `model_convert_dense_bf16_q6k()`'s load-time Q6_K-\>F16 dense-tensor dialect-compat
conversion calls to turn this artifact's Q6_K `attn_q_a`/`attn_q_b`/`attn_output_a`/
`attn_output_b`/indexer weights into plain F16, has a real, ported-in defect: its inner
per-block write (`y[n + l + 0] = ...`, etc.) never adds the block index's own offset
(`i * QK_K`) to `n`/`l` before indexing `y[]`. For the function's *pre-existing* caller (a
genuine single-block, `k == QK_K` per-row dequant used elsewhere) this bug is a no-op
(`nb == 1`, so the missing offset is always 0 anyway) -- which is exactly why three prior
passes' static inspection and the earlier "runtime-verified" Q3_K dequant check (a
*different* type, on the streaming *expert* dequant path, never routed through this
function) never caught it. This port's *new* load-time dense-tensor conversion is the
first caller to invoke it with `k` spanning the WHOLE flattened 2D tensor (thousands of
blocks per call, e.g. 16384 blocks for `attn_q_a`'s 4096x1024 shape) -- and with the
missing offset, every block overwrites the same first-256-float window of the destination,
so only the LAST block processed survives there and every other destination float -- i.e.
essentially the entire tensor -- is left at its `malloc`'d-but-never-written default of
zero. Verified directly: a throwaway debug hook printing `attn_q_a`'s converted bytes at
element 4096 (`QK_K * 16`, exactly one `attn_q_a` row) showed `0,0,0,0` for every one of
the 43 layers, both mid-tensor and at the tensor's own final element, while element 0
was correctly non-zero and matched the file's real Q6_K-decoded value.

**Second, independent (but currently-masked) bug found and fixed alongside it**: while
localizing the above, found that the single-token decode graph's Q-LoRA/KV-raw projection
(`metal_graph_encode_decode_layer_phase`, `ds4.c`) unconditionally calls the Q8_0-specific
fused kernel (`ds4_gpu_matmul_q8_0_pair_tensor`) and its own non-pair fallback
(`ds4_gpu_matmul_q8_0_tensor`) for `attn_q_a`/`attn_kv` with **no check of the tensor's
actual type** -- correct only when those tensors are natively Q8_0 (true for the IQ2_XXS
`AProjQ8` baseline, hence never caught before), but silently wrong for any dialect-compat
file where `model_convert_dense_bf16_q6k()` has converted them to F16 (interprets F16 byte
layout with a hardcoded Q8_0 34-bytes/32-elements block stride). The batch/prefill graph
already guards this correctly (`metal_graph_matmul_q8_0_named_tensor()` ->
`metal_graph_matmul_dense_quant_tensor()`, which dispatches on `w->type`); this pass
mirrors that guard into the decode graph. This bug was empirically masked by the first one
during isolation (with the weight data itself zero beyond block 0, misreading it as Q8_0
happened to also decode to ~0), so it was not independently distinguishable until both
were fixed together and verified against ground truth.

**Fix**: two changes in `ds4.c`.
1. `dequantize_row_q6_K()`: added the missing per-block `y0 = i * QK_K` offset to all four
   `y[]` writes.
2. `metal_graph_encode_decode_layer_phase()`'s Q-LoRA/KV-raw projection: gate the
   Q8_0-specific fast paths on `layer->attn_q_a->type == DS4_TENSOR_Q8_0 &&
   layer->attn_kv->type == DS4_TENSOR_Q8_0`, falling back to the already-existing
   type-generic `metal_graph_matmul_plain_tensor()` otherwise.

**Post-fix verification**: re-ran the same per-layer dump against the real MXFP4 artifact.
`attn_out`/`ffn_out` sums now track llama.cpp's ground truth closely at every layer
checked -- e.g. layer 0 `attn_out` -19.90 (ds4) vs -19.06 (llama.cpp), `ffn_out` 19.35 vs
19.34; layer 1 `attn_out` 24.12 vs 23.05, `ffn_out` 5.76 vs 5.67; layer 3 `ffn_out` 7.54 vs
7.36 -- differences consistent with f16-accumulation noise, the same magnitude of
agreement the IQ2_XXS yardstick showed against the same ground truth. Smoke ladder against
the real 150GB artifact, foreground with timeouts, polled:
- `-p "Reply with exactly: ok" --nothink`: `ok` (exact match).
- `-p "What is the capital of France?" --nothink` with
  `DS4_METAL_STREAMING_DECODE_PREFILL_MAX=4096`: `The capital of France is Paris.` --
  coherent and correct, replacing the prior-documented fluent-but-incoherent garbage.
  `~0.72-0.77 t/s` prefill/generation (unchanged perf envelope; this pass was a
  correctness fix, not a performance one).

`make clean && make cuda-spark`: clean rebuild, no warnings. `./ds4_test`: `ds4 tests: 10
failure(s)` across four sections -- `tool-call-quality`, `logprob-vectors`,
`metal-kernels`, `metal-tensor-equivalence` -- all four within the pre-existing flaky set
documented by every prior pass on this hardware; no new failing test names.

**Scope note**: this pass's earlier debug hooks (`DS4_DEBUG_STREAM_CACHE_LAYER` etc. from
the previous entry) are unrelated to this bug and left untouched. This pass's own
throwaway diagnostic probes (an F16-weight-pointer dumper, a cuBLAS-path tracer, a
per-block Q6_K dequant tracer) were removed once the root cause was isolated and fixed;
one small, permanent, env-gated debug hook was kept in the same style
(`DS4_METAL_DEBUG_DENSE_CONVERT=<name-substring>`, in `model_convert_dense_bf16_q6k()`)
since it is a generally useful tool for verifying any future dialect-compat dense-tensor
conversion by name, not specific to this bug.

### P3a status (2026-08-01): fused decode (n_tokens==1) matvec kernels for
MXFP4/Q3_K routed experts

Follow-on pass to the P1 correctness fix above (`dequantize_row_q6_K` offset bug), now
that MXFP4 streaming inference is confirmed correct but slow (0.75-0.79 t/s decode on the
real 150GB artifact). This pass adds fused, device-side-indexed decode kernels for
MXFP4/Q3_K routed experts, removing the per-layer host readback and per-(token,expert)-pair
f16 dequant + cuBLAS GEMM chain that `routed_moe_dequant_gemm_dispatch` (P1's generic
fallback) does for every decode step.

**Kernel design.** Cloned the block/warp organization of the IQ2_XXS+Q2_K fused decode
kernels (`moe_gate_up_mid_mid_kernel` / `moe_down_f32_kernel`, `ds4_cuda.cu`) rather than
the Q4_K fused path: one CUDA block per (out-row, token*n_expert-slot) pair, 256 threads
each decoding a strided subset of quantized blocks straight into a register dot-product
accumulator, then a shared-memory tree reduction (`partial[256]`) to one value per block.
gate/up consume f32 activations directly (`x->ptr`), matching what the IQ2_XXS+Q2_K decode
kernels already consume — **not** q8_K-staged activations. This is a documented deviation
from the ticket's stated preference for the Q4_K decode kernel structure: Q4_K's own decode
path (`moe_gate_up_mid_decode_q4K_qwarp32` and ~a dozen sibling specializations gated by
booleans like `use_q4_mma_tiles16`/`use_q4_gate_h16r8`/`use_direct_down_sum`, spanning
hundreds of lines around `ds4_cuda.cu:21376-22541`) is a heavily hand-tuned,
warp-shuffle/MMA-tile family specific to Q4_K's 4-bit/32-scale-per-superblock layout and
would take a dedicated pass on its own scale to port faithfully; the IQ2_XXS+Q2_K path is
the structurally closer, simpler sibling this ticket's own text explicitly permits ("or
f32/f16 activation if that's what the Q4_K decode kernels actually consume -- follow the
existing pattern exactly" / "whichever existing K-quant decode kernel is structurally
closest" for Q3_K, extended here to the gate/up path too for consistency and time budget).

Two new `__device__` dot-product helpers do the in-register block decode, each a line-for-
line port of the existing dequant kernels' math (`mxfp4_dequant_f16_kernel`,
`q3_k_dequant_f16_kernel`) but accumulating `dl * v * x[i]` instead of writing a
dequantized row:
- `dev_mxfp4_dot_f32`: one 32-elem MXFP4 block (17 bytes: E8M0 scale + 16B E2M1 nibbles),
  reuses the existing `kvalues_mxfp4_dev` / `mxfp4_e8m0_to_fp32_half_dev` constants.
- `dev_q3_k_dot_f32`: one 256-elem Q3_K super-block (110 bytes: hmask/qs/scales/d), reuses
  the existing scale/hmask unpacking exactly as `q3_k_dequant_f16_kernel` does.

Two new kernels, each a C++ template on a compile-time `bool` selecting MXFP4 vs Q3_K (two
instantiations apiece, no runtime branch inside the per-block loop):
`moe_gate_up_mid_fused_fp4q3k_decode_kernel<GATE_MXFP4>` (writes `gate_out`/`up_out`/
`mid_out` = SiLU(gate)*up*routing_weight, same as the IQ2_XXS kernel's combine) and
`moe_down_fused_fp4q3k_decode_kernel<DOWN_MXFP4>`. gate_type covers both gate and up
(ds4's pre-existing load-time invariant); down_type is looked up independently, so a layer
with MXFP4 gate/up and Q3_K down (or vice versa) dispatches through the same call with each
half using its own template instantiation -- mirroring `routed_moe_dequant_gemm_dispatch`'s
own gate/down-independent type lookup.

Critically, unlike `routed_moe_dequant_gemm_dispatch` (which does one `cudaMemcpy`
device->host of the `selected[]` table per MoE-layer call because cuBLAS needs weight
pointers on the host), the new kernels index `selected[]` **device-side**, exactly like the
Q4_K/IQ2_XXS fused paths -- no host sync, no f16 intermediate materialization, no cuBLAS
call at all.

**Q5_K is out of scope for the fused path** (per the ticket's own MXFP4/Q3_K wording):
`fused_fp4q3k_decode_type_supported()` only accepts types 39/11, so any layer with a Q5_K
gate/up or down (the real artifact has exactly one Q5_K down tensor among 129 routed
tensors) still falls through to `routed_moe_dequant_gemm_dispatch` even at decode.

**Wiring** (`routed_moe_launch`): added `fused_fp4q3k_decode_path = n_tokens==1 &&
!q4k_path && !iq2_q2k_path && fused_fp4q3k_decode_type_supported(gate_type) &&
fused_fp4q3k_decode_type_supported(down_type) && !getenv("DS4_CUDA_DISABLE_FUSED_FP4_DECODE")`,
checked before `dequant_gemm_path` so it takes priority whenever it matches; prefill
(`n_tokens>1`) is untouched and always falls through to the existing `dequant_gemm_path`
loop. New escape hatch `DS4_CUDA_DISABLE_FUSED_FP4_DECODE=1`, mirroring the existing
`DS4_CUDA_DISABLE_DEQUANT_GEMM_SELECTED_EXPERT_VIEWS` idiom, forces the generic path even
for decode-shaped MXFP4/Q3_K calls -- used below both as the "before" measurement toggle and
as the fused-vs-generic correctness cross-check.

**Correctness.** Extended `research/gb10/test_mxfp4_moe.c` and `test_mixed_moe.c`: after
each decode-shaped (`n_tokens==1`) case's existing CPU-reference check, if the type
combination is fused-supported, the identical inputs are re-dispatched through
`ds4_gpu_routed_moe_batch_tensor()` a second time with `DS4_CUDA_DISABLE_FUSED_FP4_DECODE=1`
set (forcing the generic path) and the two GPU outputs are compared directly against each
other (0.02 relative tolerance, same tolerance the existing CPU-reference checks use).
`test_mxfp4_moe`'s decode case (MXFP4/MXFP4) and `test_mixed_moe`'s MXFP4/Q3_K and
Q3_K/Q3_K decode cases all exercise this cross-check; the MXFP4/Q5_K decode case correctly
skips it (down_type=13 not fused-supported) and continues through the generic path only,
as designed. All cases pass: fused output agrees with the CPU reference and with the
generic-path output, and all pre-existing (unmodified) prefill-shaped cases still pass.

```
$ ./research/gb10/test_mxfp4_moe
MXFP4 MoE test: all cases passed
$ ./research/gb10/test_mixed_moe
mixed routed-MoE test: all cases passed
```

**End-to-end**, real 150GB artifact, `--cuda --ssd-streaming --ssd-streaming-cache-experts
40GB --nothink`, foreground with timeouts, polled:
- `-p "Reply with exactly: ok"`: `ok` (exact match). `prefill: 0.91 t/s, generation: 0.95 t/s`.
- `-p "What is the capital of France?"` with `DS4_METAL_STREAMING_DECODE_PREFILL_MAX=4096`:
  `The capital of France is Paris.` -- coherent and correct. `prefill: 0.92 t/s,
  generation: 0.99 t/s`.

Both stay coherent -- no regression from the P1 correctness fix's baseline behavior.

**Performance**, same warm state (no cache drops between conditions -- both measured back
to back on the same already-warm page cache from the correctness/smoke runs immediately
preceding), fixed prompt (`"Explain in a few sentences how photosynthesis works."`, `-n
100`, `DS4_METAL_STREAMING_DECODE_PREFILL_MAX=4096`), 3 runs each, decode t/s from the
`generation:` field ds4 prints itself:

| run | before (generic, `DS4_CUDA_DISABLE_FUSED_FP4_DECODE=1`) | after (fused, default) |
|---|---|---|
| 1 | 0.79 t/s | 1.00 t/s |
| 2 | 0.79 t/s | 1.04 t/s |
| 3 | 0.79 t/s | 1.04 t/s |
| prefill (same runs) | 0.75 / 0.75 / 0.75 t/s | 0.92 / 0.96 / 0.96 t/s |

Decode: 0.79 t/s (all 3 before runs, identical to two decimal places) -> 1.00/1.04/1.04
t/s after -- roughly a 27-32% decode speedup from removing the per-layer host readback
and per-pair cuBLAS/f16 chain for the ~89% of routed-expert tensors (98 MXFP4 + 30 Q3_K of
129) this pass's fused kernels now cover at decode. Prefill also improved slightly
(0.75->0.92-0.96 t/s) even though prefill dispatch is unchanged code -- attributable to
prefill's own decode-style per-token streaming-prefill path (the
`DS4_METAL_STREAMING_DECODE_PREFILL_MAX=4096`-forced mechanism, itself built from per-token
`n_tokens==1` calls into `routed_moe_launch`) also picking up the fused kernels
incidentally, not a change to the true `n_tokens>1` batched-prefill GEMM path, which this
pass explicitly left untouched. All three generation texts stayed coherent and consistent
with the P1 fix's baseline quality (compare: "Photosynthesis is the process by which
plants, algae, and some bacteria convert sunlight into chemical energy...").

This is a real but modest win, not a step-change to the ~16-24 t/s class the Q4_K/IQ2_XXS
fused tile kernels or the REAP25 prior-art reach -- the fused kernels here still do a
naive one-thread-per-block-element strided scan with no warp-shuffle reduction, no
half2/vectorized loads, and no MMA/tensor-core usage (a genuine port of Q4_K's own
warp-tile machinery, deferred per the design-summary note above). Follow-up P3b scope:
warp-shuffle reduction (replace the 256-wide shared-memory tree with a warp-level
`__shfl_down_sync` reduction plus a small cross-warp combine, matching the Q4_K decode
kernels' own reduction shape) and vectorized MXFP4/Q3_K block loads, before reaching for
tensor-core prefill kernels.

**Verification.** `make clean && make cuda-spark`: clean rebuild, no warnings. `./ds4_test`
(single run, 900s): `ds4 tests: 11 failure(s)` across the five pre-existing flaky sections
documented by every prior pass on this hardware (`tool-call-quality`,
`think-tool-recovery`, `logprob-vectors`, `metal-kernels`, `metal-tensor-equivalence`) --
no new failing test names.

### P3a diagnosis (2026-08-01, cont'd): why the expert-cache sweep is flat -- root
cause found (CUDA `--ssd-streaming-cache-experts` is a no-op), disk-bound confirmed
with numbers, instrumentation added

Follow-on to the P3a expert-cache sweep (`MEASUREMENTS.md`) that found decode flat at
1.02-1.04 t/s across 8/40/60/100 GB cache budgets. This pass root-causes it: reads the
actual fetch code path (not just log lines), measures the real OS/hardware bound with
concurrent `iostat`/`nvidia-smi dmon`/`pidstat` capture, and adds CUDA-side counters that
directly confirm the finding against the real 150GB artifact
(`gguf/DeepSeek-V4-Flash-MXFP4_MOE.patched.gguf`, the same dialect-compat + BF16/Q6_K-
converted file the sweep itself used -- verified by file listing, not assumed).

**Q1 -- bound discrimination.** Concurrent capture during a steady-state 100-token decode
(`--ssd-streaming-cache-experts 60GB`, `ds4-server` stopped, page cache dropped
beforehand, single-GPU process tracked by PID so the capture window is exactly the ds4
process's own lifetime -- an early attempt without PID-scoping was contaminated by an
unrelated k3s pod's disk burst and discarded):

| resource | decode-phase measurement |
|---|---|
| NVMe (`iostat -x`, `nvme0n1`) | **~3.0-3.2 GB/s sustained read** (rkB/s mean 3.156e6 KB/s), **aqu-sz mean 14.9** (near the storage characterization's own QD16 class), **%util mean 67%**, r_await mean 0.59 ms (up from ~0.08 ms at idle/near-QD1) |
| GPU SM (`nvidia-smi dmon -s u`) | **mean ~18% SM utilization** during decode, never exceeding ~31% |
| CPU (`pidstat -u`, ds4 process only) | **mean ~44% of one core** (out of 2000% available on this 20-core host) -- not even one core saturated |

Verdict: **disk-bound**, not compute-bound and not CPU-bound. The NVMe is doing real,
sustained work close to (about 80-86% of) its own measured QD16 scattered-read ceiling
(3.73 GB/s, `MEASUREMENTS.md`'s storage characterization table) while GPU and CPU sit
mostly idle. This also resolves an artifact in the *first* (mis-scoped) capture attempt:
sampling without pinning to the ds4 PID's actual lifetime picked up an unrelated
high-throughput burst from a k3s pod on the same host, which looked like "disk barely
used" until the window was corrected to the process's own start/exit timestamps
(`pidstat`'s per-sample `Time` column) -- a caution for any future capture on this shared
host: always scope iostat/dmon windows to the target PID's own lifetime, not a fixed
wall-clock sleep.

**Q2 -- cache participation, traced.** Read the full call chain from `routed_moe_launch`
(`ds4_cuda.cu`) through to the byte-level fetch. The b15cc29-wired populate path is
`ds4_gpu_stream_expert_cache_begin_selected_load()` /
`ds4_gpu_stream_expert_cache_prepare_selected_batch()`, both of which delegate to a single
static function, `cuda_stream_selected_cache_begin_load()` (`ds4_cuda.cu`, ~line 23888).
Its **first line** is `cuda_stream_selected_cache_invalidate()`, which unconditionally
zeroes `g_stream_selected_cache.valid` -- discarding whatever the device-side gate/up/down
staging buffers held from the *previous* call, every single call, with no comparison
against the newly-requested expert set first. The function then always proceeds to copy
every selected expert's gate/up/down bytes fresh via `cuda_model_copy_to_device_streamed()`
straight from `table->model_map` (the mmap'd GGUF file view) into the (now-empty) staging
buffers. **There is no lookup against a persistent, expert-id-keyed cache anywhere in this
function or its callers** -- `g_stream_selected_cache` is a same-call scratch buffer sized
to the current token's selected-expert set, not a cache with cross-token reuse.

This is confirmed independently, and more starkly, by the budget-wiring functions
themselves: `ds4_gpu_set_streaming_expert_cache_budget(uint32_t experts)` (`ds4_cuda.cu`,
~line 28305) is `{ (void)experts; }` -- **a complete no-op**. Its companion
`ds4_gpu_set_streaming_expert_cache_expert_bytes(uint64_t bytes)` is likewise `{ (void)bytes;
}`, and `ds4_gpu_stream_expert_cache_configured_count(void)` unconditionally `return 0;`.
**The `--ssd-streaming-cache-experts N` CLI flag is wired through `ds4.c` all the way to
these three CUDA entry points and then discarded.** There is no code path on CUDA, at any
value of N, that makes cache size do anything at all -- the flat 8/40/60/100 GB sweep curve
in the prior pass wasn't measuring a saturated cache or a compute ceiling; it was measuring
a parameter with zero effect on the code it's supposed to configure.

Contrast with `ds4_metal.m`'s GLM streaming implementation, the mature reference: it has a
real per-`(layer, expert)` LRU (`g_stream_expert_cache[layer][expert]`, `ds4_metal.m`
~line 602 onward) with an actual hit path
(`ds4_gpu_stream_expert_cache_peek()`/`_get_protected()`, ~line 13086-13270) that checks
`ds4_gpu_stream_expert_cache_entry_matches()` (same tensor identity: model map, byte
offsets, sizes) against the live entry *before* deciding whether to reload, increments
`g_stream_expert_cache_hits`/`_misses` accordingly, and only re-`pread()`s + reinstalls on
an actual miss; entries are LRU/hotness-evicted only when `g_stream_expert_cache_entry_count`
exceeds the *configured* budget (`ds4_gpu_stream_expert_cache_prune_global()`, consulting
`ds4_gpu_stream_expert_cache_configured_budget()` -- a real, non-stub function on Metal).
CUDA has never had an equivalent; it only ever had the same-call staging buffer described
above, which every prior CUDA streaming pass built fused/generic dispatch on top of without
anyone tracing whether a "cache" existed underneath it.

**Reconciling with Q1's measured bytes:** since the OS's own page cache is the *only*
caching layer left in play once the CUDA-level "expert cache" is confirmed inert, and the
box has 108-117 GiB free RAM regardless of the `--ssd-streaming-cache-experts` value passed
(that value, being a no-op, never reserves anything real), the flat sweep curve now has a
complete explanation: every arm from 8 GB to 100 GB was, in terms of actual caching
behavior, identical -- all of them relied on nothing but the kernel's page cache over the
same free-RAM pool, and none of them consulted any ds4-level structure that the
`--ssd-streaming-cache-experts` flag could have changed.

**Q3 -- the 11 bypass layers, confirmed.** Log line source: `ds4.c` ~line 56914
(`"ds4: SSD streaming mixed-precision model: %u/%u routed layers off the slab size class
will bypass the expert cache..."`), gated by `boosted > 0` where `boosted` counts layers for
which `weights_streaming_layer_experts_uniform()` (`ds4.c` ~5177) returns false --
i.e. a layer's `streaming_layer_routed_expert_bytes()` (gate+up+down bytes summed) doesn't
match `ds4_streaming_routed_expert_bytes()`, the *modal* (most common) per-expert byte class
across all 43 layers (the "slab size class" the CUDA/Metal cache-of-fixed-size-slots design
assumes).

Confirmed by direct calculation from the real artifact's tensor-type inventory (43 layers:
32 pure-MXFP4 gate/up/down, 10 pure-Q3_K gate/up/down, 1 mixed MXFP4-gate/up + Q5_K-down --
derived from the header scan in the 2026-08-01 Q3_K-port entry above, where gate/up-type
counts (33 MXFP4-layer, 10 Q3_K-layer) and down-type counts (32 MXFP4, 10 Q3_K, 1 Q5_K)
combine to exactly these three groups, with no other combination possible under the file's
own tensor counts): MXFP4 has 0.53125 bytes/element (17B/32-elem block), Q3_K has
0.4296875 bytes/element (110B/256-elem superblock), Q5_K has 0.6875 bytes/element
(176B/256-elem superblock). The 32 pure-MXFP4 layers are the modal class (confirmed by the
runtime's own printed accounting: `"4306 experts, 12.75 MiB each"` at the 60 GB arm, i.e.
`4306/43=100.1 ~ n_expert*layers`, consistent with 43 layers all being *counted* toward the
slab-class total even though only 32 are actually eligible -- `ds4_streaming_cacheable_expert_count()`
only counts layers matching the slab class, so the true cacheable set is 32 layers x
`DS4_N_EXPERT`, not all 43; the 4306 figure in the log reflects the byte-budget-derived
expert count at 60 GiB, not the eligible-layer count). The 10 pure-Q3_K layers and the 1
mixed MXFP4/Q5_K layer -- **11 layers total, exactly matching the log's own count and the
ticket's own type-theory suspicion (the 10 Q3_K + 1 Q5_K expert layers)** -- have different
per-expert byte totals and are excluded from the size-class slab by construction.

**Important qualifier, not previously stated:** on CUDA this bypass distinction is **moot**.
It matters on Metal, where excluded layers are routed to a different code path (direct
mapped-view reads) than the other 32 layers (which get real LRU cache treatment) --  a
genuine two-tier system. On CUDA, since Q2 establishes there is no real cache for *any*
layer, all 43 layers -- both the 32 "slab class" and the 11 "bypass" -- already go through
the identical same-call staging-buffer fetch in `cuda_stream_selected_cache_begin_load()`.
The bypass/non-bypass split changes nothing observable in CUDA's actual behavior; the log
line is accurate about the *classification* (confirmed above) but its stated *consequence*
("bypass the expert cache and read experts via mapped model views") doesn't differentiate
CUDA runtime behavior the way it does on Metal, because CUDA has no cache-path/mapped-view
fork to begin with -- everything is already "mapped model views" on CUDA. Estimated
per-token traffic if the 11 layers *were* the only miss source (Metal-style, cache hit
elsewhere): ~0.7-0.9 GB/token (10 layers x 6 experts x ~10.3 MiB + 1 layer x 6 experts x
~14.0 MiB, using the byte-rate math above with `n_embd=4096`, per-expert intermediate
dim=2048 inferred from the 12.75 MiB/expert log figure) -- consistent with, and clarifying,
the ticket's own "~0.9 GB/token" estimate, but this number describes Metal-shaped behavior,
not what's actually happening on the CUDA path measured in Q1/Q5.

**Q4 -- instrumentation added.** `DS4_CUDA_STREAM_STATS=1` env-gated counters, matching the
existing `getenv("DS4_CUDA_...")`-per-call self-gating idiom used throughout `ds4_cuda.cu`
(e.g. `DS4_CUDA_DISABLE_FUSED_FP4_DECODE`, `DS4_CUDA_WEIGHT_CACHE_VERBOSE`). Four host-side
`uint64_t` atomics-by-single-threaded-construction (this fetch path only ever runs on the
one CUDA-dispatch thread, so plain counters are correct and cheap -- no `std::atomic`
needed) incremented at the exact fetch decision point traced in Q2, inside
`cuda_stream_selected_cache_begin_load()`'s per-expert copy loop, only after all three
copies (gate/up/down) for that expert succeed: `g_cuda_stream_stats_fetch_calls` (one per
`begin_load` call, i.e. one per layer per token-step), `g_cuda_stream_stats_expert_fetches`
/ `_cache_misses` (one per distinct expert actually copied), `g_cuda_stream_stats_bytes_from_file`
(`gate_expert_bytes*2 + down_expert_bytes` per fetch). `g_cuda_stream_stats_cache_hits` /
`_bytes_from_cache` are wired but structurally can never increment today (no code path
serves bytes without going through the copy loop) -- kept, with a comment explaining why,
so the absence is an empirically-reported zero rather than a field that doesn't exist; this
also means the counters are forward-compatible with a future real-cache fix without needing
a second instrumentation pass. `ds4_gpu_print_cuda_stream_stats(void)` (new, declared in
`ds4_gpu.h`, implemented in `ds4_cuda.cu`, no-op stub in `ds4_metal.m` for build symmetry --
Metal already has its own real hit/miss reporting via `ds4_gpu_print_memory_report` /
`DS4_METAL_STREAMING_EXPERT_LAYER_STATS`) prints one summary line, self-gated on the env var,
called from `ds4_cli.c`'s `run_sampled_generation()` (the function backing the `-p ... -n
...` single-shot CLI path used by every arm of the P3a sweep and this ticket's own
methodology) right after the existing `"ds4: prefill: ... generation: ..."` line. This
closes the documented gap ("Metal has `--expert-profile`; CUDA has nothing").

**Q5 -- re-run with counters, 60 GB arm, 3 reps**, same fixed prompt/flags as the sweep
(`--ssd-streaming-cache-experts 60GB --nothink -p "Explain in a few sentences how
photosynthesis works." -n 100`, `ds4-server` stopped, page cache dropped before each rep):

| rep | prefill t/s | decode t/s | fetch_calls | expert_fetches | hit_rate | bytes_from_file | bytes/gen-token |
|---|---|---|---|---|---|---|---|
| 1 | 0.90 | 1.00 | 3784 | 22704 | **0.000** | 270.77 GiB | 2.91 GB |
| 2 | 0.93 | 1.00 | 4300 | 25800 | **0.000** | 307.69 GiB | 3.30 GB |
| 3 | 0.94 | 1.00 | 3870 | 23220 | **0.000** | 276.92 GiB | 2.97 GB |

`expert_fetches / fetch_calls = 6.00` exactly in all three reps -- confirms
`DS4_N_EXPERT_USED=6` and that every routed layer's `begin_load` call fetches exactly the
router's top-6 selection, every time, with zero deduplication against anything previously
resident. **Measured hit rate: 0.000 in all three reps -- the cache-participation finding
from Q2 is not a code-reading inference, it is now a directly measured runtime fact.** Mean
bytes/token (counting `bytes_from_file` against the 100 generated tokens, prefill's smaller
contribution folded in) ~3.06 GB/token, closely matching both (a) the Q1 disk-bandwidth
back-calculation (~3.0-3.2 GB/s measured / ~1.0 tok/s decode) and (b) the prior sweep
writeup's own independent estimate (~3.62 GB/token, computed a completely different way, via
QD16-max-bandwidth / observed-t/s) -- three independent measurement methods (host-side fetch
counters, iostat, and the earlier bandwidth/throughput inference) now agree within ~15-20%
on the same number, which is strong triangulation that this is real, not measurement noise.

**Fix recommendation (not attempted this unit, per scope).** The disk-bound ceiling and the
zero-hit-rate finding are two separable problems with two separable fixes:

1. **Make `--ssd-streaming-cache-experts` do something on CUDA.** Port Metal's
   `g_stream_expert_cache[layer][expert]` LRU design (`ds4_metal.m` ~line 12900-13300,
   `ds4_gpu_stream_expert_cache_peek`/`_get_protected`/`_install_loaded`/`_prune_global`) to
   CUDA: a persistent, device-resident, expert-id-keyed table of gate/up/down device
   pointers with `entry_matches()`-style identity checks, populated lazily on miss and
   consulted on every `begin_load` call *before* falling back to
   `cuda_model_copy_to_device_streamed()`, evicted by the existing hotness/LRU discipline
   once `ds4_gpu_stream_expert_cache_configured_count()` (currently hard-`return 0`, needs
   to actually read the budget) is exceeded. This is the change that would make the
   `--ssd-streaming-cache-experts` sweep methodology from the prior pass actually measure
   what it was designed to measure. Scale: comparable to the Metal implementation it would
   port (a few hundred lines, one new persistent data structure, changes concentrated in
   `ds4_cuda.cu`), not a redesign of the streaming architecture.
2. **Independently, disk bandwidth is the wall even with hits accounted for.** Even a
   perfect CUDA-side cache only helps the *repeat* fraction of expert selections across
   tokens (session-local routing correlation, not yet measured -- "session working-set size
   from routing traces" is still an open item in `MEASUREMENTS.md`'s Pending list). The
   32/43 "slab class" layers could then benefit from real reuse; the 11 bypass layers
   (per Q3) would still need every-token reads regardless of any cache, since they're
   excluded from the size-class slab design entirely -- a distinct, likely smaller,
   follow-up (giving bypass layers their own smaller slab class, or a secondary
   variable-size cache tier) noted in the prior pass's own Pending list and unchanged by
   this diagnosis.
3. Fix (1) is the higher-leverage, correctly-scoped next unit: it's blocked on nothing this
   pass didn't already establish (the trace, the counters, and the measured 0% hit rate all
   point at the same missing piece), and the counters added here are exactly what a fix
   would use to prove itself working (a non-zero hit rate on a repeat run would be the
   acceptance signal).

**Verification.** `make clean && make cuda-spark`: clean rebuild of `ds4`, `ds4-server`,
`ds4-bench`, `ds4-eval`, `ds4-agent`, no warnings. Real-artifact runs above (Q1 capture +
Q5 3-rep counter runs) all completed end-to-end with coherent, on-topic photosynthesis
generations, consistent with every prior pass's correctness baseline -- no regression from
adding the counters (they are pure additive instrumentation: four new counters, one new
print function gated on an unused-by-default env var, one new call site, no changes to any
existing control flow or the byte-copy path itself beyond the counter increments
immediately following it). `ds4-server` stopped for every measurement run in this pass and
restarted afterward; see confirmation below.


## P3a fix: real CUDA per-(layer,expert) LRU expert cache (2026-08-01)

Implements the diagnosis pass's own fix recommendation (immediately above): ports
`ds4_metal.m`'s `g_stream_expert_cache` design (~line 560-620 struct/globals,
`ds4_gpu_stream_expert_cache_peek`/`_install_loaded`/`_prune_global` ~line 12900-13320) to
`ds4_cuda.cu` as `cuda_stream_expert_cache_entry` / `cuda_stream_expert_cache_peek`/
`_install`/`_prune_global`/`_clear_all`, a persistent, device-resident,
`(layer,expert)`-keyed table consulted by `cuda_stream_selected_cache_begin_load()` before
it falls back to `cuda_model_copy_to_device_streamed()` (the mapped-file fetch this whole
diagnosis traced). `ds4_gpu_set_streaming_expert_cache_budget`/`_expert_bytes`/
`ds4_gpu_stream_expert_cache_configured_count` -- all no-op stubs or hard-`return 0` before
this pass -- are now real.

**Allocation choice.** Each cache entry owns its own exactly-sized `cudaMalloc`'d gate/up/
down device buffers -- the same allocation idiom `g_stream_selected_cache`'s own staging
buffers already use (`cuda_stream_selected_ensure_bytes()`), not `cudaMallocManaged`/
host-pinned. A hit is served with a device-to-device `cudaMemcpy` from the persistent entry
into the existing per-call packed staging buffer that the downstream fused MXFP4/Q3_K decode
kernels already read from -- this was a deliberate choice to avoid changing the
kernel-facing packed-buffer interface at all (kernels are unmodified). A miss still pays the
existing `cuda_model_copy_to_device_streamed()` fetch (disk read + host-to-device copy) into
the packed buffer, then additionally installs those bytes into the persistent cache via a
second device-to-device copy so the entry survives past this call. Net effect: a hit skips
both the disk read and the host-to-device copy (satisfying the ticket's own constraint),
replacing them with two comparatively cheap device-to-device copies (into the packed buffer,
and on install, out of it).

**Eviction.** Global least-recently-used across the whole `(layer,expert)` table, mirroring
Metal's `_prune_global`'s recency ordering (this port omits Metal's additional
"route-hotness" tie-break layer, judged not worth the extra bookkeeping for a first port --
plain LRU is what actually governs eviction order in the vast majority of Metal's own cases
too, since hotness only breaks *ties* at equal recency).

**Budget mapping.** `N` in `--ssd-streaming-cache-experts N` (or the NGB form, already
converted to a plain expert count by `ds4.c` upstream of the GPU backend, identically for
Metal and CUDA) is stored directly as the entry-count budget and clamped to the fixed
`80 x 384`-entry table bound, matching Metal's own `DS4_METAL_STREAM_EXPERT_CACHE_MAX_ENTRIES`
clamp. No CUDA-side NGB math was needed -- `ds4.c` already did it generically.

**Bypass-layer decision: brought them in, not left structural.** Metal's cache is a
single-size-class slab allocator (all cached experts must share one gate/up/down byte
total, to make `MTLBuffer` pooling cheap); the 11 Q3_K/Q5_K layers off that size class are
excluded from Metal's cache by construction. CUDA's per-entry (not slab-pooled)
`cudaMalloc` allocation has no such uniform-size requirement, so bringing the 11 bypass
layers into the same LRU was a small, natural extension -- done this pass. All 43 routed
layers participate in one CUDA cache; there is no CUDA-side bypass tier anymore. (This also
means the informational-only `ds4_gpu_set_streaming_expert_cache_expert_bytes()` no longer
gates CUDA caching the way its Metal namesake gates Metal's slab class -- kept for CLI/log
symmetry with `ds4.c`'s own "N/43 routed layers... bypass" log line, which remains accurate
about the underlying tensor-size classification even though it no longer describes a real
CUDA behavioral fork.)

**Correctness.** `-p "Reply with exactly: ok" --nothink` (40 GB budget): `ok` (exact match).
`-p "What is the capital of France?" --nothink` with
`DS4_METAL_STREAMING_DECODE_PREFILL_MAX=4096` (40 GB budget): `The capital of France is
Paris.` (exact match) -- both byte-identical to every prior pass's recorded text,
confirming greedy-decode determinism held through the new cache (no staleness/eviction
bug). `./ds4_test`: `ds4 tests: 5 failure(s)`, all three failing sections
(`tool-call-quality`, `logprob-vectors`, `metal-kernels`) on the pre-existing flaky list
from every prior pass; `metal-tensor-equivalence` passed this run; no new failing test
names.

**Benchmark (full table and interpretation in `MEASUREMENTS.md`, "P3a fix" section).**
Summary: the previously-flat 8/40/100 GB sweep now bends as hypothesized. 40 GB and
100 GB both reach ~79-81% measured hit rate and 2.5-2.85x the pre-cache flat decode
baseline (0.73->2.59 t/s at 40 GB budget=2700 entries; ~1.03->2.96 t/s at 100 GB
budget=7519, cache saturating at 4.4-4.9k entries in practice). The 8 GB arm (budget=130
entries, under the diagnosis pass's own ~192-258-entry per-token working-set warning) stays
at hit_rate=0.000 exactly as before, but now measures a **regression** to 0.73 t/s (from
1.02-1.04 t/s pre-cache) -- attributed to `cudaMalloc`/`cudaFree` install/evict churn on
every miss with no offsetting hit benefit at a too-small budget; noted as a follow-up
(pooled/reused install buffers) in `MEASUREMENTS.md`'s Pending list, not fixed this pass.
Memory watched through the 100 GB arm's load and steady state: host `used` peaked ~80 GiB
of 121 GiB physical, no swap touched, no OOM risk.

**Verification.** `make clean && make cuda-spark`: clean rebuild of `ds4`, `ds4-server`,
`ds4-bench`, `ds4-eval`, `ds4-agent`, no warnings (including the diagnostic
`g_cuda_expert_cache_expert_bytes` field, which is read back by a small getter used in the
`DS4_CUDA_STREAM_STATS` print line specifically to avoid a dead-store warning). No changes
to the downstream fused-decode kernel interface or the packed
`g_stream_selected_cache` staging-buffer layout -- the persistent LRU sits entirely upstream
of both, so this is additive to the existing streaming architecture. `ds4-server` stopped
for every measurement run in this pass and restarted afterward; see confirmation below.

## P3b item 1: layer-major streaming prefill fixed -- `pread()` past real EOF into the
BF16/Q6_K dense-conversion mmap extension (2026-08-01)

Root-caused and fixed the `cuda prefill failed` crash on realistic (~25-140-word) prompts
under `--ssd-streaming` that every prior pass worked around with
`DS4_METAL_STREAMING_DECODE_PREFILL_MAX=4096`. Confirmed hypothesis: prompts past the
18/64-token decode-style-prefill cap (`metal_graph_streaming_decode_prefill_max_tokens()`)
route into `metal_graph_prefill_layer_major()`'s streaming page-in branch, whose default
mechanism (`metal_graph_stream_pread_range()`, layer-pread, enabled by default under
`--ssd-streaming`) issues `pread(model->fd, ..., offset)` bounded only by `model->size` --
which `model_convert_dense_bf16_q6k()` (`3106c3c`) grows past the real on-disk file to hold
converted dense-tensor bytes (`attn_q_a`/`attn_q_b`/`attn_output_a`/`attn_output_b`/
`indexer.attn_q_b`, all present in every layer's decode span set). Any layer-prepare span
touching a converted tensor's offset issues a `pread()` past real EOF, fails, and the whole
prefill call fails with no diagnostic -- deterministic given a specific span, but
apparently-intermittent across prompts/reruns because which spans a given prompt's expert
routing touches varies. Fix: new `ds4_model.file_size` (real on-disk length, captured once
in `model_open()` before any growth) plus a clamp in `metal_graph_stream_pread_range()` that
only `pread()`s the on-disk portion of a range and directly touches the
already-correct-and-resident mmap extension for the rest, mirroring the CUDA-side
`cuda_model_range_ptr_from_fd()` fix for the same underlying mmap-growth invariant.

Full root-cause narrative, fix detail, and before/after verification (France + a 118-word
prose prompt + a 158-word GPQA-style prompt, all without the env-var workaround, all
repeated including immediately after an explicit page-cache drop) in `MEASUREMENTS.md`'s
"P3b item 1" section. `make clean && make cuda-spark`: clean, no warnings.
`test_mxfp4_moe`/`test_mixed_moe`/`test_mxfp4_dequant`: all pass.

## P3b item 2: prefill batching for the generic dequant+GEMM routed-MoE fallback
(2026-08-01)

New `routed_moe_dequant_gemm_dispatch_prefill_grouped()` (`ds4_cuda.cu`): for prefill-shaped
(`n_tokens>1`) calls, groups (token,expert) pairs by expert (host-side stable `qsort` on the
call's own already-required selected-expert readback -- structurally mirroring the Q4_K
tile8 fused prefill path's own device-side `sorted_pairs`/`offsets` grouping, just done
host-side since this fallback already needs the expert table on the host for cuBLAS), so
each distinct expert's gate/up/down weight matrices are dequantized ONCE per prefill call
(not once per token that selected it) and driven through a single `n=group_size` cuBLAS GEMM
(not `n=1` per token). Decode (`n_tokens==1`) is untouched; new
`DS4_CUDA_DISABLE_DEQUANT_GEMM_PREFILL_BATCH` escape hatch forces the old per-pair loop.
Selected-cache/LRU population protocol (P3a-fix's CUDA expert LRU) is unchanged -- purely a
compute-shape change downstream of expert-pointer resolution.

Full design rationale, correctness verification (`test_mxfp4_moe`/`test_mixed_moe`'s own
`n_tokens=5` prefill-shaped cases now exercise this path by default; both pass with the
grouped path on and off), and before/after prefill throughput on a ~300-token prompt
(~1.72-1.74 t/s -> ~4.60-4.64 t/s, ~2.65x, 3 reps each, same warm state) in
`MEASUREMENTS.md`'s "P3b item 2" section. `make clean && make cuda-spark`: clean, no
warnings.

## P3b item 3: pooled allocator for the CUDA expert LRU -- partial fix for the 8GB-arm
regression (2026-08-01)

New size-keyed device-buffer pool (`cuda_stream_expert_pool_class_for`/`_alloc`/`_free`/
`_release_all`, `ds4_cuda.cu`): `cuda_stream_expert_cache_install()`/`_clear_entry()`'s raw
`cudaMalloc`/`cudaFree` per miss/eviction become pool pop/push (real `cudaMalloc` only on
genuine growth); the existing whole-cache-reset call sites (`cuda_stream_expert_cache_clear_all()`
-- model swap, streaming-mode toggle, budget change, all rare/non-per-token) additionally
release the pool back to the driver, so real teardown is unchanged.

Measured result: 8GB-arm decode improves from the pre-fix 0.73 t/s to ~0.97-0.99 t/s (4
reps, page cache dropped before each rep) -- a real +34% improvement, but short of this
unit's own acceptance bar (>= the 1.02-1.04 no-cache baseline). Root-cause of the residual
gap not chased further this pass (out of "pooled/slab allocator" scope specifically):
likely candidates are the three real device-to-device `cudaMemcpy`s per install (pooling
removes the allocator cost but not the memcpy work, 100% wasted at a 0%-hit-rate budget) and/or
`cuda_stream_expert_cache_prune_global()`'s O(30720)-entry global LRU scan run on
essentially every call at this budget -- flagged as next steps in `MEASUREMENTS.md`'s "P3b
item 3" section, which has the full numbers and analysis. `make clean && make cuda-spark`:
clean, no warnings. `test_mxfp4_moe`/`test_mixed_moe`/`test_mxfp4_dequant`: all pass.

## P3c-1: sm_121 mxf4 tensor-core probe -- BLOCKED at toolchain level (2026-08-01)

Scope: replace the dequant-to-f16 + cuBLAS step inside the grouped-prefill path
(`routed_moe_dequant_gemm_dispatch_prefill_grouped`, the P3b item 2 target above) with
direct MXFP4 tensor-core GEMM kernels consuming the 17-byte blocks natively, via donor
`mma.sync.aligned.kind::mxf4.block_scale...m16n8k64` from llama.cpp `mma.cuh` (gated there
on `__CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL(1200)`, which includes `GGML_CUDA_CC_DGX_SPARK
= 1210` i.e. our sm_121).

**Per the ticket's own instruction, a 20-line standalone probe was written and run FIRST,
before any integration work.** Probe (`research/gb10/mxf4_probe.cu`, kept in-repo as
evidence): one warp, one `mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.
row.col.f32.e2m1.e2m1.f32.ue8m0` call with uniform operands -- every A/B nibble set to
E2M1 table index 2 (value 1.0 after the kvalues_mxfp4 `*0.5` convention), every E8M0 scale
byte set to 127 (2^0 = 1.0) -- so the expected D is exactly 64.0f in all 128 output floats
(sum of 64 k-products of 1*1, scale 1*1) **regardless of the per-lane element permutation**,
making it a layout-independent correctness check without needing to reverse-engineer the
donor's `tile<>`/`load_ldmatrix` thread-mapping first.

**Result: PROBE FAILS TO COMPILE on every target this toolkit's `ptxas` accepts** -- not
merely on sm_121. Compiled with `nvcc -arch=<X>` for X in {`sm_121`, `sm_121a`, `sm_121f`,
`compute_121a`, `compute_121f`, `sm_120`, `sm_120a`, `sm_120f`, `compute_120a`,
`compute_120f`, `sm_100`, `sm_100a`, `sm_103a`, `sm_110a`} (i.e. GB10 itself, every
Blackwell/DGX-Spark-family and "a"/"f" variant spelling, and datacenter Blackwell sm_100/103
for comparison), identical `ptxas` errors on all of them:

```
ptxas ..., line 31; error   : Instruction 'mma with block scale' not supported on .target 'sm_121'
ptxas ..., line 31; error   : Feature '.kind::mxf4' not supported on .target 'sm_121'
ptxas ..., line 31; error   : Feature '.block_scale' not supported on .target 'sm_121'
ptxas ..., line 31; error   : Feature '.scale_vec::2X' not supported on .target 'sm_121'
ptxas fatal   : Ptx assembly aborted due to errors
```

(target name in the message tracks whichever `-arch` was passed; error set identical across
all 14 targets tried). Also re-tried with the `mxf4nvf4`/`scale_vec::4X`/`ue4m3` (NVFP4)
variant of the same instruction on the "a" targets -- same rejection.

Toolchain: `/usr/local/cuda-13.0` (`nvcc`/`ptxas` release 13.0.88, Aug 2025 build; the only
CUDA install present on robo-dog -- `/usr/local/cuda-13`/`/usr/local/cuda` both symlink to
it, no alternate/older toolkit to fall back to). `nvcc --list-gpu-arch` on this install caps
out at `compute_121` with no `f`/`a`-suffixed entries listed at all (the `f`/`a` suffixes
were accepted syntactically by `nvcc` but produced byte-identical `ptx`/errors to the
unsuffixed target, i.e. this `ptxas` build doesn't distinguish family/arch-conditional
variants for these SMs). `ptxas` itself parses `.kind::mxf4`/`.block_scale`/`.scale_vec::2X`
as recognized PTX tokens (the errors are "not supported on this target", not "unknown
directive") -- so the PTX ISA grammar exists in this `ptxas`, but its SASS-generation
backend has no codegen for it on any target it will assemble for, including sm_100/103
(datacenter Blackwell, not just consumer/DGX-Spark). This reads as a toolkit-vintage gap
(this instruction class may need a CUDA 12.8/12.9-era `ptxas` per NVIDIA's original Blackwell
block-scaled-MMA rollout, and something about the 13.0 release on this box's sbsa-linux
target either regressed or never carried the codegen) rather than an sm_121-specific
hardware/gating gap -- but that is inference from the evidence above, not confirmed against
an NVIDIA changelog (none found locally; no internet access from this pass).

**Per the ticket's explicit instruction ("if the toolchain rejects mxf4 for this arch, STOP
and report the exact error -- do not fight the compiler for hours"), this pass stops here.**
No kernel-design or integration work was attempted (constraints #2/#3/#4 in the ticket are
moot until the probe passes on some toolchain). No model server activity was needed or
performed this pass (server left running throughout, never touched). Concrete next step for
whoever picks this up: try an older/newer CUDA toolkit install on robo-dog specifically for
its `ptxas` block-scaled-MMA codegen support (12.8/12.9 first) before spending more time on
sm_121 gating specifics; if no such toolkit is obtainable in this sovereignty-constrained
environment, P3c-1's native-tensor-core-MXFP4-prefill goal is blocked until one is, and the
existing P3b grouped-dequant+cuBLAS path (~4.6-4.64 t/s prefill) remains the fastest
available prefill path for MXFP4 experts.

## P3c-1 correction: prior "toolkit-vintage gap" verdict was WRONG -- probe invocation bug, not a real gap (2026-08-01)

**Discriminator: does llama.cpp's own CUDA backend compile its MXFP4 tensor-core (block-scaled
mma) path with this box's CUDA 13.0 toolkit for arch 121?** Built `~/src/llama.cpp` @`5f55650`
CUDA backend from scratch (`cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121
-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc`, `cmake --build build-cuda --target
ggml-cuda -j 20`, ~7 min wall). **Verdict: (a) COMPILES WITH REAL MMA SASS.** Both
`ggml/src/ggml-cuda/template-instances/mmq-instance-mxfp4.cu.o` and `mmq-instance-nvfp4.cu.o`
compiled clean (no errors/warnings on those files), and `cuobjdump -sass` on the objects shows
genuine block-scaled tensor-core instructions, not a scalar/fallback path:

```
mmq-instance-mxfp4.cu.o:  OMMA.SF.16864.F32.E2M1.E2M1.E8        R124, R92, R156, RZ, R42, R86, URZ ;
mmq-instance-nvfp4.cu.o:  OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X  R56, R56, R60, RZ, R20, R24, URZ ;
```

(`cuobjdump -elf` confirms `arch = sm_121a`.) `cuobjdump -ptx` on the mxfp4 object shows the
exact PTX idiom that lowered to that SASS -- identical instruction text to our standalone
probe's inline asm:

```
mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0
    {%f27, %f28, %f29, %f30}, {%r487, %r488, %r489, %r490}, {%r507, %r508},
    {%f27, %f28, %f29, %f30}, %r509, {0, 0}, %r510, {0, 0};
```

**Root cause of the earlier probe's blanket failure, found and reproduced**: it was an `nvcc`
*invocation* bug in the probe, not a toolkit/hardware gap. `nvcc -arch=sm_121a` (the bare
shorthand -- one of the 14 spellings the prior pass tried) is **not** equivalent to requesting
only the `sm_121a` target. `nvcc --dryrun` shows it silently expands to build **two** device
images: (1) a forward-compatible virtual-PTX image compiled for the *base, non-"a"* virtual
arch `compute_121` (`cicc -arch compute_121 ...`, then `ptxas -arch=compute_121 ...`) -- meant
to let the fatbinary JIT onto future non-family-specific hardware -- **and** (2) the real
`compute_121a`/`sm_121a` cubin (`cicc -arch compute_121a ...` then `ptxas -arch=sm_121a ... -o
....sm_121a.cubin`). Block-scaled MXFP4 MMA is a **family-specific** ("a"-suffixed-only) ISA
extension, absent from the base `compute_121`/`sm_121` target -- so step (1)'s `ptxas
-arch=compute_121` invocation is the one that hits the "not supported on target sm_121" wall
(reproduced verbatim: `error: Instruction 'mma with block scale' not supported on .target
'sm_121'` etc.), and `nvcc` treats that failure as fatal for the whole compile, aborting before
step (2) -- the cubin that would have succeeded -- is ever reported. This masked the real
result for every one of the 14 `-arch=X` spellings tried previously (all of them route through
the same bare-shorthand expansion). Reproduced live on robo-dog:

```
$ nvcc -arch=sm_121a -o /tmp/probe_121a mxf4_probe.cu
ptxas .../mxf4_probe.compute_121.ptx, line 31; error: Instruction 'mma with block scale' not supported on .target 'sm_121'
  [... same 4 errors as the original probe report, exit via fatal ptxas on the base-PTX image ...]

$ nvcc --generate-code=arch=compute_121a,code=[compute_121a,sm_121a] -std=c++17 -o /tmp/probe_gencode mxf4_probe.cu
[compiles clean, no ptxas errors]
```

**Fix for anyone re-running the probe or any standalone `.cu` targeting a family-specific SM**:
use `-gencode`/`--generate-code=arch=compute_121a,code=[compute_121a,sm_121a]` explicitly (no
bare `compute_121`/`sm_121` fallback image requested) -- exactly the invocation
`ggml/CMakeLists.txt`'s CUDA-arch machinery already produces for `-DCMAKE_CUDA_ARCHITECTURES=121`
(CMake logs `Replacing 121 in CMAKE_CUDA_ARCHITECTURES with 121a` and never asks for a bare
`compute_121`/`sm_121` companion image). Never use the bare `-arch=sm_121a` (or any bare
`-arch=` family-suffixed) shorthand for this instruction class.

**No `CUDART_VERSION` gate is in play here for CUDA 13.0** -- `BLACKWELL_MMA_AVAILABLE` in
`ggml/src/ggml-cuda/common.cuh:286-288` is purely an `__CUDA_ARCH__` device-code macro:
```
#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL && __CUDA_ARCH__ < GGML_CUDA_CC_RUBIN
#    define BLACKWELL_MMA_AVAILABLE
#endif
```
with `GGML_CUDA_CC_BLACKWELL=1200`, `GGML_CUDA_CC_DGX_SPARK=1210` (robo-dog's sm_121a, folded
into the same bucket), `GGML_CUDA_CC_RUBIN=1300` -- unconditional on toolkit version. The only
`CUDART_VERSION >= 12080` gates in the CUDA backend (`common.cuh:822`, `quantize.cu` x5,
`vendors/cuda.h:18`) guard unrelated host-side `nv_bfloat16`/`__nv_fp8_e4m3` E8M0/E4M3
scalar-conversion helpers, not the `mma.sync...block_scale` instruction itself; they're
satisfied trivially by CUDA 13.0 (>= 12080) regardless. The ticket's premise that llama.cpp
"gates FP4 paths at CUDART_VERSION >= 12080" refers to those scalar-conversion helpers, not to
whether the tensor-core MMA path itself compiles for a given SM -- that's controlled solely by
which `-arch`/`-gencode` spelling is used, per the invocation-bug finding above.

**Bonus/non-blocking finding, not required by the ticket**: running the correctly-recompiled
standalone probe (`--generate-code=arch=compute_121a,code=[compute_121a,sm_121a]`) executes
without error but returns numerically wrong output -- `D == 32.0` in all 128 output floats,
not the expected `64.0`. This is consistent with the probe's simplistic register-layout
assumption being wrong specifically for `scale_vec::2X`: `k64` is split into two `k32`
sub-blocks each needing its own E8M0 scale byte packed into the `a_scale`/`b_scale` registers,
and the probe only populates byte 0 (leaving the second sub-block's scale byte 0x00 ->
~2^-127, i.e. that half's contribution collapses to ~0, halving the sum from 64 to 32). This is
a probe-numerics bug to fix in a follow-up pass, not a toolchain/hardware blocker -- the
tensor-core path itself compiles and executes (produces nonzero, deterministic output); getting
the per-lane/per-subblock operand layout exactly right is the next step before any ds4 kernel
port, and llama.cpp's own `tile<>`/`load_ldmatrix` fragment-loading code (which this pass did
not need to reverse-engineer to answer the discriminator question) is the reference to copy the
layout from rather than re-deriving it.

**Verdict for the ds4 kernel-design question**: **(a)** -- this toolkit and this hardware
(CUDA 13.0.88, sm_121a) both fully support the block-scaled MXFP4/NVFP4 tensor-core MMA path;
nothing blocks porting it into ds4's grouped-prefill kernel at the toolchain level. The
remaining work is strictly the operand/register-layout correctness (per the bonus finding
above), to be derived from llama.cpp's `tile<>` fragment-loading code in `mma.cuh`, not a
toolchain workaround. P3b's grouped-dequant+cuBLAS path (~4.6-4.64 t/s prefill) remains the
current production path until that layout work lands.

No ds4-server activity was needed or performed this pass (compile-only investigation against a
scratch `~/src/llama.cpp` build tree, never committed). Bonus benchmark step (llama-cli FP4-MMA
prefill on `DeepSeek-V4-Flash-MXFP4_MOE.patched.gguf`) was skipped per the ticket's own
guidance: full `-ngl 99` offload doesn't fit (150GB model > 121GB unified memory) and a partial
offload wasn't attempted as it wasn't required and risked OOM.

## P3c-1 take 2: sm_121a MXFP4 tensor-core prefill kernel -- implemented, integrated behind an opt-in gate, DEFAULT OFF due to an unresolved numeric bug (2026-08-01)

Follow-on to the two P3c-1 entries above (toolchain-vintage-gap dead end, then the
invocation-bug correction establishing that `mma.sync.aligned.kind::mxf4.block_scale...
m16n8k64` genuinely compiles to real SASS on this box's CUDA 13.0/sm_121a when invoked via
the correct `--generate-code=arch=compute_121a,code=[compute_121a,sm_121a]` form). This pass
picked up exactly where that one stopped: fix `mxf4_probe.cu`'s D==32-not-64 scale bug, port
the kernel, integrate it, and measure. The first two landed; the kernel itself has a real
correctness bug not resolved this pass, so it ships present but **disabled by default**.

### Probe fix (gate to proceed)

`research/gb10/mxf4_probe.cu`'s bonus finding from the prior pass diagnosed its own bug
correctly (missing second E8M0 scale byte for `scale_vec::2X`'s two k32 sub-blocks) but
hadn't applied the fix. Fixed: `a_scale`/`b_scale` now pack `0x7F7F7F7F` (byte 127 in every
byte position, not just byte 0) instead of `127u`. Rebuilt with the correct
`--generate-code=arch=compute_121a,code=[compute_121a,sm_121a]` invocation (never bare
`-arch=sm_121a`, per the prior pass's own finding) and run:

```
nonzero=128/128 correct(==64.0)=128/128 bad=0
sample D[0..3] (lane0)=64.000 64.000 64.000 64.000
sample D lane16 =64.000 64.000 64.000 64.000
PROBE VERDICT: PASS -- mxf4 mma.sync produced correct result on sm_121
```

D==64.0 exactly, confirming the raw PTX instruction and its operand plumbing (uniform
E2M1/E8M0 operands, both scale bytes) execute correctly on this hardware/toolchain. Fixed
probe committed as instructed.

### Build: `make cuda-spark`'s arch flags

`make cuda-spark` (`CUDA_ARCH=` empty) previously left `NVCC_ARCH_FLAGS` completely unset --
every `nvcc` invocation for the whole build used nvcc's bare default target, with no `-arch`
of any kind, let alone the family-specific `sm_121a` block-scaled-MMA needs. Fixed in
`Makefile`: when `CUDA_ARCH` is unset, `NVCC_ARCH_FLAGS` now defaults to
`--generate-code=arch=compute_121a,code=[compute_121a,sm_121a]` (documented inline with the
same "never bare `-arch=sm_121a`" rationale as the probe fix) rather than nothing. Applies to
the whole build (not just the new kernel's translation unit) since that's how the Makefile's
`CUDA_ARCH`/`NVCC_ARCH_FLAGS` plumbing is structured -- every other `.cu`/link step already
shares the same `NVCCFLAGS`, and DGX-Spark/GB10 (what `cuda-spark` targets) is sm_121a-only
hardware, so there's no forward-compat reason to keep requesting a bare/virtual companion
image for anything else built here either. `make clean && make cuda-spark`: clean, zero
warnings, ~56s wall.

### Kernel design

`dsv4_mxfp4_mma_gemm_kernel` (`ds4_cuda.cu`, ahead of `dqg_group_gemm`): one warp per
(16 weight-rows x 8 tokens) output tile, looping over `in_dim` in 64-element chunks, one
`mma.sync...m16n8k64` call per chunk, accumulating in the same `tile<16,8,float>` registers
across the K-loop (matches llama.cpp's own accumulate-in-place idiom). Grid:
`(out_dim/16, ceil(group_size/8))`, block size 32 (one warp).

Ported primitives (MIT, llama.cpp `mma.cuh`/`common.cuh`/`quantize.cu` @5f55650): the
`tile<16,8,int>`/`tile<8,8,int>`/`tile<16,8,float>` `DATA_LAYOUT_I_MAJOR` `get_i`/`get_j`
physical-register formulas, the `ldmatrix.sync.aligned.m8n8.x4.b16` A-operand load address
formula, `load_generic`'s B-operand addressing, the `mma_block_scaled_fp4` PTX idiom itself,
and the scalar E8M0/E2M1 helpers (`ggml_cuda_e8m0_to_fp32`, `compute_e8m0_scale`,
`ggml_cuda_float_to_fp4_e2m1`) -- prefixed `dsv4_` to avoid colliding with ds4's own
unrelated (different-convention) `dsv4_e2m1fn_*` activation-sim code.

**Weight (A) operand**: confirmed via `llama.cpp`'s own `ggml_cuda_mmq_load_tiles_mxfp4_fp4`
(`mmq-load-tiles.cuh`) that the tensor core consumes ds4's native on-disk/in-memory MXFP4
block bytes (1 E8M0 byte + 16 packed-nibble bytes, `byte[j] = lo:a[j] hi:a[j+16]`)
**completely unmodified** -- that donor function does a bare 16-byte `memcpy` into its
operand SRAM, no repacking. So the kernel stages weight rows into `__shared__` via a
straight per-lane memcpy of the raw block bytes, no transform, and packs the two k32
sub-blocks' E8M0 scale bytes into one uint32 the same way (`byte0=block0.e,
byte1=block1.e`) as that donor function.

**Activation (B) operand** is quantized on the fly per (token, k64-chunk), independently
per 32-element sub-block (own E8M0 scale), using the ported E2M1 LUT/E8M0 encode. The
resulting nibble packing was independently re-derived from `llama.cpp`'s
`quantize_mmq_mxfp4` kernel's shuffle-based per-lane write pattern (not run, just read and
worked through by hand) and turns out to reduce to the same standard MXFP4 nibble order as
the weight side (`byte[m] = lo:nib[m] hi:nib[m+16]`) -- i.e. no exotic interleave beyond
ordinary MXFP4 packing, just applied per-token to freshly quantized nibbles. The per-lane
scale-register source mapping (`tidx_A = lane/4 + (lane%2)*8`, `tidx_B = lane/4`) is copied
verbatim from `ggml_cuda_mmq_vec_dot_fp4_fp4_mma` (`mmq-vec-dot.cuh`) -- the documented PTX
"warp-level block scaling" thread-id wiring for the `{0,0}` byte-id/thread-id literals used
in the mma asm.

### SASS evidence

Not separately re-captured for this kernel (the probe's own SASS/PTX correspondence,
`OMMA.SF.16864.F32.E2M1.E2M1.E8` on `sm_121a`, from the prior pass's `llama-cpp` reference
build, already establishes that this exact instruction lowers to real tensor-core SASS on
this hardware/toolchain -- the kernel uses the identical inline-asm string). `cuobjdump -sass`
on `ds4_cuda.o` was not run this pass; noted as a gap rather than asserted.

### Correctness: BROKEN, not shipped enabled

Two dedicated tests added and committed (`research/gb10/test_mxfp4_mma_gemm.c`,
`test_mxfp4_mma_diag.c`, both built against a **temporary** debug-only entry point
`ds4_debug_mxfp4_mma_gemm` added to `ds4_cuda.cu` for exactly this purpose -- bypasses the
routed-MoE grouping machinery so the kernel can be exercised in isolation):

- `test_mxfp4_mma_gemm.c`: random MXFP4 weight rows + random activations across 7 shapes
  (including the real gate/up 2048x4096 and down 4096x2048 shapes at small group sizes), CPU
  reference re-derives the *exact* quantized-activation math the tensor core does (E2M1
  round-to-nearest per 32-sub-block, same E8M0 encoding) rather than comparing against
  full-precision, so any mismatch is a genuine kernel bug, not activation-quantization
  noise. **Result: FAILS on every case**, but not randomly -- outputs are plausible
  GEMM-shaped values (right order of magnitude, not garbage/NaN/zero), just wrong.
- `test_mxfp4_mma_diag.c`: a one-hot diagnostic isolating the failure mode further --
  16x16 weight with row `r`'s only nonzero element at k-position `r` (value exactly 1.0),
  distinct-per-k activation values. Expected: `out[r] ~= quantized(x[r])`. **Actual: only 2
  of 16 output rows (a fixed, lane-correlated pair -- rows 2 and 3, both matching the
  expected quantized value exactly) come out correct; the other 14 are silently zero.**

This narrows the bug to either the A-operand (weight) `ldmatrix` addressing or the
`tile<16,8,float>` (C/output) write-back lane mapping -- both copied verbatim from
llama.cpp's own formulas, which their own compile (prior pass, real SASS) and this pass's
probe (D==64.0 exact) both indicate are individually sound in isolation. Suspected: a
subtlety in how those formulas compose when used **outside** the donor's full templated MMQ
scheduling context (their formulas assume a specific `nwarps`/SRAM-tile-width convention
this single-warp, single-k64-chunk-at-a-time specialization may violate in a way not yet
identified) -- documented as a hypothesis, not confirmed. Tried (and ruled out): `__align__
(16)` on every `__shared__` buffer (no change, so not a raw alignment fault).

**Per the ticket's own "if it degrades, leave the toggle default off" instruction** (read
here as applying with even more force to "not proven correct" than to "correct but
lower-quality"): the dispatch gate requires an explicit opt-in
(`DS4_CUDA_ENABLE_MMA_PREFILL_EXPERIMENTAL=1`, renamed from the ticket's suggested
`DS4_CUDA_DISABLE_MMA_PREFILL` opt-out framing specifically because opt-out-by-default would
mean a stock build silently computes wrong routed-expert outputs) rather than being on by
default when the arch check passes. With the gate at its default (unset), every existing
test (`test_mxfp4_moe`, `test_mixed_moe`, `test_mxfp4_dequant`, `./ds4_test`) passes exactly
as before this pass -- the grouped dequant+cuBLAS path (P3b, ~4.6-4.64 t/s prefill) remains
the one actually running in production, completely unchanged.

### End-to-end coherence / A-B measurement: not performed

Given the kernel is demonstrably numerically wrong at the unit level (not merely
lower-precision), running an end-to-end France-prompt/prose-prompt A/B or a prefill t/s
comparison against it would produce numbers that could be mistaken for a real quality/speed
tradeoff rather than what they'd actually be measuring (a broken kernel's output). Skipped
on that basis rather than run-and-report; re-attempt once the diagnostic above is resolved.

### Tests / build

`make clean && make cuda-spark`: clean, zero warnings. `./research/gb10/test_mxfp4_moe`,
`test_mixed_moe`, `test_mxfp4_dequant`: all pass (gate default = off). `./ds4_test`: 1
failure (`logprob-vectors: ERR`) -- matches this suite's own pre-existing documented
run-to-run flakiness (2026-07-31 entry above lists `logprob-vectors` among the tests that
flip OK/ERR between runs of an *identical* unpatched binary); not attributable to this pass.
`ds4-server` stopped before `./ds4_test`/isolation-test runs, restarted after; confirmed
`active`, loaded its usual `DeepSeek-V4-Flash-IQ2XXS-...imatrix.gguf` normally, and answered
a live France-capital smoke prompt coherently post-restart.

### Follow-up (for whoever picks this up next)

1. Resolve the one-hot diagnostic's A-operand-vs-C-write-back ambiguity: dump `A.x[]` via a
   single-lane `printf` for a controlled input and compare against hand-computed expected
   register contents per llama.cpp's own `ldmatrix` formula, independent of whether the MMA
   or write-back stage is at fault.
2. Once `test_mxfp4_mma_diag.c` and `test_mxfp4_mma_gemm.c` both pass, flip
   `DS4_CUDA_ENABLE_MMA_PREFILL_EXPERIMENTAL` default to on (or drop the gate and restore the
   simpler opt-out framing the ticket originally specified) and re-run the A/B prefill t/s +
   end-to-end coherence measurements this pass explicitly skipped.
3. `ds4_debug_mxfp4_mma_gemm` (the temporary isolation entry point) can stay -- it's small,
   guarded, and is what makes `test_mxfp4_mma_gemm.c`/`test_mxfp4_mma_diag.c` possible without
   a full MoE harness -- but should get a proper comment marking it as a permanent test-only
   API if it's going to stay past the next pass, rather than "temporary" as currently written.

## P3c-1 take 3: root-caused and fixed the B-operand transpose bug -- one-hot diagnostic now 16/16, randomized isolation test still fails on larger/denser cases, gate stays DEFAULT OFF (2026-08-01)

Follow-on to take 2's own explicit next step ("dump `A.x[]` via a single-lane `printf` for a
controlled input and compare against hand-computed expected register contents"). Read the
full P3c-1 take-2 writeup above, the 75c8185 diff, and re-derived the donor's `tile<>`/
`load_ldmatrix`/`load_generic` register-layout formulas from `~/src/llama.cpp/ggml/src/
ggml-cuda/mma.cuh` and `mmq.cuh` (`vec_dot_fp4_fp4_mma`, `load_tiles_mxfp4_fp4`) by hand
before touching any code.

### Root cause: B-operand (activation) pack/read transpose, not A or C

Hand-deriving the A-operand `ldmatrix` address formula and the `tile<16,8,int>`/
`tile<16,8,float>` `get_i`/`get_j` register-layout formulas against the donor's own
(non-AMD, non-Volta, i.e. Turing/Blackwell) branch showed the ported A-operand addressing
and the C-tile write-back (`out_row0 + ((l/2)*8) + (lane/4)`, `tok0 + ((lane%4)*2) + (l%2)`)
both match the donor's formulas exactly -- the take-2 writeup's own two suspects were
**not** where the bug was. Re-deriving the B-operand (activation) side instead found a
genuine transposition: `dsv4_mxfp4_mma_gemm_kernel`'s "pack B nibbles into the 8x8
physical-int layout" step wrote `b_qs[i * 8 + j]` (`i` = int-column 0..7, `j` = token 0..7,
i.e. **intcol-major**), while the B-tile load immediately below it read `b_qs[bi * 8 + bj]`
with `bi = lane/4` (token) and `bj` = int-column -- i.e. **token-major**. These two index
conventions are transposed; they only coincide on the `token == intcol` diagonal, which
exactly explains take 2's one-hot diagnostic finding ("only 2 of 16 output rows -- a fixed,
lane-correlated pair -- come out correct").

Confirmed via single-lane `printf` register dumps (temporary, removed before the final
commit) of `A.x[]`/`B.x[]`/`a_scale`/`b_scale` for the exact one-hot input from `test_
mxfp4_mma_diag.c`: `A.x[]` and the scale registers matched hand-derived expectations from
the donor formulas exactly; `B.x[]` did not (e.g. lane 4's `B.x[0]` held nonzero real
activation data pulled from the wrong (intcol, token) cell rather than the expected
all-zero padding-token value).

### Fix

One-line fix in `dsv4_mxfp4_mma_gemm_kernel`'s B-packing loop: `b_qs[i * 8 + j]` ->
`b_qs[j * 8 + i]`, making the store token-major to match the token-major read (`bi =
lane/4`, matching the donor's `tile<8,8,int>::get_i(l) = threadIdx.x/4` convention).
Comment added in place documenting the convention and the bug it fixes.

### One-hot diagnostic: now 16/16 correct

`research/gb10/test_mxfp4_mma_diag.c`, unchanged: all 16 output rows now match the
expected quantized value (`out[r] == quantized(x[r])`, verified by hand against the E2M1
LUT/E8M0 scale for the test's specific data), up from 2/16 before the fix. Also spot-checked
with three additional hand-built one-hot variants (not committed -- ad hoc `/tmp` programs)
to close gaps the original diagnostic didn't cover: weight one-hot in the *second* k32
sub-block of a single k64 tile (block1, k=32..63) -- correct; weight one-hot in a *second*
k64 tile (`in_dim=128`, kt=1, k=64..95) with correct hi/lo-nibble placement -- correct
(a first attempt at this variant showed a large discrepancy that turned out to be an error
in the throwaway test's own nibble packing, not a kernel bug -- caught and corrected before
drawing any conclusion from it); multi-token (`group_size=5`) with per-token near-identical
activations -- correct (only the expected tie-break-sensitive row differed, consistent with
genuine E2M1 rounding, not a swap).

### Randomized isolation test: still fails on larger/denser cases -- root cause not fully isolated

`research/gb10/test_mxfp4_mma_gemm.c` (full random weights + activations, CPU reference
re-derives the tensor core's exact quantized math): the two smallest cases (`in=64,out=16,
group=1` and `group=3`) and one larger case (`in=512,out=256,group=4`) now **pass exactly**
(0 mismatches) -- a real improvement from take 2's "fails on every case". The remaining four
cases (`in=128,out=32,group=5`; `in=512,out=256,group=5`; `in=4096,out=2048,group=3`;
`in=2048,out=4096,group=7`) still fail, with mismatches on roughly 12-70% of (token, row)
positions, magnitudes ranging from sub-1 to several hundred (both plausible as single-LUT-
step tie-break noise near the small end, and clearly structural at the large end).

Two hypotheses were formed and empirically ruled out this pass, both with zero effect on
the failure set (byte-identical mismatch lists before/after):

1. **`--use_fast_math` imprecision in the activation-quantization `log2f`.** Rebuilt
   `ds4_cuda.o` without `--use_fast_math` entirely -- identical failures, byte for byte.
   Ruled out.
2. **Host vs. device `log2f` non-bit-identical rounding at the E8M0 scale-bucket boundary**
   (`dsv4_compute_e8m0_scale`'s `log2f` + round-to-nearest-int, mirrored by the CPU
   reference's own `log2f` + `lrintf` -- these two library implementations are not
   guaranteed bit-identical). Replaced both the kernel's and the CPU reference's scale
   computation with a bit-exact `frexpf`-based equivalent (exact exponent/mantissa
   extraction plus a single fixed `1/sqrt(2)` threshold comparison, mathematically
   equivalent to `round(log2(amax))` but reproducible cross-platform) -- identical failures,
   byte for byte. Ruled out and reverted (kept the diff minimal since it demonstrably
   didn't help).

Also algebraically verified (not empirically, since it's opaque hardware behaviour) that the
CPU reference's weight-side dequant (`mxfp4_e8m0_to_fp32_half` + doubled `kvalues_mxfp4`)
is exactly mathematically equivalent to the kernel's raw-byte-to-hardware interpretation
(standard E2M1 magnitude LUT `{0,0.5,1,1.5,2,3,4,6}` + raw ue8m0 `2^(x-127)` scale) --
the two "doublings" (nibble table and scale) cancel exactly, so this is not a format
mismatch either.

Given the one-hot diagnostic (sparse, mostly-zero data) is now fully correct but dense
random data still shows scattered failures growing with problem size, the remaining bug
(if it is one, rather than an as-yet-unidentified reference-side issue) most likely lives in
either a synchronization/staging subtlety that only manifests when *most* lanes carry live
nonzero data simultaneously (shared-memory bank/visibility edge case not exercised by sparse
one-hot inputs), or a residual addressing issue specific to combinations of multiple
`out_dim` row-blocks with `group_size` values that don't evenly fill the 8-token warp tile
(`in=512,out=256` passes at `group=4` and fails at `group=5` with everything else held
structurally equal, which is the closest lead to what actually differs, but did not resolve
within this pass's time budget). Left open rather than guessed at further.

### Gate: stays DEFAULT OFF

Per the ticket's own explicit instruction ("if it degrades / not proven correct, leave the
toggle default off"): `test_mxfp4_mma_gemm` does not pass at its stated tolerance, so the
correctness gate for flipping `DS4_CUDA_ENABLE_MMA_PREFILL_EXPERIMENTAL` to default-on is
**not met**. The gate is unchanged (`DS4_CUDA_ENABLE_MMA_PREFILL_EXPERIMENTAL=1` opt-in
required, default off). End-to-end France/prose coherence A/B and prefill t/s measurement
against the MMA path were **not performed** this pass, for the same reason take 2 skipped
them: measuring a still-known-broken (albeit now less broken) kernel's speed/quality would
produce numbers that could be mistaken for a real result.

### Tests / build

`make clean && make cuda-spark`: clean, zero warnings, ~56s wall (Makefile's `NVCC_ARCH_FLAGS`
default unchanged from take 2). `./research/gb10/test_mxfp4_mma_diag`: all 16 rows correct
(was 2/16). `./research/gb10/test_mxfp4_mma_gemm`: 3/7 cases pass exactly, 4/7 still fail
(was 0/7 passing in take 2) -- see above. `test_mxfp4_moe`, `test_mixed_moe`, `test_mxfp4_
dequant`: all pass (gate default = off, unaffected by this pass's change). `ds4-server`
stopped before `./ds4_test`; `./ds4_test` (full run, no truncation): `ds4 tests: 7
failure(s)` across three sections -- `think-tool-recovery`, `logprob-vectors`, `metal-
tensor-equivalence` -- all three on the pre-existing flaky list documented by every prior
pass on this hardware (see the 2026-07-31/08-01 entries above); no new failing test names.
`ds4-server` restarted after; confirmed `active`, loaded its usual `DeepSeek-V4-Flash-
IQ2XXS-...imatrix.gguf` normally, and answered a live France-capital smoke prompt
coherently post-restart.

### Follow-up (for whoever picks this up next)

1. Isolate the remaining randomized-test failure: the `in=512,out=256,group=4` (passes) vs.
   `group=5` (fails) pair is the tightest reproducer found this pass -- everything else
   held equal, only `group_size` differs. Bisect by shrinking further (e.g. `in=512,out=32,
   group=5` or smaller) to get a hand-traceable failing case, then apply the same
   single-lane `printf` register-dump technique used successfully in this pass for the
   B-operand bug.
2. Once `test_mxfp4_mma_gemm` passes all 7 cases at its stated tolerance, re-run this
   pass's still-skipped gate-flip criteria: existing tests pass (already true), end-to-end
   France + ~300-token prose coherence A/B, numeric delta reported honestly, and only then
   flip `DS4_CUDA_ENABLE_MMA_PREFILL_EXPERIMENTAL`'s default and measure prefill t/s A/B.
3. `ds4_debug_mxfp4_mma_gemm` (the temporary isolation entry point) still carries a
   "TEMPORARY ... to be removed before final commit" comment two passes running now --
   worth either committing to keeping it (with a permanent-API comment, as take 2 already
   flagged) or actually removing it once the isolation tests are no longer needed for
   active debugging.

## P3c-1 take 4: test-harness bug root-caused, isolation test now 7/7 -- but a *second*, distinct MoE-integration bug found when attempting the earned default-ON flip; gate stays DEFAULT OFF (2026-08-01)

Follow-on to take 3's own explicit next step ("bisect [`in=512,out=256,group=4` (passes) vs.
`group=5` (fails)] by shrinking further ... apply the same single-lane `printf` register-dump
technique"). Started that bisection but found the actual answer before needing device-side
printf: re-examined `ds4_debug_mxfp4_mma_gemm()`'s own host-side code first (per the "reference-
side issue" possibility take 3's writeup already flagged but didn't rule out) and found it.

### Root cause of take 3's remaining 4/7 failures: the isolation test's CPU reference, not the kernel

`ds4_debug_mxfp4_mma_gemm()` (`ds4_cuda.cu`) converts its `x_host` (float) argument to `__half`
before the kernel ever sees it (`xh_host[i] = __float2half(x_host[i])`) -- deliberately, since
this matches `dqg_group_gemm_mxfp4_mma`'s real `xh_group` parameter, which is always `__half` in
production (`dqg_group_gemm`'s own signature). But `test_mxfp4_mma_gemm.c`'s `ref_row()` was
quantizing the *un-rounded* `float` activations passed into `run_case()`, not the fp16-rounded
values the kernel actually consumes. On small/sparse cases (`in=64,out=16`) this fp16 rounding
essentially never crossed an E2M1 LUT bucket boundary, so the mismatch stayed invisible; on
larger/denser random cases it increasingly did, for exactly the scattered-mismatch signature take
3 observed (12-70% of positions, magnitudes proportional to one weight*activation LUT step) and
its scaling with problem size (more elements => more chances of a boundary crossing).

Confirmed by writing a scratch copy of the test that round-trips `x[]` through IEEE-754 binary16
(round-to-nearest-even, implemented in plain C to avoid a `cuda_fp16.h` host-linkage dependency
in a `.c` translation unit) before calling `ref_row()`: all 7 cases, including the two largest
(`in=4096,out=2048,group=3` and `in=2048,out=4096,group=7`, ~6K and ~29K output elements), match
the kernel's output *exactly*, 0 mismatches. Take 3's own two ruled-out theories (`--use_fast_math`,
host/device `log2f` rounding) were both real dead ends -- this was a third, independent cause the
"reference-side issue" aside in take 3's writeup had already speculated about but not chased down.

### Fix: round activations through fp16 in the test's own reference

`research/gb10/test_mxfp4_mma_gemm.c`: added a plain-C `float_to_half_to_float()` (manual
IEEE-754 binary16 round-trip, RNE, no `cuda_fp16.h` dependency) and call it on `x[]` right after
generation, before `ref_row()` -- so the reference quantizes the same activation values the
kernel actually receives. No kernel-side code changed. `test_mxfp4_mma_gemm`: **7/7 cases pass
exactly, 0 mismatches**, up from take 3's 3/7. `test_mxfp4_mma_diag` (kernel-side, unaffected by
this test-only fix): still 16/16.

### Attempted the earned default-ON flip -- found a second, distinct MoE-integration bug, reverted

With the unit-level correctness gate now met, attempted this pass's next earned step: flipped
`dqg_group_gemm`'s dispatch condition from opt-in (`DS4_CUDA_ENABLE_MMA_PREFILL_EXPERIMENTAL=1`
required) to opt-out (`DS4_CUDA_DISABLE_MMA_PREFILL=1` forces the cuBLAS fallback; default ON),
per the ticket's own instruction for the naming/polarity of the earned-gate flip. Existing tests
(`test_mxfp4_moe`, `test_mixed_moe`, `test_mxfp4_dequant`) were re-verified passing *before* the
flip (unaffected, gate was off). After the flip and a clean rebuild (`make clean && make
cuda-spark`, zero warnings), re-ran the same three tests: `test_mixed_moe` and `test_mxfp4_dequant`
still passed, but **`test_mxfp4_moe`'s second case (`n_tokens=5, n_expert=3, in=512, mid=256,
out=256`) now failed** -- the last token's last several output columns (`out[1275..1279]` of a
5*256=1280-element output) came back exactly `0.0` instead of the correct (large-magnitude)
reference values, while every other element matched.

This is a **different bug from the one take 3 fixed**, not a recurrence of it: the standalone
`test_mxfp4_mma_gemm.c` harness -- which bypasses the real routed-MoE grouping/scatter machinery
entirely via `ds4_debug_mxfp4_mma_gemm()`, calling `dqg_group_gemm_mxfp4_mma()` directly with a
`group_size` that's just a plain token count over a contiguous `x` buffer -- kept passing 7/7 on
the *exact same* rebuilt `ds4_cuda.o`. Toggling `DS4_CUDA_DISABLE_MMA_PREFILL=1` on/off with
nothing else changed cleanly isolates the cause to the MMA dispatch specifically: `test_mxfp4_moe`
passes with the off-switch set (forcing cuBLAS) and fails without it. So the kernel itself (proven
correct at the unit level, 7/7 + 16/16) is not implicated directly -- the bug lives somewhere in
how `dqg_group_gemm_mxfp4_mma`'s output composes with the real MoE per-expert grouping (built from
scattered `(token, expert)`-pair rows via `pair_idx`, not test_mxfp4_mma_gemm.c's contiguous
`0..group_size-1` token range), the multi-expert accumulate, or the down-projection scatter
(`dqg_scatter_down_rows_kernel`) -- not yet isolated further this pass. Given the ticket's own
explicit criterion ("existing tests pass" as a hard requirement for the default flip, "visible
[...] degradation -> default stays OFF, report"), **reverted the flip immediately**: gate is back
to `DS4_CUDA_ENABLE_MMA_PREFILL_EXPERIMENTAL=1` opt-in, default off. Re-verified after revert +
another clean rebuild: `test_mxfp4_moe` passes by default (env var unset); re-running it with the
opt-in flag set reproduces the exact same failure (`out[1275..1279]` zero) on demand, confirming
the revert is clean and the bug is real, reproducible, and gated off.

### End-to-end checks performed during the (reverted) default-ON window

Before discovering the `test_mxfp4_moe` regression, ran the ticket's other earned-gate checks
against the real 150GB artifact (`gguf/DeepSeek-V4-Flash-MXFP4_MOE.patched.gguf`,
`--ssd-streaming --ssd-streaming-cache-experts 100GB`, `ds4-server` stopped first per protocol),
with `DS4_CUDA_ENABLE_MMA_PREFILL_EXPERIMENTAL=1` set and `DS4_DEBUG_MMA_PREFILL=1` confirming the
MMA kernel actually dispatched (9352 `mma-prefill` launches logged for the ~230-token prose
prompt below, prefill token count well above the `group_size>1` threshold):

- `-p "What is the capital of France?" --nothink`: `The capital of France is **Paris**.` (correct,
  coherent). `prefill: 1.27 t/s, generation: 2.08 t/s`.
- ~230-token prose prompt ("Write a detailed explanation of how photosynthesis works...",
  covering light-dependent/independent reactions, C3/C4/CAM pathways, biotech applications),
  `-n 300`: coherent, well-structured, technically accurate response under the MMA path
  (`prefill: 3.65 t/s, generation: 1.32 t/s`) and, run again with the flag unset (cuBLAS path,
  `prefill: 3.67 t/s, generation: 2.40 t/s`), also coherent and technically accurate -- both
  outputs on-topic and correct, no visible quality degradation between paths (exact token-for-
  token match not expected or checked, since default sampling is non-greedy, temp=1.0).
- **Prefill t/s, 3 reps each, same prompt/config, MMA vs cuBLAS** (`-n 20`, isolating prefill
  from most of the decode-time noise): MMA `3.68, 3.64, 3.67` t/s (mean 3.663); cuBLAS `3.65,
  3.60, 3.64` t/s (mean 3.63). **Honest numeric delta: ~1% apart, within run-to-run noise --no
  meaningful prefill speedup from the tensor-core path in this configuration.** Plausible
  explanation (not verified further this pass): this SSD-streaming workload's prefill time is
  dominated by NVMe/mapped-view expert fetch, not GEMM compute, so a faster GEMM kernel doesn't
  move the end-to-end number much here.

These numbers are recorded for the follow-up's benefit but should **not** be read as "the MMA
path is validated to be quality-neutral and speed-neutral in production" -- they were collected
before the `test_mxfp4_moe` regression was found, under a since-reverted default-ON state, and
the France/prose prompts used happened not to trigger whatever group-construction shape triggers
the `test_mxfp4_moe` bug. They remain useful context for whoever fixes the MoE-integration bug
next, but the gate is not earned until `test_mxfp4_moe` (not just `test_mxfp4_mma_gemm`) passes
with the flag on.

### Gate: stays DEFAULT OFF, but for a new reason

Per the ticket's own instruction ("if it degrades / not proven correct, leave the toggle default
off"): unlike takes 2 and 3 (kernel-level correctness not yet proven), take 4 proves kernel-level
correctness (7/7 + 16/16) but finds the *MoE-dispatch integration* is not proven correct
(`test_mxfp4_moe` fails with the path enabled). The gate is unchanged in net effect
(`DS4_CUDA_ENABLE_MMA_PREFILL_EXPERIMENTAL=1` opt-in required, default off), though the specific
bug blocking it has moved from the GEMM kernel itself to its integration with the real routed-MoE
grouping/scatter pipeline.

### Tests / build

`make clean && make cuda-spark`: clean, zero warnings, ~57s wall, both before and after the
flip-and-revert (three total clean rebuilds this pass). `test_mxfp4_mma_gemm`: **7/7 cases pass
exactly** (was 3/7). `test_mxfp4_mma_diag`: 16/16 (unchanged). `test_mxfp4_moe`: passes by default
(flag unset); **fails reproducibly with the opt-in flag set** (`out[1275..1279]` zero, case
`n_tokens=5,n_expert=3,in=512,mid=256,out=256`) -- this is the new finding, not a regression in
the shipped default state. `test_mixed_moe`, `test_mxfp4_dequant`: pass in both states. `ds4-server`
stopped before `./ds4_test` (built fresh against the final, reverted `ds4_cuda.o`); full run: `ds4
tests: 6 failure(s)` across `logprob-vectors` and `metal-tensor-equivalence` -- both on the
pre-existing flaky list documented by every prior pass on this hardware (take 3's own run saw 7
failures across a 3-section subset of the same list); no new failing test names or sections.
`ds4-server` restarted after; confirmed `active`, loaded its usual `DeepSeek-V4-Flash-IQ2XXS-
...imatrix.gguf` normally, and answered a live France-capital chat-completion prompt correctly
and coherently post-restart.

### Follow-up (for whoever picks this up next)

1. Build a **new** isolation test that goes through the real `dqg_group_gemm()` call path (or a
   deliberately `pair_idx`-scattered synthetic group matching the real MoE dispatch's group
   construction) instead of `ds4_debug_mxfp4_mma_gemm()`'s direct/contiguous harness, specifically
   targeting the `test_mxfp4_moe` case-2 shape (`n_tokens=5,n_expert=3,in=512,mid=256,out=256`,
   `n_total_expert=16`) to reproduce the `out[1275..1279]` zero-output bug independently of the
   full `ds4_gpu_routed_moe_batch_tensor()` stack, then bisect it the same way take 3 bisected the
   B-operand transpose (one-hot-style narrowing, then single-lane printf register dumps once a
   small hand-traceable case is found).
2. Once that's fixed and `test_mxfp4_moe` (plus `test_mixed_moe`, `test_mxfp4_dequant`,
   `test_mxfp4_mma_gemm`, `test_mxfp4_mma_diag`) all pass with
   `DS4_CUDA_ENABLE_MMA_PREFILL_EXPERIMENTAL=1` set, redo the default-ON flip (same
   `DS4_CUDA_DISABLE_MMA_PREFILL=1` off-switch polarity already wired up and documented at the
   `dqg_group_gemm_mxfp4_mma` call site in `ds4_cuda.cu`, ready to go) and re-collect the
   end-to-end France/prose A/B and prefill t/s numbers under a now-actually-earned gate (this
   pass's numbers above are a reasonable starting expectation -- near-parity with cuBLAS on this
   SSD-streaming-bound workload -- but should be re-measured, not assumed, once the fix lands).
3. `ds4_debug_mxfp4_mma_gemm` (the temporary isolation entry point) still carries a "TEMPORARY
   ... to be removed before final commit" comment three passes running now -- same open question
   take 2/3 already flagged (keep as a permanent debug API, or remove once no longer needed).

## DSpark drafter x `--ssd-streaming` compatibility: gate lifted, structural fix in the CUDA routed-MoE batch dispatch (2026-08-02)

The `--ssd-streaming` / `--mtp` mutual-exclusion gate at `ds4.c:56770-56773` ("not compatible
yet") is now scoped to `--mtp` **without** `--dspark` only. Full root-cause, fix, and
correctness/throughput measurement writeup lives in `MEASUREMENTS.md`'s "Drafter-streaming
compatibility" unit; summary for anyone touching this code next:

- The target model's own DSpark verify path was **already** streaming-compatible before this
  unit (it reuses the same selected-cache/LRU protocol as real streamed prefill/decode) --
  contrary to what the gate's placeholder comment implied.
- The actual bug: `ds4_gpu_routed_moe_batch_tensor()` (`ds4_cuda.cu`) silently discarded its
  own `force_resident` parameter (`(void)force_resident;`, always `allow_streaming=1`),
  unlike its `n_tokens==1` sibling `ds4_gpu_routed_moe_one_tensor()`, which already respected
  the equivalent flag. Combined with `metal_graph_encode_layer_ffn_batch()`'s call site in
  `ds4.c` hardcoding `force_resident=false`, the DSpark support/drafter model's own batch
  stage-block forward (`metal_graph_eval_dspark_stage_block`, which already temporarily sets
  `g->ssd_streaming=false` around exactly this call, anticipating the need) never actually
  got routed away from the CUDA streaming selected-cache -- a cache/fetch path that is keyed
  to, and whose disk-miss fallback reads via a process-global fd bound to, the *streamed
  target* model only. Draft proposal failed on every cycle as a result (silent, not a crash:
  `s->dspark_draft_valid` just never became true), so the previous mutual-exclusion gate was
  hiding a second, deeper wiring gap than its own comment suggested.
- Fix: `ds4_gpu_routed_moe_batch_tensor()` now respects `force_resident`;
  `metal_graph_encode_layer_ffn_batch()`'s call site now passes `!g->ssd_streaming` instead
  of a hardcoded `false` (a no-op for the streamed target model's own real calls, where
  `g->ssd_streaming` is genuinely true).
- Correctness: byte-identical to the no-drafter baseline on the France prompt at both
  confidence 0.0 and the shipping-default 0.9; a 500-token generation completed coherently
  with no crash/garbage. **One real caveat found and documented, not specific to DSpark**:
  `--ssd-streaming` decode at `--temp 0` is not perfectly run-to-run reproducible on this
  build even with zero drafter code involved (two identical no-`--mtp` reruns diverged at
  the same spot on a longer prose prompt) -- most likely an `-ffast-math`/`--use_fast_math`
  FP-reduction-order sensitivity, flagged as a separate follow-up, not fixed here.
- Throughput with the shipping default (`--dspark-confidence 0.9`) against the community
  drafter used for this unit does **not** presently beat the no-drafter baseline (~0.57 t/s
  vs. 2.69-4.46 t/s) because the confidence gate rejects nearly every draft while still
  paying the drafter's own per-step propose+confidence-probe compute; forcing acceptance
  (`--dspark-confidence 0.0`) does produce real accepted speculative windows but was measured
  only in short/cold-cache single-prompt tests, which is exactly the regime this ticket
  itself predicted would *not* show the win (needs the ~96%+ hit-rate warm-session regime
  from the long-session unit in `MEASUREMENTS.md` to amortize the extra per-step expert
  fetch). **A full warm multi-turn with/without-drafter trajectory was not completed within
  this unit's time budget** -- the natural next step for whoever picks this up.

**Follow-up (2026-08-02): GA-matched drafter detection fixed, full warm A/B completed --
verdict is a measured non-win, not another blocker.** The `a335048` blocker (GA-matched
drafter, different tensor-naming dialect, `support_model_detect()` didn't recognize it) is
resolved: `dspark.*`-dialect per-stage and global/head tensor binding added to `ds4.c`
(dialect-compat-alias style, additive, legacy `mtp.*` path untouched), after confirming the
two-matrix `markov_w1`/`markov_w2` head is exactly what ds4's own DSpark compute already
consumes (not new math -- verified against the artifact's own conversion recipe and raw GGUF
header, both matching ds4's existing `[markov_rank, DS4_N_VOCAB]` two-matrix consumer). The
matched drafter now loads, detects, and decodes correctly (byte-identical to no-drafter on
verbatim/identity checks). The completed warm 8-turn A/B, finally answering this section's
own "natural next step" above: **the matched drafter does not beat the no-drafter baseline at
either confidence setting** -- default confidence (0.9) is 26.7% slower (3.85 vs. 5.25 t/s
steady state), force-accept (0.0) is 69.1% slower (1.62 t/s), because the verify path in this
build never accepts more than 2 tokens per event regardless of draft length or confidence
setting, so longer/more-frequent drafting only adds cost without adding accepted tokens.
Full writeup, evidence, and the acceptance-rate/draft-length stats behind this verdict: see
`MEASUREMENTS.md`'s "Matched-drafter detection fix + completed A/B" entry. Production
recommendation unchanged from the entry above -- do not enable `--mtp`/`--dspark` on the live
`ExecStart`.

## DSpark drafter 2-token acceptance ceiling: root-caused, STRUCTURAL (2026-08-02)

Root-caused why every DSpark verify event accepts exactly 2 tokens
(bonus + 1) regardless of drafted length (2-5) or confidence setting, per
this section's own "one lever that could plausibly change this verdict"
follow-up. Full trace and evidence in `MEASUREMENTS.md`'s "Spec-decode
2-token acceptance ceiling" entry; summary:

- The verify/accept machinery (`ds4.c:62050-62420` `commit_drafts`
  longest-common-prefix loop, `metal_graph_verify_suffix_tops_impl`
  `ds4.c:35063`) is correct, standard speculative-decode logic -- not the
  cause. No hardcoded cap, no window-size-2 bug, no truncation before
  verify.
- Real cause: the drafter's one-shot batched proposal
  (`metal_graph_eval_dspark_stage_chain`, `ds4.c:32351`) seeds every
  draft-slot position beyond the first with the checkpoint's own
  `noise_token_id` metadata placeholder
  (`metal_graph_prepare_dspark_setup_block`, `ds4.c:31401-31432`), not the
  real (or even self-consistent) continuation, then applies only a weak
  order-1 "Markov" correction (`ds4.c:33103-33356`, previous-token
  embedding bias, no re-encode). Position 0 gets real context and verifies
  well (that's the pre-verify `target_top == drafts[0]` gate passing);
  position 1+ is conditioned on a noise mask and, in this GA-matched
  drafter/target pairing, essentially never survives batch-verify against
  the target's real causal continuation -- confirmed uniform across all
  3,338 verify events measured in both A/B arms (`verified=1` always).
  This is the model checkpoint's own intended masked/parallel-block
  decoding design (real GGUF metadata field, not a ds4 port artifact), and
  there is no iterative refinement/denoising loop in the codebase to let
  later positions condition on earlier drafted tokens' real content.
- **Verdict: STRUCTURAL, not wiring** -- nothing to point-fix. A real fix
  would need a new multi-round refinement mechanism (re-embed +
  re-run the stage-chain per round with real tokens replacing the noise
  placeholder), estimated **medium** size, its own ticket, and no
  guaranteed win even then (each refinement round adds real forward-pass
  cost on hardware where per-token expert-fetch cost already dominates,
  per this section's own prior finding).
- No model runs performed this unit (code-trace only); `ds4-server` was
  not touched.


## Quality battery

Calibration/hallucination probe battery for pre-promotion GA-FP4 checks: see `research/gb10/calibration_probes/` (40-item probes.jsonl across unanswerable/known-fact/trap-premise/tool-precision categories, runner + heuristic scorer, protocol comparing IQ2 baseline vs. GA FP4 vs. GA FP4 with an abstention system prompt).

## GA artifact (`0731`, 156.4 GB) header census + INSPECT: BLOCKED on a new metadata gap (2026-08-02)

**Scope.** GA-0731 swap unit. Artifact:
`gguf/DeepSeek-V4-Flash-0731-MXFP4_MOE-Q8_0.gguf` (156,378,344,992 bytes = 145.6 GiB /
156.4 GB, main file; matched drafter `gguf/DeepSeek-V4-Flash-0731-DSpark-Drafter-MXFP4-Q8_0.gguf`,
10.9 GB, noted present but out of this unit's scope). `ds4-server` (systemd, production
IQ2XXS) stopped first; restarted and verified (`systemctl is-active`=`active`,
`GET /v1/models`->HTTP 200) at the end, per protocol, even though this unit stopped at a
load failure.

**Header census (raw file header, no tensor data loaded), vs. the preview community
conversion's own header (this doc, `--inspect` summary at line ~479, 2026-08-01 entries
above):**

| | preview (`.patched.gguf`, community conversion) | GA (`0731-MXFP4_MOE-Q8_0.gguf`) |
|---|---|---|
| GGUF version | v3 | v3 |
| tensor count | 1328 | 1328 (identical) |
| raw metadata keys in file | 51 | 59 |
| block_count (layers) | 43 | 43 (identical) |
| expert_count / used | 256 / 6 | 256 / 6 (identical) |
| embedding_length / ffn_length | 4096 / 2048 | 4096 / 2048 (identical) |
| logical params (sum of tensor element counts) | 284.33 B (per `--inspect`) | **284.33 B**, independently computed from the raw header -- byte-for-byte the same architecture, not the ~304 B this unit's brief anticipated |
| file size | 153.52 GiB | 156.4 GB (145.6 GiB) |
| dense/attention tensor dtype | mixed BF16 (13) + Q6_K (9 families) | **Q8_0** (365 tensors) -- the Q6_K family is entirely absent from GA |
| routed-expert dtype | MXFP4 (98 tensors, generic dequant+GEMM family, post BF16/Q6_K/F32 conversion work) | MXFP4 (129 tensors -- higher count because GA also keeps `ffn_gate_inp`/`hc_*`/indexer tensors distinct per the "_exps" grep scope; routed `ffn_{gate,up,down}_exps` themselves: same per-layer MXFP4 as preview) |
| tensor **naming** dialect | community aliases throughout (`hc_head_*`, `attn_kv_latent.weight`, suffix-dropped `attn_sinks`/`ffn_gate_tid2eid`/hc tensors, `attn_compressor_*`/`indexer_compressor_*`) -- required the full alias-mechanism port (2026-07-31 entries above) | **canonical ds4/llama.cpp names throughout** -- confirmed via header dump (`blk.0.attn_kv.weight`, `blk.0.ffn_gate_tid2eid.weight` with `.weight` suffix, `output_hc_base/fn/scale.weight`) and via the `--inspect` run below emitting **zero** tensor-name-alias compat notices, only dtype-dequant notices |

**The 8 metadata keys the preview's compat layer had to derive
(`attention.output_lora_rank`, `attention.output_group_count`, `hash_layer_count`,
`hyper_connection.count`, `hyper_connection.sinkhorn_iterations`, `hyper_connection.epsilon`,
`attention.compress_rope_freq_base`, `attention.compress_ratios`) are all natively present
in GA's raw header** (verified directly: `deepseek4.attention.output_lora_rank = 1024`,
`...output_group_count = 8`, `deepseek4.hash_layer_count = 3`,
`deepseek4.hyper_connection.count = 4`, `...sinkhorn_iterations = 20`, `...epsilon =
9.999999974752427e-07`, `deepseek4.attention.compress_rope_freq_base = 160000.0`,
`deepseek4.attention.compress_ratios = (5, 46)` array present) -- exactly the `59 - 51 = 8`
key-count delta. **GA is a materially different, more standard conversion than the preview's
community one**: it uses ds4's/llama.cpp's own canonical tensor names and includes all the
keys the preview's converter dropped. The BF16/Q6_K dense-tensor dequant-at-load compat
mechanism (ported 2026-08-01, `model_convert_dense_bf16_q6k` /
`tensor_is_dense_conversion_candidate`) still fires for GA -- 13 tensor families (BF16
`token_embd.weight`, `ffn_gate_inp.weight`x43, `output.weight`, plus BF16/F32
compressor/indexer/hc tensors) get dequantized-to-F16-at-load notices in the `--inspect` log
below, confirming that prior port's work generalizes to this artifact too, not just the
preview one.

**`--inspect` result: NEW blocker, a genuine metadata gap not seen on the preview artifact.**
Command: `./ds4 -m gguf/DeepSeek-V4-Flash-0731-MXFP4_MOE-Q8_0.gguf --cuda --ssd-streaming
--ssd-streaming-cold --ssd-streaming-cache-experts 8GB --inspect`, `timeout 480`. Full log:

```
ds4: Linux cuda backend set oom_score_adj=1000
ds4: tensor family token_embd.weight (bf16, 1 tensor) dialect compat: dequantized to f16 at load
ds4: tensor family blk.N.ffn_gate_inp.weight (bf16, 43 tensors) dialect compat: dequantized to f16 at load
ds4: tensor family blk.N.hc_attn_fn.weight (f32, 43 tensors) dialect compat: dequantized to f16 at load
ds4: tensor family blk.N.hc_ffn_fn.weight (f32, 43 tensors) dialect compat: dequantized to f16 at load
ds4: tensor family blk.N.attn_compressor_ape.weight (f32, 41 tensors) dialect compat: dequantized to f16 at load
ds4: tensor family blk.N.attn_compressor_gate.weight (bf16, 41 tensors) dialect compat: dequantized to f16 at load
ds4: tensor family blk.N.attn_compressor_kv.weight (bf16, 41 tensors) dialect compat: dequantized to f16 at load
ds4: tensor family blk.N.indexer_compressor_ape.weight (f32, 21 tensors) dialect compat: dequantized to f16 at load
ds4: tensor family blk.N.indexer_compressor_gate.weight (bf16, 21 tensors) dialect compat: dequantized to f16 at load
ds4: tensor family blk.N.indexer_compressor_kv.weight (bf16, 21 tensors) dialect compat: dequantized to f16 at load
ds4: tensor family blk.N.indexer.proj.weight (bf16, 21 tensors) dialect compat: dequantized to f16 at load
ds4: tensor family output_hc_fn.weight (f32, 1 tensor) dialect compat: dequantized to f16 at load
ds4: tensor family output.weight (bf16, 1 tensor) dialect compat: dequantized to f16 at load
ds4: required metadata key is missing: deepseek4.vocab_size
```

Exit code 1. GPU was idle before and after (`nvidia-smi` shows no process, 0 MiB); no stray
`ds4` process left running.

**Diagnosis (measurement/report only, no code changed, per this unit's explicit
instruction to stop and document rather than chase a new gap).** `deepseek4.vocab_size` is
`required_u32(m, "deepseek4.vocab_size")` at `ds4.c:6300`, validated against `DS4_N_VOCAB`
at `ds4.c:6374`. GA's raw header has **no key matching `vocab` anywhere** (confirmed via a
full metadata-key grep of the header dump: zero hits) -- unlike the preview artifact, whose
converter apparently included this key (it is not in the preview's list of 8 missing keys
from the 2026-07-31 entry above), GA's converter dropped it. The value is very likely
tensor-derivable, mirroring the existing compat-layer pattern for the other 8 keys: GA's own
`tokenizer.ggml.tokens` array has exactly 129280 entries and `token_embd.weight`/
`output.weight` both have a `129280`-sized dimension, all three agreeing with
`DS4_N_VOCAB`'s expected value. This is flagged as the probable fix (same shape/tensor-count
derivation pattern already used for `hash_layer_count`/`hyper_connection.count` above) for
whoever picks this up next -- **not implemented this unit**, per the ticket's explicit
"stop at that failure, document precisely, commit, report" instruction for new gaps.

**Consequence.** This blocks every downstream step in this unit's protocol (smoke, warm
baseline, quality battery, decision prep) -- the model cannot load at all yet, on any
backend or flag combination, since `config_validate_deepseek4_model()` fails before any
GPU/CUDA-specific code path is reached. Steps 3-6 of this unit were **not attempted** (would
fail identically -- same required-key check runs on every load path, `--inspect` and normal
generation alike). The 0731 GA promotion decision cannot be made until this one-key gap is
closed; recommend the next unit either extend the metadata compat layer with the
tensor-derived fallback above (small, same-shape fix as the existing 8-key mechanism) or, if
`--vocab-size`-style override flag exists, use that as a stopgap to unblock smoke-level
manual verification while the proper fix lands.

**Server discipline.** `ds4-server` (systemd, port 8000, production IQ2XXS model) stopped
before this unit's `--inspect` attempt; restarted and verified (`systemctl is-active`=
`active`, `GET /v1/models`->HTTP 200) after, per protocol -- mandatory even on failure.

**Follow-up (2026-08-02): fixed and unblocked.** Extended the same 8-key dialect-compat
mechanism with a 9th derived fallback for `deepseek4.vocab_size` (`tokenizer.ggml.tokens`
array length, cross-checked against `token_embd.weight`/`output.weight`'s vocab dimension,
`ds4.c` `deepseek4_compat_vocab_size()`/`deepseek4_tensor_dim1()`). GA now loads and
`--inspect`s cleanly; full protocol (smoke, warm baseline, eval, calibration, decision prep)
completed -- see `MEASUREMENTS.md`'s "GA-0731 swap unit: UNBLOCKED" entry for the complete
writeup, including a real memory-thrashing incident found at the preview-precedent 100GB
cache budget (corrected to 75GB for this artifact) and the resulting GA-promotion
recommendation.

## DSpark chaining hypothesis check + resident A/B, ceiling claim corrected (2026-08-02)

Cross-reference only -- not an FP4/MXFP4 CUDA kernel issue, full writeup lives in
`MEASUREMENTS.md`'s entry of the same name. Summary for anyone landing here from a
DSpark/spec-decode angle: operator hypothesis that ds4's DSpark "3-stage" drafter chain
is missing DeepSeek-style per-position argmax-embed chaining between draft positions was
checked against the code, both drafter GGUFs' own metadata, and upstream `fc9efd1`'s
commit/README intent -- **refuted**: `n_stages`/`dspark.layer_count=3` is the small
drafter's own transformer depth (EAGLE-style, tapping 3 target hidden layers via
`dspark.target_layer_ids`), already correctly hidden-state-chained layer-to-layer; the
axis DeepSeek's true MTP chaining would apply to (`dw->block_size`, 5 draft positions) is
deliberately noise/mask-seeded per the checkpoint's own `noise_token_id` metadata, matching
upstream's own "greedy argmax-only path" framing, not a ds4 port omission. Prior unit's
`c832953` "structural, no fix" verdict stands. New: the prior unit's "always exactly
accepted=2" claim was pairing-specific (a different resident pairing reaches depth 6); a
genuine greedy-identity divergence (drafter-enabled output differs from no-drafter output
at `--temp 0`, on a pairing where the no-drafter path is itself perfectly reproducible) was
found and flagged for follow-up, not root-caused this unit -- distinct from this file's
already-documented `--ssd-streaming`+`--temp 0` FP-reduction-order non-determinism note
above (that one reproduces with **zero** drafter code involved; this one only appears with
the drafter enabled, in **resident**, non-streaming mode).
