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
