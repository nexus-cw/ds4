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
