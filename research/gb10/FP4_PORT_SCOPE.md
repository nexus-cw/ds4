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
- **P2 — NVFP4**: same skeleton, E4M3 sub-scales (worth ~3.1 PPL per FP4 research); blocked
  upstream on quantize-tool emit (ftype 39 reserved in llama.h but no CLI entry at donor
  HEAD) unless we target an existing NVFP4 GGUF.
- **P3 — speed**: native decode dot-product (vecdotq) then sm_121 tensor-core MMA prefill
  path. Only after P1 quality is proven.

## Target artifact

Phase 1 needs an MXFP4_MOE GGUF of an architecture ds4 implements (DeepSeek-V4-Flash or
GLM-5.2). Options (checkpoint hunt in progress):

- Ready-made FP4 GGUF from a quantizer account, if one exists.
- Self-convert: BF16/FP8 original → `convert_hf_to_gguf.py` → `llama-quantize MXFP4_MOE`.
  Needs staging disk for the source checkpoint (dMon has the fast striped storage; robo-dog
  frees ~429GB when the colibri int4 copy is reclaimed).

Expected size, DeepSeek-V4-Flash at MXFP4_MOE (~4.4-5.0 whole-file bpw): ~165-195GB —
bigger than the 121GB memory pool, i.e. a genuine streaming target, which is the point.
