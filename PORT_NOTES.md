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

## M3 results (CUDA v1, robo-dog 2026-08-08)

Design: kernels dereference the mmap'd GGUF directly (GB10
cudaDevAttrPageableMemoryAccess=1 verified) — no weight copies, no
pinning, page cache is the streaming layer. Production ds4-server was
never touched; the 82GB model was never made resident (deferred, below).

- Kernel correctness gate (ds4-inkling-cuda --selftest N, real weights,
  double-accumulation reference; PASS = GPU no farther from the exact
  dot than CPU-fp32 is, 4x allowance):
  layers 0 (dense Q5_K/Q6_K/Q8_0/Q4_K), 2 (MoE IQ2_XXS/IQ3_XXS + shexp
  Q5_K/Q6_K + router F32), 40 (IQ4_XS down), 41 (IQ2_S gate/up +
  IQ4_XS down): ALL PASS. Typical numbers: gpu_vs_ref ~1-4e-6 vs
  cpu_vs_ref ~1.3-6.7e-5 — the GPU tree reduction is ~10x closer to the
  exact dot than the CPU's sequential fp32 accumulation. All 9 artifact
  quant types covered.
- End-to-end GPU greedy ("The capital of France is", n=5, c=512):
  ids 12650/13/12650/382/290 = " Paris. Paris is the" — identical to
  the CPU path and the llama.cpp reference; logits match CPU within
  ~2e-5. Timing (paged, production co-resident): prefill 137s,
  decode 41-81s/token.
- DEFERRED: full-speed resident benchmark (weights pre-faulted into
  unified memory) waits for the operator-scheduled production swap
  window. Nothing in the code needs to change for it: residency is a
  page-cache/prefault property, not an engine mode.

## M4 results

- Serving shape chosen: separate binary `ds4-inkling-server` wrapping
  ds4_inkling.c. ds4_server.c was judged too entangled with the
  deepseek4 engine (ds4_engine/ds4_session API is deepseek-shaped:
  MTP/GLM/expert-streaming switches through every handler) to reuse
  without touching the hot path. Endpoints: POST /v1/chat/completions
  (greedy or temperature sampling; stop on eos/<|end_message|>; JSON
  errors; streaming 400s in v1), GET /v1/capabilities, GET /v1/models.
  Single-session, serialized requests, no auth changes.
- Batched prefill: ink_forward_batch — weight rows dequantized once per
  batch (ink_matmat), causal in-batch attention, sequential shortconv
  advance. 5-token prefill went from ~15min (token-by-token) to 137s.
- Chat template: hand-rendered v1 of the artifact's jinja
  (message_system/user/model + content_text framing, model turns closed
  with <|content_model_end_sampling|>), verified against llama.cpp
  --jinja rendering of the same GGUF. Tool calls / thinking-effort
  blocks are NOT implemented (flagged for later).
- Shortconv state x KV snapshot decision (v1): the 4-tap rolling states
  are part of the session state exactly like KV; ink_state_save/load
  snapshot pos + KV + all conv states as one blob at chain boundaries.
  Rewind to an arbitrary position is NOT supported (conv states are
  recurrent); v1 servers re-prefill instead. Deeper option (per-position
  conv-state history for O(1) rewind, ~3*(2*kvw+2*n_embd) floats per
  position) flagged for later.

## Server smoke + resident-benchmark disposition (2026-08-08)

- ds4-inkling-server round-trip on robo-dog (CPU, ctx 512, port 8099):
  GET /v1/capabilities 200; POST /v1/chat/completions 200 with a valid
  OpenAI-shape body — 16 prompt tokens (chat-template framing), greedy
  completion opens "The user is" (the model's thinking-style preamble,
  matching llama.cpp chat mode on this artifact). Request served in
  423s at paged-CPU speed. Test server torn down afterwards; production
  ds4-server stayed active the whole time.
- Resident full-memory GPU benchmark: NOT RUN. The operator cleared a
  swap window, but the local permission layer denied `systemctl stop
  ds4-server` from this session, and stopping production was not
  worked around. Everything needed for the window is ready: binaries
  build on robo-dog, layer selftests all green, e2e GPU parity proven
  paged. Protocol for the window: stop ds4-server; pre-fault the 82GB
  file (vmtouch or cat >/dev/null); ./ds4-inkling-cuda parity run (5
  tokens) then a ~300-token prompt with -n 64 -c 512; report prefill
  t/s and decode t/s as DS4_PREFILL_TPS_REFERENCE /
  DS4_DECODE_TPS_REFERENCE candidates; restart ds4-server and curl
  capabilities for 200.

## M5: true resident mode (2026-08-08)

Root cause of the window failure (evidence-based hypothesis, not fully
reproduced): the v1 GPU path dereferenced file-backed mmap pages
directly via pageable memory access. In the prefaulted, memory-
saturated state (82GB of page cache on a 128GB unified-memory box) the
kernel reclaims and refaults those pages continuously; GPU ATS reads
racing reclaim returned bad data, NaNs propagated through the forward,
and the (then-unguarded) argmax over an all-NaN/-inf vector emitted
token id 0 with logit -inf. Under production-normal memory the same
binary is correct — mmap-paged GPU reads only fail near saturation.
Active reproduction (re-saturating memory under production) was
deliberately not attempted. Owned memory removes the failure class.

Changes:
- `--resident` (ds4-inkling-cuda): copies every tensor into
  cudaMallocManaged memory once at load, then the forward never touches
  the file mapping. `--resident-budget GiB` makes a tensor prefix
  resident and pages the rest (validation mode). Load fails loudly if
  the request does not fit MemAvailable + 4 GiB margin.
- `--resident` (ds4-inkling-server): same via malloc (CPU path).
- `ink_logits_guard`: every decode (CPU CLI, GPU CLI, server) now
  aborts with a corruption diagnostic on NaN-poisoned or all--inf
  logits instead of silently emitting token 0. Exit code 3.

Validation under production (robo-dog, serving untouched):
- `--resident-budget 6`: resident 6.0 GiB in 256/960 tensors (20.8s
  copy-in), then exact parity: ids 12650/13/12650 with logits matching
  the M3 run to the last digit. Copy-in + mixed resident/paged forward
  both correct.
- Full 82GB resident not run here (does not fit beside production).

Window rerun invocation (update /data/gguf/inkling-window.sh):
  1. sudo systemctl stop ds4-server
  2. cd /home/jacinta/src/ds4 && env OMP_NUM_THREADS=8 \
       ./ds4-inkling-cuda -m /data/gguf/inkling/Inkling-Small-UD-IQ2_XXS.gguf \
       -p "The capital of France is" -n 5 -c 512 --resident \
       > /data/gguf/inkling-window-m5.log 2>&1
     (NO cat-prefault step: --resident does the one-time streamed
     copy-in itself, ~3-5 min from SATA; expect a "resident 76.x GiB in
     960/960 tensors" line. Parity bar: " Paris" "." " Paris" " is"
     " the". Any corruption now aborts with exit 3 instead of token 0.)
  3. Benchmark: append a second run with the ~300-token prompt and
     -n 64 -c 512 --resident; report prefill s and decode s/token.
  4. sudo systemctl start ds4-server && curl -s localhost:8080/v1/... 200.

## M6: kernel optimization pass (2026-08-08)

Baseline (operator window, fully resident, v1 scalar kernels): decode
0.197s/token (5.1 t/s, ~13GB/s effective of 273GB/s), prefill 3.3 t/s
(32-token chunks re-walking weights with per-element dequant).

Changes: warp-per-row matvec/matmat kernels with whole-32-subgroup
register dequant (adapted llama.cpp CUDA vecdot patterns, MIT),
shared-memory activation cache (in <= 4096), batched matmat that
dequantizes each weight row once per chunk with an 8-token register
unroll (chunk cap 32 -> 128), grouped MoE expert launches (3 per stage
instead of 18; host keeps the microsecond top-k qsort), per-stage
cudaEvent accounting (--bench / INK_BENCH=1), and --bench-layers L
(resident per-tensor kernel GB/s, disk-independent).

Correctness: full selftest matrix PASS on all 9 quant types incl. new
batch-matmat and grouped-matvec checks (max abs diff 4.6e-5 / 3.8e-5 vs
CPU); e2e parity ' Paris' reproduced paged, partial-resident, and
partial-resident+bench.

Kernel microbenchmarks (resident copies, layer 2; old kernels were
~13GB/s effective end-to-end): Q5_K 53-66 GB/s, Q8_0 60, F32 111,
Q4_K ~66, Q6_K/IQ3_XXS ~38, IQ2_XXS 22-24 GB/s. Single-tensor launches
(2048 rows) underutilize the GPU; real decode uses 6-expert grouped
launches, so these are lower bounds.

Full-model decode estimate from per-tensor bytes/rates: routed experts
~68ms + attention ~20ms + shared experts ~9ms + head ~7ms + dense/router
~6ms => ~110ms/token, ~9 t/s (conservative; grouped-launch occupancy
should land it in 9-15 t/s). The dominant remaining lever is the
IQ2_XXS kernel (carries most decode bytes at only ~22GB/s — needs
wider/vectorized block loads and 2 rows/warp). Prefill becomes
weight-reuse-bound: each row dequantized once per 128 tokens, so
prefill t/s should now exceed decode t/s; measured properly only at the
next window.

One NaN incident during M6 validation (partial-resident + warm page
cache): the logits guard aborted exactly as designed; the identical
config passes deterministically with normal cache state — consistent
with the M5 paged-read-under-reclaim root cause, NOT a kernel bug
(selftests + reruns pass). Full residency avoids the class entirely.

Window invocation for the M6 verification run (same as M5 protocol,
plus bench self-reporting):
  env OMP_NUM_THREADS=8 INK_BENCH=1 ./ds4-inkling-cuda \
    -m /data/gguf/inkling/Inkling-Small-UD-IQ2_XXS.gguf \
    -p "The capital of France is" -n 5 -c 512 --resident \
    > /data/gguf/inkling-window-m6a.log 2>&1
  then the ~300-token prompt with -n 64 -c 512 --resident, INK_BENCH=1
    > /data/gguf/inkling-window-m6b.log 2>&1
  Parity bar unchanged (" Paris" "." " Paris" " is" " the"); the bench
  report at exit gives per-stage ms + GB/s for the before/after table.

## M7: decode-path round 2 (2026-08-08)

Why decode didn't improve in M6: not bandwidth. Measured GB10 read
ceilings (2GiB grid-stride sum, in-binary --bench-membw + standalone
probe agree): cudaMallocManaged host-touched 162-164 GB/s, cudaMalloc
225-237 GB/s, plain malloc (ATS) 165 GB/s. The 27.7/15.3 GB/s pool
rates in the M6 window were launch+sync overhead: ink_forward_gpu
issued ~785 kernel launches per decode token, each wrapper ending in
cudaDeviceSynchronize (~100us+ apiece on Grace) because host-side
rmsnorm/silu/rel-projection/residual math sat between GPU ops.

Changes: single CUDA stream, wrappers no longer sync; rmsnorm (vector +
per-head), silu-mul (+gamma variant), residual add, scale,
rel-projection, and MoE weighted-accumulate became device kernels; KV
cache writes are stream-ordered copy kernels; host syncs per decode
token: 1 per MoE layer (top-k routing qsort) + 1 final = ~41, from 785.
Gate+up+silu fused into fewer grouped launches. Bench report split into
prefill/decode phase tables (--bench itself re-adds per-launch syncs
for attribution, so async wins show only in plain wall-clock). Core
gained ink_model_make_resident_ex (per-tensor allocator choice) so a
follow-up can place big quant tensors in cudaMalloc (+45% streaming vs
managed) while host-read tensors (token_embd/gscale/exp_probs_b) stay
host-accessible - NOT yet wired into --resident.

Two real bugs found by the gate during M7 (both committed as fixes):
1. Stale-KV race: cudaMemcpyAsync between two pageable host pointers
   executes synchronously on the calling thread, jumping ahead of the
   queued sconv/k-norm kernels producing the source; the KV cache read
   stale data and parity broke (' Paris' -> 'es'). Fixed with a
   stream-ordered copy kernel.
2. Use-after-free: per-layer MoE host temporaries were freed while
   queued kernels still read them (heap corruption). Hoisted to
   per-forward-call lifetime, freed after the final sync.

Validation (production untouched): selftest matrix all 9 types +
matmat + grouped PASS; paged e2e parity exact 5/5 (' Paris . Paris is
the', logits within 2e-5 of CPU); partial-resident (6GiB) parity exact
with phase-split bench working. Paged decode wall-clock 26-43s/token
vs 41-81s in M6 under similar cache state (disk-bound; indicative
only).

Projected resident decode: bytes/token ~2.4GiB; with the sync wall
removed (785 -> 41) and managed streaming measured at 162 GB/s, decode
lands at 10-25 t/s depending on the true grouped-kernel rate at decode
shapes (medium confidence in >=10 t/s; the remaining unknown is kernel
efficiency, not overhead or bandwidth). If the window still shows the
group pool slow, next levers are cudaMalloc placement via
make_resident_ex and IQ2_XXS load vectorization.

M7 window invocation (operator verification): TWO runs -
  a) plain wall-clock (the real numbers):
     env OMP_NUM_THREADS=8 ./ds4-inkling-cuda -m .../Inkling-Small-UD-IQ2_XXS.gguf \
       -p "<300-token prompt>" -n 64 -c 512 --resident > inkling-window-m7a.log 2>&1
  b) attribution (per-launch syncs forced, slower, per-phase tables):
     same with INK_BENCH=1 > inkling-window-m7b.log 2>&1
  Plus the 5-token parity bar first (" Paris" "." " Paris" " is" " the").

## M8: group-matvec gap (2026-08-08)

Changes: vectorized hot dequants (IQ2_XXS u32-pair header loads,
IQ3_XXS u32, Q6_K 16 u32 loads with folded qsel addressing), sign
application via IEEE sign-bit XOR (bitwise identical to *-1.f), float4
X reads in the warp-dot loop; --resident now splits the arena via
make_resident_ex: non-F32 tensors (minus token_embd) -> cudaMalloc,
F32/host-read -> host malloc, boot line prints the GiB split;
--bench-layers adds the 6-expert grouped launch at real decode shape;
bench copy-in via UVA cudaMemcpy (device-arena tensors faulted host
memcpy); decode/prefill prints now ms-resolution.

Kernel before/after at decode shapes (bench-layers, layer 2):
  IQ2_XXS single 22-24 -> 62 GB/s; IQ3_XXS 38 -> 75 (managed) / 98
  (device arena); grouped 6-expert: gate/up 47-54 GB/s, down 42-45 --
  ~identical managed vs device arena, so the grouped path is now
  COMPUTE-bound (dequant ALU), not placement- or bandwidth-bound.
  Window-effective baseline for the same pool was 28 GB/s.

Decode arithmetic/token from measured rates: routed grouped ~39ms +
Q5_K pool ~26ms + Q6_K ~5ms + head ~5-7ms + router/attention/sconv/
sync ~9ms => ~84-90ms/token, ~11-12 t/s. Confidence medium-high for
the >=10 t/s bar (every component measured at decode shape; residual
risk is end-to-end interference). Next lever if the window disagrees:
the grouped kernel lacks the float4/shared-X treatment of the single
matvec path, and IQ2_XXS dequant ALU (grid+sign unpack) is the
per-weight cost floor.

Gate: selftest matrix all types + matmat + grouped PASS (float4
reassociation tightened gpu_vs_ref to ~2e-6); paged parity exact;
partial-resident split-arena parity exact (5.5 GiB device / 0.5 GiB
host boot line confirmed). One paged run hit the known
warm-cache/reclaim NaN class (prefill 19s = fully cached = saturated);
guard aborted correctly; cold rerun exact. Full residency remains the
fix for that class.

M8 window invocation: unchanged from M7 (parity bar, then plain
wall-clock run a, then INK_BENCH=1 run b) -- the binary now prints
ms-resolution decode lines, and with --resident expect the boot line
"resident 76.x GiB in 960/960 tensors (NN.N GiB device / NN.N GiB
host)". Watch device-free: the WARNING line is advisory; cudaMalloc
failure dies loudly.

## M9: productization (2026-08-09)

Served window decision: 65536 (interactive convention). KV math: 42
layers x 2 x 1024 floats x 4B = 336 KiB/token -> 21.5 GiB at 65536,
+ ~5 MB conv states; with the 76.0 GiB device arena that is ~98 GiB of
128, ~25 GiB headroom in a dedicated window. Long-window plumbing
validated under production at -c 16384 with a 1294-token prompt
(chunked batched prefill x11, SWA windows past pos 512, global rel
extents; greedy continuation " Paris." of an "In summary, the capital
of France is" tail). Full 65536 allocation is window-only (21.5 GiB
KV does not fit beside production).

Server hardening (ds4_inkling_server.c): FIFO slot pool (one worker
thread owns the single engine state; slots bound queue depth, 503
overloaded), OpenAI SSE streaming byte-compatible with ds4-server
(role delta / content deltas / finish / optional usage / [DONE]),
auth mirrors ds4-server exactly (completions open; only select gated
by ACCRETION_ADMIN_TOKEN: 405 admin_disabled unset, 401 wrong),
/v1/capabilities with configured-vs-trained context (trained 1048576
read from the header), /v1/activity with measured prefill tps +
derived eta (omitted until tps>0), /v1/models/available with sidecar
mode tags, SIGTERM drain (second signal _exit 130). Tested live on
port 8099 under production (paged): capabilities/models/activity
shapes, 401/405, 503 overload with 2 slots, SSE frame-exact stream,
activity mid-prefill, clean drain. usage counts sampled tokens (fix
committed after the first SSE test showed a skipped special token).

Arch dispatch (the swap): new env key DS4_ARCH (inkling|deepseek4,
absent=deepseek4) + wrapper /opt/accretion/bin/accretion-serve execd
by the systemd unit; select in EITHER server writes DS4_MODEL/DS4_ARCH/
DS4_CTX/DS4_CACHE_BUDGET/DS4_EXTRA_FLAGS and exits 42; Restart=
on-failure re-runs the wrapper which execs the matching binary.
Cross-family models report loadable:"yes" only when the unit env sets
ACCRETION_ARCH_WRAPPER=1 (honest gating on the wrapper being armed).
ds4-server side: model_arch_loadable + select env-write extended
(ds4_server.c, commit b83d8a5, on inkling-port after merging latest
platform which carries the sidecar layer). Inkling side mirrors the
full select choreography so the console can swap back.

Accretion repo (branch inkling-serve, pushed): build/accretion-serve
wrapper, unit template ExecStart -> wrapper, install.sh installs
wrapper + seeds DS4_ARCH/ACCRETION_ARCH_WRAPPER in the env file,
build-release.sh builds/stages ds4-inkling-server (guarded: aborts
naming the missing engine sync until inkling-port lands in platform),
docs/CONSOLE.md architecture-swap section, docs/ARCH_SEAM.md inkling
row. Engine subtree sync (inkling-port -> platform -> subtree pull) is
a pending operator step.

Operator steps to make Inkling console-live:
1. Merge/land ds4 fork inkling-port into platform (or cherry-pick),
   run accretion scripts/sync-engine.sh, merge accretion inkling-serve.
2. Release: scripts/build-release.sh; install.sh on robo-dog (installs
   accretion-serve, refreshes unit to ExecStart wrapper; env file gains
   DS4_ARCH=deepseek4 + ACCRETION_ARCH_WRAPPER=1 if regenerating -- for
   the existing env file ADD those two lines by hand).
3. Ensure the inkling sidecar sits NEXT TO the served gguf as
   <model>.gguf.env with DS4_ARCH=inkling, DS4_CTX=65536,
   DS4_DECODE_TPS_REFERENCE=12.3 (the banked file at
   /data/gguf/models/inkling-small/ has the right keys; note the
   scanner keys off "<path>.env" of the actual gguf).
4. systemctl daemon-reload && restart ds4-server (window); then the
   swap is console-driven: pick Inkling in MODELS -> select -> exit 42
   -> wrapper starts ds4-inkling-server resident at 65536 (copy-in
   ~343s before first token; /v1/capabilities appears when up).
5. Full resident multi-session validation in that window (not
   testable beside production).

## M10: the server was running the CPU engine (task #36, 2026-08-10)

The bug: Makefile built ds4-inkling-server from ds4_inkling_server.c +
ds4_inkling.c with $(CC) and NO CUDA translation unit, so console-live
Inkling was served by the correctness-reference CPU engine -- ~2.1 s/
token and 0.88 t/s prefill against the CLI's 81.5 ms/token and 11.0
t/s. The banked 12.3 t/s sidecar value was a CLI number the server
could never deliver. It was invisible because nothing in the API
reported which engine was live; the M9 API gate checked response
SHAPES and never timed a token.

Fixes:
- ds4-inkling-server now links ds4_inkling_cuda.cu via nvcc (main()
  gated by DS4_INKLING_NO_MAIN) and dispatches prefill AND decode
  through ink_forward_gpu with the split device arena
  (ink_cuda_make_resident, exported for the server). Prefill batches at
  the GPU chunk (128), not 32.
- --cpu / DS4_INKLING_BACKEND=cpu selects the reference engine;
  ds4-inkling-server-cpu builds it CUDA-free for hosts without nvcc.
  The CPU engine is NOT deleted -- it is the correctness oracle.
- /v1/capabilities reports serving.backend ("cuda"|"cpu") so this class
  of mistake is visible from the console instead of a stopwatch.
- accretion scripts/build-release.sh refuses to cut a release without
  nvcc rather than silently shipping a CPU-linked server.
- Logits corruption now fails the REQUEST, not the process:
  ink_logits_ok() is the non-fatal form; checked once after prefill
  (clean 500, nothing sent yet) and per decode step (500, or an SSE
  error event when frames are already streaming). CLIs keep the fatal
  ink_logits_guard.

Verified beside production (could not restart it, see below):
- Server starts on the GPU engine: "backend cuda", split arena line
  "3.5 GiB device / 0.5 GiB host", /v1/capabilities serving.backend
  "cuda".
- --cpu path starts and serves: "backend cpu", API returns content
  "The user" for {"messages":[{"role":"user","content":"Capital of
  France?"}],"max_tokens":3,"temperature":0} (286.9 s -- the CPU
  engine's real speed, and the reference string for the window check).
- Guard-to-500 proven for real: the paged test instance tripped the
  known reclaim-NaN class and returned
  {"error":{"message":"logits corruption detected..."}} with HTTP 500,
  and the PROCESS SURVIVED (capabilities 200, activity idle, pid
  alive). Before this change that was exit(3).

NOT measured: through-the-API decode/prefill t/s. That needs the
service restarted onto the new binary, which this session's permission
layer denies (systemctl blocked, as in M5/M9). The binary is staged at
/opt/accretion/bin/ds4-inkling-server.new. The sidecar's
DS4_DECODE_TPS_REFERENCE=12.3 / prefill 11.0 remain CLI-measured
numbers and should be REPLACED with the through-API measurements from
the run below; expect them slightly under the CLI (per-request state
reset, chat-template tokens, HTTP framing).

Operator steps (deploy + measure):
  sudo install -m 755 /opt/accretion/bin/ds4-inkling-server.new \
      /opt/accretion/bin/ds4-inkling-server
  sudo systemctl restart ds4-server
  # confirm the engine is the GPU one BEFORE trusting any number:
  curl -s localhost:8000/v1/capabilities   # serving.backend must be "cuda"
  # correctness (must return content "The user"):
  curl -s -X POST localhost:8000/v1/chat/completions \
    -H Content-Type:application/json \
    -d '{"messages":[{"role":"user","content":"Capital of France?"}],"max_tokens":3,"temperature":0}'
  # decode t/s through the API (32 tokens, watch total time):
  time curl -s -X POST localhost:8000/v1/chat/completions \
    -H Content-Type:application/json \
    -d '{"messages":[{"role":"user","content":"Write one paragraph about rivers."}],"max_tokens":32,"temperature":0}'
  # prefill t/s through the API: poll /v1/activity during a long-prompt
  # request and read prefill.tokens_per_second.
Then update the sidecar DS4_DECODE_TPS_REFERENCE to the measured value.

## M11: int8/dp4a fast path -- fast, numerically sound, and NOT SHIPPABLE

Built (llama.cpp ggml-cuda/vecdotq.cuh technique, MIT, adapted): the
grouped expert matvecs quantize the activation once per grouped call to
int8 blocks of 32, then do 8 __dp4a integer dots per 32 weights instead
of 32 float FMAs, with the codebooks staged in shared memory. Applied
to IQ2_XXS + IQ3_XXS (the decode carriers) only. Scaling stays pure
float with the verified db formulas (deliberately not llama.cpp's
integer sumi*ls/8). Sign application uses a scalar per-byte negate, not
__vcmpne4/__vsub4 -- a known perf residual.

Speed (bench-layers 2, 6-expert grouped, real decode shape):
  gate_exps IQ2_XXS  FAST 69.3 GB/s  vs EXACT 50.8  (+36%)
  up_exps   IQ2_XXS  FAST 69.3       vs EXACT 44.8  (+55%)
  down_exps IQ3_XXS  FAST 60.5       vs EXACT 42.5  (+42%)

Numerics at matvec granularity: err/ref_rms = 0.297-0.300% against the
CPU f32 reference, with ref_rms ~4.8-5.5 -- i.e. EXACTLY the int8
activation-quantization floor. The kernel is not buggy at this level.
(The harness now prints ref_rms/ref_absmax so this is read off, not
inferred: an earlier reading of mine called maxreldiff=39.7 "noise near
zero" and moved on, which was the wrong instinct -- the loose-bound
flag was right to fire.)

END-TO-END, HOWEVER, IT CHANGES GENERATIONS:
  prompt "The capital of France is" -> FAST and EXACT identical
    (12650/13/12650/382/290, " Paris. Paris is the")
  prompt "The three largest planets in the solar system are"
    EXACT: " the three largest planets"  (top-1 logit 14.52)
    FAST:  "The user is asking"          (top-1 logit 12.22)
A 2.3-logit, 16%-lower top-1 is not what 0.3% per-matvec noise looks
like after one layer; it is what it looks like after 40 MoE layers of
accumulation into a 2-bit model's residual stream, or what an
only-in-full-forward bug looks like. Both hypotheses are open.
DEFAULT IS THEREFORE THE EXACT PATH (committed); INK_FAST_DEQUANT=1
opts in for investigation only. Nothing was deployed: the staged
/opt/accretion/bin/ds4-inkling-server.new is the exact-default build
(an earlier staging with FAST as default was replaced -- do not deploy
any binary built before 2026-08-10 18:05).

Projected gain had it been shippable: grouped pool ~39ms -> ~28ms of an
81.5ms CLI decode => ~70ms CLI, ~10.3 t/s through the API (+12% over
9.2). Modest, because the remaining decode time is now dominated by
pools dp4a never touched: Q5_K attention/shexp (~26ms) and the Q4_K
output head (~7ms).

Answer to the "is it an ALU wall / should we repack to MXFP4" question:
NOT YET, and the evidence says the ALU levers are not exhausted. Even
on the fast path the kernels sit ~3.3x below the 225 GB/s device
ceiling, and three known-unexhausted causes remain, in my order of
expected value:
  1. scalar sign construction (ink_pack_signed4 does a 4-iteration
     per-byte negate where llama.cpp does __vcmpne4 + XOR + __vsub4 --
     roughly 4x the ALU ops in the innermost loop);
  2. no ILP/occupancy work at all (rows-per-warp, wider per-thread
     work; the builder deferred it as too risky to write blind);
  3. redundant requantization -- gate and up quantize the SAME
     activation vector twice per layer.
MXFP4 repacking doubles bytes/token (2.4 -> ~4.8 GiB) to buy near-free
dequant; at a realistic 150-200 GB/s that is ~24-32ms/token, which a
fully-optimized dp4a path (2.4 GiB at 120-150 GB/s = 16-20ms) should
still beat. I would exhaust 1-3 before scoping a prepare-pipeline
change.

Next-round recommendation (in order): fix the divergence (bisect by
running the fast path in ONE layer at a time to see whether error
accumulates smoothly or jumps -- that discriminates accumulation from
bug); then SIMD sign construction; then extend dp4a to the K-quants,
which is where the remaining decode time actually is.

## M12: divergence bisect -- verdict ACCUMULATION, not a bug

Harness: INK_FAST_LAYERS=none|all|N|LO-HI gates the dp4a path per layer,
INK_FAST_TYPES=both|iq2|iq3 gates it per quant type (for this artifact
iq2 = gate/up, iq3 = down), --logits-out dumps the full logits vector so
every step is diffed against the all-exact reference. One generated
token per run (the logits after prefill are what matter).

Sweep, prompt "The three largest planets in the solar system are"
(reference top-1 id=290 ' the', logit 14.5164):
  fast layers   top1     logit     max|dlogit|   token
  none          290      14.5164   0.00000       same
  0             290      14.5164   0.00000       same
  0-1           290      14.5164   0.00000       same
  0-3           79575    13.9833   2.66460       CHANGED
  0-7           279      10.7750   5.46189       CHANGED
  0-15          976      12.2558   7.42159       CHANGED
  0-31          976      12.2265   7.37926       CHANGED
  all           976      12.2241   7.39743       CHANGED
Same sweep, "The capital of France is": token NEVER changes, max|dlogit|
stays in 0.18-0.60 -- a high-confidence prompt absorbs the perturbation.

Follow-ups (all layers unless noted), planets prompt:
  iq2 only (gate/up)   top1 290 same   max|dlogit| 1.59518
  iq3 only (down)      top1 290 same   max|dlogit| 2.12886
  layers 32-41 only    top1 290 same   max|dlogit| 0.06307
  layers 20-41 only    top1 290 same   max|dlogit| 0.09527
  both types, all      top1 976 CHANGED max|dlogit| 7.39743

VERDICT: accumulation. Five independent pieces of evidence, no
discontinuity anywhere:
1. Layers 0-1 are the DENSE blocks -- no expert matvecs, and the delta
   is exactly 0.00000, so the gating is sound and nothing leaks.
2. Growth across the sweep is monotone then saturating (2.66 -> 5.46 ->
   7.42 -> 7.38 -> 7.40). A bug in one layer/expert/edge case would show
   as a jump at a specific step; there is none.
3. Neither quant type dominates (1.60 vs 2.13) -- inconsistent with a
   format-specific coding defect in one dequantizer. It also refutes my
   own heavy-tail hypothesis: post-SiLU inputs (iq3) are only modestly
   worse than post-rmsnorm inputs (iq2), not categorically worse.
4. The effect is governed by DEPTH POSITION, smoothly: the last 10
   layers cost 0.063, the last 22 cost 0.095, all 40 cost 7.40. Error
   injected early is amplified by the remaining layers; error injected
   late has no depth left to amplify it. A bug would not track position
   this cleanly.
5. The two types are strongly super-additive (1.60 + 2.13 = 3.7 alone,
   7.40 together), the signature of nonlinear amplification through a
   deep 2-bit model rather than additive noise.
So the int8 activation floor (measured 0.30% per matvec on synthetic
data) is real and irreducible at this precision, and 40 MoE layers of a
2-bit model amplify it past the point where top-1 survives on
lower-confidence prompts.

IS dp4a USABLE ANYWHERE? Yes, but the honest win is small:
- SAFE: late layers. Layers 20-41 (22 of 40 MoE layers, ~55% of expert
  traffic) move the logits by 0.095 and change no token on either
  prompt. The output head (Q4_K, the very last matvec) is by the same
  argument the safest place of all -- nothing downstream to amplify.
- NOT SAFE: early/middle layers, at any precision int8 can offer.
- Expected gain if we took the safe part: expert pool ~39ms of an
  81.5ms decode, 55% of it at +40% => ~6ms, i.e. ~75ms CLI / ~9.8 t/s
  through the API, +7% over 9.2. Adding the head is maybe +3ms more.
Recommendation: do NOT adopt dp4a for a ~7-10% gain that costs a
behavioural change we would have to re-validate on every prompt class.
Spend the next round on the numerics-PRESERVING levers instead, which
are untouched and which the exact path also benefits from: ILP and
occupancy (rows-per-warp, wider per-thread work) on the exact kernels,
which still sit at 42-54 GB/s against a 225 GB/s device ceiling -- a 4x
gap that costs nothing in accuracy to attack. Revisit dp4a only if a
late-layers-only mode is wanted after that, and only with the sweep
above rerun as its gate.

Higher-precision activations: not promising. The failure is depth
amplification of a per-matvec error, so a 2x-finer activation grid buys
roughly one extra sweep step of headroom, not a category change; fp16
activations would abandon dp4a (which is the entire point of the
exercise). Nothing shipped this round: exact remains the default, and
no binary was deployed.

## M13: numerics-preserving round -- expert pool +16-49%, identity held

Changes (all in ink_matvec_row_warp + three dq32_*, shared by the exact
single AND grouped kernels; no fast/dp4a path, quantizer or server
touched):
1. 2-way ILP: the lane's dependent dequant->dot->acc chain became two
   independent accumulators over paired subgroups (sg, sg+32), merged in
   fixed order before the unchanged shuffle reduction. This is a bounded
   deterministic REASSOCIATION, not bit-identical -- of the same kind
   already present in the warp reduction and the float4 grouping.
2. dq32_q5_K: ql/qh read as 8x4-byte words instead of 32 byte loads,
   loop-invariant nibble-mask selection hoisted.
3. dq32_iq2_s: sign-bit XOR (bitwise identical to *-1.f), which the
   other IQ dequantizers have had since M8.
Declined by the builder, reasoning accepted: cross-block activation
dedup (shared memory is per-block; needs a persistent/two-pass kernel)
and dynamic extern __shared__ sizing (unverifiable blind; silent shared
corruption is exactly what this round guards against).

IDENTITY GATE -- PASS:
  selftest matrix (layers 0, 2): all PASS at unchanged tight bounds.
  planets prompt: top-1 id 290 -> 290, logit 14.516440 -> 14.516437,
                  full-vector max|dlogit| = 2.06e-05
  france prompt:  top-1 id 12650 -> 12650, logit 18.536530 -> 18.536531,
                  full-vector max|dlogit| = 7.63e-06
Both deltas are fp-reassociation noise, below the selftest's own
gpu_vs_cpu spread (~4e-05). No token changed.

SPEED, grouped 6-expert at decode shape (rotating over 12 distinct
expert sets so nothing sits in L2):
  gate_exps IQ2_XXS  45.30 -> 56.16 GB/s  (+24%)
  up_exps   IQ2_XXS  48.80 -> 56.61       (+16%)
  down_exps IQ3_XXS  41.81 -> 62.39       (+49%)
Per 6-expert set per layer: 1.0132 ms -> 0.7691 ms, i.e. the routed
expert pool is 24% cheaper. Over 40 MoE layers that is ~40.5 ms ->
~30.8 ms, saving ~9.8 ms of an 81.5 ms decode.

MEASUREMENT WARNING (recorded in the harness output too): the
SINGLE-TENSOR bench numbers moved the OTHER way -- wq/wo Q5_K 175 ->
142 GB/s, gate_exps 70 -> 60. That is the builder's register-pressure
risk showing up exactly where it should: a single 2-19 MB slice hit 20x
is L2-RESIDENT, so 2-way ILP costs registers/occupancy while hiding
latency that does not exist in that regime. Real decode streams six
experts chosen fresh per layer out of a 76 GiB arena with no reuse --
the rotating grouped table is the representative one. Earlier notes'
single-tensor GB/s figures were partly measuring cache and should be
read with that caveat.

Projected: ~9.8 ms saved on the expert pool, plus whatever the Q5_K
wide loads give attention in the streaming regime (unmeasurable here
because the single-tensor bench is in-cache) => ~70-72 ms CLI decode,
~10.2-10.6 t/s through the API vs 9.2 measured (+11-15%). NOT measured
end-to-end: the permission layer denies systemctl restart, so the
staged binary could not be exercised through the API.

Where the time sits afterward, and the bandwidth question: nothing is
bandwidth-bound yet. The expert pool now runs at 56-62 GB/s against a
225 GB/s device ceiling, and the dp4a path -- which removes dequant ALU
entirely -- reaches only 63-66 GB/s on the same shapes. That
convergence is the informative part: after M13 the exact path is within
7-11% of the ALU-free path on gate/up and has CAUGHT it on down_exps
(62.4 vs 62.8). So dequant ALU is no longer the dominant cost; whatever
now limits both paths is shared (memory latency/occupancy at these
shapes), and dp4a's remaining advantage has essentially evaporated --
another reason not to revisit it.
Next lever, if more decode speed is wanted, is therefore NOT more ALU
work on these kernels: it is occupancy/latency (persistent kernels,
dynamic shared sizing, or fusing gate+up so one weight stream feeds two
outputs), or the prepare-time MXFP4 repack. I would measure a resident
end-to-end run first -- three of the last four rounds' projections were
distorted by benchmarking artifacts, and only the API number settles it.

Operator steps to measure M13 through the API:
  sudo install -m 755 /opt/accretion/bin/ds4-inkling-server.new \
      /opt/accretion/bin/ds4-inkling-server
  sudo systemctl restart ds4-server
  curl -s localhost:8000/v1/capabilities        # backend must be "cuda"
  curl -s -X POST localhost:8000/v1/chat/completions -H Content-Type:application/json \
    -d '{"messages":[{"role":"user","content":"Capital of France?"}],"max_tokens":3,"temperature":0}'
    # expect content "The user" (unchanged from M10 -- identity gate)
  time curl -s -X POST localhost:8000/v1/chat/completions -H Content-Type:application/json \
    -d '{"messages":[{"role":"user","content":"Write one paragraph about rivers."}],"max_tokens":32,"temperature":0}'
    # decode t/s: expect ~10.2-10.6 vs 9.2 baseline
If decode does NOT improve, revert ONLY the ILP hunk (keep the Q5_K
wide loads and the iq2_s sign XOR, which are strict wins) -- the ILP is
the only change whose benefit depends on the streaming regime.

## M14: the console follows the model (task #37)

The gap: ds4_console.h was wired only into ds4-server, so with Inkling
serving, GET / answered "no such endpoint". Selecting Inkling from the
browser removed the browser -- the swap-back API worked, but only by
curl. Accretion's rule is that box operations are clicks.

Fix, one implementation not two: ds4_console.h gained
ds4_console_render(), which takes a backend-neutral ds4_console_facts
struct and writes through a caller-supplied ds4_console_write_fn (the
two servers have different buffer types, so a callback rather than a
shared buf). ds4-server's inline rendering moved onto it; its now-dead
buf_html_escape was removed. ds4-inkling-server gained GET / and
/console using the same call. Neither server can fork the page.

Honest degradation: the page's JS polled /v1/routing-stats
unconditionally, which the inkling backend does not implement. The
renderer now emits window.DS4C={routing,expert_cache,prewarm} from the
facts, and the routing section reports "not available for this
architecture" instead of polling a 404 and sitting blank. Nothing is
fabricated -- fields left NULL/0 are simply not rendered.

Gate (test instance on :8099 beside production):
- GET / and GET /console -> 200, 12090 bytes, contains "ds4 console".
- Rendered facts match the JSON: architecture inkling; backend cuda;
  sessions 2 == /v1/capabilities serving.sessions; "2048 configured /
  1048576 trained" == context.configured/trained; resident rendered as
  "streamed" for a non---resident test instance (truthful).
- window.DS4C={routing:false,expert_cache:false,prewarm:false} and the
  routing section renders "not available for this architecture".
- Unknown paths still 404 (routing table intact).
- Swap direction inkling -> deepseek4, full choreography: models list
  shows DeepSeek entries loadable:yes (wrapper armed) and laguna/
  deepseek4-dspark honestly loadable:no; select returned 200
  {"status":"swapping"}; the env file was rewritten to
  DS4_MODEL=/data/gguf/DeepSeek-V4-Flash-MXFP4_MOE.gguf and
  DS4_ARCH=deepseek4; the server drained and exited 42.
- Reverse direction (deepseek4 -> inkling) is the same shared code path
  in ds4_server.c (M9, commit b83d8a5) and its binary now REBUILDS
  clean with the console refactor, but it could not be exercised at
  runtime here: starting ds4-server needs an 82 GB DeepSeek model
  resident beside production. It needs the operator window; the
  coordinator has already verified its list/select_enabled side from
  production.
Production was untouched throughout (prod=200 after).

Staged for the operator (nothing installed by me):
  /opt/accretion/bin/ds4-inkling-server.new  (console + M13 kernels)
  /opt/accretion/bin/ds4-server.new          (console refactor)
  sudo install -m 755 /opt/accretion/bin/ds4-inkling-server.new \
      /opt/accretion/bin/ds4-inkling-server
  sudo install -m 755 /opt/accretion/bin/ds4-server.new \
      /opt/accretion/bin/ds4-server
  sudo systemctl restart ds4-server
  curl -s -o /dev/null -w '%{http_code}\n' localhost:8000/    # expect 200
  # then in a browser: STATUS shows inkling/cuda/resident/65536, MODELS
  # lists both families, and picking a DeepSeek model swaps back with the
  # console still present on the other side.
Known cosmetic: building the CUDA TU into the server emits four
"declared but never referenced" remarks for the CLI-only bench/selftest
helpers (they are used by ds4-inkling-cuda's main, compiled out here).
Harmless; left alone rather than restructuring working code late.
