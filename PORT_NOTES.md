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
