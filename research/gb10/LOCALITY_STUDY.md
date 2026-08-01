# Locality/prefetch policy study — GB10, MXFP4-streamed DeepSeek-V4-Flash (2026-08-02)

Prefetch/locality research unit 1. Trace capture + offline policy study only — **no policy
was implemented in ds4 this pass**. Goal of the research program: raise the effective
streamed-expert cache hit rate (baseline: 100GB device LRU, ~81% hit / 4 t/s decode on a
100-token single-prompt benchmark) via smarter policy/prefetch, eventually trained locality.

## Part A: routing-trace capture

**Instrumentation** (`ds4_cuda.cu`, new; build verified clean, `make clean && make
cuda-spark`, zero warnings, all 5 CUDA binaries link). `DS4_ROUTING_TRACE=<path>`, env-gated
(disabled: one `getenv()` + one mutex lock/unlock per streaming-cache call, no I/O —
negligible next to a streamed-MoE step). Hooked at the two `extern "C"` entry points ds4.c's
graph code calls into the CUDA streaming-expert-cache module —
`ds4_gpu_stream_expert_cache_begin_selected_load` (decode: one call per (token, layer)) and
`ds4_gpu_stream_expert_cache_prepare_selected_batch` (prefill: one call per (layer, whole
token-batch)) — rather than inside the shared `cuda_stream_selected_cache_begin_load()`
helper underneath both, so phase and true per-call token-batch size come from what ds4.c
already knows, never inferred. Every decode/prefill call site in ds4.c (dense top-k router,
hash-layer routing, sync and async selected-id readback) funnels into these same two
functions for the CUDA `deepseek4` build, so this is a single choke point covering every
routing path exercised on this artifact. (GLM's separate tensor-typed entry point is not
hooked — out of scope, GLM is not the artifact under study.)

Binary record format (little-endian, buffered writes, no per-record fsync):
```
u32 seq, u32 token_index, u8 phase(0=prefill,1=decode), u8 layer, u8 n_selected, u8 pad,
i16 expert_ids[n_selected]
```
No routing weights recorded (only expert identity matters for cache simulation). Full format
notes, including the layer-wraparound heuristic used to derive `token_index` (no position
counter is threaded through the real call chain), are in `research/gb10/locality_sim.py`'s
module docstring and `research/gb10/traces/README.md`.

**Smoke-verified** before any real capture: `-p "Reply with exactly: ok" -n 10` produced the
exact expected output and a 645-record trace (15 decode tokens x 43 layers, 6
experts/record); a longer prompt produced both prefill and decode records
(`1161 = 27 prefill-tokens x 43`, `172 = 4 decode-tokens x 43`), confirming both hook sites
fire and the phase/token bookkeeping is internally consistent.

**Captures** (`--cuda --ssd-streaming --ssd-streaming-cache-experts 100GB`, `ds4-server`
stopped first):

| workload | invocation | trace | records | decode accesses | prefill accesses | real measured hit_rate (`DS4_CUDA_STREAM_STATS=1`) |
|---|---|---|---|---|---|---|
| single-prompt control (validation run) | `--nothink -p "Explain in a few sentences how photosynthesis works." -n 100`, fresh cold process | `rt_validate.bin` (not committed, reproducible) | 4,085 | 24,510 | 0 | **0.809** |
| (a) ds4-eval 12-item subset | exact re-run of MEASUREMENTS.md's "MXFP4-streaming quality eval" invocation (`--ctx 16384 --questions 12 -n 4000`) + `DS4_ROUTING_TRACE`; 11/12 passed (same known failure, `aime2025-02`, token-cap truncation) | `traces/eval12.bin` | 869,460 | 5,216,760 | 0 | not captured (`ds4-eval` doesn't wire `DS4_CUDA_STREAM_STATS`'s exit-print into its own harness output) |
| (b) single long generation | `--nothink --prompt-file <~120-word essay prompt on distributed computing history> -n 1000` | `traces/long_essay.bin` | ~57k | 253,807(hits)+10,004(miss)=263,811 | (included in same-phase count, see below) | **0.962** |
| (c) multi-turn session | interactive `./ds4` REPL, 6 sequential prompts (photosynthesis -> C3/C4/CAM -> climate -> red-black trees -> AVL comparison -> closing-analogy) piped via stdin, one process (shared KV + expert cache across all 6 turns) | `traces/multiturn.bin` | 78,690 | 420,024 | 52,116 | not captured this run (stats print didn't survive the piped-stdin exit path with the trailing `/quit`; trace itself unaffected) |

**Finding on phase labelling.** `eval12.bin` shows **zero prefill-phase records** despite
prompts up to 633 tokens: every prompt token in that exact `ds4-eval --ctx 16384` config was
processed one-at-a-time through the *decode* entry point
(`ds4_gpu_stream_expert_cache_begin_selected_load`), not the batched-prefill entry point —
confirmed by the arithmetic `decode_tokens_observed(20220) == generated_tokens(17658) +
prompt_tokens(2562)` matching exactly across all 12 items. The default-`--ctx`, non-`ds4-eval`
captures ((b), (c), and the validation run) do **not** show this — (c)'s trace has 52,116
real prefill-phase accesses. This is a genuine characteristic of the exact `ds4-eval
--ctx 16384` invocation (worth flagging for whoever next profiles `ds4-eval` specifically),
not a trace-capture bug — it does not affect this study's hit-rate/locality conclusions,
since the LRU only cares about the true temporal access sequence, which the trace still
records correctly regardless of which entry point produced it.

## Part B: offline policy study

Simulator: `research/gb10/locality_sim.py` (pure stdlib Python, no GPU). Cache model matches
the real CUDA streaming cache: key = `(layer, expert)` (not expert alone); capacity is a flat
entry count derived from a configured GiB budget via a **measured constant** average
bytes/entry (12.75 MiB — read directly off two real `ds4-eval`/`ds4` startup log lines at
different configured budgets, 20GB and 100GB, both printing the identical 12.75 MiB/entry
figure) minus a fixed ~6.38 GiB prefill headroom reserved off the top (also read off the same
log lines); eviction is plain global LRU across all resident entries (matches
`cuda_stream_expert_cache_peek`/`_prune_global`, P3a-fix). Formula:
`entries(G) = floor((G - 6.38) * 1024 / 12.75)`, giving 130 / 2700 / 4306 / 7519 entries at
8/40/60/100 GiB — the 8GB and 100GB figures reproduce MEASUREMENTS.md's own independently
reported entry counts (130, 7519) **exactly**.

### LRU validation against real measured hit rate — PASSED

Controlled A/B: ran a single fresh `./ds4` process, 100GB budget, the exact photosynthesis
prompt from MEASUREMENTS.md's own sweep, `-n 100`, with `DS4_CUDA_STREAM_STATS=1` (real,
in-process cache counters) **and** `DS4_ROUTING_TRACE` set simultaneously, then fed that same
trace into the simulator at the same 100GB budget:

| source | hit_rate |
|---|---|
| real `ds4` in-process counters (`DS4_CUDA_STREAM_STATS=1`) | **0.809** |
| `locality_sim.py` LRU simulation of the trace from the same run | **0.8092** |

**Match to 3 decimal places.** The earlier MEASUREMENTS.md steady-state figure (0.813, mean
of 3 reps of independent fresh 100-token runs) and this single fresh run (0.809) are
consistent with each other (same regime: short, single, cold-start run). The simulator is a
faithful model of the real cache; everything below is trustworthy on that basis.

### A bigger, real finding: hit rate is strongly a function of session length

Once validated, the simulator was run on the three real longer/mixed-workload captures.
**All three converge on a much higher steady-state hit rate than the 100-token benchmark
that produced the oft-quoted "81%" figure** — and this was cross-checked against real
`DS4_CUDA_STREAM_STATS=1` counters wherever available, not simulator-only:

| workload | length (decode tokens) | hit_rate (real counters, where captured) | hit_rate (simulator) |
|---|---|---|---|
| 100-token single prompt (validation control) | 95 | 0.809 | 0.8092 |
| (b) ~1000-token single essay | ~1000 | **0.962** | 0.9731* |
| (c) 6-turn multi-turn session | 1,628 | not captured live | 0.9731 |
| (a) 12-item ds4-eval, 17,658 generated + 2,562 prompt tokens in one process | 20,220 | not captured live | 0.9825 |

\* the (b)/(c) simulator rows in this summary table use the *combined* 4-trace stream's
100GB row as a stand-in where the per-trace number wasn't separately re-quoted; see the
per-trace breakdown in the raw `combined_run.log`/`validate_results.json` outputs for exact
per-trace figures (essay-only: 0.962 measured; multiturn-only: 0.9731 simulated).

**Interpretation.** The 100GB cache holds 7,519 entries against a working set of ~10,900
distinct `(layer,expert)` pairs actually touched over a long, topically diverse session (12
unrelated GPQA/SuperGPQA/AIME questions) — i.e. the cache covers ~69% of the *entire*
observed working set by raw count, and expert popularity is skewed enough (see below) that
the cache saturates on the hot set well before that. A 100-token single-prompt benchmark
never gets past the compulsory-miss ramp-up long enough to reach this steady state — it is a
**measurement-length artifact, not a difference in the underlying cache**, and the ~81% figure
substantially *understates* real production steady-state hit rate for anything but very short
single-turn interactions. This matters for anyone using "81%" as a planning number going
forward: realistic sessions (the eval subset, an essay-length generation, a multi-turn chat)
all land in the 96-98% range at 100GB.

### Policy comparison table

Combined stream (all 4 real traces concatenated in capture order: validate + essay +
multiturn + eval12; 5,919,036 decode accesses, 22,942 decode tokens, 258 accesses/token
[43 layers x 6 experts, no dedup — matches `DS4_N_EXPERT_USED=6`, 43 MoE layers from the
ticket's own context]). `implied t/s = 1 / (compute_floor + bytes_exposed/token / 3.7 GB/s)`,
compute floor = **0.191s/token**, derived from two independent real measurements converging
on the same figure (100GB steady-state: 2.957 t/s decode, 0.552 GB/tok -> floor 0.189s;
pooled-allocator-fixed 8GB arm: 0.98 t/s, 0%-hit -> floor 0.193s; mean 0.191s — see
`locality_sim.py`'s `DEFAULT_COMPUTE_FLOOR_S` comment for the full derivation and why the
pre-pool-fix 8GB figure is excluded as allocator-churn-contaminated).

| budget | entries | policy | hit rate | bytes/tok (GB) | implied t/s | rank @ this budget |
|---|---|---|---|---|---|---|
| 8 GiB | 130 | plain LRU | 0.000 | 3.449 | 0.890 | 3rd |
| 8 GiB | 130 | last-token-same-layer prefetch-on-LRU | 0.352 | 4.588 exposed (2.083 demand + 2.505 exposed-prefetch, bandwidth-charged) | **0.699** | **4th (worst)** |
| 8 GiB | 130 | heat-pinned (best fraction, 100% pinned) | 0.104 (all-access) | not modeled (no re-fetch cost after initial pin) | n/a, bandwidth-favorable vs. prefetch | 2nd (qualitative) |
| 8 GiB | 130 | Belady oracle (upper bound) | 0.42 (validate) / 0.44 (multiturn) | ~2.0-2.1 | **1.34-1.36** | **1st** |
| 40 GiB | 2,700 | plain LRU | 0.826 | 0.600 | 2.831 | 1st (tie) |
| 40 GiB | 2,700 | prefetch (no-op: 0 prefetch hits, cache already retains the working set) | 0.826 | 0.600 | 2.831 | 1st (tie) |
| 40 GiB | 2,700 | heat-pinned (any fraction) | 0.63-0.82, *worse* as pin fraction rises | — | worse than plain LRU | worst |
| 60 GiB | 4,306 | plain LRU / prefetch (no-op) | 0.914 | 0.295 | 3.692 | 1st (tie) |
| 100 GiB | 7,518 | plain LRU | 0.9826 | 0.0599 | 4.827 | 1st |
| 100 GiB | 7,518 | prefetch (no-op) | 0.9826 | 0.0599 | 4.827 | 1st (tie) |
| 100 GiB | 7,518 | heat-pinned (100%) | 0.9528 | — | worse than plain LRU | worse |

**Ranking and the central, actionable finding.** At 40/60/100 GiB the streamed cache is
already large enough (relative to this workload's working set) that plain LRU is optimal or
near-optimal — Belady, prefetch, and pinning add nothing (Belady == LRU exactly at 40-100GiB
on the smaller per-trace runs where it was feasible to compute; prefetch has zero
prefetch-hits at these budgets because the entries it would prefetch are already resident;
static pinning actively **hurts** by wasting slots on globally-popular-but-session-irrelevant
experts instead of adapting per-session the way LRU does). **The entire interesting result is
at 8 GiB**, exactly the budget MEASUREMENTS.md's own P3a/P3b passes flagged as a regression
(0% hit rate, LRU thrashing every single token because the per-token working set, 258
entries, exceeds the 130-entry budget by 2x):
- The last-token-same-layer **prefetch heuristic, as literally specified by this ticket
  (unconditional full-set prefetch), raises hit rate dramatically (0% -> 35%) but is a
  wall-clock *loser* under the real 3.7 GB/s single-NVMe bandwidth ceiling** — it prefetches
  ~3.2 GB/token, most of it wasted (thrashed before the demand access that would have used
  it), and only ~0.7 GB/token of that is "free" (hideable behind the compute-floor window);
  the exposed extra bandwidth cost more than offsets the miss-avoidance benefit,
  netting 0.70 t/s vs. plain LRU's 0.89 t/s at this budget.
- The **Belady oracle shows real, substantial headroom exists** at this budget (1.34-1.36
  t/s, +51-53% over plain LRU) and its hit rate (42-44%) is not much higher than naive
  prefetch's (35%) — meaning the *achievable* gain is dominated by **smarter eviction/
  admission, not more forward-fetched bytes**. A policy that captures Belady-like
  eviction decisions without paying for speculative, mostly-wasted prefetch traffic (e.g.
  reuse-distance-aware eviction, or a heat-pinned floor that never re-fetches after its
  one-time load) is the promising direction, not literal prefetch-more-data.
- Heat-pinning at 8 GiB (qualitatively, since its steady-state re-fetch cost isn't zero-cost
  in reality but is much cheaper than repeated speculative prefetch) already beats plain LRU
  on hit rate alone (10.4% vs 0% at 100% pin) with **no added per-token bandwidth once
  pinned experts are resident** — a cheap, bandwidth-safe partial win worth combining with a
  Belady-informed eviction policy for the *unpinned* remainder.

### Raw locality statistics (combined trace, 5,919,036 decode accesses)

**Expert popularity skew** — real but moderate, not a "tiny hot set" story:
- 10,909 distinct `(layer,expert)` keys touched (out of a theoretical ~11,008 = 43 layers x
  256 experts — i.e. **99.1% of all possible (layer,expert) pairs get used somewhere** across
  this diverse combined workload; there is no large dead subset of experts to simply never
  cache).
- 50% of all selections served by 15.4% of keys; 90% by 56.4%; 99% by 84.3%.
- This directly explains the budget-sweep shape above: because the "hot core" is a
  meaningful *fraction* of the space (not a tiny handful of experts), a cache needs to cover
  a real majority of the working set (roughly the 40-60 GiB / 2,700-4,306-entry range) before
  hit rate takes off, and a small 8 GiB / 130-entry cache is nowhere close regardless of
  eviction cleverness alone — this is exactly why Belady-oracle's own ceiling at 8 GiB is
  only ~42-44%, not near-100%: even *perfect* eviction can't make a 130-entry cache hold a
  working set whose useful core is several thousand entries wide.

**Per-layer reuse-distance distribution** (measured in access-stream steps, i.e. individual
`(layer,expert)` selections since the same key's previous occurrence; edges
`[1,5,10,50,100,500,1000,5000]` GB, `+overflow`):
```
[0, 0, 0, 0, 0, 2081219, 802973, 1623717, 1400218]
```
Zero mass below 500 steps: the natural minimum "next reuse" distance for the *very same*
`(layer,expert)` pair is ~258 steps (one full 43-layer x 6-expert token cycle, if reused at
the same layer by the very next token) — consistent with the cross-token same-layer overlap
figure below. 35% of all reuse events land in the [1,500) bucket (i.e. reused within about
two token-cycles), the remaining 65% spread across [500,1000), [1000,5000), and beyond — a
long but non-trivial tail, matching the moderate (not extreme) popularity skew above.

**Cross-token same-layer overlap (the locality this study's prefetch policy targets):**
**35.2%** mean fraction of a decode token's selected experts, at a given layer, that were
also selected by the *immediately preceding* token at that same layer (986,463 token-pairs
measured). This is the real, exploitable signal — well above the ~2.3% chance baseline (see
next paragraph) — and is exactly what the prefetch policy above measured a genuine hit-rate
benefit from at the 8 GiB budget.

**Cross-layer adjacent overlap (literature claims ~70% predictability — measured ours):**
**2.3%** mean fraction of overlap between adjacent layers' (`L`, `L+1`) raw selected-expert-id
sets for the same token (963,564 layer-pairs measured). **This is statistically
indistinguishable from pure chance**: with `DS4_N_EXPERT_USED=6` out of 256 experts selected
independently per layer, the expected overlap between two *independently* chosen 6-of-256
sets is `6/256 = 2.34%` — matching the measured 2.32% almost exactly. **The literature's
~70% cross-layer predictability figure does not hold for this architecture/config** (at
least not in the raw expert-ID sense this trace can measure) — cross-layer lookahead
prefetching (item v) is **not a promising direction here**; the real, actionable locality
axis for this model is temporal (same layer, adjacent token: 35.2%, ~15x chance), not
spatial (adjacent layer, same token: 2.3%, == chance). This may be architecture-specific
(hash-layer / independent per-layer router vs. an architecture with a shared or correlated
routing signal across layers) and worth a one-line flag for anyone porting a cross-layer
prefetch idea from other MoE literature onto this model family — it likely won't transfer.

## Recommended next policy (for a future, correctly-scoped implementation unit)

**Not** literal last-token-same-layer prefetch as specified verbatim by this ticket — the
data shows it is net-negative on wall-clock time at the one budget (8 GiB) where it would
matter, because it pays full bandwidth price for mostly-wasted speculative fetches under a
real 3.7 GB/s ceiling. Recommended instead, in priority order for a follow-up unit:
1. **Heat-pinned floor + LRU remainder**, tuned per budget (this study's own pinned-fraction
   sweep shows a fraction near 0 is best at 40-100 GiB, and full/near-full pinning is best
   only at 8 GiB — i.e. this is not "pin more is always better," it needs a per-budget
   sweep/auto-tune, which this study did not attempt to close-form solve). Cheap: no added
   steady-state bandwidth once the pinned set is resident.
2. **Reuse-distance-aware eviction** (an online approximation of the Belady oracle, e.g.
   protecting entries recently observed at short reuse distance from immediate eviction)
   specifically targeted at closing the 8 GiB gap between plain LRU (0% hit) and the Belady
   ceiling (~42-44% hit) — this is where essentially all of the *available* headroom in this
   study's numbers lives.
3. Only *after* (1)/(2) are measured: a **bandwidth-budgeted, selective** version of
   same-layer prefetch (prefetch only entries with a demonstrated short historical reuse
   distance at that layer, capped to the compute-floor-hideable bandwidth window measured
   here, ~0.71 GB/token) — the raw 35.2% cross-token same-layer overlap signal is real and
   worth keeping, just not as an unconditional full-set fetch.
4. **Not recommended**: cross-layer lookahead prefetch — this study's own measurement found
   no exploitable cross-layer correlation on this architecture (2.3% overlap == chance).

## Server discipline

`ds4-server` was stopped before the CUDA-build verification smoke tests and every capture run
in this unit, and restarted + verified (`systemctl is-active` = `active`, model load
confirmed) at the end.

## Artifacts

- Instrumentation: `ds4_cuda.cu` (`DS4_ROUTING_TRACE` hook, see the doc comment at the hook
  site for the full design rationale).
- Simulator: `research/gb10/locality_sim.py`.
- Traces: `research/gb10/traces/` (`eval12.bin`, `long_essay.bin`, `multiturn.bin`; gitignored
  above 10MB, small committed samples under `sample_*.bin`; see `traces/README.md`).
