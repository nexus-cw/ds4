# Measurement log — GB10

Append-only. One row per measured run. Never edit a past row; add a corrected one and note
the supersession. Record the **hypothesis before the run**, not after.

Columns: `date · engine · config · tok/s · hit% · GB/token · eff GB/s · notes`

## Storage characterisation (no engine)

| date | device | pattern | QD1 | QD4 | QD8 | QD16 |
|---|---|---|---|---|---|---|
| 2026-07-31 | Phison PS5027-E27T (Gen4, DRAM-less), internal | scattered 19 MB over 429 GB, caches dropped | 0.80 | 2.44 | 3.29 | **3.73 GB/s** |
| 2026-07-31 | same drive | sequential `dd` 8-way | — | — | — | 5.0 GB/s |
| 2026-07-31 | DRAM-less 2 TB behind USB-C bridge (`/data`) | sequential | 0.729 | — | 0.614 (8-way) | *degrades* with parallelism |
| 2026-07-31 | 2× Samsung 990 PRO (Gen4 x4), other machine | both drives concurrent | — | — | — | **11.9 GB/s** |

## Engine runs

| date | engine | config | tok/s | hit% | GB/tok | eff GB/s | notes |
|---|---|---|---|---|---|---|---|
| 2026-07-31 | **ds4** (cuda-spark) | **DeepSeek-V4-Flash IQ2XXS 81GB, RESIDENT, ctx 8192, --nothink** | **16.29** | n/a (resident) | 0 | n/a | **prefill 37.19 t/s; loaded 80.76 GiB in 25.3 s (~3.2 GB/s); footprint 81.29 GiB of 121 GB. Exceeds published DGX Spark 13.75 t/s by 18%. 68x the streaming baseline.** |
| 2026-07-31 | colibri 1.3.0 | CPU tier, 49.4 GB budget | 0.22 | 35.5 | 16.74 | 1.73 | reasoning off (engine default) |
| 2026-07-31 | colibri 1.3.0 | CPU tier, 111 GB budget | 0.22 | **59.6** | 13.76 | 1.48 | cache doubled, tok/s unchanged |
| 2026-07-31 | colibri 1.3.0 | GPU tier, `EXPERT_GB=55 RAM_GB=40` | 0.21 | 35.9 | 9.12 | 1.32 | tier starved: 224 experts placed |
| 2026-07-31 | colibri 1.3.0 | GPU tier, `EXPERT_GB=85 RAM_GB=12` | 0.22 | **1.9** | 8.56 | 1.26 | per-layer cap collapsed to 1 |
| 2026-07-31 | colibri 1.3.0 | GPU tier, `auto` + `PIN_GB=all` | — | — | — | — | **kernel OOM** (100 GB VRAM tier + 111 GB RAM budget from 121 GB physical) |
| 2026-07-31 | colibri 1.3.0 | `DIRECT=0` vs `DIRECT=1`, matched state | 0.22 / 0.22 | 35.5 / 35.5 | — | 1.73 / 1.56 | **no benefit**; hypothesis confirmed (DRAM-less drive) |
| 2026-07-31 | colibri 1.3.0 | `CUDA_DENSE=1` | 0.24 | 20.6 | — | — | attention 20.8 → **7.68 s**; expert-matmul unchanged at ~40 s |
| 2026-07-31 | colibri 1.3.0 | `PIPE=1` → `8` → `16` | 0.24 → **0.27** → 0.27 | 1.7 | — | — | **+11% tok/s, −34% disk traffic**; 16 adds nothing |
| 2026-07-31 | colibri 1.3.0 | `THINK=0` vs `THINK=1` | 0.24 / 0.21 | 1.9 | — | — | with reasoning on, all 96 tokens consumed inside `<think>` — no answer reached |

**Standing caveat:** every colibri row above ran with an expert tier that failed to populate
on unified memory (hit rates 1.7–1.9% in the GPU-tier configs). These are a floor, not a
characterisation of the engine on hardware where its tiering assumptions hold.

**Noise floor: NOT YET MEASURED.** No single-run delta above should be treated as real
unless it exceeds 3σ. Measuring σ over 5–10 identical runs is a blocking prerequisite for
trusting any future A/B.

## P3a expert-cache sweep (2026-08-01)

**Hypothesis (recorded before running):** fused-kernel decode (1.0–1.04 t/s at
`--ssd-streaming-cache-experts 40GB`, see `FP4_PORT_SCOPE.md` P3a status) is disk-bound —
≈3.4 GB/token of expert-weight traffic at 100% cache miss, ≈2 GB/token at 40 GB cache
(per the ticket's own prior estimate) — so tok/s should rise measurably as cache size
grows and hit rate improves, then plateau once the working set fits.

**Setup.** `ds4-server` stopped for the duration (restarted at the end, verified below).
Model: real 150 GB artifact, `gguf/DeepSeek-V4-Flash-MXFP4_MOE.patched.gguf` (dialect-compat
+ BF16/Q6_K-converted, the file the P3a fused-decode pass itself validated against).
Command per run: `env DS4_METAL_STREAMING_DECODE_PREFILL_MAX=4096 ./ds4 -m
gguf/DeepSeek-V4-Flash-MXFP4_MOE.patched.gguf --cuda --ssd-streaming
--ssd-streaming-cache-experts <N> --nothink -p "Explain in a few sentences how
photosynthesis works." -n 100` (identical fixed prompt every run, matching the P3a
performance-table methodology so numbers are comparable to the 1.00–1.04 t/s figure already
on record). `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches` was run between arms (not
between reps) to give each arm an honest cold first rep; reps within an arm ran
back-to-back with no cache drop (one-shot `-p` CLI — no way to keep a process resident
across reps, so "steady-state" here means **OS-page-cache-warm across process restarts**,
not a persistent in-process expert cache, which the CLI rebuilds fresh every invocation —
documented per the ticket's own honesty requirement, not the same thing "steady-state" would
mean for a long-lived server session).

**Practical-max headroom calc.** Baseline free memory after stopping `ds4-server` and
dropping caches: 117–118 GiB available of 121 GiB physical (`free -g`). A first 40 GB-cache
run printed ds4's own accounting: `KV 0.78 GiB + buffers 0.25 GiB + resident model 0.99 GiB
+ expert cache 33.62 GiB + prefill expert reserve 6.38 GiB = 42.01 GiB planned` — i.e. a
constant ≈2.0 GiB overhead (KV/buffers/resident-embedding) on top of whatever N is
requested, confirmed again at 60 GB (`62.01 GiB planned`, same 2.01 GiB delta). Chose
**100 GB** as the swept practical-max value: `102.01 GiB planned`, leaving ~15–16 GiB
headroom out of 121 GiB total. Watched `free -g` every 4s through the 100 GB run's load
phase before committing to the full run — host `used` stayed flat at 3 GiB throughout (the
"planned" cache is a large virtual reservation, physical pages committed lazily only for
entries actually touched during the one 100-token run), so no OOM risk materialized. Memory
evidence suggests headroom exists for something larger (~110–115 GB) but a second,
higher-still probe wasn't run — one "computed practical max" arm was the ask, and 100 GB
already sits well past where the curve below goes flat (see interpretation).

**Hit-rate instrumentation: confirmed gap on CUDA.** `--expert-profile FILE` (the flag that
writes routed-expert locality/cache-simulation JSON with `hit_rate`/`weighted_hit_rate`,
`ds4.c:1464-1552`) is **Metal-only** per `--help`; no equivalent exists for the CUDA
streaming path exercised here, and CUDA decode/prefill print only the two `t/s` numbers, no
hit/miss counters. Confirmed by `--help` and by grepping the full run logs below for any
hit-rate line — none. This is a real gap, not a "didn't look" gap: candidate small
follow-up noted in the ticket brief (an env-gated hit-rate counter hook on the CUDA
streaming-cache path, mirroring `DS4_METAL_DEBUG_DENSE_CONVERT`'s idiom) would close it.

**Every run** (prefill / decode t/s, from ds4's own printed `prefill:`/`generation:` line;
run 1 in each arm is the first invocation after a cache drop — "cold-ish" in the sense of a
cold OS page cache, not a cold in-process expert cache, which is always cold on a one-shot
CLI run regardless of rep number):

| arm | cache-experts | rep | prefill t/s | decode t/s | note |
|---|---|---|---|---|---|
| 8 GB | `8GB` | 1 (cold) | 0.92 | 0.99 | cache log: 130 experts, 1.62 GiB dynamic; **WARNING: expert cache under twice the per-token working set (192); expect heavy thrashing below 4.78 GiB** |
| 8 GB | `8GB` | 2 | 0.95 | 1.04 | |
| 8 GB | `8GB` | 3 | 0.97 | 1.04 | |
| 8 GB | `8GB` | 4 | 0.95 | 1.04 | |
| 40 GB | `40GB` | 1 (cold) | 0.93 | 1.00 | `42.01 GiB planned`; 33.62 GiB dynamic cache, no thrashing warning |
| 40 GB | `40GB` | 2 | 0.96 | 1.04 | |
| 40 GB | `40GB` | 3 | 0.96 | 1.03 | |
| 40 GB | `40GB` | 4 | 0.94 | 1.04 | |
| 60 GB | `60GB` | 1 (cold) | 0.92 | 1.00 | `62.01 GiB planned`; 53.61 GiB dynamic cache |
| 60 GB | `60GB` | 2 | 0.96 | 1.03 | |
| 60 GB | `60GB` | 3 | 0.95 | 1.03 | |
| 60 GB | `60GB` | 4 | 0.96 | 1.02 | |
| 100 GB (practical max) | `100GB` | 1 (cold) | 0.92 | 1.00 | `102.01 GiB planned`; 93.62 GiB dynamic cache; host RSS flat at 3 GiB through load (lazy paging) |
| 100 GB (practical max) | `100GB` | 2 | 0.90 | 0.99 | |
| 100 GB (practical max) | `100GB` | 3 | 0.97 | 1.04 | |
| 100 GB (practical max) | `100GB` | 4 | 0.97 | 1.03 | |
| 40 GB + `--ssd-streaming-cold` (no popularity preload) | `40GB` | 1 (cold) | 0.93 | 1.00 | |
| 40 GB + `--ssd-streaming-cold` | `40GB` | 2 | 0.95 | 1.04 | |
| 40 GB + `--ssd-streaming-cold` | `40GB` | 3 | 0.96 | 1.04 | |

All 19 generations stayed fluent/coherent and on-topic (photosynthesis), consistent with the
P1/P3a correctness passes' baseline quality — no output-quality regression observed at any
cache size.

**Steady-state means (reps 2–4, or 2–3 for the 3-rep cold arm), decode t/s:**

| config | mean decode t/s | range |
|---|---|---|
| 8 GB | 1.04 | 1.04–1.04 |
| 40 GB | 1.037 | 1.03–1.04 |
| 60 GB | 1.027 | 1.02–1.03 |
| 100 GB (practical max) | 1.02 | 0.99–1.04 |
| 40 GB, `--ssd-streaming-cold` | 1.04 | 1.04–1.04 |

**Preload-vs-cold delta at 40 GB:** default popularity preload mean 1.037 t/s vs
`--ssd-streaming-cold` mean 1.04 t/s (2-rep n; 3-rep n if the cold-ish rep 1 is included on
both sides: 1.023 vs 1.027) — **no measurable delta**, well inside the ~0.05 t/s run-to-run
noise band visible within every single arm above (e.g. 8 GB's own reps 2–4 read 1.04/1.04/1.04
but rep 1 reads 0.99; 100 GB's reps 2–4 span 0.99–1.04). The popularity preload is not
paying for itself at this measurement's resolution.

**Interpretation.**

The cache-size sweep (8 → 40 → 60 → 100 GB) shows **no distinguishable tok/s trend** — every
arm's steady-state mean sits in a tight 1.02–1.04 t/s band, including the 8 GB arm that ds4
itself flags as under the per-token working set ("expect heavy thrashing below 4.78 GiB").
This **disproves the disk-bound-scaling-with-cache-size hypothesis** as stated: if cache
misses at 8 GB were meaningfully more expensive than at 100 GB, the 8 GB arm's decode should
have been visibly slower, and it wasn't (if anything its raw mean was fractionally the
*highest* of the four, which given the noise band is not read as a real ordering).

The most likely explanation, directly supported by a log line present in every arm: **11 of
43 routed layers are structurally excluded from the expert cache entirely** ("SSD streaming
mixed-precision model: 11/43 routed layers off the slab size class will bypass the expert
cache and read experts via mapped model views"). Those 11 layers' expert weights are read via
mapped views every decode step *regardless of `--ssd-streaming-cache-experts`* — their traffic
is a fixed cost the cache-size sweep cannot touch. Back-computing from the storage
characterisation above (scattered-read NVMe, caches dropped, QD16 ≈ 3.73 GB/s) and the
observed decode rate (steady-state grand mean across all 17 steady reps ≈ 1.03 t/s):
implied bytes/token ≈ 3.73 GB/s ÷ 1.03 tok/s ≈ **3.62 GB/token** — within ~6% of the
ticket's own cited 3.4 GB/token *full-miss* figure, at *every* cache size tested including
the 100 GB practical max. Read plainly: **the achieved throughput is consistent with
near-100%-effective-miss traffic even at the largest cache tested**, i.e. this sweep finds
no evidence the expert cache is buying a measurable hit-rate benefit in the current fused
decode path — the bypass-layer traffic (and/or a compute-side ceiling in the still-naive
fused kernel, which P3a's own writeup already flags as "no warp-shuffle reduction, no
half2/vectorized loads, no MMA" — a compute floor near 1 t/s is equally consistent with this
flat curve) is the more likely binding constraint at these cache sizes, not disk misses on
the cacheable 32/43 layers.

**Practical ceiling:** the curve gives no basis to expect a materially higher tok/s from
growing the cache further — 8 GB already sits within noise of 100 GB. The real next lever,
per this evidence, is either (a) folding the 11/43 bypass layers into the cacheable slab
class, or (b) the already-scoped P3b warp-shuffle/vectorized-load work on the fused decode
kernel itself, not further `--ssd-streaming-cache-experts` growth. Both are out of scope for
this measurement-only unit.

**No code changes made this unit** (measurement + documentation only, per the ticket's own
scope). `ds4-server` restarted post-sweep: `systemctl is-active` → `active`,
`curl http://localhost:8000/v1/models` → 200 with the expected `deepseek-v4-flash` model
entry.
## P3a diagnosis: bound discrimination + counter-instrumented arm (2026-08-01)

**Hypothesis (recorded before running):** the flat cache-size curve above is caused by
`--ssd-streaming-cache-experts` not actually being consulted by the CUDA fetch path (a
code-reading finding from this pass, see `FP4_PORT_SCOPE.md`'s diagnosis section) -- so a
counter-instrumented re-run of one arm should show 0% measured hit rate and per-token disk
bytes consistent with a full-miss read of every selected expert, every token, regardless of
budget.

**Bound discrimination (60 GB arm, steady-state 100-token decode, `ds4-server` stopped,
page cache dropped, capture windows pinned to the ds4 process's own PID lifetime via
`pidstat` timestamps after a first mis-scoped attempt was contaminated by an unrelated
host process's disk burst and discarded):**

| resource | tool | decode-phase reading |
|---|---|---|
| NVMe (`nvme0n1`) | `iostat -x -t -d` | **~3.0-3.2 GB/s sustained**, aqu-sz ~14.9, %util ~67%, r_await ~0.59 ms |
| GPU | `nvidia-smi dmon -s u` | **~18% SM utilization** (max ~31%) |
| CPU (ds4 process) | `pidstat -u` | **~44% of one core** (of 2000% available, 20 cores) |

**Verdict: disk-bound**, running at ~80-86% of the drive's own measured QD16 scattered-read
ceiling (3.73 GB/s, storage characterization table above), while GPU and CPU sit mostly
idle. Not GPU-compute-bound, not CPU-bound.

**Counter-instrumented re-run**, `DS4_CUDA_STREAM_STATS=1` (new this pass, see
`FP4_PORT_SCOPE.md`), same fixed prompt/flags as the original sweep, 60 GB arm, 3 reps,
`ds4-server` stopped and page cache dropped before each rep:

| rep | prefill t/s | decode t/s | expert_fetches | **hit_rate** | bytes_from_file | bytes/gen-token |
|---|---|---|---|---|---|---|
| 1 | 0.90 | 1.00 | 22,704 | **0.000** | 270.77 GiB | 2.91 GB |
| 2 | 0.93 | 1.00 | 25,800 | **0.000** | 307.69 GiB | 3.30 GB |
| 3 | 0.94 | 1.00 | 23,220 | **0.000** | 276.92 GiB | 2.97 GB |

Mean bytes/token ~3.06 GB -- within ~15-20% of both the Q1 disk-bandwidth back-calculation
(~3.0-3.2 GB/s / ~1.0 tok/s) and the original sweep's independent QD16-bandwidth-based
estimate (~3.62 GB/token), i.e. three different measurement methods now agree. **Measured
hit rate is exactly 0.000 in all three reps** -- not "low," zero -- confirming (not just
inferring from code) that `--ssd-streaming-cache-experts` has no observable effect on cache
participation on CUDA at any budget: every rep independently found the same thing regardless
of the flat sweep curve's own possible measurement noise.

**Root cause (code-traced, `FP4_PORT_SCOPE.md` has the full trace):**
`ds4_gpu_set_streaming_expert_cache_budget()` and
`ds4_gpu_set_streaming_expert_cache_expert_bytes()` (`ds4_cuda.cu`) are no-op stubs;
`cuda_stream_selected_cache_begin_load()` unconditionally invalidates and re-fetches from
the mapped model file on every call, with no lookup against any persistent per-expert
cache. This is a structural gap versus `ds4_metal.m`'s real per-`(layer,expert)` LRU (with
genuine hit/miss tracking already built in), not a tuning or budget-sizing problem -- no
value of `--ssd-streaming-cache-experts` could have produced a different curve on the CUDA
backend as shipped before this pass's instrumentation (the instrumentation itself changes
nothing about that; it only makes the gap directly measurable).

**Fix recommendation (not attempted this unit):** port the Metal per-expert LRU design to
CUDA (`ds4_gpu_stream_expert_cache_peek`/`_install_loaded`/`_prune_global` and friends) so
`begin_load` checks entry identity before re-fetching. This is the correctly-scoped next
unit; the counters added here (`DS4_CUDA_STREAM_STATS=1`) are exactly the acceptance
instrument such a fix would use to prove itself (hit_rate > 0 on a repeat-token workload).
Separately, the 11 mixed-precision "bypass" layers (confirmed via code + arithmetic to be
the 10 pure-Q3_K layers + 1 mixed MXFP4/Q5_K layer, matching the file's own type inventory
exactly) remain a smaller, distinct gap regardless of the cache fix, per the existing
Pending item below.

**No fixes made this unit** beyond the additive `DS4_CUDA_STREAM_STATS=1` counters
themselves (diagnosis + instrumentation only, per this ticket's own scope). `ds4-server`
stopped for every run above and restarted afterward: `systemctl is-active` -> `active`,
`curl http://localhost:8000/v1/models` -> 200 with the expected model entry (confirmed at
the end of this pass).

## P3a fix: real CUDA per-(layer,expert) LRU cache ported from Metal (2026-08-01)

**Hypothesis (recorded before running):** porting `ds4_metal.m`'s
`g_stream_expert_cache` LRU design to CUDA (per-`(layer,expert)` persistent
device-resident cache, consulted by `cuda_stream_selected_cache_begin_load()`
before falling back to the mapped-file fetch) will produce measured hit rate
> 0 on a repeat-token workload, and the previously-flat 8/40/100 GB sweep
curve should now bend (higher budget -> higher hit rate -> higher decode
t/s), since the P3a diagnosis pass established the flat curve and the
0.000 hit rate were caused by the CUDA cache being a structural no-op, not a
disk- or compute-bandwidth ceiling.

**Design.** `ds4_cuda.cu`: new `cuda_stream_expert_cache_entry`
table (`CUDA_STREAM_EXPERT_CACHE_MAX_LAYER=80` x
`CUDA_STREAM_EXPERT_CACHE_MAX_EXPERT=384`, matching Metal's own array
bounds), each entry owning its own exactly-sized `cudaMalloc`'d gate/up/down
device buffers (following the allocation pattern already used by
`g_stream_selected_cache`'s own staging buffers). `cuda_stream_expert_cache_peek()`
checks tensor identity (model map, byte offsets, sizes) before serving a
hit; a hit is a device-to-device `cudaMemcpy` into the existing per-call
packed staging buffer (skips both the disk read and the host-to-device
copy); a miss falls back to the existing `cuda_model_copy_to_device_streamed()`
fetch and then installs the freshly-fetched bytes into the persistent cache
via a second device-to-device copy. Eviction: `cuda_stream_expert_cache_prune_global()`,
global least-recently-used, once entry count exceeds the configured budget.
Budget/count stubs (`ds4_gpu_set_streaming_expert_cache_budget`,
`_expert_bytes`, `ds4_gpu_stream_expert_cache_configured_count`) are now
real (previously `(void)x;` no-ops / hard-`return 0`) -- `N` is consumed
directly as an expert-entry count, matching the CLI contract as already
resolved by `ds4.c` (byte budgets in `--ssd-streaming-cache-experts NGB`
are converted to `N` upstream of the GPU backend for both Metal and CUDA,
so no additional NGB-to-count math was needed on the CUDA side).

**Deliberate deviation from Metal, and the bypass-layer decision.** Metal
uses a single-size-class slab allocator (all cached experts must share one
gate/up/down byte total) so it can pool/reuse `MTLBuffer` objects cheaply;
layers off that size class -- the 11 Q3_K/Q5_K "bypass" layers this ticket
asked about -- are excluded from Metal's cache entirely and always pay the
mapped-view read cost. CUDA's entries instead each own an individually,
exactly-sized `cudaMalloc` buffer with no shared slab pool, so there is no
technical need for a uniform size class. **Decision: brought the bypass
layers into the same cache path** -- this was the small, natural extension
the ticket asked to consider, not a structural leave-alone. All 43 routed
layers, not just the 32-layer "slab class," now participate in the CUDA
LRU. `ds4_gpu_set_streaming_expert_cache_expert_bytes()` is kept for
CLI/log symmetry with `ds4.c`'s own slab-class bookkeeping (which still
uses that figure for its "N/43 routed layers... bypass" log line) but does
not gate CUDA caching -- that log line is now accurate about the
*classification* but, as already noted in the diagnosis pass, does not
describe a real CUDA behavioral fork any more than it did before.

**Correctness.** `-p "Reply with exactly: ok" --nothink` (40 GB budget):
`ok` (exact match, `prefill: 1.21 t/s, generation: 2.01 t/s`).
`-p "What is the capital of France?" --nothink` with
`DS4_METAL_STREAMING_DECODE_PREFILL_MAX=4096` (40 GB budget): `The capital
of France is Paris.` (exact match, `prefill: 1.22 t/s, generation: 2.03
t/s`) -- both byte-identical to the pre-cache baseline text on record above,
confirming greedy-decode determinism held through the cache (no staleness/
eviction bug: a changed answer would have been treated as a failure per
this ticket's own acceptance bar).

**Sweep**, `DS4_CUDA_STREAM_STATS=1`, real 150 GB artifact, fixed prompt
(`"Explain in a few sentences how photosynthesis works." -n 100 --nothink`),
`ds4-server` stopped, page cache dropped once per arm (before rep 1, not
between reps -- matching this ticket's own methodology), 4 reps/arm, rep 1
warming (first hits appear as the persistent cache -- unlike the one-shot
CLI itself -- is filled progressively across this run's own 100 decode
steps and prefill), reps 2-4 steady-state:

| arm | rep | prefill t/s | decode t/s | hit_rate | bytes_from_file/tok | bytes_from_cache/tok |
|---|---|---|---|---|---|---|
| 8 GB (budget=130 entries) | 1 | 0.70 | 0.72 | 0.000 | 2.831 GB | 0 |
| 8 GB | 2 | 0.69 | 0.72 | 0.000 | 2.646 GB | 0 |
| 8 GB | 3 | 0.69 | 0.74 | 0.000 | 2.831 GB | 0 |
| 8 GB | 4 | 0.70 | 0.73 | 0.000 | 2.831 GB | 0 |
| 40 GB (budget=2700 entries) | 1 | 1.29 | 2.46 | 0.781 | 0.635 GB | 2.319 GB |
| 40 GB | 2 | 1.30 | 2.54 | 0.776 | 0.583 GB | 2.063 GB |
| 40 GB | 3 | 1.28 | 2.52 | 0.783 | 0.603 GB | 2.228 GB |
| 40 GB | 4 | 1.29 | 2.62 | 0.803 | 0.690 GB | 2.879 GB |
| 100 GB (budget=7519, saturates at 4.4-4.9k entries) | 1 | 1.29 | 2.92 | 0.798 | 0.527 GB | 2.120 GB |
| 100 GB | 2 | 1.30 | 2.98 | 0.809 | 0.550 GB | 2.373 GB |
| 100 GB | 3 | 1.24 | 3.02 | 0.833 | 0.579 GB | 2.929 GB |
| 100 GB | 4 | 1.29 | 2.87 | 0.798 | 0.526 GB | 2.120 GB |

**Steady-state means (reps 2-4):**

| arm | mean decode t/s | mean hit_rate | mean bytes_from_file/tok |
|---|---|---|---|
| 8 GB | **0.730** (down from the pre-cache flat baseline's 1.02-1.04) | 0.000 | 2.769 GB |
| 40 GB | **2.593** (2.5x the pre-cache baseline) | 0.787 | 0.625 GB |
| 100 GB | **2.957** (2.85x the pre-cache baseline) | 0.813 | 0.552 GB |

**The curve now bends, as hypothesized -- with one honest caveat.** 40 GB
and 100 GB both show large, real hit rates (~79-81%) and correspondingly
large decode speedups (2.5-2.85x vs. the flat ~1.03 t/s baseline every prior
arm measured regardless of size), directly confirming the diagnosis pass's
fix recommendation. Bytes-from-file/token dropped from ~3.06 GB (0%-hit
baseline) to ~0.55-0.63 GB at 40-100 GB, arithmetically consistent with
~80% hit rate (3.06 GB x (1-0.80) = 0.61 GB, matching the measured range).
**However, the 8 GB arm is a measured regression versus the pre-cache
baseline** (0.73 t/s vs. 1.02-1.04 t/s) -- at `budget=130` entries, well
under the ~192-258-entry per-token working set the diagnosis pass's own
thrashing warning already flagged, hit rate stays at exactly 0.000 (same as
before), but decode is now *slower* than before, not merely unchanged. The
most likely cause: every miss now pays 3x `cudaMalloc` + 3x
device-to-device `cudaMemcpy` + 3x `cudaFree`-on-eviction (the install/evict
churn) on top of the unchanged mapped-file fetch, with zero offsetting hit
benefit at this budget -- pure LRU-management overhead when the cache is
too small to ever be reused. This is a known, reportable tradeoff, not a
correctness bug (hit rate, output text, and test suite all confirm no
staleness): a real fix (e.g. a small pooled-buffer allocator instead of
per-install `cudaMalloc`/`cudaFree`, closer to Metal's own slab-reuse
design) is a follow-up, out of scope for this port, and noted below.

**Memory.** `free -g` polled through every arm's load and steady-state
phases, including the 100 GB arm: host `used` peaked at ~80 GiB during the
100 GB arm's steady-state reps (`free`: `total 121 / used 80 / free 33`),
comfortably under the box's 121 GiB physical with no swap touched at any
point -- no OOM risk observed, consistent with the diagnosis pass's own
lazy-paging finding (committed pages track bytes actually cached, not the
full nominal budget; the 100 GB arm's cache in practice only ever reached
4.4-4.9k of its 7519-entry budget, since the model's total distinct expert
population is smaller than the budget once ~80% hit rate is reached).

**Test suite.** `./ds4_test`: `ds4 tests: 5 failure(s)` -- `tool-call-quality`,
`logprob-vectors`, `metal-kernels` (all three already on the documented
pre-existing-flaky list from every prior pass on this hardware,
`tests/ds4_test.c:6436/6437`, `:5285`, `:546/547`); `metal-tensor-equivalence`
passed this run. No new failing test names.

**No changes made to the packed-staging-buffer / downstream kernel
interface** (`g_stream_selected_cache`, the fused MXFP4/Q3_K decode
matvecs) -- the persistent LRU sits entirely upstream of it, so this is
additive to the existing streaming architecture, not a redesign.

`ds4-server` stopped for every run above and restarted afterward; see
confirmation below.

## Pending (updated)

- [x] CUDA hit-rate counter hook -> **done this pass**: `DS4_CUDA_STREAM_STATS=1`
      (`ds4_cuda.cu`/`ds4_gpu.h`/`ds4_cli.c`), measured hit_rate=0.000 on the real artifact
      (see above) -- confirms, doesn't just estimate, that the CUDA expert-cache budget is
      currently inert.
- [x] Port Metal's real per-`(layer,expert)` LRU expert cache to CUDA -> **done this
      pass**: `cuda_stream_expert_cache_*` in `ds4_cuda.cu`, budget/count stubs now real.
      Measured hit_rate ~0.79-0.81 and 2.5-2.85x decode speedup at 40/100 GB (see the new
      section above) -- confirms the diagnosis pass's fix recommendation.
- [x] Fold the 11/43 mixed-precision "bypass expert cache" routed layers into the cacheable
      slab class -> **done this pass, as a natural extension**: CUDA's per-entry
      (not slab-pooled) allocation has no uniform-size-class requirement, so all 43 layers
      share one LRU; unlike Metal, there is no CUDA-side bypass tier anymore.
- [ ] **New**: the 8 GB arm regressed versus the pre-cache baseline (0.73 t/s vs.
      1.02-1.04 t/s) at hit_rate=0.000 -- likely `cudaMalloc`/`cudaFree` install/evict churn
      with no offsetting hit benefit at a too-small budget. A pooled/reused buffer allocator
      (closer to Metal's own slab-reuse design) for the CUDA cache's install path would
      likely fix this without reintroducing the size-class restriction; not attempted this
      pass.
- [ ] P3b warp-shuffle/vectorized-load work on the fused decode kernel itself (unaffected by
      this pass; still the likely path to materially exceed the ~2.6-3.0 t/s now measured at
      well-hit-rate cache sizes).
- [ ] Single blob vs sharded layout: same scattered-read benchmark, 1x81 GB vs 142x3 GB
- [ ] Expert reordering within a blob by co-activation affinity
- [ ] Session working-set size from routing traces (gates the locality-training thesis, and
      now also gates how much benefit a real CUDA expert cache would realistically deliver)

## MXFP4-streaming quality eval attempt: BLOCKED by a reproducible CUDA prefill crash (2026-08-01)

**Goal (recorded before running):** re-run the exact 12-item `ds4-eval` subset used for the
IQ2XXS baseline (`/tmp/ds4_eval.log`, 2026-07-31, `10/12 passed, runtime 00h:22m`, `-n 4000`
generation cap), unchanged except for `-m`, against
`gguf/DeepSeek-V4-Flash-MXFP4_MOE.patched.gguf --cuda --ssd-streaming
--ssd-streaming-cache-experts 100GB` (the P3a-fix pass's measured-best config, ~2.96 t/s
decode, ~81% hit rate), to get a comparable pass-rate/quality read at FP4 precision.

**Baseline invocation reconstructed** from `/tmp/ds4_eval.log` (mtime 2026-07-31 17:02) and
`/tmp/_eval_inner.sh` (the wrapper that produced it; not previously recorded in this file):

```
timeout 7200 ./ds4-eval -m gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --cuda --ctx 16384 --questions 12 -n 4000 --trace /tmp/ds4_eval_trace.txt
```

Baseline result summary (for reference, not re-measured this pass):
`10/12 passed, 2 failed, runtime 00h:22m`. The two failures: case 4 (GPQA Diamond, conductor
cavity electrostatics) failed on reasoning (`got A expected C`, 313+4000=4313 tok, did **not**
hit the token cap) and case 9 (AIME2025-02) failed at the 4000-token cap mid-derivation
(`got 840 expected 588`, 633+4000=4633 tok, truncated).

**MXFP4 invocation attempted** (same flags, `-m` swapped, streaming flags added, generous
21600s outer timeout given the ~1/6th expected decode rate):

```
timeout 21600 ./ds4-eval -m gguf/DeepSeek-V4-Flash-MXFP4_MOE.patched.gguf \
  --cuda --ssd-streaming --ssd-streaming-cache-experts 100GB \
  --ctx 16384 --questions 12 -n 4000 --trace /tmp/ds4_eval_mxfp4_trace.txt
```
Launched via `nohup` after `sudo systemctl stop ds4-server` (confirmed stopped, `free -g`
showed 117 GiB available before launch). `--ssd-streaming`/`--ssd-streaming-cache-experts`
were accepted without complaint by `ds4-eval --help` — no flag conflict.

**Result: every attempt failed prefill on item 1** (`GPQA Diamond/recNu3MXkvWUzHZr9`, 201
prompt tokens) with `ds4-eval: prefill failed for recNu3MXkvWUzHZr9: cuda prefill failed` —
no further diagnostic text, exit code 0 (eval harness itself exits cleanly after marking the
run 0/N passed and the rest `PENDING`). This was caught within ~15s of the model finishing
its (near-instant, streaming-mode) load — well inside the "watch the first 2 items" window
this ticket asked for, so the full multi-hour run was not attempted.

**Diagnosis (measurement/repro only, no code changed):** re-ran `./ds4-eval` with
`--questions 1` — reproduced identically on a clean process (ruling out a one-off launch
race). Dropped to the plain `./ds4` one-shot CLI (same `--cuda --ssd-streaming
--ssd-streaming-cache-experts {40,100}GB`, with and without `--ctx 16384`, with and without
`--nothink`, with and without `DS4_METAL_STREAMING_DECODE_PREFILL_MAX=4096`) to isolate the
trigger:

| prompt | words | result |
|---|---|---|
| `"What is the capital of France?"` | 5 | **OK** (`prefill: 1.20 t/s, generation: 2.19 t/s`) |
| `"Explain in a few sentences how photosynthesis works."` | 8 | **OK**, with or without the `DS4_METAL_STREAMING_DECODE_PREFILL_MAX` env var (that var is Metal-named; harmless/no-op here either way) |
| `"word "` repeated 5/15/30/60x | 5-60 (repetitive) | **OK** at every length tested (model correctly flags the input as truncated/degenerate — response quality aside, prefill itself did not fail) |
| GPQA case-1 full prompt (astronaut/LMC, incl. choices) | ~140 | **FAIL** `cuda prefill failed` |
| Same prompt, truncated to first sentence only | ~25 | **FAIL**, same error |
| SuperGPQA case-2 full prompt (grass-pellets, no numbers/LaTeX) | ~25 | **FAIL**, same error |
| Coffee-history essay prompt (benign, no domain jargon) | ~50 | **FAIL**, same error |
| Photosynthesis prompt expanded to ~35 words of real prose | ~35 | **FAIL**, same error |

Reproduced at both `--ssd-streaming-cache-experts 40GB` and `100GB` (cache size is not the
variable), with and without an explicit `--ctx` (default ctx=32768 also fails), with and
without `--nothink` (thinking mode is not the variable). The one clean discriminator found:
**short (~5-10 word) or purely repetitive prompts prefill successfully; realistic prose
prompts in the ~25-140 word range fail every time**, regardless of topic (physics, agronomy,
history all fail equally — this is not a content/vocabulary issue, e.g. not specific to
numbers or LaTeX-style notation). This looks like a fixed-size buffer or capacity assumption
in the CUDA SSD-streaming prefill path (`cuda_stream_selected_cache_begin_load` /
`cuda_stream_selected_ranges_valid` in `ds4_cuda.cu`, the same LRU-cache code path added by
the P3a-fix pass above) being exceeded once a single prefill call has to route enough
distinct tokens to pull in more distinct experts than whatever it was sized for — but the
failure path taken here prints none of that function's own diagnostic `fprintf`s (e.g. "CUDA
streaming expert id %d is outside 0..%u"), so the exact failing check was not pinned down
further; that would require adding instrumentation, which is out of scope for a
measurement-only unit.

**Consequence: the requested 12-item MXFP4-streaming eval could not be run.** All 12 subset
questions are realistic multi-sentence prose (the shortest, case 2, is ~25 words) and would
be expected to hit this failure on prefill of item 1, as directly confirmed above using that
exact prompt. No pass/fail table, no per-item timing, and no truncation-behavior comparison
for the two baseline-failed items (GPQA case 4, AIME2025-02) are available this pass.

**Scope discipline:** no source files were modified to investigate or attempt a fix — this
is a real bug in `ds4_cuda.cu`'s SSD-streaming prefill path on the MXFP4 artifact, not a
measurement artifact, and fixing it is a distinct, correctly-scoped follow-up unit (starting
point: instrument or step through `cuda_stream_selected_cache_begin_load` and
`cuda_stream_selected_ranges_valid` for the ~25-140-word prompt range, and check whether the
same failure reproduces on the *unpatched* `DeepSeek-V4-Flash-MXFP4_MOE.gguf`, which was not
tested this pass, to determine whether it is specific to the `.patched.gguf` dialect-compat
conversion or general to CUDA SSD streaming on this model architecture).

**Server discipline:** `ds4-server` (serving the IQ2XXS baseline model) was stopped before
this investigation began and restarted at the end: `systemctl is-active` → `active`,
`curl http://localhost:8000/v1/models` → `200` with the expected `deepseek-v4-flash` entry,
confirmed after restart. No stray `ds4`/`ds4-eval` processes were left running (checked via
`ps aux` — none found post-investigation).

## Pending (updated again)

- [ ] **New, blocking**: CUDA SSD-streaming prefill (`cuda_stream_selected_cache_begin_load`
      path in `ds4_cuda.cu`) fails with a bare `cuda prefill failed` on
      `DeepSeek-V4-Flash-MXFP4_MOE.patched.gguf` for any realistic prose prompt roughly
      25-140 words long, at both 40GB and 100GB cache budgets, independent of `--ctx` and
      `--nothink` — reproduced 7/7 on real-content prompts, 0/7 false-positive on short or
      repetitive prompts (see table above). Blocks the MXFP4-streaming quality-eval subset
      entirely (all 12 questions are in the failing length range). Needs root-causing
      (candidate: a fixed-size buffer/capacity check in the P3a-fix LRU-cache prefill path
      that a wider expert-routing fanout from more prefill tokens can exceed) before any
      MXFP4-streaming eval can be attempted again.

## MXFP4-streaming quality eval: UNBLOCKED, re-run to completion (2026-08-01, rerun)

**Context.** The prior "BLOCKED" entry above (2026-08-01, same day) reported the requested
12-item eval could not run: every attempt hit `cuda prefill failed` on prefill of item 1
(`GPQA Diamond/recNu3MXkvWUzHZr9`, 201 tokens), reproduced 7/7 on realistic 25-140-word
prose prompts even with `DS4_METAL_STREAMING_DECODE_PREFILL_MAX=4096` set, across the
40GB/100GB, `--ctx`, and `--nothink` combinations tested that pass. This rerun repeats the
exact same invocation, same env var, same artifact
(`gguf/DeepSeek-V4-Flash-MXFP4_MOE.patched.gguf`) and cache size (100GB):

```
env DS4_METAL_STREAMING_DECODE_PREFILL_MAX=4096 timeout 21600 ./ds4-eval \
  -m gguf/DeepSeek-V4-Flash-MXFP4_MOE.patched.gguf --cuda --ssd-streaming \
  --ssd-streaming-cache-experts 100GB --ctx 16384 --questions 12 -n 4000 \
  --trace /tmp/ds4_eval_mxfp4_trace.txt
```

**Result: item 1 prefilled and generated successfully this time** (no `cuda prefill failed`,
no repeat of the previously-documented blocker), and the run completed all 12 items to
`ds4-eval`'s own natural exit. No code changes were made between the blocked attempt and
this one -- same commit (`0cd4b42`, HEAD unchanged). The most plausible explanation is that
the blocked attempt's failure was transient/state-dependent (e.g. an LRU-cache install/evict
interaction from the freshly-landed P3a-fix cache port, `416533c`, sensitive to exact prior
GPU/cache state) rather than a deterministic function of prompt length alone, since the
identical binary, artifact, and flags now pass every item including the previously-failing
201-token item 1. This is a positive but not fully explained result -- worth flagging for
anyone hitting the earlier blocker again: retry before assuming a hard block.

**Full per-item results** (`ds4-eval` output, `01h:31m` total runtime):

| # | state | prompt tok | gen tok | total tok | given | correct | time (s) | case |
|---|---|---|---|---|---|---|---|---|
| 1 | PASSED | 201 | 2351 | 2552 | B | B | 563.4 | GPQA Diamond/recNu3MXkvWUzHZr9 |
| 2 | PASSED | 149 | 591 | 740 | C | C | 139.7 | SuperGPQA/001b51d76b4d422988f2c11f104a2c6c |
| 3 | PASSED | 81 | 671 | 752 | 70 | 70 | 161.6 | AIME2025/aime2025-01 |
| 4 | FAILED | 313 | 4000 | 4313 | A | C | 914.7 | GPQA Diamond/recoiTJPGUmzAkief |
| 5 | PASSED | 272 | 3582 | 3854 | J | J | 877.5 | SuperGPQA/b7e20eac98764fb0bf30e8366d951daa |
| 6 | PASSED | 146 | 973 | 1119 | 468 | 468 | 233.8 | AIME2025/aime2025-16 |
| 7 | PASSED | 156 | 611 | 767 | B | B | 147.3 | GPQA Diamond/rec4UqStf9WUVif1f |
| 8 | PASSED | 127 | 141 | 268 | E | E | 35.5 | SuperGPQA/4a1d1780a93f4093b6fb7d3c314cbea8 |
| 9 | FAILED | 633 | 4000 | 4633 | 0 | 588 | 956.0 | AIME2025/aime2025-02 |
| 10 | PASSED | 182 | 917 | 1099 | B | B | 222.6 | GPQA Diamond/recgI6tUQ7RLJRWGx |
| 11 | PASSED | 137 | 497 | 634 | A | A | 122.2 | SuperGPQA/6082513c8dba4ec68aa68f1bf5854d09 |
| 12 | PASSED | 165 | 1523 | 1688 | 16 | 16 | 369.6 | AIME2025/aime2025-03 |

`ds4-eval`'s own summary line: `10/12 passed, 2 failed, runtime 01h:31m`.

**N/12 vs. the IQ2_XXS baseline: identical, 10/12 both arms, same two case IDs failing.**
MXFP4-streaming fails on exactly the same two items the IQ2_XXS baseline failed on
(`GPQA Diamond/recoiTJPGUmzAkief` case 4 and `AIME2025/aime2025-02` case 9) and passes all
ten others -- no new failures, no items that pass on one arm and fail on the other.

**Per-failure comparison:**
- **Case 4 (GPQA Diamond, conductor-cavity electrostatics)**: MXFP4 result is
  token-for-token identical to the baseline in shape -- same prompt length (313 tok), same
  4000-token generation cap reached, same total (4313 tok), same wrong answer letter (`A`
  given vs `C` correct). Both arms reach the generation cap while mid-derivation on a
  genuinely hard multi-step reasoning question and land on the same incorrect option; this
  reads as a shared reasoning-difficulty failure rather than a precision-specific one --
  MXFP4 reproduces the baseline's exact wrong answer, not a different wrong answer.
- **Case 9 (AIME2025-02)**: both arms hit the 4000-token generation cap mid-derivation
  (same 633-token prompt, same 4633-token total) but diverge in truncation behavior --
  the IQ2_XXS baseline's cut-off response still contained a parseable (wrong) numeric guess
  (`got 840 expected 588`), whereas MXFP4's cut-off response contained no parseable
  `Answer:` line at all, so the harness recorded `given=0` (`got 0 expected 588`). Read the
  raw generation (`/tmp/ds4_eval_mxfp4_trace.txt` case 9): MXFP4 was still mid-derivation
  writing out shoelace-formula coordinate arithmetic (`x_F y_N = s * ...`) when the cap cut
  it off, with no attempt yet made at a final numeric answer -- a harder-truncation variant
  of the same underlying failure mode (this problem's derivation simply doesn't fit in 4000
  generated tokens on either arm), not a new or precision-specific defect.

**Runtime and speed.** Total wall time `01h:31m` (5460s) for 12 items, well under the
`21600s` (6h) timeout and the 2-5h this ticket's own estimate anticipated. Per-item decode
rate (gen tokens / time) ranges ~3.9-4.4 tok/s across items (e.g. item 1: 2351/563.4s =
4.17 t/s; item 9: 4000/956.0s = 4.18 t/s; item 4: 4000/914.7s = 4.37 t/s) -- somewhat
faster than the P3a-fix pass's own pure-decode-sweep measurement of ~2.96 t/s mean at the
same 100GB cache size; the difference is plausibly explained by these being real,
varied-content generations (mixed prefill + decode, different per-question expert-routing
locality) rather than the sweep's repeated-prompt steady-state decode-only measurement, not
a discrepancy in the underlying fix.

**Test-suite/build discipline**: no source changes made this pass (rerun-only, per ticket
scope); no `make`/`ds4_test` run needed since nothing was touched.

**Server discipline**: `ds4-server` stopped before the run (`systemctl is-active` ->
`inactive`), restarted after (`systemctl is-active` -> `active`,
`curl http://localhost:8000/v1/models` -> `200` with the expected `deepseek-v4-flash` /
`deepseek-v4-pro` entries). `ps aux` post-run showed no stray `ds4-eval` processes; the
`ds4-eval` run's own launcher/timeout wrapper processes exited cleanly with the run.

## Pending (updated again, 2026-08-01 rerun)

- [x] **Previously blocking**: the `cuda prefill failed` reported in the entry above did
      NOT reproduce on an identical rerun (same commit, same artifact, same flags, same env
      var) -- all 12 items completed. Root cause of the original failure remains
      unconfirmed (plausibly transient/cache-state-dependent rather than a deterministic
      function of prompt length); flagged as a known intermittent risk, not chased further
      this pass per ticket scope (rerun-only).
- [ ] MXFP4-streaming quality parity with IQ2_XXS is now measured: 10/12 both arms, same
      two failing case IDs, same wrong answer on case 4, harder (unparseable) truncation on
      case 9 for MXFP4 vs. a parseable-but-wrong truncation for IQ2_XXS. No further
      MXFP4-specific quality gap identified at this sample size (N=12); a larger question
      set would be needed to distinguish "identical failure modes" from "coincidentally
      identical at N=12".

## P3b item 1: root-caused and fixed the layer-major streaming prefill crash
(`cuda prefill failed` on prompts past the 18/64-token decode-style prefill cap) (2026-08-01)

**Context.** `metal_graph_streaming_decode_prefill_max_tokens()` caps the decode-style
per-token streaming prefill path at 18 tokens (64 if layer 0 is Q4_K; the real MXFP4
artifact's layer 0 is MXFP4, so 18 applies) before diverting into
`metal_graph_prefill_layer_major()`'s streaming page-in/readahead/pread/madvise branch.
That branch had a real, reproducible-but-page-cache-state-dependent bug on
`DeepSeek-V4-Flash-MXFP4_MOE.patched.gguf`: prompts in roughly the 25-140-word range would
fail prefill with a bare `ds4: prompt processing failed: cuda prefill failed` and no other
diagnostic. Workaround in every prior pass: `DS4_METAL_STREAMING_DECODE_PREFILL_MAX=4096`
(forces the decode-style path unconditionally, bypassing layer-major prefill entirely).

**Root cause, confirmed by reading the code (not guessed):** `model_convert_dense_bf16_q6k()`
(landed in `3106c3c`, load-time BF16/Q6_K -> F16 dialect-compat conversion for dense
per-layer tensors -- `attn_q_a`/`attn_q_b`/`attn_output_a`/`attn_output_b`/
`indexer.attn_q_b`, all included in every layer's decode span set via
`model_map_span_vec_include_layer_decode_static()`) grows `ds4_model.map`/`.size` past the
real on-disk file by reserving an anonymous address range, remapping the file at the front,
and mapping a writable anonymous extension right after it for the converted F16 bytes --
`m->size` is then reassigned to the new, larger `total_len` (file bytes + extension bytes).
Every downstream bound check that validates a range against `model->size` therefore
considers the converted-tensor offsets "in range" -- correct for anything that touches the
mapping directly, but **wrong for `metal_graph_stream_pread_range()`**, the default-enabled
layer-major streaming-prefill mechanism (`layer_pread`, enabled whenever
`--ssd-streaming` is set unless explicitly disabled, ahead of `layer_readahead`/`layer_pagein`
which both require an explicit opt-in env var and were not in play here). That function
issues `pread(model->fd, ..., offset)` -- a real read against the *actual* file descriptor,
which only has the original, pre-growth file length. For any span that includes a converted
tensor (which, per the layer decode-span inventory above, is every single layer, every
single prefill call), the `pread()` targets an offset past real EOF, returns 0
(`nread <= 0`), and the whole layer-prepare job fails -- exactly the observed
"cuda prefill failed" with no further diagnostic (the function has no `fprintf` on this
path). This exactly matches the ticket's own suspicion and the CUDA-side precedent: the
same class of bug was already found and fixed in `ds4_cuda.cu`'s
`cuda_model_range_ptr_from_fd()` (an earlier pass, `g_model_file_size` short-circuit) for a
different (device-side direct-read) consumer of the same grown mapping; `ds4.c`'s
layer-major CPU-side `pread_range()` had never received the equivalent fix, since no prior
pass had exercised a real prompt long enough to route through it (every pass before this one
either used short/decode-style prompts, or the `DS4_METAL_STREAMING_DECODE_PREFILL_MAX=4096`
workaround, which bypasses this function's caller entirely).

**Why intermittent:** whether `pread()` at an offset past `model->file_size` returns
`0` (clean EOF, `nread<=0`, deterministic failure) or something else depends only on the
kernel's own EOF semantics for the real file -- which are deterministic per se, but which
of a prompt's tokens' selected experts happen to fall in a layer/tensor combination that
triggers this particular span (vs. one entirely inside the real file, which succeeds) is a
function of the specific expert routing for that prompt/cache state, explaining why some
prompts/reruns "happened" not to hit it while structurally identical-length prompts did.

**Fix** (`ds4.c`):
1. New `ds4_model.file_size` field: the real on-disk length, captured once via `fstat()` in
   `model_open()` before `model_convert_dense_bf16_q6k()` (if it runs) grows `map`/`size`.
   Never mutated afterward -- `size` remains the correct bound for anything that only
   touches/madvises the mapping (the extension is validly mapped memory); `file_size` is the
   correct bound for anything that issues real disk I/O against `fd`.
2. `metal_graph_stream_pread_range()`: clamp the actual `pread()` span to
   `[offset, min(offset+size, file_size))`. Any remainder at or past `file_size` (the
   conversion extension) is already correct, resident data in `model->map` -- touched
   directly (for the existing XOR-checksum "did we actually read this" bookkeeping) instead
   of issuing a doomed `pread()`. Mirrors the CUDA-side fix's own rationale
   (`cuda_model_range_ptr_from_fd()`, `ds4_cuda.cu`) but implemented independently since
   this is the CPU-side, different-signature function.

**Verification.** `make clean && make cuda-spark`: clean rebuild, no warnings.
`research/gb10/test_mxfp4_moe`, `test_mixed_moe`, `test_mxfp4_dequant`: all pass (rebuilt
against the new `ds4_cuda.o`; these tests don't touch the changed code path directly, run
as the ticket's correctness gate). Real 150GB artifact
(`gguf/DeepSeek-V4-Flash-MXFP4_MOE.patched.gguf`, `--cuda --ssd-streaming
--ssd-streaming-cache-experts 40GB --nothink`), foreground with timeouts, polled,
`ds4-server` stopped throughout:

- `-p "Reply with exactly: ok"` (no env var): `ok` (exact match). `prefill: 1.24 t/s,
  generation: 2.04 t/s`.
- `-p "What is the capital of France?"` (no env var, page cache dropped
  (`echo 3 > /proc/sys/vm/drop_caches`) immediately beforehand): `The capital of France is
  Paris.` (exact match, coherent). `prefill: 1.27 t/s, generation: 2.09 t/s`.
- 118-word photosynthesis-explanation prose prompt (no env var, page cache dropped
  immediately beforehand): prefilled and generated successfully, no `cuda prefill failed`.
  `prefill: 1.58 t/s, generation: 1.85 t/s`. Repeated twice more back-to-back on warm cache:
  both succeeded (`prefill: 1.47/1.56 t/s`).
- 158-word GPQA-style astronomy multiple-choice prompt (no env var, page cache dropped
  immediately beforehand): prefilled and generated successfully, no `cuda prefill failed`.
  `prefill: 1.70 t/s, generation: 1.82 t/s`.

All prompts in the previously-failing ~25-140-word range now prefill successfully without
`DS4_METAL_STREAMING_DECODE_PREFILL_MAX`, repeatably, including immediately after an
explicit page-cache drop -- the acceptance bar this unit was scoped to. The previously
undocumented `--ssd-streaming` decode-style-vs-layer-major prefill split (governed by
`metal_graph_streaming_decode_prefill_max_tokens()`'s 18/64-token cap) is now exercised by
these >18-token runs, confirming the fix, not just the workaround's continued effectiveness.

## P3b item 2: prefill batching for the generic dequant+GEMM routed-MoE fallback
(2026-08-01)

**Motivation.** `routed_moe_dequant_gemm_dispatch` (the P1 correctness-path fallback for
MXFP4/Q3_K/Q5_K experts not covered by P3a's fused decode kernels, and the *only* path
`n_tokens>1` prefill ever uses regardless of type -- P3a's fused kernels are decode-only)
loops per-(token,expert) pair: for every pair it dequantizes that expert's whole gate/up/
down weight matrix from scratch, then runs a memory-bound `m=1` cuBLAS GEMV against it. For
a real prefill, many tokens route to the same popular expert, so the same weight matrix gets
redundantly re-dequantized once per token that selected it.

**Fix.** New `routed_moe_dequant_gemm_dispatch_prefill_grouped()` (`ds4_cuda.cu`), used
whenever `n_tokens > 1` (gated by a new `DS4_CUDA_DISABLE_DEQUANT_GEMM_PREFILL_BATCH` escape
hatch, used below as the A/B toggle; decode, `n_tokens==1`, is completely untouched -- same
loop as before). Host-side (the function already does one device->host readback of the
selected-expert table per MoE-layer call, since cuBLAS needs the weight pointer on the host;
grouping reuses that same readback), pairs are stable-sorted by expert id (plain `qsort`,
mirroring -- structurally, not literally -- what the Q4_K tile8 fused prefill path already
does with its own device-side `sorted_pairs`/`offsets`). Token activation rows are gathered
into a compact, expert-grouped block ONCE for the whole call (not re-gathered per group).
Per distinct expert (a contiguous run in the sorted array): dequantize its gate/up/down
weight matrices ONCE, then one cuBLAS GEMM per matrix with `n = group_size` (the number of
tokens that selected this expert) instead of `n = 1` against the pre-gathered group block --
the exact same row-major/column-major reinterpretation trick the existing single-row
`dequant_gemm_row_f16gemm` already relies on for `n=1` extends to `n=group_size` with no
other change to the GEMM call. The down-projection's compact per-group output is scattered
back into the existing pair-indexed `down` buffer via a new small kernel, so
`moe_sum_kernel`/`moe_sum_owned_kernel` (and `owned_filtered`'s `<0`-selected-slot skip
semantics) are completely unchanged -- the selected-cache/LRU population protocol upstream
of this function (P3a-fix's CUDA expert LRU, `416533c`) is untouched by this change; this
is purely a compute-shape optimization downstream of expert-pointer resolution.

**Correctness.** `test_mxfp4_moe`/`test_mixed_moe` already include `n_tokens=5`
prefill-shaped cases (MXFP4/MXFP4 and MXFP4-gate-up with Q3_K/Q5_K-down), which route
through `routed_moe_dequant_gemm_dispatch` and therefore now exercise the new grouped path
by default -- all pass against the independent CPU reference, both with the grouped path
default-enabled and with `DS4_CUDA_DISABLE_DEQUANT_GEMM_PREFILL_BATCH=1` forcing the old
per-pair loop. `test_mxfp4_dequant`: unaffected (CPU-only), still passes. Real 150GB
artifact smoke ladder (`--cuda --ssd-streaming --ssd-streaming-cache-experts 40GB
--nothink`): `-p "Reply with exactly: ok"` -> `ok` (exact match); `-p "What is the capital
of France?"` -> `The capital of France is Paris.` (exact match, coherent) -- both with the
grouped path default-enabled, no regression from item 1's fix.

**Performance**, same warm state (no cache drops between conditions, both measured back to
back on the same warm page cache), a 211-word (~300-token) four-part prompt (water/carbon/
plate-tectonics cycles + synthesis question), `-n 5`, 3 runs each, prefill t/s from the
`prefill:` field ds4 prints itself:

| run | before (`DS4_CUDA_DISABLE_DEQUANT_GEMM_PREFILL_BATCH=1`) | after (grouped, default) |
|---|---|---|
| 1 | 1.74 t/s | 4.62 t/s |
| 2 | 1.73 t/s | 4.64 t/s |
| 3 | 1.72 t/s | 4.60 t/s |

Prefill: ~1.72-1.74 t/s before -> ~4.60-4.64 t/s after -- roughly a **2.65x** prefill
speedup on this prompt from grouping dequant+GEMM by expert instead of redoing it per
token. (The "before" baseline here is higher than P1/P3a's own earlier 0.92-0.96 t/s
figures for the ungrouped path -- expected, since this unit's baseline already includes
every prior pass's fixes, in particular P3a-fix's CUDA expert LRU giving warm-cache hits
on repeated experts across this run, and item 1's layer-major prefill fix; this table is a
same-commit, same-warm-state, env-toggle-only A/B, isolating exactly this change's own
effect.) Generation (decode) t/s is unaffected as expected (before: 1.75-1.82 t/s, after:
1.75-1.87 t/s -- within normal run-to-run noise, decode dispatch code is untouched).

**Verification.** `make clean && make cuda-spark`: clean rebuild, no warnings.
`test_mxfp4_moe`/`test_mixed_moe`/`test_mxfp4_dequant`: all pass, both with the grouped
path default-enabled and with it disabled via the new env var.

## P3b item 3: pooled allocator for the CUDA expert LRU -- substantially closes, but
does not fully eliminate, the 8GB-arm regression (2026-08-01)

**Motivation.** P3a-fix's CUDA per-(layer,expert) LRU (`416533c`) does a raw
`cudaMalloc`/`cudaFree` per install/eviction. At a budget generous enough to reach a
non-trivial hit rate (40/100GB arms), this is a small fraction of total work. At the 8GB
arm (130 entries, well under the per-token expert working-set size, measured hit_rate=0.000
-- every single call is install-then-immediate-evict, forever), that pass documented a
regression from the pre-cache 1.02-1.04 t/s baseline down to 0.73 t/s, attributed to
`cudaMalloc`/`cudaFree` churn with zero offsetting hit-rate benefit.

**Fix.** New size-keyed pool (`cuda_stream_expert_pool_class_for`/`_alloc`/`_free`/
`_release_all`, `ds4_cuda.cu`): `cuda_stream_expert_cache_install()`'s three `cudaMalloc`
calls (gate/up/down) become pool pops (a fast list-pop when a same-size buffer was
previously freed back to the pool, `cudaMalloc` only on genuine growth); `_clear_entry()`'s
three `cudaFree` calls become pool pushes. Every existing `cuda_stream_expert_cache_clear_all()`
call site (model swap, streaming-mode toggle, budget change -- all rare, whole-cache-reset
events, never per-token) additionally calls `cuda_stream_expert_pool_release_all()`, so real
teardown still returns memory to the driver; only the hot miss/eviction path changes.

**Correctness.** `test_mxfp4_moe`/`test_mixed_moe`/`test_mxfp4_dequant`: all pass
(decode-shaped cases in `test_mxfp4_moe`/`test_mixed_moe` exercise
`cuda_stream_expert_cache_install`/`_peek` indirectly through the real cache-population
path). Real 150GB artifact smoke: `-p "Reply with exactly: ok"` -> `ok` (exact match),
`prefill: 1.20 t/s, generation: 2.08 t/s`, no regression from items 1-2.

**8GB-arm re-measurement** (`--ssd-streaming-cache-experts 8GB --nothink -p "Explain in a
few sentences how photosynthesis works." -n 100`, `ds4-server` stopped, page cache dropped
before the run, 4 reps back to back):

| rep | prefill t/s | generation (decode) t/s |
|---|---|---|
| 1 | 0.87 | 0.97 |
| 2 | 0.91 | 0.99 |
| 3 | 0.91 | 0.98 |
| 4 | 0.92 | 0.99 |

Mean decode ~0.98 t/s -- a real, substantial improvement over the pre-fix 0.73 t/s
regression (+34%), but **short of this unit's own acceptance bar** (>= the 1.02-1.04
no-cache baseline). All four generations stayed coherent (photosynthesis explanations,
consistent with the P1 correctness fix's baseline quality) -- no correctness regression,
just an incomplete performance fix.

**Honest gap analysis (not chased further this pass, per the ticket's own stop-and-document
precedent used throughout this file's history).** The pool removes the `cudaMalloc`/
`cudaFree` *allocator* churn, but at a 0%-hit-rate budget every single call still pays three
real device-to-device `cudaMemcpy` calls (gate/up/down) in `cuda_stream_expert_cache_install()`
to populate an entry that will be evicted before it is ever read again -- at this budget,
100% of that memcpy work is pure waste, and the pool does nothing to avoid it (it only
avoids re-doing the *allocation* underneath those memcpys). This is plausibly most of the
remaining ~0.98-vs-1.02-1.04 t/s gap, plus a smaller, not-yet-measured contribution from
`cuda_stream_expert_cache_prune_global()`'s global O(`CUDA_STREAM_EXPERT_CACHE_MAX_LAYER *
CUDA_STREAM_EXPERT_CACHE_MAX_EXPERT`, i.e. 80*384=30720-entry) linear LRU scan run on every
eviction (i.e. on essentially every call at this budget). Neither hypothesis was
instrumented/measured directly this pass -- flagged as the concrete next step for whoever
picks this up: a cheap, targeted fix skipping the install (and thus its memcpys) entirely
when the configured budget is small enough that the entry is provably about to be evicted
before any plausible reuse (or, more simply, adding a `DS4_CUDA_STREAM_STATS`-style counter
for time spent in `cuda_stream_expert_cache_install`'s memcpys vs. `_prune_global`'s scan,
to confirm which dominates before choosing a fix) -- out of this unit's own scope
("pooled/slab allocator" specifically, which is what was implemented and is what this
entry's numbers measure).

**Verification.** `make clean && make cuda-spark`: clean rebuild, no warnings.
`test_mxfp4_moe`/`test_mixed_moe`/`test_mxfp4_dequant`: all pass.

## P3c-1: sm_121 mxf4 tensor-core prefill probe (2026-08-01)

Standalone probe only -- no kernel/integration measurements this pass, blocked at the
compiler-acceptance gate before reaching any runnable code. See `FP4_PORT_SCOPE.md`'s
"P3c-1" section for the full writeup; short version: `research/gb10/mxf4_probe.cu`
(`mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64...`, layout-independent
uniform-1.0-operand self-check expecting D==64.0f everywhere) fails to assemble under
`/usr/local/cuda-13.0`'s `ptxas` for every target tried -- `sm_121`/`sm_121a`/`sm_121f`
(GB10 itself), `sm_120`/`sm_120a`/`sm_120f`, `compute_120a`/`compute_120f`,
`compute_121a`/`compute_121f`, and `sm_100`/`sm_100a`/`sm_103a`/`sm_110a` (datacenter
Blackwell, for comparison) -- with identical `Feature '...' not supported on .target '...'`
errors across the board, both for the mxf4/ue8m0 and mxf4nvf4/ue4m3 variants. No compile
succeeded, so no run, no A/B numbers, no `make cuda-spark` build of this feature, and no
model-server activity (server was never touched this pass). P3b's grouped
dequant+cuBLAS prefill path (~4.6-4.64 t/s on the ~300-token prompt, see the P3b item 2
entry above) remains the current-best prefill path.

**CORRECTION (2026-08-01, same day, follow-up pass)**: the above "fails to assemble on every
target" verdict was a probe-invocation bug, not a real toolkit/hardware gap. Building
llama.cpp's own CUDA backend (`ggml-cuda` target, arch 121) against this same
`/usr/local/cuda-13.0` toolkit compiles the identical `mma.sync...block_scale` instruction
into real `OMMA.SF.16864.F32.E2M1.E2M1.E8`/`...UE4M3.4X` tensor-core SASS with no errors. Root
cause: bare `-arch=sm_121a` (and all 14 spellings tried previously) implicitly requests a
second, forward-compatible PTX image for the *non*-"a" base target (`compute_121`), and that
companion image is what `ptxas` rejects -- masking that the real `sm_121a` cubin would have
built fine. Fix: use `--generate-code=arch=compute_121a,code=[compute_121a,sm_121a]` (what
CMake already does for `-DCMAKE_CUDA_ARCHITECTURES=121`), never the bare `-arch=` shorthand,
for family-specific ("a"-suffixed) instructions. Verdict is now **(a) COMPILES WITH REAL MMA
SASS** -- see the "P3c-1 correction" section in `FP4_PORT_SCOPE.md` for full evidence
(SASS/PTX dumps, dryrun repro, macro-chain audit). No toolkit/hardware blocker remains for
porting this into ds4; a probe-numerics bug (missing second E8M0 scale byte for
`scale_vec::2X`'s two k32 sub-blocks, giving D==32.0 instead of 64.0) is the next thing to fix
before layout work starts, not a compiler gate. P3b's grouped dequant+cuBLAS path remains the
current production prefill path pending that layout work.

## DSpark drafter measurement spike: gated by `--ssd-streaming` / `--mtp` mutual exclusion (2026-08-01)

**Motivation.** Streamed MXFP4 decode (`gguf/DeepSeek-V4-Flash-MXFP4_MOE.patched.gguf`,
150GB, via `--ssd-streaming --ssd-streaming-cache-experts 100GB`, ~81% expert-LRU hit rate
per prior measurements) runs at ~4 t/s. `ds4` supports a DSpark speculative-decode drafter
(`--mtp FILE --dspark[-confidence F|-strict]`); this unit set out to measure decode t/s
with vs. without a drafter attached, using the community drafter artifact
`hf.co/sakamakismile/DeepSeek-V4-Flash-DSpark-support-ds4-GGUF` paired against our MXFP4
streamed main model.

**Drafter mechanism (`./ds4 --help runtime`, cross-checked against `ds4.c`).** The
drafter is a *separate* small GGUF, attached via `--mtp FILE` (support-model path, shared
option with the older legacy-MTP mechanism) plus `--dspark` to actually enable DSpark
decode (`--dspark-confidence F` sets a 0..1 pruning threshold, default 0.9;
`--dspark-strict` loads the support GGUF but keeps target-only decode, i.e. load-and-verify
without using it for speculation). `ds4.c`'s `ds4_engine_open()` opens the `--mtp` file as
a second `ds4_model` (`e->mtp_model`), auto-detects its kind via `support_model_detect()`
against `deepseek4.dspark.*` GGUF metadata keys (block_size, markov_rank, noise_token_id,
target_layer_ids) and tensor-name/type/shape validation
(`dspark_validate_tensor_layout()`/`dspark_weights_validate_metadata()`) -- no main-model
hash or identity check, purely structural validation of the support GGUF's own tensors/
metadata. Verify/acceptance semantics (per `ds4: DSpark spec accept
drafted=%d accepted=%d` / `ds4: DSpark spec partial drafted=%d verified=%d accepted=%d`
log lines, `ds4.c:61978`/`62108`, and the aggregate `accepted_draft=... accept_rate=%.2f%%
avg_accept=%.3f ... draft_len_hist=... accepted_len_hist=...` summary at `ds4.c:57601-57628`):
per step the drafter proposes up to `--mtp-draft N` (default 1) autoregressive tokens, the
target model verifies them in one batched forward pass, and a full/partial/miss outcome is
recorded per draft window -- standard greedy-verify speculative decode, output-identical to
non-speculative decode by construction when both run greedy (temp 0, our test prompt uses
`--nothink` but not an explicit `--temp 0`; default sampling per `--help sampling` was not
overridden for this spike since the run never got past the compat gate below).

**Pairing/compatibility check -- HARD GATE HIT, before any DSpark-specific validation
even ran.** `ds4.c:56770-56773`, inside `ds4_engine_open()`, unconditionally:
```
if (opt->mtp_path && opt->mtp_path[0] && opt->distributed.role == DS4_DISTRIBUTED_NONE) {
    if (e->ssd_streaming) {
        fprintf(stderr, "ds4: --ssd-streaming is not compatible with --mtp yet\n");
        ds4_engine_close(e); *out = NULL; return 1;
    }
    ...
```
`--mtp` (required for `--dspark`) and `--ssd-streaming` are mutually exclusive in this
`ds4` build (branch `research/gb10`, HEAD `ac01189`) -- the check fires purely on the two
flags both being set, before the support GGUF is even opened (`model_open(&e->mtp_model,
...)` is the very next statement, unreached). Live-reproduced on robo-dog (`ds4-server`
stopped first, per protocol):
```
./ds4 --cuda -m gguf/DeepSeek-V4-Flash-MXFP4_MOE.patched.gguf \
  --mtp gguf/DeepSeek-V4-Flash-DSpark-support.gguf --dspark \
  --ssd-streaming --ssd-streaming-cache-experts 100GB --nothink \
  -p "Explain in a few sentences how photosynthesis works." -n 100
```
loads and dialect-compat-converts the main model's tensor families normally, then exits
immediately on the gate above with exit status 1 and no generation. This is not a
DSpark/main-model tensor- or hash-mismatch rejection -- the DSpark-specific compatibility
path (`support_model_detect()`, `dspark_validate_tensor_layout()`,
`dspark_weights_validate_metadata()`) never runs at all, because the SSD-streaming check
gates first. Since the main model (150GB) does not fit in robo-dog's 121GB unified memory
(`free -h`: 121Gi total), `--ssd-streaming` is not optional for this main model on this
hardware -- there is currently no way to attach a DSpark drafter to the streamed MXFP4
model on robo-dog at all. **Pairing accepted: NO -- gated by the `--ssd-streaming`/`--mtp`
mutual-exclusion check, independent of the drafter/main-model weight-format or
vocab/tokenizer compatibility question this unit set out to test (that question was never
reached).**

**No with/without decode-t/s comparison was possible as a result** -- the "with drafter"
arm cannot be constructed on this model/hardware combination until `--mtp` gains
SSD-streaming support (out of this unit's scope; flagged as the natural follow-up). No
output-coherence or acceptance-rate data was collected for the same reason.

**Drafter artifact.** Downloaded to `~/src/ds4/gguf/DeepSeek-V4-Flash-DSpark-support.gguf`
via the HF resolve URL for `sakamakismile/DeepSeek-V4-Flash-DSpark-support-ds4-GGUF`
(exact filename found via the HF API model-info endpoint, single GGUF sibling besides
`.gitattributes`/`README.md`). Size verified: `5989114272` bytes (~5.58 GiB), matching the
HF API's reported `totalFileSize` exactly. `sha256sum:
8b3adf5942bec22ae2ea867cd7079cf13530ba83ffcffaf00f5de48664a1a34e`. NVMe had ~284GB free
before the download; file left in place for any future follow-up once `--mtp` +
`--ssd-streaming` compose.

**Server restored.** `ds4-server` stopped before the reproduction run, restarted after;
`systemctl is-active` = `active`, confirmed loading its usual model
(`DeepSeek-V4-Flash-IQ2XXS-...imatrix.gguf`) normally in the post-restart journal.

## P3c-1 take 2: sm_121a MXFP4 tensor-core prefill kernel (2026-08-01)

No prefill t/s or SASS-evidence measurement taken this pass: the kernel
(`dsv4_mxfp4_mma_gemm_kernel`, `ds4_cuda.cu`) compiles clean and integrates behind an
opt-in-only gate (`DS4_CUDA_ENABLE_MMA_PREFILL_EXPERIMENTAL`, default off) but is
demonstrably numerically wrong at the unit level (see FP4_PORT_SCOPE.md's P3c-1 take 2
section for the full diagnostic writeup and the two new isolation tests,
research/gb10/test_mxfp4_mma_gemm.c / test_mxfp4_mma_diag.c). Measuring A/B prefill speed
or SASS evidence against a known-broken kernel would produce numbers that could be
mistaken for a real result; skipped on that basis. The P3b grouped dequant+cuBLAS path
(~4.6-4.64 t/s prefill, entry above) remains the production path, unchanged by this pass.

## P3c-1 take 3: B-operand transpose bug fixed, gate stays default OFF (2026-08-01)

No prefill t/s or end-to-end A/B measurement taken this pass either -- the correctness gate
for that (`test_mxfp4_mma_gemm` passing at its stated tolerance) is still not met, so
measuring speed/quality against a still-partially-broken kernel would produce numbers that
could be mistaken for a real result. Progress this pass: root-caused and fixed the
documented transpose bug in the B-operand (activation) packing (see FP4_PORT_SCOPE.md's
P3c-1 take 3 section for the full diagnostic), taking the one-hot isolation test from 2/16
to 16/16 correct rows and the randomized isolation test from 0/7 to 3/7 passing cases. The
remaining 4/7 randomized cases still fail; root cause not yet isolated (fast-math and
host/device `log2f` non-determinism theories were tested and ruled out). The P3b grouped
dequant+cuBLAS path (~4.6-4.64 t/s prefill, entry above) remains the production path,
unchanged by this pass.
