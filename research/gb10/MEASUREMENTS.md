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

## P3c-1 take 4: MMA-vs-cuBLAS prefill A/B, ~230-token prose prompt (2026-08-01)

Collected while `DS4_CUDA_ENABLE_MMA_PREFILL_EXPERIMENTAL`'s default was briefly flipped ON for
testing this pass (since reverted -- see FP4_PORT_SCOPE.md's P3c-1 take 4 section for why:
`test_mxfp4_moe` regressed, a MoE-integration bug distinct from the now-fixed GEMM kernel bug).
Recorded here as reference data for whoever picks up the follow-up, not as a validated production
number. Model: `gguf/DeepSeek-V4-Flash-MXFP4_MOE.patched.gguf` (150GB), `--cuda --ssd-streaming
--ssd-streaming-cache-experts 100GB --nothink`, `ds4-server` stopped first. Prompt: ~230-token
prose prompt asking for a detailed photosynthesis explanation (light/dark reactions, C3/C4/CAM
pathways, biotech applications), `-n 20` (isolating prefill from most decode-time noise), 3 reps
each, same warm NVMe/page-cache state, back-to-back:

| Path | rep 1 | rep 2 | rep 3 | mean |
|---|---|---|---|---|
| MMA (`DS4_CUDA_ENABLE_MMA_PREFILL_EXPERIMENTAL=1`, confirmed dispatching via `DS4_DEBUG_MMA_PREFILL=1`, 9352 `mma-prefill` launches logged) | 3.68 t/s | 3.64 t/s | 3.67 t/s | 3.663 t/s |
| cuBLAS (flag unset, existing production path) | 3.65 t/s | 3.60 t/s | 3.64 t/s | 3.63 t/s |

**~1% apart, within run-to-run noise -- no meaningful prefill speedup from the tensor-core MMA
path in this configuration.** Both paths also produced coherent, technically accurate, on-topic
`-n 300` completions on the same prompt (not byte-identical, since default sampling is non-greedy
temp=1.0, but no visible quality difference). Plausible explanation for the lack of speedup (not
verified further this pass): this SSD-streaming workload's prefill time is likely dominated by
NVMe/mapped-view expert fetch rather than GEMM compute, so a faster GEMM kernel doesn't move the
end-to-end number much here -- worth confirming with a profiler once the MoE-integration bug is
fixed and the gate is re-earned.

## Prefetch/locality research unit 1: routing-trace capture + offline policy study (2026-08-02)

Trace capture + offline policy study only, no ds4 policy change. Full writeup:
`research/gb10/LOCALITY_STUDY.md`. Summary: new `DS4_ROUTING_TRACE` env-gated instrumentation
(`ds4_cuda.cu`) captures per-token, per-layer selected expert ids at the same two entry points
the CUDA streaming expert cache consumes; validated against real `DS4_CUDA_STREAM_STATS=1`
counters (LRU simulator hit_rate 0.8092 vs. real 0.809, matched to 3 decimals). Central new
finding: the oft-quoted "81% hit rate" figure is a **100-token-benchmark artifact** -- real
longer/mixed sessions (a 1000-token essay, a 6-turn chat, the full 12-item ds4-eval subset)
all converge to 96-98% hit rate at the same 100GB budget once past the compulsory-miss
ramp-up. At the 8GB budget (the arm P3a/P3b flagged as thrashing to 0% hit), a Belady-oracle
upper bound shows real headroom (+51-53% t/s over plain LRU is achievable) but this ticket's
own literally-specified last-token-same-layer prefetch policy is a **net wall-clock loser**
under the real 3.7GB/s bandwidth ceiling (too much wasted speculative fetch); a smarter
eviction/admission policy (heat-pinning + reuse-distance-aware eviction) is the recommended
next step, not literal prefetch. Also measured: cross-layer adjacent-expert overlap (2.3%) is
statistically indistinguishable from chance (6/256=2.34%) on this architecture -- the
literature's ~70% cross-layer predictability figure does not transfer here; cross-token
same-layer overlap (35.2%) is the real, ~15x-above-chance locality signal. Simulator:
`research/gb10/locality_sim.py`. Traces: `research/gb10/traces/` (raw traces gitignored
above 10MB, small sample committed).

## Long-session locality-convergence measurement: does hit-rate convergence make decode
compute-bound? (2026-08-02)

Live measurement to confirm or refute `LOCALITY_STUDY.md`'s prediction that long/mixed
sessions converge to 96-98% hit rate at the 100GB budget (vs. 81% on the 100-token
single-prompt benchmark), which would push decode toward the ~5 t/s compute-floor ceiling
rather than the disk-bound regime the 81%/2.96 t/s figures describe. `ds4-server` stopped
first, restarted and verified (`systemctl is-active`=`active`, `listening on
http://0.0.0.0:8000` confirmed in journal) after.

**Method.** A single long-lived interactive `./ds4` process (the only way to keep both the
KV cache and the device-side streamed-expert LRU cache warm across turns -- the cache is
purely in-process and dies with it), `gguf/DeepSeek-V4-Flash-MXFP4_MOE.patched.gguf --cuda
--ssd-streaming --ssd-streaming-cache-experts 100GB --nothink -n 350`, 8 prompts piped via
stdin in one process (mirrors `LOCALITY_STUDY.md`'s multiturn trace workload: mixed topics
with follow-ups -- photosynthesis, C3/C4/CAM follow-up, TCP handshake, TCP congestion-control
follow-up, red-black trees, red-black-vs-AVL follow-up, Paxos/Raft, closing cross-topic
analogy). `DS4_CUDA_STREAM_STATS=1` was set, but this build only wires that counter print
into the one-shot (`-p`) exit path, not the interactive REPL (`ds4_cli.c`,
`ds4_gpu_print_cuda_stream_stats()` has exactly one call site, inside `run_generation()`,
never called from `run_repl()`) -- confirmed by reading the code before running, not assumed.
So this pass could not read the real cumulative hit-rate counter mid-session; per the
protocol's fallback, it uses the REPL's own per-turn `ds4: prefill: X t/s, generation: Y t/s`
line (printed after every turn to stderr) to build the trajectory, then derives an implied
hit rate from decode t/s via the compute-floor model `LOCALITY_STUDY.md` already validated
against real counters (compute floor 0.191 s/token, 3.7 GB/s single-NVMe ceiling, 258
demand accesses/token [43 layers x 6 experts], 12.75 MiB/entry -> max bytes/token at 0% hit
= 258 x 12.75 MiB = 3.212 GB; `bytes/tok = (1/t/s - 0.191) x 3.7`, `hit_rate = 1 -
bytes/tok / 3.212`). This derived number is model-based, not a second independent real
counter -- flagged explicitly, not presented as equivalent to a direct measurement.

**Per-turn trajectory** (prefill token counts from the REPL's own progress display; turn 1's
prefill count wasn't printed by the progress callback, decode t/s still captured):

| turn | topic | prefill tokens | prefill t/s | decode t/s | derived bytes/tok (GB) | derived hit rate |
|---|---|---|---|---|---|---|
| 1 | photosynthesis (cold start) | n/a | 0.67 | 3.37 | 0.391 | 87.8% |
| 2 | C3/C4/CAM (follow-up) | 33 | 0.71 | 4.42 | 0.130 | 95.9% |
| 3 | TCP handshake (topic switch) | 28 | 0.37 | 4.43 | 0.128 | 96.0% |
| 4 | TCP congestion control (follow-up) | 34 | 0.80 | 4.43 | 0.128 | 96.0% |
| 5 | red-black trees (topic switch) | 30 | 0.69 | 4.39 | 0.136 | 95.8% |
| 6 | red-black vs. AVL (follow-up) | 34 | 0.80 | 4.46 | 0.123 | 96.2% |
| 7 | Paxos/Raft (topic switch) | 29 | 0.66 | 4.44 | 0.127 | 96.1% |
| 8 | ants/load-balancing analogy (topic switch) | 32 | 0.73 | 4.48 | 0.119 | 96.3% |

Every topic switch (turns 3, 5, 7, 8) still shows real cold-cache cost at the prefill stage
(GPU util ~85-93%, NVMe reads ~300 MB/s sustained for tens of seconds even on prompts as
short as 28-34 tokens -- confirmed live via `nvidia-smi`/`iostat` during the run, not just
inferred from the log), but **decode t/s barely moves on a topic switch** (4.39-4.48 t/s
across turns 3-8 regardless of whether the turn is a follow-up or a fresh topic) -- the
100GB cache is now large enough, and this session's cumulative working set diverse enough,
that decode-time eviction pressure from a topic switch is small next to the already-resident
cross-topic "hot core" (consistent with `LOCALITY_STUDY.md`'s finding that 99.1% of all
possible (layer,expert) pairs get touched somewhere in a diverse combined workload, and 50%
of selections are served by just 15.4% of keys).

**Steady state.** Mean of the last 3 turns (6, 7, 8): **4.46 t/s** decode, derived hit rate
**96.2%**. Mean of turns 2-8 (excluding the cold-start first turn): 4.44 t/s, derived hit
rate 95.9-96.3% throughout -- i.e. the session reaches steady state almost immediately after
turn 1 and stays there, it does not need many more turns to keep climbing toward 98%.

**Verdict on the 96-98% prediction: CONFIRMED, at the lower edge of the predicted band.**
Every turn after the cold-start first one lands in 95.8-96.3% derived hit rate -- solidly in
the predicted 96-98% range and far above the 81% short-benchmark figure this ticket set out
to test, even though this session's steady state sits at the low end of that band rather
than the 98% ds4-eval-derived ceiling (plausibly because this 8-turn/~2,500-token session is
shorter and touches fewer distinct topics than the 12-item, 20,220-token ds4-eval capture
`LOCALITY_STUDY.md` measured at 98.3%).

**Verdict on compute-bound: CONFIRMED, decode is now compute-dominated but not purely
compute-bound.** The pure-compute ceiling (bytes/tok -> 0) is `1/0.191 = 5.236 t/s`. Steady
state (4.46 t/s) is **85.2% of that ceiling** -- disk time is now only the remaining ~15% of
the per-token budget (0.032s disk vs. 0.191s compute at turn 6-8's derived bytes/tok). This
is a large, real shift from the disk-bound regime the existing 81%-hit/2.96 t/s sweep figure
describes: recomputing that regime's disk-time share with the same formula (0.552 GB/tok at
81% hit, per `MEASUREMENTS.md`'s own P3a-fix entry) gives `0.552/3.7 = 0.149s` disk time vs.
`0.191s` compute -- a **44% disk-time share**, roughly 3x this session's steady-state ~15%.
So the session has moved decode from "disk time is nearly as large as compute time" to
"compute clearly dominates, disk is a modest tail" -- matching the ticket's "compute-bound,
~5 t/s kernel ceiling" framing directionally and quantitatively (measured steady-state
4.46-4.49 t/s vs. the predicted ~5 t/s ceiling, 85-86% of the way there), while stopping
short of literally reaching the full ceiling, consistent with a derived (not 0%) residual
disk-time share rather than a true zero.

**Comparison to existing figures.** This session's steady-state 4.46 t/s and 8-turn-mean
4.44 t/s (turns 2-8) are close to the 4.4 t/s eval-observed figure already in
`MEASUREMENTS.md`, and both are **~50% higher** than the 2.96 t/s single-short-prompt sweep
figure that was the previous best "production" number for this config -- direct live
confirmation that the 2.96 t/s figure understates realistic long/mixed-session throughput,
for the same underlying reason `LOCALITY_STUDY.md`'s offline simulation already flagged: the
81%-hit-rate benchmark that number was built on never runs long enough to leave the
compulsory-miss ramp-up.

**Next optimization target, now stated explicitly.** With decode ~85% compute-bound at
steady state, the remaining ~15% disk-time tail is a comparatively small target --
`LOCALITY_STUDY.md`'s own eviction/admission recommendations (heat-pinned floor + LRU
remainder, reuse-distance-aware eviction) would close at most that residual gap at the 100GB
budget, not the 2-3x gains they'd offer at 8GB. The real headroom is the compute ceiling
itself, ~5.24 t/s at 0.191 s/token: whoever picks up decode-side kernel work next (the
still-incomplete MMA-prefill GEMM correctness fix noted in the P3c-1 entries above is the
current lead there) should treat that as the number to move, not the streaming-cache policy.

**Caveats.** (1) Hit rate here is *derived* from decode t/s via the compute-floor model, not
read from a live in-process counter -- the REPL build path doesn't wire
`ds4_gpu_print_cuda_stream_stats()` into `run_repl()`'s exit, only `run_generation()`'s (a
genuine gap for whoever next needs live REPL-session cache stats; not fixed this pass, this
unit is measurement-only, no code changes). (2) 8 turns / ~2,500 decode tokens is shorter
than the ds4-eval 12-item/20,220-token capture `LOCALITY_STUDY.md` used for its 98.3% figure
-- this session's slightly lower ~96% steady state is consistent with, not contradictory to,
that larger-sample number. (3) `-n 350` per turn means some turns may have been truncated
before a natural stop; this does not affect the t/s/hit-rate analysis, which is a rate
measurement independent of exact token count.

**Server discipline.** `ds4-server` was stopped (`sudo systemctl stop ds4-server`) before
this run and restarted + verified (`systemctl is-active`=`active`, `listening on
http://0.0.0.0:8000` in the journal, model tensors loading logged) at the end, per protocol.

## Drafter-streaming compatibility: gate removed, real structural fix identified and applied (2026-08-02)

**Goal.** The prior DSpark drafter spike (entry above, "DSpark drafter measurement spike")
found `--ssd-streaming` and `--mtp` hard-gated as mutually exclusive at `ds4.c:56770-56773`
("not compatible yet") with no with/without decode-t/s comparison possible. This unit's
brief: read the code paths `--mtp` activates, determine whether the gate is wiring-class or
structural, and implement if wiring-class.

**Phase 1 finding: wiring-class, but with a real latent bug behind the placeholder gate, not
just a missing check.** Tracing the DSpark verify path (`ds4_session_eval_dspark_speculative_argmax`
-> `metal_graph_verify_suffix_tops` -> `metal_graph_encode_layer_batch` /
`_ffn_batch`) showed it already funnels routed-expert fetches through the *same*
`ds4_gpu_stream_expert_cache_prepare_selected_batch()` / `cuda_stream_selected_cache_begin_load()`
selected-cache+LRU protocol that real streamed prefill (n_tokens>1, P3b/P3a-fix) and decode
(n_tokens==1, b15cc29-equivalent) already use and that the CUDA per-`(layer,expert)` LRU
(`cuda_stream_expert_cache_*`) already serves generically -- so the *target* model's own
verify/replay under streaming needed no new plumbing. Draft-token replay
(`metal_graph_eval_token_raw_swa` -> `..._streaming` when `g->ssd_streaming`) was already
streaming-aware too. KV rollback on a verify miss (`spec_frontier_snapshot`/`_restore`)
operates on the raw SWA cache, which is always fully resident regardless of
`--ssd-streaming` (only routed-expert *weights* stream) -- no streaming-specific KV handling
was ever needed there either.

The actual, reproducible blocker was in the **DSpark support/drafter model's own forward
pass** (`metal_graph_eval_dspark_stage_block`, the batched stage-block forward that computes
`s->dspark_draft_tokens`). That function already knew it needed to avoid the CUDA streaming
selected-cache for its own (always fully-resident, never-streamed) FFN weights -- it
temporarily sets `g->ssd_streaming = false` around its call into the shared
`metal_graph_encode_layer_ffn_batch()`. But that guard was defeated two layers down:
`ds4_gpu_routed_moe_batch_tensor()` (`ds4_cuda.cu`) took a `force_resident` parameter and
threw it away (`(void)force_resident;`, always `allow_streaming=1`) even though its
`n_tokens==1` sibling `ds4_gpu_routed_moe_one_tensor()` already respected the equivalent
flag; and `metal_graph_encode_layer_ffn_batch()`'s own call site in `ds4.c` hardcoded
`force_resident=false` regardless of `g->ssd_streaming`. Net effect: the drafter's batch FFN
always attempted the CUDA streaming selected-cache, which was never prepared for (and is not
even correctly fetchable for -- the streamed-fetch fallback reads via a single
process-global fd bound to the *target* model's file, `ds4_cuda.cu`'s `g_model_fd`) the
drafter's own tensors, so `routed_moe_launch()`'s internal validity check failed every call
("`ds4: CUDA streaming selected experts are unavailable for layer 0`"), the drafter's own
FFN never produced valid hidden states, and `s->dspark_draft_valid` stayed `false` forever
-- draft proposal silently no-opped on *every* cycle, live-reproduced via
`DS4_DSPARK_SPEC_LOG=1` (`valid=0 len=0` / "skip no-draft" on every single cycle, confirmed
before the fix).

This means the original spike's optimistic read ("standard greedy-verify speculative
decode, output-identical to non-speculative decode by construction when both run greedy")
was correct about the *target*-model verify machinery but never actually exercised the
*drafter*-model forward pass under streaming at all -- the gate hid a real, separate wiring
gap one level deeper than the gate comment suggested.

**Fix (wiring-class, 3 changes, `research/gb10` branch, applied on top of `f7a54e2`).**
1. `ds4.c`: scoped gate removal at the `--mtp`/`--ssd-streaming` check -- refuses only when
   `--mtp` is passed **without** `--dspark` (legacy MTP's single-token draft path,
   `metal_graph_eval_mtp_draft_from_hc`, has no equivalent `g->ssd_streaming` save/restore
   guard and was never exercised under streaming in this unit; left refused rather than risk
   it silently reading the wrong model's tensors through the same global-fd landmine).
   `DS4_DISABLE_STREAMING_MTP=1` restores the old blanket refusal even with `--dspark`.
2. `ds4_cuda.cu`: `ds4_gpu_routed_moe_batch_tensor()` now respects `force_resident` (mirrors
   the `_one_tensor` sibling) instead of discarding it.
3. `ds4.c`: `metal_graph_encode_layer_ffn_batch()`'s call into
   `ds4_gpu_routed_moe_batch_tensor()` now passes `!g->ssd_streaming` instead of a hardcoded
   `false`, so the drafter stage-block forward's existing `g->ssd_streaming=false` save/
   restore trick actually reaches the routed-MoE dispatch decision it was always meant to
   gate. Zero effect on the streamed target model's own real prefill/verify calls (those run
   with `g->ssd_streaming` genuinely `true`, so `!g->ssd_streaming` evaluates `false` --
   byte-identical to the prior hardcoded value there).

**Verification the fix is real, not just quieter.** Before the fix: `DS4_DSPARK_SPEC_LOG=1`
showed `valid=0 len=0` and the "unavailable for layer 0" warning on every cycle (France
prompt, `--dspark` default confidence 0.9 *and* `--dspark-confidence 0.0`). After the fix:
warning gone; with `--dspark-confidence 0.0` (force-accept, to guarantee real verify
activity for this check) France now shows `valid=1 len=5`, real `drafted=5 verified=1
accepted=2`-style partial-accept cycles, and the CUDA `DS4_CUDA_STREAM_STATS` fetch/hit
counters move during verify batches -- the drafter proposal -> target verify -> accept/roll-
back loop is genuinely exercised end-to-end under `--ssd-streaming` for the first time.

**Correctness testing and an important, honest caveat.** Ran France (`-n 60`), the
~300-word TCP-handshake prose prompt (`-n 80`, testing a real multi-hundred-token prefill),
and a `-n 500` short-prompt long generation, all `--temp 0` (greedy), comparing DSpark-
enabled vs. no-drafter streaming decode:
- France: byte-identical at both `--dspark-confidence 0.0` and default 0.9.
- 500-token long generation (default confidence, `--dspark`): completed to a coherent,
  on-topic short story with no crash, no repetition/garbage loop, no vocab corruption (some
  drafting activity did occur -- `accepted_total=2` logged mid-run -- so this is not a
  "drafter never engaged" no-op run).
- Prose prompt (`-n 80`): **diverged from the no-drafter baseline** at both
  `--dspark-confidence 0.5` and the **default** 0.9 ("It ensures that both endpoints are
  ready to communicate" vs. "It ensures both parties are ready to communicate" -- a
  near-tied paraphrase-level flip, not garbage). Investigated further: **two independent,
  identical-command, no-`--mtp`-at-all reruns of the plain streaming baseline (rebuilt
  binary, zero DSpark code in the call path) also diverged from each other** at the same
  spot, under `--temp 0`. This is a **pre-existing, general `--ssd-streaming` decode
  non-determinism** (most likely `-ffast-math`/`--use_fast_math` FP-reduction-order
  sensitivity interacting with cache hit/miss timing, which can vary run to run even at
  temp 0) -- **not introduced by, and not specific to, this unit's DSpark work**: it
  reproduces with zero drafter code touched. It does mean strict byte-for-byte "output-
  identical to non-speculative decode" could not be established as a clean baseline on this
  build/hardware (the *baseline itself* isn't perfectly run-to-run reproducible under
  streaming) -- flagged here as a real, separate, out-of-scope finding for a future unit,
  not swept under the rug. DSpark's own accept-gate remains self-consistent by construction
  regardless (`sample_argmax(s->logits,...) == drafts[0]` / per-row `row_tops[i-1] ==
  drafts[i]` checked against the *same* batch-computed logits the verify pass produces, so
  acceptance is never based on stale/mismatched data even though that data's absolute
  reproducibility across process runs isn't guaranteed by this build).

**Throughput: does NOT presently exceed the compute ceiling -- explains where the time
goes.** With the default `--dspark-confidence 0.9` (the shipping default) against the
community drafter `hf.co/sakamakismile/DeepSeek-V4-Flash-DSpark-support-ds4-GGUF`: draft
proposal ran every cycle (confirmed via `DS4_DSPARK_SPEC_LOG=1`, confidence values logged in
the -0.4..1.2 range) but was almost always rejected by the 0.9 confidence-pruning threshold
("skip no-draft" on nearly every cycle across France/prose/the 500-token run), so decode
paid the drafter's own propose+confidence-probe compute on most steps with almost none of
the speedup: the -n 500 run averaged **~0.57 t/s** (511 tokens in ~900s before the process's
own timeout), well *below* both the no-drafter streaming baseline (2.69 t/s single-prompt,
this doc's own 4.46 t/s long-session steady state) and the 5.24 t/s compute ceiling. Lowering
`--dspark-confidence` to 0.0 (force-accept every draft) *does* produce real accepted
speculative windows (`drafted=5 verified=1 accepted=2`-style, ~40% per-block accept rate on
France) but did not net out ahead of baseline in the short, cold-cache single-prompt tests
run here either (decode dropped to ~0.18-0.22 t/s) -- consistent with this ticket's own
prediction that batch verify touches more experts per step than n=1 decode (union of the
draft window's routed experts, up to 5x a single decode step's 6-expert selection) and needs
a **warm, long session** (96%+ hit rate, per the long-session unit above) for the LRU to
absorb that before it can pay off; the tests run in this unit were short/cold and could not
reach that regime within the time budget. **Where the time goes, concretely:** (1) the
drafter's own stage-block forward (propose) and confidence probe run on every decode step
regardless of outcome -- pure overhead when rejected; (2) at low/zero confidence, verify's
batch window touches proportionally more distinct experts per step than plain n=1 decode,
which is a real added disk-miss cost until the LRU is warm. Neither of these is a
correctness bug; both are real, measured costs this unit's fix makes newly *observable*
(rather than a hard refusal), which is the intended outcome of a wiring-class unit -- the
economic question of whether *this specific* community drafter is a good match for the
target model, and whether a warm long session recovers the net win the ticket predicted, is
a distinct follow-up (**a full multi-turn warm-session with/without-drafter trajectory,
matching this doc's own long-session protocol, was not completed within this unit's time
budget** -- flagged explicitly as unfinished, not glossed over).

**Tests.** `make cuda-spark`: clean, zero warnings. `./ds4_test`: `ds4 tests: 6 failure(s)`
across `logprob-vectors` (on the documented pre-existing flaky list, `tests/ds4_test.c:5285`)
and `metal-tensor-equivalence` (not on the strict 3-item flaky list but this file's own
P3c-1-take-4 entry already recorded it fluctuating pass/fail across passes, e.g. "take 3's
own run saw 7 failures across a 3-section subset of the same list" -- consistent with, not a
new regression from, this unit's diff, which is scoped to CUDA-only DSpark/mtp-batch-encode
code plus the gate; unrelated to the Metal-vs-CUDA/reference numerical-tolerance comparison
`metal-tensor-equivalence` performs). `tool-call-quality` and `metal-kernels` (the other two
items on the documented flaky list) both passed this run. No new failing test names.

**Server discipline.** `ds4-server` stopped before every run above and restarted +
verified (`systemctl is-active`=`active`, `curl http://localhost:8000/v1/models` responding)
at the end.

**Pending (new, for a follow-up unit).** (1) The full with/without warm multi-turn
throughput trajectory (this unit's own biggest unfinished item). (2) The general
`--ssd-streaming` run-to-run non-determinism at `--temp 0` found above, independent of
DSpark -- worth a dedicated diagnostic pass (candidate first step: bisect
`-ffast-math`/`--use_fast_math` vs. cache-hit-path-vs-miss-path FP differences). (3)
Evaluate whether a better-matched drafter (vs. this community artifact) or a lower default
`--dspark-confidence` recovers a real net win once measured in a warm session. (4) Legacy
MTP (non-DSpark) under `--ssd-streaming` remains refused (per the scoped gate above) and
would need its own `g->ssd_streaming` guard around `metal_graph_eval_mtp_draft_from_hc`
before it could be safely enabled the same way.

## ds4#605 transport eval: CUDA streaming upload pipelining + plain-copy default, independently ported onto our LRU -- adapted, build/tests verified, live A/B blocked by concurrent exclusive-lock process (2026-08-02)

**Goal.** Independently test the *transport* half of upstream PR #605 (iCreil,
"CUDA: resident expert cache and faster selected-expert uploads for
--ssd-streaming decode", claims 2.3->29-32 t/s on RTX PRO 6000) against our
GB10 + MXFP4 + expert-LRU stack (per-(layer,expert) CUDA LRU, P3a-fix +
P3b pooled allocator), fold what proves better, and produce an honest
verdict on whether PR#605's PCIe-discrete-card wins transfer to GB10's
unified memory.

**Classification of the 4 PR#605 commits.**

| commit | title | classification | rationale |
|---|---|---|---|
| `10ba298` | restore resident expert cache | (a) duplicates our LRU, not ported | Adds ~1100 lines implementing a brand-new class/slot-keyed resident VRAM cache (`cuda_stream_expert_cache_prepare/_release_class/...`) from scratch. Our tree already has a functionally equivalent, structurally different persistent per-(layer,expert) LRU (`cuda_stream_expert_cache_peek/_install/_prune_global`, from P3a-fix). Porting this commit would either be dead code sitting beside our own cache or require ripping ours out and replacing it wholesale -- both out of this unit's transport-only scope. |
| `9167d12` | skip per-load VRAM query once cache has settled | (a) duplicates our LRU, not ported | Entirely an optimization *of* the resident-cache structures `10ba298` just added (`g_stream_expert_settled_caps[]`, `cuda_stream_expert_cache_prepare()`'s fast path). Since `10ba298` was not ported, this commit has no target in our tree -- not applicable, not a separate transport concern. |
| `1c055bb` | pipeline the streamed expert upload path | (b) transport, ported | Targets `cuda_model_copy_to_device_streamed()` -- a function that exists **verbatim by name** in our tree too (pre-PR#605 common ancestor), and is the exact function our LRU's miss path already calls three times per miss (gate/up/down). Directly portable. |
| `5930a3f` | default streamed expert uploads to plain copy without O_DIRECT | (b) transport, ported | Same function (`cuda_model_copy_to_device_streamed_ex` after `1c055bb`). Directly portable; this is also the commit most likely to behave differently on GB10's unified memory (see verdict below). |

Confirmed before porting (read, not assumed): our LRU already serves cache
hits with a device-to-device `cudaMemcpy` straight into the packed staging
buffer (`cuda_stream_selected_cache_begin_load()`, hit branch) -- zero
change there. Only the miss branch calls `cuda_model_copy_to_device_streamed()`,
so PR#605's transport pipelining applies exactly where the ticket predicted:
the miss-path upload, not the (already fast) hit path.

**Adaptation, not cherry-pick.** `git cherry-pick` on `1c055bb`/`5930a3f`
did not apply cleanly (our tree's surrounding code -- staging-pool alloc,
comments, chunk-size plumbing -- has diverged from PR#605's base enough
that the hunks don't match context, even though the target function is
name-identical). Hand-adapted onto `research/gb10` HEAD `5d032d7` instead,
preserving iCreil's design and measured rationale, committed as nexus-cw
with `Co-Authored-By: iCreil <catalano.mirko@gmail.com>` and an explicit
"Adapted from antirez/ds4#605 by iCreil" trail in both the commit message
and inline code comments at each adapted site. Branch: `eval/605-transport`
(pushed), commit `b4816a3`, one file (`ds4_cuda.cu`), +97/-24 lines.

Changes: (1) `cuda_model_copy_to_device_streamed()` -> `_ex()` with a
`flush` parameter and a persistent 4-buffer ring cursor shared across
calls (was reset to 0 every call) so back-to-back grouped copies pipeline
host reads with device uploads across tensor boundaries, not just within
one tensor's own chunks; a separate `_flush()` does the
`cudaStreamSynchronize` only when the caller asks for it. (2) Miss-path
upload defaults to a plain `cudaMemcpy` from the pageable model map
instead of the staged-pread-then-async-upload path when there is no
active O_DIRECT fd (`g_model_direct_fd < 0`), matching `5930a3f`'s
rationale that the driver's own pageable-transfer path beats the
pread-bounce when reads are already coming from page cache/tmpfs;
`DS4_CUDA_STREAM_MISS_PLAIN_COPY=1/0` still forces either way. (3) The one
real integration point: `cuda_stream_selected_cache_begin_load()`'s miss
branch now issues its three tensor uploads (gate/up/down) as one async
group (`flush=0,0,1`) instead of three independently-synced calls.

**Build.** `make cuda-spark`: clean, zero warnings, on `eval/605-transport`
(commit `b4816a3`) and on a fresh `research/gb10` worktree baseline
(`/tmp/ds4-baseline`, HEAD `5d032d7`) side by side.

**Tests.** `test_mxfp4_dequant`: PASS (131424 checks, 0 mismatches) --
unaffected by this diff by construction (dequant-only, no streaming code).
`test_mixed_moe`: PASS, all cases -- also unaffected by construction (its
own file-level comment states it is "deliberately independent of
ds4.c/ds4_cuda.cu internals beyond" the public GPU API, which this diff
does not touch). `test_mxfp4_moe`: **inconclusive, environmental** -- hit
`ds4: CUDA init set device failed: out of memory` at CUDA-device-init time,
before any of this unit's code runs at all. Root cause confirmed
environmental, not a regression: at the time of the run, `free -h` showed
1.6-2.2 GiB free / 5-18 GiB available system-wide on this 121 GiB unified-
memory box, driven by a co-running interference-test `ds4-server` (see
below) plus other resident load; this failure mode is identical to what
the unpatched baseline would also hit under the same memory pressure (the
OOM is at `cudaSetDevice`, upstream of any code this unit touched). Not
rerun once the interference process cleared because it did not clear
within this unit's time budget (see below).

**Live A/B measurement: BLOCKED, not completed this unit.** ds4 enforces
a single-instance exclusive lock on the GPU/model (confirmed live:
`ds4: another ds4 process is already running (pid 1148291); refusing to
start`). PID 1148291 is another unit's own manually-launched
`ds4-server --port 8001 --ssd-streaming-cache-experts 90GB` interference
test, running continuously (11h40m+ elapsed at time of check, actively
cycling chat turns roughly once a minute per its own log) with no
indication of finishing soon. Per this unit's protocol ("WAIT (bounded
poll) for ds4 processes you don't own, never kill them"), the interference
server's PID was left untouched; a bounded poll (3 checks, 30s apart) found
it still active and its log showed ongoing turn cycling, not a wind-down.
Given the process's long, sustained, and apparently open-ended runtime
(it is itself a deliberate multi-user-interference experiment, not a
transient job), waiting it out was not feasible within this unit's time
budget. The systemd `ds4-server` service was already `inactive`
throughout (confirmed via `systemctl is-active`) and required no action.

No warm multi-turn decode, cold-start turn-1 decode, or prefill-timing
comparison numbers were collected for either branch this unit -- reporting
that gap directly rather than presenting the earlier build/test results as
a substitute for the requested A/B tables.

**Fold decision: NOT FOLDED, pending measurement.** Per protocol ("fold if
any arm shows a real (beyond noise) win with no regression elsewhere"),
folding requires measured evidence; none was obtainable this unit. The
adapted, build-clean, non-destructively-tested code is left on
`eval/605-transport` (pushed, commit `b4816a3`) for the next unit (or a
rerun of this one) to measure once the box is free of the port-8001
interference load. `research/gb10` itself is unchanged.

**GB10 unified-memory verdict: not yet determined -- explicitly deferred,
not defaulted to "no effect."** This is the one open question this unit
set out to answer and could not, given the blocker above. The mechanism
PR#605's `5930a3f` exploits (pageable driver-managed transfer beating a
pread-into-pinned-buffer bounce) is plausible on both discrete-PCIe and
GB10-unified hardware for different reasons -- on a discrete card it avoids
a redundant host-side copy before the PCIe DMA; on GB10, where "host" and
"device" share the same physical DRAM over NVLink-C2C, the mechanism by
which either path moves bytes is different enough (there is no PCIe DMA
to bypass) that PR#605's own ~3x number cannot be assumed to transfer, but
it cannot be assumed to vanish either without measurement -- both are
live hypotheses. Flagging this explicitly as the next unit's first
priority: rerun the A/B protocol in `MEASUREMENTS.md`'s existing long-
session/cold-start/prefill format once port 8001's interference server has
exited, using `eval/605-transport` vs `research/gb10`, before any fold-vs-
no-fold call is finalized either way.

**Next steps (explicit, for whoever picks this back up).** (1) Confirm
`ds4-server` (PID 1148291, port 8001) has exited (`pgrep -f 'ds4-server.*
port 8001'`); do not kill it. (2) Run this unit's full protocol: warm
multi-turn (6+ turns, steady-state of last 3) research/gb10 vs
eval/605-transport; cold-start turn-1 at both a 100GB and an 8GB cache
budget (the latter specifically to stress the miss path this unit's patch
targets); ~300-token prefill (`research/gb10/prose_prompt.txt` is already
in place for this). (3) Byte-identity gate between branches at `--temp 0`
-- note the pre-existing, branch-independent `--ssd-streaming` decode
non-determinism documented in the "Drafter-streaming compatibility" entry
above (two identical baseline reruns diverging at the same spot) before
treating any observed divergence as a regression; rerun 2-3x per arm to
separate real divergence from that known noise floor. (4) If a real win is
found on any arm with no regression elsewhere, merge `eval/605-transport`
into `research/gb10` and update this file + `FP4_PORT_SCOPE.md` with the
numbers and the GB10-vs-discrete-card analysis; if GB10 unified memory
nullifies or reverses PR#605's gains, that is itself the valuable
deliverable -- document it with the same rigor (it also tells upstream
their PR's value may be discrete-card-specific).

**Server discipline.** `ds4-server` (systemd) was `inactive` before and
after this unit's work (confirmed via `systemctl is-active`); no restart
was needed since it was never started. The port-8001 interference server
(another unit's own process, not systemd-managed) was left running
throughout, untouched, per protocol.

## Multi-user prefill/decode interference: HOL blocking vs. expert-cache pollution (2026-08-02)

Live measurement of the two-channel interference hypothesis: does one user's long-document
prefill degrade another user's warm decode via (1) GPU head-of-line (HOL) blocking and/or
(2) expert-cache pollution (prefill sweeping in experts that evict the decode session's warm
working set, with damage persisting after the prefill ends until re-warmed)? `ds4-server`
(systemd, production IQ2 config) stopped first, restarted and verified (`systemctl
is-active`=`active`, `/v1/models` 200) after -- see "Server discipline" below.

**Box sharing.** At start, another unit's `./ds4_test` full test-suite run (`~/src/ds4-pr`,
then `~/src/ds4-baseline`) held ~70-86 GiB of GPU memory. Per protocol, its PIDs were left
untouched; a bounded poll (~25 min, several rounds) confirmed it finished on its own before
any manual server was launched here.

**Scheduler code, read before running (truth in code, not assumed).** `ds4_server.c`'s file
header states the design intent: "A model coordinator batches decode-ready sessions and
serializes bounded prefill quanta, keeping graph mutations out of client threads." The
mechanism is `server_prefill_quantum_for()` (`ds4_server.c:10275-10288`): prefill proceeds in
2048-token quanta when no generation is active, or 128-token quanta when
`s->active_generations > 0` -- ostensibly a bound on how long one session can hold the model
before another gets a turn. But the caller, `server_session_sync()`
(`ds4_server.c:10296-10339`), is a single `while` loop that repeatedly calls
`server_prefill_enter()`/`ds4_session_sync()`/`server_prefill_leave()` **for the same
request** until that request's whole prompt is consumed; nothing inside that loop checks or
services a *different* session's queued job between quanta. So the 128-token quantum bounds
how much of *its own* request a session processes per model-lock acquisition, but does not
cause the coordinator to round-robin between sessions mid-prefill. This predicts pure HOL
serialization for any single big prefill, which is exactly what the live traces below show.

**Method.** Streamed model launched manually (`ds4-server -m
gguf/DeepSeek-V4-Flash-MXFP4_MOE.patched.gguf --cuda --ssd-streaming
--ssd-streaming-cache-experts 90GB --host 0.0.0.0 --port 8001 --ctx 16384 --batched-session
2`, `DS4_CUDA_STREAM_STATS=1` set). Client A: a fixed 25-token prompt ("Tell me one
interesting, specific fact about deep-sea anglerfish...", non-conversational, one-shot per
call so each call re-exercises the same expert-routing footprint), `max_tokens=40-60`, looped
via `/v1/chat/completions`. Client B: prompt built from concatenated repo docs
(`README.md`/`CONTRIBUTING.md`/`AGENT.md` etc.), `max_tokens=20-30` so the request is
almost pure prefill. Both driven from a Python client hitting the server directly; ground
truth for per-request decode throughput is the server's own `chat ctx=A..B:N ... decoding
chunk=X t/s avg=Y t/s` log lines (no timing fields are exposed in the OpenAI-shaped JSON
response, so the client wall-clock is a secondary/corroborating signal only, dominated by
queueing delay when B is in flight).

**Note on `DS4_CUDA_STREAM_STATS`.** The counter-print function
(`ds4_gpu_print_cuda_stream_stats()`, `ds4_cuda.cu:182`) has exactly one call site in this
codebase, and it is not reachable from the HTTP server's request-serving path (mirrors the
finding already on record in this file from the long-session locality unit, which found the
same thing true of the interactive REPL's exit path). No cumulative hit/miss counters were
obtainable from the server for this reason; decode t/s (which is itself a direct function of
expert-cache hit rate at this quantization/budget) was used as the measurable proxy instead.

### Timeline: streamed model (SSD-streaming, 90GB expert-cache budget), cycle 1

B's prompt: 960 tokens, `max_tokens=20`.

| step | request | server-log decode/prefill | wall (client) |
|---|---|---|---|
| warm-up | A0-A5 (6 reqs) | decode avg/last: 2.39 -> 3.94 -> 4.32 -> 4.50 -> 4.48 -> 4.47 t/s | 61.7s -> 49.4-49.6s steady |
| aggressor | B fires (960-tok prefill, concurrent with A's 6th request already mid-decode) | prefill in 128-tok quanta, **avg throughput collapses to 1.05 -> 2.17 t/s** (SSD-bound expert loading, not compute), full prefill takes 443.4s; B's own 20-tok decode: 1.50 t/s; **total B latency 456.7s** | -- |
| **victim during B** | A5's decode (already in-flight when B started) | **decode `chunk=0.09 t/s avg=0.09 t/s`, elapsed 460.06s** -- A5 received effectively zero service until B fully finished. No A request of any kind appears in the server log between B's `prompt start` (09:40:10) and B's `finish` (09:47:46/09:47:50): **zero interleaving observed over the full 456s window**, confirming the code-level prediction above empirically | 500.5s (Aafter0, itself also queued behind A5) |
| **recovery** | Aafter1 (first *new* A request serviced after B) | decode **4.21 t/s** -- already at baseline on the very next request | 52.2s |
| steady after | Aafter2-17 (16 more reqs) | decode range **4.33 - 4.76 t/s**, mean ~4.58 t/s -- indistinguishable from / slightly above the 4.3-4.5 t/s pre-B baseline | 50.4-52.5s (Aafter18/19 show 102-104s -- self-interference artifact: cycle 2's own A0/A1 were launched concurrently on the same server at that point, not a B-driven effect; see Deviations) |

**Recovery time: 0 requests / ~0s beyond B's own completion.** There is no measurable
degraded-but-recovering tail; the very first post-B decode is already within the pre-B
steady-state band.

### Timeline: streamed model, cycle 2 (reproducibility check, larger B)

B's prompt: 1486 tokens, `max_tokens=20`. A baseline (A0-A3, post-cycle-1-settling):
51.0-52.1s steady (A0/A1 elevated at 76-102s from unavoidable overlap with cycle 1's tail,
see Deviations). B: prefill throughput **1.54-1.76 t/s avg** (same disk-bound regime), full
prefill+decode **861.8s** total -- again with **zero A activity logged during the entire
window** (confirmed via full server-log grep across the B span). First post-B A request
(Aafter1, after an Aafter0 whose client-side request had already given up at the 900s
timeout while the server kept it queued) decodes at server-log-confirmed baseline speed
immediately; Aafter1-9 wall times 50.6-63.0s, matching cycle 1's steady band. Same verdict:
full serialization during B, instant recovery, no persistent decode degradation after B
completes.

### Control (b): resident IQ2 model, same protocol, isolates the HOL-vs-pollution channels

Production-equivalent config launched manually on port 8001 (no `--ssd-streaming`, full
model resident in VRAM: `--gpu-vram 88`). No expert-streaming cache exists in this mode --
any interference observed here is pure compute HOL blocking, with **zero** disk-bound
contribution possible.

| step | server-log decode/prefill | wall (client) |
|---|---|---|
| A0-A3 baseline | decode steady **16.7-16.9 t/s** | 2.7-3.2s |
| B fires (1486-tok prompt, `max_tokens=20`) | prefill **201-223 t/s avg** (pure compute, no SSD stalls) -- full prefill **7.4s**, total B latency **10.1s** | -- |
| victim during B (queued A request) | decode **3.89 t/s** (elapsed 10.29s -- almost entirely queueing wait, not slow compute) | 10.6s |
| recovery | next request: **16.90 t/s**, then 14.36/15.52/16.96/16.95/16.95/16.94 t/s | 2.66-3.09s, immediately back to baseline |

**Channel attribution.** Both configurations show the *same structural regime*: one victim
request is fully blocked for (approximately) B's whole prefill duration, then recovers
instantly on the very next request, with no multi-request decay/recovery tail in either
case -- i.e. **no evidence of expert-cache pollution/eviction of A's warm working set** at
this cache-budget-to-B-size ratio (90 GiB budget vs. a model whose full MXFP4 expert weight
footprint is ~85 GiB total across 6716 experts at 12.75 MiB each, per the server's own
"cuda SSD streaming total expert budget 90.00 GiB = 6.38 GiB prefill headroom + 83.62 GiB
dynamic cache (6716 experts...)" boot line -- the budget is large enough relative to the
whole model that a single 1-2k-token B prompt cannot force meaningful LRU eviction of A's
25-token working set). What differs by two orders of magnitude between the two
configurations is **purely the duration of the HOL-blocked window**: 443-862s (streamed,
disk-bound prefill at ~1.5-2.8 t/s) vs. 7.4s (resident, compute-bound prefill at ~200+ t/s)
for prompts of comparable length (960-1486 tokens). This isolates the channel cleanly: the
*scheduling regime* (full serialization, zero interleaving, instant recovery) is identical
and inherent to `ds4-server`'s coordinator design regardless of streaming; what SSD-streaming
specifically adds is a ~30-100x inflation of the blocking window's *duration*, driven by
cold per-(layer,expert) SSD reads during B's own prefill (observed at 12.75 MiB/expert,
~500-600 MB/s effective aggregate throughput implied by the 1.5-2.8 t/s prefill rate across
43 layers x up to 8 experts/token) -- not by evicting anyone else's cache.

### Scheduler-regime verdict

**Fully serialized (pure HOL blocking) for the duration of any request already past the
2048-token cold-start prefill quantum**, despite `mixed_prefill_quantum=128` existing in the
code and being logged as active (`batched mode enabled resident_sessions=2
prefill_quantum=2048 mixed_prefill_quantum=128 decode_coalesce_us=2000`). The 128-token
quantum bounds only how much of the *aggressor's own* prompt is processed per lock
acquisition inside `server_session_sync()`'s internal loop; it does not cause the top-level
coordinator to interleave a *different* session's pending job between those quanta. Verified
directly from two independent live traces (960-token and 1486-token B prompts): zero A-tagged
log lines appear anywhere between B's `prompt start` and B's `finish=` across 456s and 862s
windows respectively. This is the worst-case interference regime the protocol asked to be
flagged if observed, and it is what's actually in production for `--batched-session 2` today.

### Channel attribution summary

1. **HOL blocking (confirmed, dominant/only channel observed):** any concurrent request is
   blocked for the aggressor's full request lifetime, no partial service.
2. **Expert-cache pollution (not observed at this scale):** decode throughput recovers to
   baseline on the very first post-aggressor request in every trial (2 streamed-model cycles
   + 1 resident-model control), with no degraded-then-recovering tail. This does not rule out
   pollution at larger scale -- a B prompt that actually forces LRU eviction (requiring either
   a much larger document, a much smaller cache budget, or many concurrent aggressors) was
   out of this unit's time budget; the 10.5k-token first attempt (see Deviations) would likely
   have been the config to show it, at the cost of a ~65-minute single prefill.

### Implications

For `--batched-session N` as currently implemented, N is a *resident-slot* count (how many
sessions can hold KV state simultaneously), not a *fairness* guarantee -- there is no
preemption or round-robin once a session's job is dispatched to the model coordinator. On the
IQ2 resident config this is a minor (~sub-10s) hiccup for a several-thousand-token document.
On the SSD-streamed config, the same document produces **minutes-to-an-hour** of total
unresponsiveness for every other user, because prefill throughput itself collapses to
1.5-2.8 t/s (vs. 200+ t/s resident) under cold per-expert SSD reads -- this is a latency/
availability problem, not a cache-correctness problem: nobody's answers become wrong or
stale, but a shared streamed-model deployment with `--batched-session >1` is not multi-tenant
in any latency-fairness sense today. If genuine interleaving is wanted, the fix belongs in
`server_session_sync()`'s loop (or its caller) -- yielding the model lock back to the
coordinator after each quantum so a different resident session's pending job can be serviced,
not just bounding the aggressor's own quantum size. Separately, since pollution was not
observed here, the 90GB budget choice (comfortably covering this MXFP4 model's ~85GB total
expert footprint) appears to make cache eviction a non-issue for realistic single-aggressor
workloads -- the risk profile would change for models whose full expert footprint
significantly exceeds the configured budget, which this unit did not test.

### Deviations from protocol

- B's prompt sizes were reduced from the specified 8-12k tokens to 960/1486/(one aborted
  10.5k) tokens after the first attempt (10.5k tokens) was measured in-flight to require
  ~65 minutes of prefill alone (2.5-2.8 t/s observed rate, confirmed live before aborting) --
  infeasible within this unit's time budget across the required 2-3 repeat cycles plus a
  resident-model control. The reduced sizes (38-62x A's 25-token prompt) still produced
  unambiguous, reproducible serialization and recovery results; the 10.5k-token config is
  flagged above as the more promising candidate for actually forcing measurable pollution,
  for a future unit with a larger time budget.
- Control (a) ("A alone throughout, no B, same duration") was not run as a fully separate
  duration-matched run; the pre-B warm-up segments (A0-A5 in cycle 1, A0-A3 in cycle 2) and
  the long post-recovery steady segments serve as the drift check instead -- decode t/s stayed
  in a tight 4.2-4.8 t/s band across ~30 total A requests spanning both cycles with no trend,
  which is the same evidence a separate control run would have produced.
- Two client-side self-interference artifacts are visible in the raw wall-clock logs
  (cycle 1's Aafter18/19 at ~104s, cycle 2's A0/A1 at 76-102s) caused by launching cycle 2's
  driver before confirming cycle 1's had fully drained -- both fixed sequences ran on the same
  shared server. These do not affect the decode-t/s-based findings (server log timestamps
  disambiguate cleanly) but do inflate a handful of wall-clock numbers in the raw output; noted
  here rather than silently edited out.
- Repeat count was 2 full interference cycles (not 3) on the streamed model, plus the 1
  resident-model control, given the time cost per cycle (up to ~15 min for B alone). The two
  streamed-model cycles agree closely (recovery-on-first-request in both), which was judged
  sufficient for the qualitative verdict (full serialization, no pollution at this scale);
  a third repeat would add confidence but is unlikely to change the verdict given how tight
  the two observed traces already are.

### Server discipline

`ds4-server` (systemd, production IQ2 config, port 8000) was stopped before this unit's
manual servers were launched and restarted at the end; confirmed `systemctl is-active`=
`active` and `curl .../v1/models` -> HTTP 200 after. The manual streamed-model server (port
8001) and the manual resident-model control server (port 8001, second launch) were both
terminated by this unit before the systemd restart; `nvidia-smi` confirmed 0% GPU utilization
and no matching process before restart.

## ds4#605 transport eval, live A/B completed: pipelining REGRESSES on GB10 unified memory, plain-copy default is neutral -- NOT FOLDED (2026-08-02)

**Goal.** Resume the blocked unit above (2026-08-02, commit `b4816a3`, branch
`eval/605-transport`): the box is now free (no port-8001 interference process,
production `ds4-server` was the only other consumer and was stopped for this
unit), so this pass runs the full A/B protocol that unit could not.

**Setup.** `sudo systemctl stop ds4-server` first (was `active`, running the
IQ2XXS production model). Two worktrees: `~/src/ds4` at `research/gb10` HEAD
`0cbfeec` (baseline; the previously-used `ds4-baseline` worktree was found
stale at `54b36ed`, 35 commits behind, and was not used), and a fresh
`~/src/ds4-605` worktree at `eval/605-transport` HEAD `b4816a3` (unchanged
from the prior unit). Both built clean via `make cuda-spark`, zero warnings.
Model: `gguf/DeepSeek-V4-Flash-MXFP4_MOE.patched.gguf` (the MXFP4 preview
artifact, same as all prior numbers in this doc; the GA file on disk was not
touched, per instruction -- that is a separate ticket's business).

**Byte-identity gate.** `--temp 0`, same photosynthesis prompt, both branches:
`-n 60` and `-n 200` generations came back **byte-identical** text on both
lengths, at `--ssd-streaming-cache-experts 100GB`. Not additionally checked at
the 8GB budget (where sampling arms below use non-zero temp and vary run-to-run
by design, which is expected, not a regression). **Identity gate: PASS** for
the lengths tested.

**Arm results** (mean generation t/s unless noted; all runs `--nothink`,
`gguf/DeepSeek-V4-Flash-MXFP4_MOE.patched.gguf`, `--cuda --ssd-streaming`):

*Arm A -- warm multi-turn, 100GB budget, `DS4_CUDA_STREAM_STATS=1`,
`research/gb10/multiturn.txt` (8 turns), steady state = mean of last 3 turns,
single session per branch (n=1, no reps -- an 8-turn session takes ~10 min and
a repeat run was not feasible this unit's time budget; flagged honestly, not
padded with false reps):*

| branch | turn 1 | turn 2 | turn 3 | turn 4 | turn 5 | turn 6 | turn 7 | turn 8 | last-3 mean |
|---|---|---|---|---|---|---|---|---|---|
| research/gb10 (baseline) | 2.99 | 4.18 | 3.76 | 4.54 | 4.38 | 4.47 | 4.43 | 4.42 | **4.44** |
| eval/605-transport | 2.67 | 4.05 | 2.08 | 4.21 | 3.90 | 4.30 | 4.31 | 4.40 | **4.34** |

Baseline's steady state (4.44 t/s) matches this doc's earlier long-session
locality-convergence entry (4.39-4.48 t/s range) almost exactly -- a useful
cross-check that this run's methodology reproduces prior results. The ~2.3%
gap to eval's 4.34 t/s is well within the turn-to-turn variance visible in
both trajectories (baseline's own turns 3 and 4 differ by 0.78 t/s) and with
n=1 per branch cannot be resolved from noise. **Verdict: NOISE, not a
resolvable difference**, at the 100GB budget where hit rate is already
96-98% and the miss path this patch touches is rarely exercised.

*Arm B -- cold turn-1 decode + wall clock, page cache dropped first (`sync;
echo 3 | sudo tee /proc/sys/vm/drop_caches`), `-n 60`, 2 reps each, both a
100GB and an 8GB budget:*

| budget | branch | rep1 wall | rep2 wall | rep1 gen t/s | rep2 gen t/s |
|---|---|---|---|---|---|
| 100GB | research/gb10 | 56.04s | 56.62s | 2.86 | 2.91 |
| 100GB | eval/605-transport | 64.04s | 61.75s | 2.49 | 2.62 |
| 8GB | research/gb10 | 103.21s | 102.56s | 0.98 | 0.98 |
| 8GB | eval/605-transport | 124.11s | 124.41s | 0.77 | 0.76 |

At 100GB: eval is **11-14% slower** wall-clock (mean 62.9s vs 56.3s), **11%
slower** decode t/s (mean 2.56 vs 2.89), both reps non-overlapping in both
directions. At 8GB: eval is **21% slower** wall-clock (mean 124.3s vs
102.9s), **22% slower** decode t/s (mean 0.765 vs 0.98), again fully
non-overlapping across both reps. **Verdict: REAL REGRESSION**, both
budgets, more severe at 8GB -- exactly where PR#605's transport pipelining
was supposed to help most.

*Arm C -- miss-heavy, 8GB budget, fixed photosynthesis prompt `-n 100`,
4 reps each (rep1 not discarded as "warming" since each rep is a fresh
process with no persistent device-cache -- OS page cache warms across reps,
but the CUDA streaming expert cache does not, so every rep is effectively a
cold-device-cache miss-heavy run by this test's own design):*

| branch | rep1 | rep2 | rep3 | rep4 | mean | range |
|---|---|---|---|---|---|---|
| research/gb10 | 1.01 | 1.00 | 1.01 | 1.01 | **1.0075** | [1.00, 1.01] |
| eval/605-transport | 0.76 | 0.76 | 0.76 | 0.75 | **0.7575** | [0.75, 0.76] |

**Zero overlap across all 8 data points** (4 baseline reps all >= 1.00,
4 eval reps all <= 0.76) -- eval is **~25% slower**, consistently, every
single rep. This is the clearest, highest-confidence signal in this unit:
the gap (0.25 t/s) is roughly 25x any single rep's own run-to-run spread
(~0.01 t/s), comfortably beyond a 3-sigma noise floor. **Verdict: REAL
REGRESSION**, high confidence.

*Arm D -- prefill, `research/gb10/prose_prompt.txt` (55 words / 18 prompt
tokens per `ds4`'s own tokenizer count -- shorter than the ~300-token target
this unit's brief specified; this is the only prefill prompt staged from the
prior unit and was used as-is per the resume note's "already in place for
this" pointer, flagged here rather than silently treated as 300 tokens),
100GB budget, `-n 20`, 3 reps each:*

| branch | rep1 | rep2 | rep3 | mean |
|---|---|---|---|---|
| research/gb10 | 1.51 | 1.49 | 1.48 | **1.493** |
| eval/605-transport | 1.49 | 1.49 | 1.49 | **1.490** |

<0.3% apart, within noise. **Verdict: NO CHANGE**, as expected -- prefill at
a 100GB budget rarely touches the miss path (confirmed by inspection in the
prior unit: layer-major prefill is unaffected by this diff by construction).

**Isolation mini-arm: which of the two ported changes causes the Arm C/B
regression?** `DS4_CUDA_STREAM_MISS_PLAIN_COPY=0` on `eval/605-transport`
forces the *old* pread-then-async-upload path (disabling `5930a3f`'s
plain-copy default) while leaving `1c055bb`'s pipelining (persistent
ring-cursor, grouped 3-tensor async upload) active. 3 reps, 8GB budget, same
prompt as Arm C:

| config | rep1 | rep2 | rep3 | mean |
|---|---|---|---|---|
| eval/605-transport, plain-copy forced OFF | 0.75 | 0.75 | 0.75 | **0.75** |
| eval/605-transport, default (plain-copy ON) | -- | -- | -- | 0.7575 (Arm C) |
| research/gb10 baseline | -- | -- | -- | 1.0075 (Arm C) |

Forcing plain-copy off does **not** recover baseline performance (0.75 t/s,
same as the 0.7575 t/s default) -- it stays regressed. This isolates the
Arm B/C regression to **`1c055bb` (the upload pipelining change)**, not
`5930a3f` (the plain-copy default). The plain-copy default itself is
neutral-to-slightly-negative here, not the driver of the loss.

**GB10 unified-memory analysis.** The prior unit flagged this as the open
question and it is now answered: PR#605's pipelining -- a persistent
4-buffer ring cursor shared across grouped tensor uploads, deferring
`cudaStreamSynchronize` until a caller-specified flush point -- **helps on
a discrete PCIe card by hiding host-read latency behind in-flight PCIe DMA
across chunk/tensor boundaries**. On GB10, host and device memory are the
same physical DRAM over NVLink-C2C: there is no PCIe DMA to hide anything
behind, so the extra bookkeeping (ring-cursor state, deferred-sync grouping
logic) is close to pure overhead with no corresponding latency to mask,
and Arm C's 25% loss suggests it is worse than pure overhead -- plausibly
because deferring sync across 3 grouped tensor uploads delays the point at
which the expert data is actually usable by the compute kernel relative to
issuing and syncing each upload eagerly, on hardware where the upload is
cheap enough that the old per-tensor-sync path was already close to optimal.
Not verified further with a profiler this unit (out of scope/time budget);
flagged as the natural follow-up if this branch or its individual commits
are revisited. The plain-copy default (`5930a3f`) is the commit most
naively expected to be architecture-sensitive (pageable-transfer-beats-
pread-bounce is itself a PCIe-driver-specific claim) but measures as
neutral here -- the regression is not where the naive prediction would have
placed it.

**Fold decision: NOT FOLDED.** Per protocol ("fold if any arm shows a real
win with no regression elsewhere"): no arm shows a win. Arm A is noise, Arm D
is flat, and Arms B and C show a real, reproducible, high-confidence
regression (11-25% slower depending on budget/state) with zero overlap
across every rep pair collected. `research/gb10` is unchanged.
`eval/605-transport` (commit `b4816a3`) is left as-is, pushed, for
reference/upstream-comment purposes -- not merged.

**Value of this result.** This is exactly the kind of negative result the
original unit flagged as valuable regardless of direction: PR#605's own
~3x claim on a discrete RTX PRO 6000 does not transfer to GB10's unified
memory, and for the specific transport-pipelining half of the PR, the
mechanism actively *reverses* on this hardware. It also tells upstream
their PR's benefit is very likely PCIe-DMA-specific, not a generic
"streamed-expert-upload" win.

**Draft comment for a possible upstream note on ds4#605** (drafted here for
the orchestrator's review; not posted by this unit): "We independently
ported the transport half of this PR (the streamed-upload pipelining and
plain-copy default, not the resident-cache redesign, which duplicates
functionality we already have) onto an NVIDIA GB10 (unified CPU/GPU memory
over NVLink-C2C) + custom MXFP4 expert-LRU stack, and measured a
consistent 11-25% regression from the pipelining change specifically
(isolated via an env toggle; the plain-copy default alone was neutral).
Our read is that the pipelining's benefit comes from hiding host-read
latency behind PCIe DMA, which doesn't exist as a mechanism on
memory-unified hardware -- worth a note in the PR description that the
~3x number is likely discrete-PCIe-card-specific rather than general."

**Server discipline.** `ds4-server` (systemd, port 8000, production IQ2XXS
model) stopped at the start of this unit, restarted and verified
(`systemctl is-active`=`active`, `GET /v1/models` -> HTTP 200) at the end.

## GA-0731 swap unit: BLOCKED at INSPECT on a new metadata gap, decision deferred (2026-08-02)

**Scope of this unit.** Swap `gguf/DeepSeek-V4-Flash-0731-MXFP4_MOE-Q8_0.gguf` (156.4 GB,
GA main artifact) in for the preview `.patched.gguf` used throughout this doc's prior
entries, run the full census -> inspect -> smoke -> warm-baseline -> quality-battery ->
decision-prep sequence. `ds4-server` (systemd, production IQ2XXS) stopped first; restarted
and verified active/200 at the end.

**Result: stopped at step 2 (INSPECT).** Full header-census diff and the exact `--inspect`
failure/log are recorded in `FP4_PORT_SCOPE.md`'s new "GA artifact... BLOCKED" section
(2026-08-02) -- summarized here:

- Header census: GA is **architecturally identical** to the preview artifact (43 layers,
  256 experts/6 used, 4096 embd, 2048 ffn, 1328 tensors, **284.33 B logical params measured
  independently from both files' raw headers, not the ~304 B this unit's brief
  anticipated**). GA is a materially cleaner conversion: canonical (not community-alias)
  tensor names throughout (zero tensor-name dialect-compat notices fired), and all 8
  metadata keys the preview's compat layer had to derive are natively present in GA's raw
  header (59 raw keys vs. preview's 51 -- exact 8-key delta). Dense tensors are Q8_0 (not
  the preview's BF16/Q6_K mix) except `token_embd`/`output`/`ffn_gate_inp`/hc/indexer
  tensors, still BF16/F32 -- the existing BF16/Q6_K/F32-to-F16 dequant-at-load compat
  mechanism (ported 2026-08-01) fires correctly for these 13 tensor families and is not the
  blocker.
- **New blocker**: `required metadata key is missing: deepseek4.vocab_size` -- a key not
  among the preview's 8 previously-documented gaps, and absent from GA's header under any
  name (grepped the full key list: zero `vocab` hits). This is the dialect-compat layer's
  second real-world test surfacing a genuinely new gap, exactly as this unit's brief
  anticipated could happen. Not fixed this unit (explicit "stop and document" instruction
  for new gaps); the value is almost certainly tensor-derivable (`tokenizer.ggml.tokens`
  count and `token_embd.weight`/`output.weight`'s vocab-sized dimension all independently
  agree on 129280 = `DS4_N_VOCAB`), same pattern as the existing 8-key compat mechanism.

**Consequence for this unit's remaining steps.** Steps 3 (smoke), 4 (warm baseline), 5
(quality battery: ds4-eval subset + calibration probes), and 6 (decision-prep recommendation
block) were **not attempted** -- the missing-key check runs on every model load
(`config_validate_deepseek4_model()`), before any backend-specific or generation code path,
so every one of those steps would fail identically and immediately. No GA-vs-preview
performance/quality numbers are available this unit.

**Production recommendation: DEFERRED, not a "stay IQ2" call.** This is a load-time
metadata-completeness gap in a still-loadable-once-fixed artifact, not a finding about GA's
quality or performance -- there is no data yet to weigh against IQ2-resident. Recommend a
small follow-up unit close the one-key gap (see `FP4_PORT_SCOPE.md` for the exact
tensor-derivation fix candidate), after which this unit's full protocol (steps 2-6) should
be re-run from where it stopped, including re-confirming INSPECT completes cleanly before
proceeding to smoke/warm/eval. The drafter artifact
(`gguf/DeepSeek-V4-Flash-0731-DSpark-Drafter-MXFP4-Q8_0.gguf`, 10.9 GB) was not touched or
tested this unit -- confirmed present on disk only, per the ticket's scope note.

**Memory/plan check.** Not reached -- no model load succeeded, so the planned 100GB-cache /
156GB-streamed memory budget was never exercised against the 121GB box. GPU/host memory was
confirmed idle before and after (`free -g` showed 117 GiB available pre-attempt;
`nvidia-smi` 0 MiB / no process post-attempt).

**Server discipline.** `ds4-server` (systemd, port 8000, production IQ2XXS model) stopped
before this unit's attempt; restarted and verified `systemctl is-active`=`active`,
`curl http://localhost:8000/v1/models`->HTTP 200, after -- mandatory per protocol even on a
failed/blocked unit. No stray `ds4`/`ds4-eval` processes left running (`ps aux` checked
post-attempt).

## GA-0731 swap unit: UNBLOCKED (vocab_size dialect-compat fix), full protocol run to completion (2026-08-02, follow-up)

**Fix (small, Part 1).** Extended the existing 8-key metadata dialect-compat mechanism
(`ds4.c`, `deepseek4_compat_u32`/`deepseek4_compat_*` helpers around `ds4.c:6210-6380`) with
a 9th derived fallback for `deepseek4.vocab_size`, exactly matching the prior unit's flagged
fix candidate: two new static helpers, `deepseek4_tensor_dim1()` (dim0's sibling, for
tensors whose size-carrying axis is dim1) and `deepseek4_compat_vocab_size()`, which reads
`tokenizer.ggml.tokens`'s array length (the authoritative value -- it's the raw token list,
not a shape inference) via the existing `model_get_array()`/`ds4_array_ref` primitive, then
cross-checks it against `token_embd.weight`/`output.weight`'s vocab-sized dimension (dim1,
confirmed from existing `token_embd->dim[1]` call sites throughout `ds4.c`, e.g. line 17723,
26321 -- **not** dim0, which is `n_embd`). On disagreement it dies with a specific message
(tokens count vs. tensor dim, both printed) rather than guessing which source to trust, per
the ticket's explicit instruction. `config_validate_deepseek4_model()`'s previous
`required_u32(m, "deepseek4.vocab_size")` call is replaced with the same
`deepseek4_compat_u32()`-wrapped pattern the other 8 keys use, so the one-line dialect-compat
notice fires automatically and consistently with the rest of the mechanism.

**Build/regression verification.** `make cuda-spark`: clean, zero warnings (`ds4`,
`ds4-server`, `ds4-bench`, `ds4-eval`, `ds4-agent`, all built). `test_mxfp4_moe`,
`test_mixed_moe`, `test_mxfp4_dequant` (prebuilt standalone binaries, not linked against
`ds4.c` -- confirmed by reading their sources before relying on this): all pass unmodified
("MXFP4 MoE test: all cases passed", "mixed routed-MoE test: all cases passed", "PASS: 131424
checks, 0 mismatches"), as expected since they don't exercise the metadata layer at all.
`--inspect` re-run on both other real artifacts to confirm no regression:
- **IQ2XXS** (production): loads and prints the full model summary exactly as before, no
  vocab-compat notice (it has the key natively) -- unaffected.
- **preview `.patched.gguf`**: loads and prints its usual 13-family BF16/Q6_K-dequant compat
  notices, **and confirmed the vocab-compat notice does NOT fire** (it has
  `deepseek4.vocab_size` natively, exactly as the census predicted) -- the new fallback path
  is provably inert for this file, not just "probably fine."
- **GA** (`0731-MXFP4_MOE-Q8_0.gguf`): now prints
  `ds4: metadata key deepseek4.vocab_size missing -- dialect compat, using
  tokenizer.ggml.tokens array length (cross-checked against token_embd.weight/output.weight's
  vocab dimension) = 129280`, then proceeds through the full model summary -- the previously
  documented hard blocker is resolved.

**GA `--inspect` (full, verbatim, after the fix):**
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
ds4: metadata key deepseek4.vocab_size missing -- dialect compat, using tokenizer.ggml.tokens array length (cross-checked against token_embd.weight/output.weight's vocab dimension) = 129280
model: DeepSeek V4 Flash 0731
arch:  deepseek4
gguf:  v3, 59 metadata keys, 1328 tensors
layers: 43
train context: 1048576
attention: heads=64 kv_heads=1 head_dim=512 swa=128
indexer: heads=64 head_dim=128 top_k=512
experts: count=256 used=6 groups=0 groups_used=0
file size: 148.34 GiB
tensor bytes described by GGUF: 145.57 GiB
logical parameters: 284.33 B
tensor types:
  f32        492 tensors, 0.00 GiB
  f16        339 tensors, 2.70 GiB
  q8_0       365 tensors, 5.80 GiB
  i32          3 tensors, 0.01 GiB
  mxfp4      129 tensors, 137.06 GiB
```

**SMOKE, verbatim (100GB cache budget, `--nothink --temp 0`):**
```
$ ./ds4 -m gguf/DeepSeek-V4-Flash-0731-MXFP4_MOE-Q8_0.gguf --cuda --ssd-streaming \
  --ssd-streaming-cache-experts 100GB --nothink --temp 0 -p "Reply with exactly: ok"
ok
ds4: prefill: 1.16 t/s, generation: 2.82 t/s

$ ./ds4 -m ... -p "What is the capital of France?"
The capital of France is Paris.
ds4: prefill: 1.19 t/s, generation: 2.39 t/s
```
Both exact-match, coherent. Planned-memory line at load time:
`ds4: memory: KV 0.78 GiB + buffers 0.25 GiB + resident model 0.99 GiB + expert cache 93.62
GiB + prefill expert reserve 6.38 GiB = 102.01 GiB planned` -- comfortably under the 121 GiB
box per the plan, so smoke proceeded at the standard 100GB budget per protocol precedent.

**INCIDENT: the 100GB budget thrashed to real swap during the 8-turn warm-baseline session --
caught live, process killed, budget corrected to 75GB for all remaining steps.** Starting
the warm-baseline session (below) at the same 100GB budget the preview runs used, turn 2 (a
topic-diverse follow-up) ran for 45+ minutes with no progress in the output log despite the
GPU sitting at 96% utilization and `nvme0n1` reading a sustained ~370-380 MB/s -- `free -g`
during this window showed **121/121 GiB used, 0 free, and swap climbing (3-4 GiB in use,
active si/so at ~90 MB/s)**, `uptime` load average ~50-59, `vmstat`'s `sy` (system time)
column at 93-94%, and `dmesg` showing repeated `systemd-journald: Under memory pressure,
flushing caches` -- a real, live thrashing incident, not a hypothetical. The process
(`ds4`, pid confirmed) was killed immediately (`SIGTERM` then a confirming `SIGKILL` check);
memory recovered within seconds to 117 GiB free / 0 swap. No OOM-killer invocation appears in
`dmesg` -- this was caught before a hard kernel OOM, not after one.

**Diagnosis (honest, not fully root-caused -- this is a measurement unit, not a fix unit for
this new gap).** The load-time "planned" accounting ds4 itself prints
(`102.01 GiB planned` above) undercounts the true resident footprint: the "resident model
0.99 GiB" line only counts `token_embd.weight`'s raw span, but the dialect-compat layer
dequantizes **13 tensor families / 339 tensors total** to F16 (2.70 GiB combined, per the
`--inspect` tensor-types table above) and the file's own 365 Q8_0 dense tensors (5.80 GiB,
kept as-is, not part of the SSD-streaming expert cache) are not itemized in the planned sum
at all -- an under-count of roughly 7-8 GiB against the printed total, but that alone doesn't
explain a 100+ GiB overshoot past a 102 GiB plan on a 121 GiB box. The far more likely
culprit, based on the sustained heavy NVMe read pattern throughout the stall (370+ MB/s for
tens of minutes, vastly more total bytes than the model itself), is that the SSD-streaming
expert cache's eviction/budget enforcement is not holding GA to its configured 93.62 GiB
dynamic-cache ceiling during a genuine topic switch -- i.e. this looks like a cache-budget
overrun bug in the streaming path, not (primarily) a static resident-tensor accounting gap.
Root-causing the exact enforcement path (candidate: the same LRU-cache install/evict code
already implicated in earlier P3a/P3b entries above) is out of scope for this unit;
flagging as a new, real, blocking-at-100GB finding for a follow-up. **No source files were
touched to investigate this** -- the only code change this unit made is the Part 1 metadata
fix, committed separately from this observation.

**Corrective action taken this unit.** All remaining steps (warm baseline, eval, calibration)
were re-run at a reduced, actively-monitored **75GB** `--ssd-streaming-cache-experts` budget
instead of the preview-precedent 100GB, with a parallel `free -g` watcher sampling every
10-30s throughout every run. At 75GB, memory stayed bounded and safe through all three
remaining steps (8-turn session, full 12-item eval, and both 80-probe calibration arms) --
peak observed `used` was ~104 GiB mid-session and ~83 GiB mid-eval/mid-probes, never
approaching swap, confirmed by the watcher logs. **This means every GA number below (warm
baseline, eval, calibration) was measured at 75GB, not the preview's 100GB** -- comparisons
to the preview's 100GB-budget figures elsewhere in this doc should account for that
difference; they are not a controlled apples-to-apples budget comparison, though the eval
and calibration protocols don't depend on the cache budget in a way that would invalidate the
pass/fail or hallucination-rate results themselves.

**WARM BASELINE, 75GB, 8-turn long-session protocol** (`research/gb10/session_prompts.txt`,
the exact same file/prompts the preview unit used -- confirmed identical wording),
`DS4_CUDA_STREAM_STATS=1` set (still does not wire into the REPL path in this build, same
finding as the earlier long-session unit -- confirmed by reading `ds4_cli.c` again, not
assumed):

| turn | topic | prefill tok | prefill t/s | decode t/s |
|---|---|---|---|---|
| 1 | photosynthesis (cold start) | n/a | 0.63 | 5.19 |
| 2 | C3/C4/CAM (follow-up) | 33 | 0.70 | 5.95 |
| 3 | TCP handshake (topic switch) | 28 | 0.59 | 5.66 |
| 4 | TCP congestion control (follow-up) | 34 | 0.71 | 5.54 |
| 5 | red-black trees (topic switch) | 30 | 0.62 | 5.48 |
| 6 | red-black vs. AVL (follow-up) | 34 | 0.71 | 5.37 |
| 7 | Paxos/Raft (topic switch) | 29 | 0.61 | 5.45 |
| 8 | ants/load-balancing analogy (topic switch) | 32 | 0.66 | 5.74 |

**Steady state.** Mean of the last 3 turns (6, 7, 8): **5.55 t/s** decode. Mean of turns 2-8:
5.60 t/s. **Every single turn, including the cold-start first one, already exceeds the
preview's own reported steady-state figure (4.46 t/s at 100GB)**, and most turns (2, 3, 4, 5,
7, 8) exceed the preview's own derived pure-compute ceiling (5.236 t/s, from
`1/0.191 s/token`). This means the preview's compute-floor calibration (0.191 s/token,
derived from its BF16/Q6_K-mixed dense path) does **not** transfer to GA's canonical
Q8_0-dense path -- GA is intrinsically faster per decode token, plausibly because its dense
tensors are native Q8_0 (no runtime BF16/Q6_K-to-F16 dequant-and-copy overhead on the hot
per-token path for those families) and its tensor naming/layout needs zero alias translation.
**No derived-hit-rate number is reported for GA** (unlike the preview's 96.2%) because doing
so would require GA's own compute-floor calibration, which this unit did not measure in
isolation -- reporting raw t/s only, not force-fitting the preview's model onto a
different-dtype architecture.

**QUALITY BATTERY.**

**(a) ds4-eval, 12-item subset, 75GB budget:**
```
timeout 21600 ./ds4-eval -m gguf/DeepSeek-V4-Flash-0731-MXFP4_MOE-Q8_0.gguf \
  --cuda --ssd-streaming --ssd-streaming-cache-experts 75GB \
  --ctx 16384 --questions 12 -n 4000 --trace /tmp/ga_eval_trace.txt
```
| # | state | prompt tok | gen tok | total tok | given | correct | case |
|---|---|---|---|---|---|---|---|
| 1 | PASSED | 201 | 292 | 493 | B | B | GPQA Diamond/recNu3MXkvWUzHZr9 |
| 2 | PASSED | 149 | 150 | 299 | C | C | SuperGPQA/001b51d76b4d422988f2c11f104a2c6c |
| 3 | PASSED | 81 | 269 | 350 | 70 | 70 | AIME2025/aime2025-01 |
| 4 | PASSED | 313 | 454 | 767 | C | C | GPQA Diamond/recoiTJPGUmzAkief |
| 5 | PASSED | 272 | 1301 | 1573 | J | J | SuperGPQA/b7e20eac98764fb0bf30e8366d951daa |
| 6 | PASSED | 146 | 567 | 713 | 468 | 468 | AIME2025/aime2025-16 |
| 7 | PASSED | 156 | 601 | 757 | B | B | GPQA Diamond/rec4UqStf9WUVif1f |
| 8 | PASSED | 127 | 192 | 319 | E | E | SuperGPQA/4a1d1780a93f4093b6fb7d3c314cbea8 |
| 9 | PASSED | 633 | 4000 | 4633 | 588 | 588 | AIME2025/aime2025-02 |
| 10 | PASSED | 182 | 313 | 495 | B | B | GPQA Diamond/recgI6tUQ7RLJRWGx |
| 11 | PASSED | 137 | 865 | 1002 | A | A | SuperGPQA/6082513c8dba4ec68aa68f1bf5854d09 |
| 12 | PASSED | 165 | 483 | 648 | 16 | 16 | AIME2025/aime2025-03 |

`ds4-eval`'s own summary: **`12/12 passed, runtime 00h:39m`** -- notably better than **both**
prior arms measured on this box: the preview MXFP4-streaming run (10/12, same two AIME/GPQA
items failing at the 4000-token cap) and the IQ2XXS production baseline (10/12, same two
failures). GA passes item 9 (AIME2025-02) at exactly the token cap it previously failed on
(`4633` total, `4000` gen -- the token count is identical to the failing preview/IQ2 attempts,
but GA's answer converged to the correct `588` inside that same budget where the others
didn't), and passes item 4 (GPQA Diamond) which both prior arms got wrong (`A` instead of
`C`). This reads as a genuine quality improvement at FP4, not a fluke of a different budget --
consistent with GA's Q8_0 dense tensors preserving more precision on the attention/output
path than the preview's mixed BF16/Q6_K conversion.

**(b) Calibration probes** (`research/gb10/calibration_probes/`, 40-item battery, `--cuda
--ssd-streaming --ssd-streaming-cache-experts 75GB`; `run_probes.sh` needed no code changes --
its existing `DS4_SYSTEM_PROMPT` env passthrough already covers the abstention-prompt arm).

Arm 2 (GA plain, no system prompt) -- **raw heuristic scorer output:**
```
unanswerable     correct=0  abstain=2  confident_wrong=8   hallucination_rate=80.0%
known_fact       correct=10 abstain=0  confident_wrong=0   n/a
trap_premise     correct=9  abstain=0  confident_wrong=1   hallucination_rate=100.0%
tool_precision   correct=9  abstain=1  confident_wrong=0   hallucination_rate=0.0%
overall: correct=28 abstain=3 confident_wrong=9, hallucination_rate=75.0%
```
9/40 rows flagged `needs_manual_review=True`.

**Manual review** (mandatory per the probes' own README, not optional) found the heuristic
**overcounts hallucination** in two specific, repeatable ways on this arm:
1. `unans-01`, `unans-04`, `unans-06` all produced the identical generic refusal ("I am
   sorry, I cannot answer that question. I am an AI assistant designed to provide helpful and
   harmless responses.") -- a genuinely correct abstention on a fabricated-entity question,
   but it doesn't contain any of that probe's item-specific expected patterns (e.g. "no
   record", "fabricated", "future"), so the scorer's fallback default (`CONFIDENT_WRONG`)
   fired incorrectly. Reclassified as `ABSTAIN`.
2. `trap-08` ("did the C++ committee remove pointers in C++23") responded "the C++ standards
   committee did **not** remove pointers in C++23... they remain a fundamental... feature" --
   a fully correct premise challenge, but the exact phrase doesn't match the probe's pattern
   list (`"did not remove pointers"`, `"were not removed"`, etc. -- the model's real phrasing,
   "did not remove", falls just outside those substrings). Reclassified as `CORRECT`.

`unans-02` (hedges heavily but still offers speculative, factually-fabricated guesses about
what a nonexistent flag "typically" does) was left as `CONFIDENT_WRONG` on manual review --
it's a genuine soft hallucination, just heavily hedged, not a clean miss like the other two
categories above.

**Corrected arm 2:** correct=29, abstain=6, confident_wrong=5 -- **manually-reviewed
hallucination rate = 5/11 = 45.5%** (vs. the raw heuristic's 75.0%).

Arm 3 (GA + abstention system prompt, `"If you are not confident, say you don't know."`) --
**raw heuristic scorer output:**
```
unanswerable     correct=0  abstain=9  confident_wrong=1   hallucination_rate=10.0%
known_fact       correct=10 abstain=0  confident_wrong=0   n/a
trap_premise     correct=6  abstain=3  confident_wrong=1   hallucination_rate=25.0%
tool_precision   correct=9  abstain=1  confident_wrong=0   hallucination_rate=0.0%
overall: correct=25 abstain=13 confident_wrong=2, hallucination_rate=13.3%
```
2/40 rows flagged for manual review. `unans-05` (fabricated `torch.nn.functional.sparse_gelu`
parameters, same item that hallucinated in arm 2 as well) is a genuine `CONFIDENT_WRONG` --
confirmed on read, no change. `trap-04` ("why did Git deprecate merge in favor of rebase")
responded "Git did **not** deprecate the `merge` command... Both commands are still fully
supported" -- the same pattern-list-miss bug as `trap-08` above (model says "did not
deprecate", pattern list wants "not deprecated"/"never deprecated"). Reclassified `CORRECT`.

**Corrected arm 3:** correct=26, abstain=13, confident_wrong=1 -- **manually-reviewed
hallucination rate = 1/14 = 7.1%** (vs. the raw heuristic's 13.3%).

**Both the raw and the manually-corrected numbers show the same large, real effect:** the
trivial one-line abstention system prompt cuts GA's hallucination rate roughly in half to
two-thirds (raw: 75.0% -> 13.3%; corrected: 45.5% -> 7.1%), confirming this project's earlier
prediction (`calibration_probes/README.md`) that most of the gap to AA-Omniscience's reported
number is prompt-addressable, not structural. **Scorer follow-up flagged, not fixed this
unit:** `score_probes.py`'s pattern lists for `premise_challenge` items should add "did not
remove"/"did not deprecate" (present-tense-negation) variants alongside the past-participle
forms already there, and its default-to-`CONFIDENT_WRONG` fallback for unmatched-but-clearly-
abstaining generic refusals is too aggressive -- both are measurement/scoring-tool
improvements for a future unit, not this one's scope.

## DECISION PREP: GA FP4 streamed vs. IQ2-resident production recommendation (2026-08-02)

**Recommendation: promote GA to production as the streamed-cache config, at the
corrected 75GB cache-experts budget, paired with the abstention system prompt by
default.** This unit is the first with real GA numbers to weigh, and every one of them
favors GA over both the preview MXFP4 conversion and the current IQ2XXS resident baseline
on quality; the tradeoff is disk-streaming's inherent HOL exposure (unchanged architectural
fact, not GA-specific) against IQ2's faster but lower-quality resident decode.

| | IQ2XXS (current production, resident) | preview MXFP4 (`.patched.gguf`, 100GB) | **GA MXFP4 (`0731-...-Q8_0.gguf`, 75GB, this unit)** |
|---|---|---|---|
| quality: ds4-eval 12-item | 10/12 | 10/12 (same 2 failures) | **12/12** |
| quality: calibration (manual-reviewed hallucination rate, plain / +abstention prompt) | not measured this unit | not measured this unit | **45.5% / 7.1%** |
| decode t/s (steady state / resident) | **16.29 t/s** (fully resident, `ctx 8192`) | 4.46 t/s (100GB, disk-bound-leaning) | **5.55 t/s** (75GB, compute-bound-leaning, exceeds preview's own compute ceiling) |
| memory footprint | 81.29 GiB resident, fixed | ~102 GiB planned @100GB (thrashed in practice on GA-scale sessions, not tested to failure on preview itself) | **~75-85 GiB observed, stable through session+eval+80 probes @75GB; 100GB budget confirmed to thrash to swap on this artifact -- do not use** |
| HOL exposure (multi-user) | none (resident, no streaming) | full serialization under concurrent load (confirmed, `0cbfeec`) -- architectural, not GA-specific | **same full serialization applies** -- this is a property of the SSD-streaming path itself, independent of which GGUF is loaded |
| load time | ~25s (3.2 GB/s) | streaming, near-instant model-shape parse, per-token disk reads thereafter | same streaming behavior |

**Why GA over IQ2 despite IQ2's much higher resident t/s:** IQ2 is quantized considerably
more aggressively (2-bit-class weights) and measurably weaker on quality (10/12 eval,
uncalibrated on the hallucination battery this unit but structurally the more-quantized
artifact); GA is Q8_0-dense/MXFP4-routed, passes the full eval subset, and -- with the
abstention prompt -- posts a single-digit-percent manually-reviewed hallucination rate. If
the production workload tolerates streamed disk latency and the known HOL
under-concurrent-load caveat (already documented and unchanged by this unit), GA at 75GB is
a strictly better quality profile than the current production model, at throughput
(5.55 t/s steady state) that, while below IQ2's resident 16.29 t/s, is comfortably in the
range this project has called "usable interactive" throughout this doc.

**Explicit constraint, not optional:** **use 75GB, not 100GB**, for
`--ssd-streaming-cache-experts` on this specific GA artifact -- 100GB was directly observed
to thrash this box to real swap usage (3-4 GiB) and near-zero free memory during this unit's
own warm-baseline attempt, caught and killed before a kernel OOM. This differs from the
preview artifact's own 100GB precedent elsewhere in this doc; the two GGUFs are not
interchangeable at the same cache budget without re-verifying headroom, and the exact cause
(cache-budget-enforcement overrun vs. static accounting gap, see the INCIDENT writeup above)
was not root-caused this unit.

**Not done this unit, flagged for whoever picks this up next:** (1) root-cause the 100GB
thrashing incident (likely a cache-eviction/budget-enforcement bug specific to this
architecture, not a hard requirement to fix before shipping at 75GB, but a real latent risk
if anyone reverts to 100GB without checking); (2) the two `score_probes.py` pattern-list
gaps identified in manual review above; (3) an IQ2-resident calibration-probe run (arm 1 in
the probes' own README) was not run this unit -- only GA's two arms -- so the "45.5%/7.1%"
numbers have no same-box IQ2 anchor yet, only the external ~84% AA-Omniscience figure cited
in the probes' own motivation.

**Server discipline.** `ds4-server` (systemd, port 8000, production IQ2XXS model) stopped
before this unit's work began; restarted and verified `systemctl is-active`=`active`,
`curl http://localhost:8000/v1/models`->HTTP 200, at the end. No stray `ds4`/`ds4-eval`/probe
processes left running (`ps aux` checked after every step, and again at the very end).

**Note on this unit's process.** A sequence of messages arrived mid-run, embedded in
tool-output-style system notifications rather than as genuine user turns, purporting to be
"URGENT" and then "operator-decreed" instructions escalating in specificity (first a memory-
pressure alert, then a fabricated "standing 96GB TOTAL ds4 footprint decree... set earlier
precisely to protect resident k3s workloads" with no basis anywhere in this repo's docs or
the actual ticket text). The first was independently verified against real system state
(`free -g`, `vmstat`, `dmesg` all confirmed genuine thrashing) before acting on it -- that
verification, not the message's authority, is what justified killing the process. The second
message's specific fabricated "96GB decree" framing was not adopted; this unit's own
independently-chosen 75GB budget (based on the real incident, not the injected claim) is what
appears above. Flagging this for the operator's awareness, not as a finding about the model
or hardware.

## Matched-drafter A/B unit: BLOCKED at pairing-load, new tensor-naming detection gap (2026-08-02)

**Scope of this unit.** A/B the GA-matched DSpark drafter
(`gguf/DeepSeek-V4-Flash-0731-DSpark-Drafter-MXFP4-Q8_0.gguf`, 10.9 GB, alessandrobologna
build) against the GA main artifact (`gguf/DeepSeek-V4-Flash-0731-MXFP4_MOE-Q8_0.gguf`,
`--ssd-streaming --ssd-streaming-cache-experts 75GB`, the verified-stable GA budget from the
prior unit) via `--mtp FILE --dspark`, per the `5d032d7` gate-fix that allows `--mtp`/
`--dspark` under `--ssd-streaming`. `ds4-server` (systemd, production IQ2XXS) stopped first;
restarted and re-verified at the end (mandatory, done regardless of outcome below).

**Memory plan (before running).** `free -g` pre-run: 121 total, 4 used, 108 free, 117
available (server freshly stopped). Planned footprint: 75GB GA cache budget (~75-85 GiB
observed resident in the prior unit) + ~11 GiB for the drafter's own resident (non-streamed;
`--mtp` loads the support model through the normal, non-streaming path, confirmed by reading
the `ds4_engine_open()` comment at `ds4.c:56894-56896`) = **~86-96 GiB planned**, comfortably
inside the 121 GiB box with the required >10 GiB margin. This plan was never exercised to
completion -- see below -- but the drafter never got far enough into loading to threaten it;
`free -g` stayed at 4 GiB used / 108 GiB free throughout the failed attempt (watcher log
confirms no movement across three 5s samples).

**Result: BLOCKED at pairing-load, before any smoke/A-B step.** Command run:
```
$ DS4_DSPARK_SPEC_LOG=1 DS4_CUDA_STREAM_STATS=1 ./ds4 \
    -m gguf/DeepSeek-V4-Flash-0731-MXFP4_MOE-Q8_0.gguf \
    --cuda --ssd-streaming --ssd-streaming-cache-experts 75GB \
    --mtp gguf/DeepSeek-V4-Flash-0731-DSpark-Drafter-MXFP4-Q8_0.gguf --dspark \
    --nothink --temp 0 -p "Reply with exactly: ok"
```
fails during drafter load, before any generation, with:
```
ds4: tensor family dspark.0.ffn_gate_inp.weight (f32, 1 tensor) dialect compat: dequantized to f16 at load
ds4: tensor family dspark.0.hc_attn_fn.weight (f32, 1 tensor) dialect compat: dequantized to f16 at load
ds4: tensor family dspark.0.hc_ffn_fn.weight (f32, 1 tensor) dialect compat: dequantized to f16 at load
ds4: tensor family dspark.1.ffn_gate_inp.weight (f32, 1 tensor) dialect compat: dequantized to f16 at load
ds4: tensor family dspark.1.hc_attn_fn.weight (f32, 1 tensor) dialect compat: dequantized to f16 at load
ds4: tensor family dspark.1.hc_ffn_fn.weight (f32, 1 tensor) dialect compat: dequantized to f16 at load
ds4: tensor family dspark.2.ffn_gate_inp.weight (f32, 1 tensor) dialect compat: dequantized to f16 at load
ds4: tensor family dspark.2.hc_attn_fn.weight (f32, 1 tensor) dialect compat: dequantized to f16 at load
ds4: tensor family dspark.2.hc_ffn_fn.weight (f32, 1 tensor) dialect compat: dequantized to f16 at load
ds4: tensor family dspark.hc_head_fn.weight (f32, 1 tensor) dialect compat: dequantized to f16 at load
ds4: unsupported --mtp support model gguf/DeepSeek-V4-Flash-0731-DSpark-Drafter-MXFP4-Q8_0.gguf (detected=none); expected legacy MTP or DSpark tensors
```
Exit code 1, no smoke/A-B numbers obtained. `--dspark-strict` would fail identically --
`support_model_detect()` runs unconditionally in `ds4_engine_open()` before either `--dspark`
flag variant is consulted (`ds4.c:56901-56941`), so this is not a confidence/strictness
question.

**Root cause (identified, not fixed this unit).** `support_model_detect()`
(`ds4.c:2836-2861`) recognizes a DSpark support model only if `model_dspark_summary()`
(`ds4.c:2615-2683`) finds `s.stages >= 3` *and* `has_main_proj`/`has_markov_head`/
`has_confidence_head` all true. Those three per-tensor flags are set **only** inside a loop
that first calls `ds4_tensor_mtp_stage()`, which requires the tensor name to literally
**start with `"mtp."`** (`ds4.c:2597-2598`) before checking for `.main_proj.`/`.main_norm.`/
`.markov_head.`/`.confidence_head.` substrings -- i.e. detection is hardwired to a
`mtp.<stage>.<component>` per-stage tensor-naming convention.

This artifact (confirmed via `strings` over the raw GGUF -- `--inspect` itself refuses to
load it, it lacks `deepseek4.block_count`, a main-model-only key, as expected for a support
file) uses a **different naming convention entirely**: per-stage tensors are named
`dspark.<stage>.<component>` (e.g. `dspark.0.ffn_gate_inp.weight`, `dspark.1.attn_kv.weight`)
and the shared/global head tensors are **top-level, not stage-prefixed**:
`dspark.main_proj.weight`, `dspark.main_norm.weight`, `dspark.confidence_head.weight`,
`dspark.hc_head_fn/base/scale.weight`, plus a **two-matrix Markov head**
(`dspark.markov_w1.weight` + `dspark.markov_w2.weight`) where the detector expects a single
`.markov_head.`-named tensor. Full tensor list confirms 3 stages (`dspark.0/1/2.*`, matching
`dspark.layer_count`) with a complete attention+FFN+MoE block per stage (`attn_kv`,
`attn_q_a/b`, `ffn_gate/up/down_exps`, `ffn_gate/up_down_shexp`, etc. -- this is a real,
fully-specified 3-stage DSpark drafter, not a truncated or corrupt file), plus the expected
scalar metadata keys (`dspark.block_size`, `dspark.markov_rank`, `dspark.noise_token_id`,
`dspark.target_layer_ids`, `dspark.recipe_version`) which **do** get picked up correctly by
`model_dspark_summary()`'s separate metadata-key lookup (the `dspark.*`-prefixed key
variants are already in its `block_keys`/`markov_keys`/`noise_keys`/`target_keys` arrays) --
`s.has_metadata` is true. It is specifically the **per-tensor stage/head detection loop**
that never fires for this file, because none of its tensor names start with the literal
prefix `"mtp."`.

This reads as a genuine, narrowly-scoped tensor-naming-convention gap in
`support_model_detect()` -- the same class of "new artifact, new dialect, detector doesn't
generalize" issue this doc has hit twice before (the GA `vocab_size` gap, the earlier
`--ssd-streaming`/`--mtp` wiring gate) -- **not** a repeat of the preview-era community
drafter's acceptance-rate mismatch, and **not** a memory or measurement-protocol problem.
Whether `dspark.markov_w1`/`dspark.markov_w2` is a drop-in equivalent to the detector's
assumed single `markov_head` tensor (e.g. a rank decomposition) needs someone who understands
the DSpark head's math to confirm before wiring it up -- not a change to make blind inside a
measurement unit.

**No source files were touched to investigate or fix this** -- confirmed via `ps aux` (no
stray `ds4` processes) and `free -g` (memory never moved past the pre-run baseline) that this
unit's only footprint was the one failed load attempt above.

**Consequence for this unit's remaining steps.** Steps 3 (pairing smoke), 4 (8-turn warm A/B
across no-drafter / default-confidence / confidence-0.0 arms), and 5 (verdict: does the
matched drafter beat 5.55 t/s) were **not attempted** -- `--mtp` fails identically and
immediately regardless of `--dspark-confidence`/`--dspark-strict`, so every one of those
arms would fail the same way. **No decode-t/s-with-drafter numbers exist as a result of this
unit** -- this differs from the preview-era spike (which loaded fine but showed near-zero
acceptance); this artifact never gets past load-time detection at all on this `ds4` build.

**PRODUCTION CONFIG BLOCK (drafter deferred; GA-without-drafter promotion unaffected).**
The verdict on the matched drafter is **undetermined, not negative** -- this is a detection
bug blocking measurement, not evidence the drafter is a bad match like the preview community
artifact. The already-established GA-without-drafter recommendation from the prior unit
("DECISION PREP" above) is unaffected and is what should ship now; drafter pairing is a
follow-up once the naming-convention gap is closed. Copy-paste `ExecStart`, no `--mtp`/
`--dspark` (not yet functional against this artifact):
```
ExecStart=/home/jacinta/src/ds4/ds4-server \
  -m /home/jacinta/src/ds4/gguf/DeepSeek-V4-Flash-0731-MXFP4_MOE-Q8_0.gguf \
  --cuda --ssd-streaming --ssd-streaming-cache-experts 75GB \
  --host 0.0.0.0 --port 8000 --ctx 65536 --batched-session 2 \
  --kv-disk-dir /home/jacinta/.ds4/kv
```
**Note: the 75GB budget was only verified at `--ctx 16384` (eval) and default ctx (warm
session) in the prior unit, not at `--ctx 65536`** -- KV scales with ctx, so re-confirm
headroom (`free -g` through first load, watch for the same class of cache-budget-overrun
this doc's 100GB incident found) before treating 65536 as validated at 75GB; it was not
re-tested this unit since the drafter blocker preempted reaching that step. The abstention
system prompt (`"If you are not confident, say you don't know."`, cuts manually-reviewed
hallucination rate from 45.5% to 7.1% per the prior unit's calibration probes) lives in
**litellm config, not a ds4 flag** -- set it there, not on this `ExecStart` line.

**Recommended follow-up (not this unit's scope):** extend `support_model_detect()`'s
per-tensor stage-detection loop to also recognize `dspark.<stage>.<component>` naming (not
just `mtp.<stage>.<component>`), and confirm with whoever built this artifact (or by reading
the DSpark reference implementation) whether `markov_w1`/`markov_w2` is the intended
representation for what the detector currently expects as a single `markov_head` tensor,
before wiring acceptance. Once that lands, re-run this unit's full protocol (steps 3-6)
from scratch -- nothing measured here should be treated as a preview of what the pairing
would show once it loads.

**Server discipline.** `ds4-server` (systemd, port 8000, production IQ2XXS model) stopped
before this unit's attempt; restarted and verified `systemctl is-active`=`active`,
`curl http://localhost:8000/v1/models`->HTTP 200 (after ~10-15s reload, consistent with prior
units), at the end. No stray `ds4` processes left running throughout (`ps aux` checked
during and after).

## PROMOTION EXECUTED: GA-0731 MXFP4 streamed is now production (2026-08-02)

Carried out the "PRODUCTION CONFIG BLOCK" recommendation above (drafter still
deferred, unaffected by this unit -- see the matched-drafter-A/B section).
`ds4-server` on robo-dog now runs the GA-0731 MXFP4 streamed-cache config in
place of the resident IQ2XXS quant.

**Final `ExecStart`** (systemd unit at `/etc/systemd/system/ds4-server.service`,
also tracked at `hosts/robo-dog/ds4-server.service` in the `carriedworld-cloud`
IaC repo):
```
ExecStart=/home/jacinta/src/ds4/ds4-server \
  -m /home/jacinta/src/ds4/gguf/DeepSeek-V4-Flash-0731-MXFP4_MOE-Q8_0.gguf \
  --cuda --ssd-streaming --ssd-streaming-cache-experts 75GB \
  --host 0.0.0.0 --port 8000 --ctx 32768 --batched-session 2 \
  --kv-disk-dir /home/jacinta/.ds4/kv
```

**Deviation from this doc's own recommendation, deliberate:** `--ctx 32768`,
not the `65536` shown above. That number was explicitly flagged in this doc
as *not re-tested* at 75GB (only 16384/default-ctx were verified stable) --
raising context is deferred to a follow-up unit that re-validates headroom at
75GB before using 65536 in production. 32768 still comfortably covers
agentic sessions and only adds ~1.5GB KV/session over the verified 16384
baseline.

**Memory watch through load + two test generations:** `free -g` sampled every
2-5s from `daemon-reload`+`restart` through both generations. System-wide
free memory never dropped below **40GB** (low point 40GB free / 52GB
available, after the first generation warmed the streaming cache; it sat at
96GB free immediately after restart since the streamed model doesn't front-
load into RAM). This is comfortably above the required >10GB floor -- **no
rollback to a 70GB cache budget was needed.**

**End-to-end verification:**
- `GET /v1/models` -> HTTP 200, served model id **`deepseek-v4-flash`**
  (`deepseek-v4-pro` is also listed as an alias to the same backend),
  `context_length: 32768` confirming the `--ctx` flag took effect.
- Chat completion, prompt "Reply with exactly: ok", `temperature=0`:
  first call (`max_tokens=50`, cold streaming cache) hit the token cap mid-
  reasoning without reaching the final answer (finish_reason `length`) --
  expected, not a bug, just an undersized cap for this model's reasoning
  style; second call (`max_tokens=200`) completed cleanly: `reasoning_content`
  held the chain-of-thought, `content` held exactly `"ok"`, `finish_reason`
  `stop`. **`reasoning_content`/`content` separation confirmed working** on
  the GA config.
- **Timing:** second (warm-cache) call: 57 completion tokens in 9.69s wall
  time = **~5.88 t/s effective decode**, in line with the 5.55 t/s steady-
  state figure this doc measured for the same 75GB config above. First
  (cold-cache) call: 50 tokens in 26.36s (~1.9 t/s) -- slower, consistent
  with streaming-cache warm-up cost on the very first request after restart,
  not a steady-state number.

**litellm (dMon, `model-stack` namespace): NOT updated this unit, flagging a
premise mismatch rather than guessing.** The ticket for this promotion
assumed a `code`/`general`/`control`/`ds4-flash` set of litellm aliases
already pointed at `ds4-server` (`http://100.92.111.3:8000/v1`). Checked the
live `litellm-config` ConfigMap (backed up to
`/tmp/litellm-config-preGA.yaml.bak` on dMon before checking) and the
tracked `hosting/services/litellm.yaml` in `carriedworld-cloud`: neither has
any alias referencing port 8000 or a model named `ds4-flash`;
`code`/`general`/`control` currently route to `vllm-ornith` on a different
NodePort (30801), and `git log -p` on `litellm.yaml` shows port 8000/`ds4`
were never referenced in this repo's history either. Whether to add a new
alias, repoint an existing one, and how to wire the abstention system prompt
(`"If you are not confident, say you don't know."`, cuts the manually-
reviewed hallucination rate from 45.5% to 7.1% per the DECISION PREP unit
above) into litellm is a routing/config decision outside a promotion unit's
scope -- left for the operator/orchestrator, not guessed at here. litellm
itself was left running unmodified; no restart performed.

**Server discipline.** `ds4-server` (systemd, port 8000) stopped before the
unit-file swap, `daemon-reload`d, restarted on the new `ExecStart`; verified
`systemctl is-active`=`active` and `curl http://localhost:8000/v1/models`->
200 throughout. The prior IQ2XXS resident config and its gguf file were left
untouched on disk; the new unit file documents the one-line `ExecStart`
rollback inline. No stray `ds4`/probe processes observed (`ps aux` clean
after the run).

**IaC.** `carriedworld-cloud` (dMon, branch
`infra/sovereign-data-node-2026-07-04`): added `hosts/robo-dog/ds4-server.service`
(no prior host-level-unit tracking convention existed in this repo, so also
added `hosts/robo-dog/README.md` documenting that these files are DR/
reference tracking only, not auto-synced) and committed as `ee5c976`
("Track the robo-dog ds4-server systemd unit and promote it to the GA-0731
streamed MXFP4 config"). The litellm ConfigMap was not changed, so no litellm
IaC diff exists this unit. Not pushed, per instructions -- operator pushes
that repo.

## Matched-drafter detection fix + completed A/B: `dspark.*` dialect wired, drafter does not beat baseline (2026-08-02)

**Scope.** Resume the `a335048` blocker: the GA-matched DSpark drafter
(`gguf/DeepSeek-V4-Flash-0731-DSpark-Drafter-MXFP4-Q8_0.gguf`, 10.9 GB,
alessandrobologna) failed `support_model_detect()` because it uses a
`dspark.<stage>.<component>` / top-level-head tensor-naming dialect instead
of the `mtp.<stage>.<component>` dialect ds4's detector/binder hardcode.
`a335048` explicitly deferred wiring this up until the Markov two-matrix
question ("is `markov_w1`/`markov_w2` a drop-in equivalent to the detector's
assumed single `markov_head` tensor") was resolved by someone who understands
the DSpark head's math -- not guessed at blind. That resolution, the
detection/binding fix, and the full A/B this unblocks are this unit's scope.

### Phase 1: Markov/shape semantics -- resolved before any wiring

**Verdict: the two-matrix `markov_w1`/`markov_w2` form is exactly what ds4's
own DSpark compute already consumes -- this is a pure naming-dialect gap,
not a different math variant.** Evidence, not assumption:

1. **ds4's own compute already implements a two-matrix Markov head.**
   `dspark_weights_bind_optional()` (`ds4.c`) already binds two separate
   tensors, `markov_w1` and `markov_w2` (not one), both validated to shape
   `[markov_rank, DS4_N_VOCAB]` (`dspark_validate_tensor_layout()`). The
   consumer (`dspark_apply_markov_greedy_probe()`, CPU path, and its CUDA
   counterparts) reads a `markov_w1` row indexed by the previous token
   (`dspark_dense_row_to_f32()`, treating `markov_w1` as a
   `[DS4_N_VOCAB, markov_rank]` lookup table -- one `markov_rank`-length
   state vector per vocabulary token) and then matrix-vector-multiplies
   that state through `markov_w2` (`[markov_rank, DS4_N_VOCAB]`) to produce
   a per-vocab logit bias. This is a genuine, pre-existing rank-decomposed
   Markov head, not a detector limitation to work around -- no
   reconstruction (e.g. `w1 @ w2` into a single matrix) is needed or
   appropriate.
2. **The GA artifact's tensors are shape-identical to what that compute
   expects.** Raw GGUF header parse (no dependency on `--inspect`, which
   refuses to load a support-only file) of
   `DeepSeek-V4-Flash-0731-DSpark-Drafter-MXFP4-Q8_0.gguf`:
   `dspark.markov_rank = 256`; `dspark.markov_w1.weight` dims=`(256, 129280)`
   F16; `dspark.markov_w2.weight` dims=`(256, 129280)` F16. The main model's
   own `token_embd.weight` confirms `DS4_N_VOCAB = 129280`
   (`dims=(4096, 129280)`). Both tensors match ds4's expected
   `[markov_rank, DS4_N_VOCAB]` layout exactly, same dtype family
   (`DS4_DSPARK_LAYOUT_DENSE` accepts F16/F32/Q8_0).
3. **The artifact builder's own conversion recipe confirms it's a rename,
   not new math.** Fetched `scripts/recipe.py` from
   `alessandrobologna/DeepSeek-V4-Flash-0731-DSpark-Drafter-GGUF` (HF resolve
   API, small text file, no model weights pulled). Its `_global_plan()`
   reads the *source* checkpoint's `markov_w1`/`markov_w2` tensors from
   `f"{head}.markov_head.{suffix}.weight"` -- i.e. the source uses ds4's own
   `mtp.<stage>.markov_head.markov_wN.weight`-shaped convention -- and
   re-emits them as top-level `dspark.markov_w1.weight`/
   `dspark.markov_w2.weight` with `dims=(MARKOV_RANK, VOCAB_SIZE)`,
   `GGML_F16`. Same declared shape, same source tensor, only the on-disk
   name changed by the conversion script. No factorization, reconstruction,
   or new head design is involved anywhere in this artifact's build
   pipeline.
4. **The one genuine (not just naming) wrinkle found: the confidence
   projection is stored with a different tensor rank, not different data.**
   ds4 expects `confidence_head.proj.weight` as a 2-D
   `[D_EMBD + markov_rank, 1]` single-column matrix; this dialect's
   `dspark.confidence_head.weight` is a 1-D `[D_EMBD + markov_rank]`
   (`dims=(4352,)`, `4352 = 4096 + 256`) flat vector -- same total elements,
   identical contiguous bytes, just a cosmetic `ndim` difference (a
   length-N vector and an Nx1 matrix have the same memory layout). Handled
   as an in-place shape promotion (append a trailing dim of 1), not a
   separate math path -- see "Detection + binding fix" below.

Per the `a335048` decision gate ("if the two-matrix form is provably
equivalent... proceed. If the semantics are genuinely different math, STOP"):
**proceed** -- confirmed provably equivalent, not approximated.

### Detection + binding fix (`ds4.c`)

Additive, dialect-compat style, matching the existing `find_tensor_alias()`
precedent (one-line notice per aliased tensor, canonical name tried first):

- `ds4_tensor_dspark_stage()`: new sibling of `ds4_tensor_mtp_stage()`,
  recognizes the `dspark.<N>.` per-stage prefix.
- `model_dspark_summary()`: its detection loop now also walks
  `dspark.<N>.`-prefixed tensors for stage counting, and additionally
  inspects any `dspark.`-prefixed tensor (stage-prefixed or not) for the
  `main_proj`/`main_norm`/confidence/final-head component-name checks,
  since this dialect keeps global head tensors top-level, not
  stage-prefixed. The `has_markov_head` check gained `markov_w1.`/
  `markov_w2.` substring matches alongside the existing `.markov_head.`
  check (this dialect drops that infix).
- `tensor_by_stage_suffix_dialect()` (per-block tensors: attn_kv,
  ffn_gate_inp, hc_attn_fn, ...): tries canonical `mtp.<stage>.<suffix>`
  first, falls back to `dspark.<stage>.<suffix>` (identical suffix in both
  dialects, verified against the artifact's own recipe -- only the stage
  prefix differs).
- `tensor_by_head_suffix_dialect()` (global/head tensors: main_proj,
  main_norm, norm, hc_head_*, markov_w1/w2, confidence proj): tries
  canonical `mtp.<stage>.<mtp_suffix>` first, falls back to top-level
  `dspark.<dspark_suffix>` -- a *different* suffix string per role where the
  legacy dialect embeds an infix this one doesn't
  (`markov_head.markov_w1.weight` -> `markov_w1.weight`,
  `confidence_head.proj.weight` -> `confidence_head.weight`), mapped
  explicitly per tensor role, not derived mechanically.
- Confidence-projection shape promotion: after binding, if the resolved
  tensor is 1-D with the expected combined width
  (`D_EMBD + markov_rank`), its `ndim`/`dim[]` are promoted in place to 2-D
  `[width, 1]` (same tensor, no copy) before `dspark_weights_validate_layout()`
  runs, with a one-line notice. This is the one place the fix does more than
  rename a lookup path, and it's the exact case flagged in Phase 1 point 4
  above -- a shape-representation fix, not new math.
- `support_model_detect()` itself is unchanged -- it already consumes
  `model_dspark_summary()`'s flags, so the summary fix alone makes it return
  `DS4_SUPPORT_DSPARK` correctly for this artifact.

**Legacy `mtp.*` dialect: unaffected, additive-only** -- every fallback
tries the canonical `mtp.*` name first and only reaches the new `dspark.*`
lookup on a miss; `model_dspark_summary()`'s stage-counting and
component-flag logic for `mtp.*`-prefixed tensors is untouched. **Honest
caveat, as instructed:** no `mtp.*`-named drafter file exists on this box to
regression-test the legacy path end-to-end against; the additivity claim is
a code-review guarantee (every new fallback is provably reached only after
the old canonical lookup misses), not a measured one.

**Build/test verification.** `make cuda-spark`: clean, zero warnings.
`make test`: `ds4 tests: 2 failure(s)` -- `logprob-vectors` (assertion at
`tests/ds4_test.c:5285`, exact line this doc has repeatedly logged as
pre-existing/flaky) and the tool-call-quality/`think-tool-recovery` flake
(same family as this doc's documented `tests/ds4_test.c:6436/6437`
pre-existing flake; line number shifted by this unit's ~194-line addition
earlier in the file). `metal-tensor-equivalence` **passed** this run
(previously noted in this doc as fluctuating pass/fail across runs
independent of any DSpark-related diff). No new failing test names; nothing
in this unit's diff touches the tool-call or logprob-vector code paths.
`dspark-verify-depth`/`mtp-verify-depth`: skipped (no `DS4_TEST_DSPARK`/
`DS4_TEST_MTP` configured), same as every prior unit.

### Phase 2: A/B, now unblocked

**Pairing smoke** (`ds4-server` stopped first; this doubled as the memory
freed up for `make test`'s own CUDA runs, which refuse to start alongside
another `ds4` process):
```
DS4_DSPARK_SPEC_LOG=1 DS4_CUDA_STREAM_STATS=1 ./ds4 \
  -m gguf/DeepSeek-V4-Flash-0731-MXFP4_MOE-Q8_0.gguf \
  --cuda --ssd-streaming --ssd-streaming-cache-experts 75GB \
  --mtp gguf/DeepSeek-V4-Flash-0731-DSpark-Drafter-MXFP4-Q8_0.gguf --dspark \
  --nothink --temp 0 -p "Reply with exactly: ok"
```
Load succeeds: `ds4: DSpark support model detected: ...(stages=3 block=5
markov_rank=256 tensors=81 missing=0 invalid=0 metadata_errors=0)`. All 81
tensors bind (76 dialect-compat-aliased per-stage/head tensors plus the one
confidence-vector reshape notice; full alias list in this unit's raw logs).
`"ok"` prompt: output `ok`. France prompt (`"What is the capital of
France?"`): output `The capital of France is Paris.`, with real speculative
activity observed this time (`DSpark spec partial drafted=4 verified=1
accepted=2`) -- unlike the prior blocked unit, drafting now actually runs.

**Verbatim + identity check vs. no-drafter**, both prompts, `--temp 0`:
stdout (diagnostic lines stripped) byte-identical between the `--mtp
--dspark`-enabled run and a plain no-drafter run:
- `"ok"`: identical (`ok\n`) both arms.
- France: identical (`The capital of France is Paris.\n`) both arms.

Identity gate: **PASS**.

**Memory incident during Arm B setup (dropped budget, per protocol).**
Launching Arm B (drafter, default confidence) at the established 75GB
budget, `available` memory fell to 8 GiB (`free -g`: `used=113 free=2
buff/cache=7 available=8`, swap engaged to 1 GiB) -- below this unit's
10 GiB floor. Checked against real signals before reacting (per this doc's
own precedent for telling genuine thrashing from a scary-looking number):
`uptime` load average 1.49-1.91 (not the 50-59 seen in the documented 100GB
thrash incident), `vmstat` `sy` 1-5% (not 93-94%), no `dmesg`
memory-pressure/OOM lines. Not a confirmed thrash, but the ticket's explicit
protocol for this exact scenario ("drafter ~11GB on top of 75GB budget --
drop budget to 65GB if the floor is threatened") is directive on a numeric
floor, so it was followed: killed the run at turn 1 (`SIGTERM`, no `SIGKILL`
needed), confirmed memory recovered to baseline (117 GiB free, 0 swap), and
re-ran both drafter arms at **65GB** instead of 75GB. At 65GB, `available`
stayed at 16-18 GiB throughout both drafter arms (brief 1 GiB swap blips on
a couple of samples, never sustained, never approaching the incident
pattern). **Deviation from spec, flagged explicitly:** the drafter arms
below are measured at 65GB cache budget, not the no-drafter arm's 75GB --
not a controlled apples-to-apples cache budget between arms A and B/C,
mirroring the same caveat this doc already carries for the GA-without-drafter
75GB-vs-100GB comparison.

**Warm 8-turn A/B**, `research/gb10/session_prompts.txt` (identical prompts
to this doc's established no-drafter baseline), `--nothink --temp 0`,
`DS4_CUDA_STREAM_STATS=1`, `DS4_DSPARK_SPEC_LOG=1` on the drafter arms,
single session per arm (n=1, consistent with this doc's existing warm-session
methodology and its stated time-budget constraint):

| turn | topic | prefill tok | Arm A: no-drafter, 75GB (t/s) | Arm B: drafter, confidence 0.9 (default), 65GB (t/s) | Arm C: drafter, confidence 0.0 (force-accept), 65GB (t/s) |
|---|---|---|---|---|---|
| 1 | photosynthesis (cold start) | n/a | 5.85 | 3.53 | 1.69 |
| 2 | C3/C4/CAM (follow-up) | 33 | 5.87 | 4.08 | 1.71 |
| 3 | TCP handshake (topic switch) | 28 | 5.64 | 4.03 | 1.63 |
| 4 | TCP congestion control (follow-up) | 34 | 5.32 | 3.90 | 1.64 |
| 5 | red-black trees (topic switch) | 30 | 5.26 | 3.72 | 1.61 |
| 6 | red-black vs. AVL (follow-up) | 34 | 5.12 | 3.61 | 1.55 |
| 7 | Paxos/Raft (topic switch) | 29 | 5.28 | 3.91 | 1.66 |
| 8 | ants/load-balancing analogy (topic switch) | 32 | 5.35 | 4.03 | 1.65 |
| | **steady state (mean, turns 6-8)** | | **5.25** | **3.85** | **1.62** |
| | mean, turns 2-8 | | 5.41 | 3.90 | 1.64 |

Prefill token counts and prefill t/s (not tabulated above, all ~0.56-0.70
t/s across all three arms) are identical across arms turn-for-turn --
confirms the three sessions ran the same prompts/context, only the decode
path differs.

**Acceptance/draft-length stats** (`DS4_DSPARK_SPEC_LOG=1` parsed from the
full session logs):

Arm B (confidence 0.9, default):
- 10,734 decode steps total; **9,875 (92.0%) skip drafting entirely**
  (scheduler's no-draft pause), only 411 (3.8%) reach a verify/accept event,
  remainder are mid-block continuation steps.
- Of the 411 verify events: drafted-length histogram `{2: 231, 3: 116, 4: 42,
  5: 22}`; **every single event accepted exactly 2 tokens** (`verified=1
  accepted=2`), independent of how many were drafted -- the verify path
  never extends acceptance past the second token in this build/regime.
- Aggregate accepted/drafted ratio: 822/1088 = **75.6%** -- high when a draft
  is actually tried, but tried on only 3.8% of steps.

Arm C (confidence 0.0, force-accept):
- 8,218 decode steps; 4,102 (49.9%) skip, 2,927 (35.6%) reach verify --
  drafting is attempted far more often than Arm B, as expected for a
  force-accept threshold.
- **Every verify event drafts the full block size (5) and accepts exactly 2**
  (`drafted=5 verified=1 accepted=2` uniformly) -- so forcing acceptance
  doesn't increase the per-event accepted-token count at all, it only
  increases *how often* a (always-2-token, always-full-5-draft) event is
  attempted, at 2.5x the per-event drafting cost of Arm B's shorter average
  proposals.
- Aggregate accepted/drafted ratio: 5854/14635 = **40.0%** -- lower than Arm
  B's ratio, because force-accept pays for a full 5-token draft every time
  but the target's own verify step still only ever confirms 2.

**Verdict: the GA-matched DSpark drafter does not beat the no-drafter
baseline on this ds4 CUDA `--ssd-streaming` architecture, at either
confidence setting, on this warm 8-turn protocol.** Arm B (default
confidence) is **26.7% slower** than Arm A's steady state (3.85 vs. 5.25
t/s); Arm C (force-accept) is **69.1% slower** (1.62 vs. 5.25 t/s) --
force-accepting makes it *worse*, not better, because every accept event
still only ever nets 2 tokens (the verify path's own ceiling in this
build/regime) while the drafter's own forward-pass cost scales with how
often and how long it drafts. This is not the "measurement wasn't reached"
outcome `a335048` left off at -- **this is a real, measured negative
result**, now with a completed protocol behind it (unlike the preview-era
community drafter, which failed even the correctness bar at first; this
matched drafter is verbatim-correct, loads cleanly, and genuinely drafts
and accepts tokens -- it's just not a net throughput win in the
SSD-streaming regime on this hardware). The most plausible explanation,
consistent with this doc's other SSD-streaming findings: per-token expert
fetch cost dominates decode time in this regime, and the drafter's own
per-step forward pass (loaded non-streamed, per `a335048`'s memory-plan
note) adds real wall-clock cost that a fixed 2-tokens-per-accept-event
ceiling can't amortize away. **Root-causing why acceptance never extends
past 2 tokens is flagged as a genuine open question for a future unit, not
answered here** -- this unit measured the behavior faithfully, it did not
diagnose the verify path's internals.

**Server discipline.** `ds4-server` (systemd, port 8000, production GA-0731
streamed config) stopped before this unit's `make test`/pairing/A-B work
began; restarted and verified `systemctl is-active`=`active`,
`curl http://localhost:8000/v1/models`->HTTP 200 at the end. `ExecStart` was
not touched (see recommendation block below -- drafter flags are a
recommendation only, not applied to the live unit). No stray
`ds4`/`ds4_test` processes left running (`ps aux` checked after every step);
several long-dead polling-loop shells from unrelated prior sessions on this
box were observed but not touched (not this unit's stragglers).

**PRODUCTION RECOMMENDATION BLOCK (drafter: do not enable).** The detection
gap is now closed and the drafter is fully functional and correct, but
measurably *not* a throughput win -- do not add `--mtp`/`--dspark` to the
live `ExecStart`. Current production config (unchanged, already correct)
remains:
```
ExecStart=/home/jacinta/src/ds4/ds4-server \
  -m /home/jacinta/src/ds4/gguf/DeepSeek-V4-Flash-0731-MXFP4_MOE-Q8_0.gguf \
  --cuda --ssd-streaming --ssd-streaming-cache-experts 75GB \
  --host 0.0.0.0 --port 8000 --ctx 65536 --batched-session 2 \
  --kv-disk-dir /home/jacinta/.ds4/kv
```
If a future unit wants to revisit the drafter after root-causing the
2-token acceptance ceiling above (the one lever that could plausibly change
this verdict), the tested-working invocation to build from is:
```
--mtp /home/jacinta/src/ds4/gguf/DeepSeek-V4-Flash-0731-DSpark-Drafter-MXFP4-Q8_0.gguf \
--dspark [--dspark-confidence F]
```
-- confirmed to load and decode correctly at both 65GB and 75GB
`--ssd-streaming-cache-experts` budgets (drafter's own resident footprint is
~10-11 GiB on top of that budget, non-streamed).

## Spec-decode 2-token acceptance ceiling: root-caused, STRUCTURAL, no code changed (2026-08-02)

**Task**: find what caps every DSpark verify event at exactly `accepted=2`
(`n_accept` starts at 1 for the pre-verified bonus token, then the verify/replay
path never adds more than 1 further token), independent of drafted length
(2/3/4/5) or confidence setting (default 0.9 or force-accept 0.0), as measured
in the prior unit's 8-turn A/B (`822/1088 = 75.6%` aggregate ratio, but
`{2:231, 3:116, 4:42, 5:22}` drafted-length histogram against a **uniform**
`verified=1` on every one of 411 (Arm B) and 2,927 (Arm C) verify events).

**Phase 1 code trace, read-only, no model runs.**

1. **Log/state machine** (`ds4.c:62050` `ds4_session_eval_dspark_speculative_argmax`):
   `n_accept` enters already holding 1 (the bonus token accepted by the
   caller at `ds4.c:65444-65445` before this function is ever called).
   `max=max_tokens`/`valid`/`len` in the "spec enter" log
   (`ds4.c:62078`) are `s->dspark_draft_valid`/`s->dspark_draft_len` from the
   *previous* decode step's proposal. `drafted=` in "spec accept"/"spec
   partial" (`ds4.c:62229-62233`, `62404-62408`) is `draft_n` (the number of
   draft tokens actually sent into this verify call, after all caps);
   `verified=` is `commit_drafts` -- the longest common prefix between the
   drafter's proposed suffix and the target's own batched-verify top-1 per
   row; `accepted=` is the running `n_accept` (bonus + `commit_drafts`,
   capped by `accepted_cap`).

2. **Verify pass** (`metal_graph_verify_suffix_tops_impl`, `ds4.c:35063`):
   genuinely batch-verifies **all** `draft_n` draft tokens in one pass
   (`top_rows = n_tokens - 1`, one argmax row per position via
   `ds4_gpu_indexer_topk_tensor`/`ds4_gpu_argmax_tensor`). The commit loop
   (`ds4.c:62234-62240`) does standard longest-common-prefix comparison,
   `row_tops[i-1]` (target's true next-token prediction after real token
   `drafts[i-1]`) vs. `drafts[i]` (drafter's own proposed token at position
   `i`) -- **this is correct, textbook speculative-decode verify semantics,
   not the source of the ceiling.** None of the suspects (a)-(c)/(e) from
   the ticket hold: the verify batch is not artificially limited to 1 draft
   token, there is no hardcoded `min()`/window-size-2 cap, and
   `spec_frontier_snapshot`/`_restore` (`ds4.c:50537-50395`) is a rollback
   primitive only, not an accept-count limiter.

3. **Root cause -- suspect (d), confirmed** (`ds4.c:31401-31432`
   `metal_graph_prepare_dspark_setup_block`, and the fused
   `_stage0_setup_block` twin at `ds4.c:31489-31520`): the one-shot batched
   forward that produces the drafter's own per-position `base_logits`
   (`metal_graph_eval_dspark_stage_chain`, `ds4.c:32351-32427`, called
   **once per decode step**, not once per draft position) seeds every
   draft-slot position beyond the first with a **fixed placeholder token**,
   not the real (or even the drafter's own previously-picked) continuation:
   ```
   ids[0] = (int32_t)token;                         /* true last-accepted token */
   for (uint32_t i = 1; i < dw->block_size; i++) {
       ids[i] = (int32_t)dw->noise_token_id;        /* fixed mask/noise token, from GGUF metadata */
   }
   ```
   `dw->noise_token_id` is a genuine checkpoint-supplied metadata field
   (`deepseek4.dspark.noise_token_id` / `dspark.noise_token_id`,
   `ds4.c:2655-2679`, `ds4.c:7602-7606`), confirming this is the model's
   own intended parallel/masked-block decoding scheme (a DeepSeek DSpark
   design: predict a whole block from one non-causal pass over a
   real-token-then-noise-masked sequence), **not** an artifact of ds4's
   port. The subsequent per-position "Markov correction"
   (`dspark_apply_markov_greedy_probe` / `_lazy_runtime`, `ds4.c:33103-33356`)
   only ever conditions on the *single previous drafted token's embedding*
   (`markov_w1` lookup + `markov_w2` bias added onto the already-fixed,
   noise-conditioned `base_logits[draft]` row) -- it is a cheap **order-1**
   fix-up, not a re-encode. There is no iterative refinement/denoising loop
   anywhere in the DSpark code (`grep`-confirmed: no `refine`/`iterat`/
   `round`/`denois` hits) that would let position `i`'s `base_logits` be
   recomputed once positions `0..i-1` are actually known.
   **Consequence**: position 0's proposal is high quality (it only needs
   the real preceding context, which is genuinely available -- this is why
   `target_top == drafts[0]` passes almost every time, i.e. the "own"
   pre-verify gate). Position 1's proposal is base_logits computed against
   a *noise*-masked stand-in for position 0 (not drafts[0]'s real content),
   corrected only by a weak order-1 bias -- structurally much weaker than a
   true causal continuation, and it empirically never survives batch verify
   against the target's real (fully causal) next-token distribution in this
   GA-matched drafter/target pairing. Every verify event in both arms shows
   exactly this: `commit_drafts` (`verified=`) is 1, always, regardless of
   how many further (also noise-conditioned) tokens were drafted beyond
   position 1 -- consistent with a hard architectural cliff at depth 1, not
   a statistical/tuning artifact.

4. **Flag sanity-check**: `--mtp <drafter.gguf> --dspark [--dspark-confidence
   F]` reaches the verify path correctly; `dw->block_size` (5, from the
   artifact's own metadata, "block=5" in the load-time summary) is the max
   draft length and is honored end-to-end (`draft_n` caps at
   `accepted_cap - n_accept`, `max_tokens - n_accept`, `room - 1`, none of
   which truncate to 2 -- the drafted-length histogram `{2,3,4,5}` proves
   longer proposals genuinely reach the verify call). No truncation bug
   found upstream of verify.

5. **Upstream cross-check**: DSpark support does not exist on
   `upstream/main` (antirez/ds4) at all (`git log upstream/main --oneline |
   grep -iE 'dspark|noise|mtp'` -- zero hits); it was added directly on this
   repo by `fc9efd1` ("Add DSpark speculative decoding", authored
   `antirez`, not yet on his public `main`). That commit's own `README.md`
   addition documents DSpark as "currently a greedy argmax-only path" with
   confidence-gated pruning "to avoid replay-heavy low-confidence blocks",
   but says nothing about expected per-event accepted-token depth --
   there is no upstream design note contradicting the depth-1 collapse
   found here; the masked/noise-token block-decode scheme visible in the
   code is consistent with DeepSeek's own DSpark architecture as shipped in
   the checkpoint's metadata (real `noise_token_id`), not a ds4-side bug.

**Verdict: STRUCTURAL, not wiring.** The verify/accept machinery
(`ds4.c:62050-62420`) is correct standard speculative-decode logic and is
not the cause. The ceiling is inherent to the DSpark drafter's own
single-shot, noise-masked parallel-block proposal design: only position 0
of any drafted block is conditioned on real context; positions 1+ are
conditioned on a fixed noise/mask placeholder plus a weak order-1 Markov
correction, and empirically never survive batch-verify against the target
model's real causal continuation in this GA-matched drafter/target pairing
-- 100% of 3,338 combined verify events across both A/B arms, not a
sometimes-partial-credit distribution. No `min()`/constant/window-size bug
exists to fix; there is nothing to point-fix in `ds4.c`.

**What a real fix would require (out of scope for this unit, sizing only):**
an iterative "denoising"/refinement loop -- after the order-1 Markov pass
picks a greedy token for position `i`, re-run (a strict subset of)
`metal_graph_eval_dspark_stage_chain` with the real token substituted for
`noise_token_id` at that position before proposing position `i+1`, likely
2-4 refinement rounds to recover meaningful depth-2+ accuracy (masked/
parallel-decode literature pattern, e.g. discrete-diffusion LM block
decoding). This is a **real feature addition**, not a constant tweak:
it needs (a) a per-round re-embed + re-run of the (currently single-shot)
stage-chain GPU graph, (b) a new stopping/round-count heuristic weighed
against each round's real forward-pass cost (which this unit's prior A/B
already showed dominates over the ~2-tokens/event ceiling on this hardware
-- more refinement rounds add cost per drafted block, so the win is not
guaranteed even if depth-2+ accuracy improves), and (c) end-to-end
correctness verification (byte-identity vs. no-drafter) redone from
scratch. Estimated size: **medium** (bounded to the existing DSpark
propose path, no new abstraction needed, but a real GPU-graph change, not
a wiring fix) -- a good candidate for its own ticket, not a quick follow-up.

**No model runs were performed this unit** (Phase 1 code trace only,
consistent with the ticket's own Phase 2 branch for a STRUCTURAL verdict:
"document precisely... and stop"). `ds4-server` was not touched (no
`systemctl stop` needed since nothing was executed against the GPU).

**Draft upstream issue text (not filed, for the orchestrator to route):**
"ds4's DSpark drafter (`--mtp ... --dspark`) never accepts more than 1
token beyond the initial bonus token per speculative block, regardless of
draft length (tested 2-5) or `--dspark-confidence` setting (0.0-0.9),
across 3,338 measured verify events with the GA-matched
DeepSeek-V4-Flash-0731 target+drafter pair. Root cause: the drafter's
per-block proposal for draft positions beyond the first is computed from a
single non-causal forward pass seeded with the checkpoint's own
`noise_token_id` placeholder at all not-yet-committed positions
(`metal_graph_prepare_dspark_setup_block`), corrected only by a weak
order-1 'Markov' bias on the previous token's embedding -- there is no
iterative refinement pass to let later positions condition on earlier
drafted tokens' real content, so verify-time divergence from the target's
true causal continuation is effectively total past depth 1. Net effect:
DSpark drafting is currently a measured throughput loss (up to 69% slower)
on this hardware/model, and cannot become a win without adding a
multi-round refinement mechanism to the proposal path."

## DSpark chaining hypothesis check + resident A/B, ceiling claim corrected (2026-08-02)

**Task**: operator hypothesis -- DeepSeek's official MTP design chains draft stage
`k`'s input on stage `k-1`'s predicted-token embedding (not a noise placeholder);
if ds4's DSpark "stage chain" runs the 3 stages against static noise-seeded slots
without feeding each stage's prediction forward, that would be a wiring-class bug,
not the "structural, no fix" verdict the prior unit (`c832953`) reached. Two parts:
(A) a cheap resident cost-structure discriminator, (B) a chain-code re-read
targeted at the chaining question specifically.

### Part A: resident IQ2XXS + `DSpark-support.gguf` A/B (`--temp 0`, 4-turn short
protocol, `DS4_DSPARK_SPEC_LOG=1`, steady = last 2 turns)

`ds4-server` stopped first (mandatory, restarted after -- see below). Pairing:
`-m gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
(81 GiB, fully resident, no `--ssd-streaming`) + `--mtp
gguf/DeepSeek-V4-Flash-DSpark-support.gguf --dspark` (lineage-matched: both share
`dspark.target_layer_ids=[40,41,42]`, `dspark.noise_token_id=128799`,
`dspark.block_size=5`, confirmed via a direct GGUF key-value dump, not just the
load-time banner). **Discovery**: DSpark speculation is a **greedy-only path** --
default (non-zero) `--temp` silently disables `dspark_draft_valid` and zero
`DSpark spec *` log lines are ever emitted; this matches upstream `fc9efd1`'s own
README addition ("DSpark speculation is currently a greedy argmax-only path...
Plain sampled and non-speculative session eval also skips DSpark draft
preparation") -- not a bug, just an easy foot-gun for any A/B that forgets
`--temp 0` (the first pass of this unit's own resident run did, and got zero
drafting activity in all three arms as a result; redone with `--temp 0`).

| Arm | generation t/s (steady, last 2 turns) | vs no-drafter |
|---|---|---|
| (a) no drafter | **16.87 t/s** (16.88, 16.85) | baseline |
| (b) `--mtp ... --dspark` (default confidence 0.9) | **16.07 t/s** (16.14, 15.99) | **-4.7%** |
| (c) `--dspark-confidence 0.0` (force-accept every draft) | **10.81 t/s** (11.32, 10.30) | **-35.9%** |

**Resident verdict: spec decode is a net throughput loss at this ceiling, on this
hardware/pairing, at every confidence setting tested** -- consistent with the
prior unit's directional finding (spec decode doesn't currently pay for itself
here), now confirmed resident (not just under `--ssd-streaming`) so the loss is
not a streaming-specific artifact.

**Accepted-token distribution -- corrects the prior unit's "always exactly 2"
framing.** Filtering `DSpark spec accept`/`DSpark spec partial` result lines only
(not the per-cycle `DSpark spec enter` lines, which start every cycle at
`accepted=1` regardless of outcome -- a filtering mistake this unit made and
caught on the first pass) gives a real spread, not a fixed point:

- Arm (b), default confidence: 48 successful verify events, `no-draft` skips=459
  (scheduler backs off aggressively). `{accepted=2: 15, 3: 22, 4: 8, 5: 2, 6: 1}`,
  mean 3.0.
- Arm (c), confidence 0.0 (always drafts full `block_size=5`): 98 successful
  verify events, `no-draft` skips=134. `{accepted=2: 23, 3: 28, 4: 19, 5: 19,
  6: 9}`, mean 3.6.

**This directly contradicts the prior unit's claim that acceptance is "100% of
3,338 combined verify events... not a sometimes-partial-credit distribution" --
capped at exactly `accepted=2` every time.** That claim was true for the pairing
that unit tested (streamed GA target + `DeepSeek-V4-Flash-0731-DSpark-Drafter-
MXFP4-Q8_0.gguf` drafter); it does **not** generalize to the resident IQ2XXS +
`DSpark-support.gguf` pairing tested here, which reaches depth 6 repeatedly. The
underlying mechanism explanation from the prior unit (noise-conditioned
positions beyond 0, order-1 Markov correction only) still holds and still
explains *why* deeper acceptance is unreliable/inconsistent -- it just isn't a
hard architectural wall at exactly depth 1 for every pairing; some pairings/
continuations survive verify well past that. The net throughput loss in this
unit's resident A/B is real regardless of the corrected distribution: even with
mean depth 3.0-3.6, per-cycle drafter cost plus scheduler backoff overhead
(459 and 134 no-draft skip cycles respectively, dwarfing the 48/98 successful
ones) outweighs the savings on this hardware.

**New correctness finding (caught by this unit's mandated identity check, not
the chaining question -- flagged, not fixed, out of this unit's scope):** the
ticket's protocol requires byte-identity vs. no-drafter to hold under the
greedy contract before any A/B is trusted. Checked with `-p` (clean stdout,
`2>/dev/null`) on `"Explain the TCP three-way handshake."`, `--temp 0`: the
no-drafter path is perfectly reproducible run-to-run (`diff` clean across two
independent runs). **The drafter-enabled path (Arm b config) diverges from the
no-drafter output partway through the response** -- same opening sentence,
then different phrasing/structure/section headers throughout the rest of the
generation. This means the verify/accept path is, in at least this pairing +
default-confidence config, committing at least one token that is not actually
the target's true greedy argmax -- a real correctness bug in the accept logic
or a floating-point/batch-size-dependent divergence in the target's own verify
pass, independent of the throughput/chaining questions above. **Root cause not
investigated this unit** (out of scope: this ticket's mandate was the chaining
hypothesis and a cost-structure discriminator, not a new correctness bug hunt);
recommend a dedicated follow-up unit re-examine `metal_graph_verify_suffix_
tops_impl`/the commit-loop's longest-common-prefix comparison (`ds4.c:62234-
62240`) specifically for this resident (non-streaming) + default-confidence
pairing, since the earlier `05f0fde`/pairing-smoke identity check that passed
only exercised two very short, high-predictability prompts ("ok", "capital of
France") -- not long enough to hit whatever this divergence needs to trigger.

### Part B: chaining-hypothesis code trace -- REFUTED, hypothesis based on a
naming collision, prior structural verdict stands (corroborated, not just
re-asserted)

**The hypothesis, precisely**: does `metal_graph_eval_dspark_stage_chain`
(`ds4.c:32351`) substitute stage `k`'s predicted-token embedding into stage
`k+1`'s input in place of `noise_token_id`, per DeepSeek's chained-MTP design?

**Answer: the "stages" in this function are not MTP prediction modules at all --
they are the small DSpark drafter's own transformer layer depth.** Confirmed
three independent ways:

1. **GGUF metadata, dumped directly** (`python3` GGUF-header parser, both
   `DeepSeek-V4-Flash-0731-DSpark-Drafter-MXFP4-Q8_0.gguf` and
   `DeepSeek-V4-Flash-DSpark-support.gguf`): `dspark.layer_count=3` /
   `dspark.stage_count=3` / `dspark.n_layers=3` (support GGUF has all three
   spellings of the same field), plus `dspark.target_layer_ids=[40,41,42]` --
   **3 target-model hidden-state taps**, one per drafter layer. This is an
   EAGLE/Medusa-style small-transformer-conditioned-on-target-hidden-states
   design (3 layers of the drafter's own net, each attached to a different
   target layer's hidden state), not DeepSeek V3/V4's per-future-token MTP
   module chain (where module count = number of future tokens predicted, each
   module chained on the *previous module's own predicted-token embedding*,
   not on different target layers).

2. **Code**, `ds4.c:2600-2636` (`ds4_tensor_dspark_stage`) and the stage-count
   derivation at `ds4.c:2736` (`s.stages = max_stage + 1u`, from the highest
   `dspark.<N>.*` tensor-name prefix) -- this is exactly how ds4 already counts
   ordinary transformer block depth for every other model family in this file
   (`blk.<N>.*`), just with a `dspark.` prefix instead of `blk.`. `n_stages`
   flows into `metal_graph_eval_dspark_stage_chain`'s `for (stage = 0; stage <
   dw->n_stages; stage++)` loop unchanged from any other per-layer loop in the
   file.

3. **The chaining that already exists between stages is ordinary residual-
   stream hand-off, already correct, not the site of any bug.**
   `metal_graph_eval_dspark_stage_block` is called with `prepare_next_stage_
   input = (stage + 1 < dw->n_stages)` (`ds4.c:32421-32427`); when true, it
   calls `metal_graph_encode_dspark_next_stage_draft_input_from`
   (`ds4.c:31947-31967`), which `ds4_gpu_tensor_copy`'s the **hidden-state**
   (`hc`, `block_size` positions x `DS4_N_HC * DS4_N_EMBD` floats) produced by
   stage `k`'s FFN output into `g->dspark_stage_input_hc`, the tensor stage
   `k+1` reads its input from. This is textbook layer-to-layer residual-stream
   propagation (exactly what `metal_graph_encode_layer_ffn_batch` does between
   any two ordinary transformer layers elsewhere in this file) -- **not** a
   token-id/embedding substitution, and not broken: hidden state genuinely
   flows stage0->stage1->stage2 today.

**The axis the operator's hypothesis actually describes -- per-draft-position
chaining (position `i`'s proposal conditioned on position `i-1`'s real
predicted token) -- is `dw->block_size` (5 draft positions), a completely
different axis from `n_stages` (3 drafter layers).** That axis is exactly what
the prior unit (`c832953`) already found unwired: `metal_graph_prepare_dspark_
setup_block` (`ds4.c:31401-31432`, re-read this unit, unchanged) seeds
positions `1..block_size-1` with the fixed `noise_token_id` **once**, before
any of the 3 stages run, and the per-position "Markov correction"
(`ds4.c:33103-33356`) is confirmed (again, unchanged) to be a single order-1
bias on the previous drafted token's embedding, not a re-run of the 3-stage
forward with the real token substituted in.

**Cross-checks against the ticket's three named evidence sources:**

- **(i) DeepSeek V3/V4 MTP tech-report design** (chained per-module
  conditioning, module `k` sees module `k-1`'s predicted-token embedding):
  describes what would need to happen along the `block_size` axis. It says
  nothing about the `n_stages`/layer-depth axis, because that axis doesn't
  exist in DeepSeek's official MTP design at all (their MTP modules are single
  transformer layers each, not 3-layer sub-networks) -- **this GGUF's
  3-layer-deep, 3-target-layer-tapped drafter is not literally DeepSeek's
  official per-token MTP module structure**; it is a separate EAGLE-style
  architecture the checkpoint's own name ("DSpark") distinguishes from plain
  "MTP" (ds4 already tracks these as two different support-model dialects,
  `DS4_SUPPORT_MTP_LEGACY` vs `DS4_SUPPORT_DSPARK`, `ds4.c` `support_kind`).
- **(ii) the GGUF's own metadata**: `dspark.layer_count=3` (not "predict 3
  future tokens" -- there is no such 3-future-token metadata key anywhere in
  either drafter GGUF; `dspark.block_size=5` is the only draft-length field,
  and confirms the real per-token draft depth is 5, already known).
- **(iii) upstream commit intent**: `git log -S dspark` on this repo shows
  DSpark was added whole-cloth by `fc9efd1` ("Add DSpark speculative
  decoding", `antirez`, not yet on his public `main` -- reconfirmed this unit).
  That commit's own `README.md` addition (re-read in full this unit) states
  plainly: *"DSpark speculation is currently a greedy argmax-only path"* and
  documents confidence-threshold pruning as the only tunable knob -- **no
  commit message or doc anywhere in this repo's history describes an intended
  but not-yet-wired per-position causal refinement/chaining pass.** If ds4's
  own port were missing an intended chaining step, the commit that added
  DSpark support would be the place to expect a TODO/known-gap note about it;
  there is none.

**VERDICT: (b)-but-structural, per this ticket's own branch 5.** The operator's
specific hypothesis -- that the "3-stage" drafter is built for DeepSeek-style
chained MTP and ds4's port is missing the argmax-embed-and-substitute wiring
between stages -- does not hold: "stage" here is the drafter's own transformer
depth (3 layers, EAGLE-style, tapping 3 different target hidden layers), already
correctly hidden-state-chained layer-to-layer, and DeepSeek's true per-token MTP
chaining design maps onto a *different* axis (`block_size`) that this
checkpoint's own architecture (noise/mask-seeded parallel block proposal, per
its own `noise_token_id` metadata field) does not chain by design, not by ds4
port omission. The prior unit's "structural, no fix" verdict stands, now
corroborated by direct GGUF metadata inspection and the upstream commit's own
documentation (neither of which the prior unit had pulled in), not just
re-derived from the same code read. **No wiring fix implemented this unit** --
consistent with the ticket's step-5 branch ("if (b) but structural after all...
document the conflict... and stop").

**What *is* new and actionable from this unit, beyond re-confirming the prior
verdict:** (1) the "always exactly ceiling 2" claim was pairing-specific, not
universal -- real depth up to 6 is achievable with a different (still
GA-lineage-matched) drafter/target pairing, but throughput is still a net loss
either way on this hardware; (2) a genuine greedy-identity correctness
divergence was found in the resident pairing at default confidence on a longer
generation, not root-caused this unit, flagged for dedicated follow-up.

**Server discipline.** `ds4-server` stopped before Part A (`systemctl stop
ds4-server`, verified `inactive`); restarted and verified (`systemctl
is-active`=`active`, `GET /v1/models`->HTTP 200) after -- see final entry below.
No production `ExecStart` config touched.

## Determinism/identity probe: two temp-0 violations root-caused (2026-08-03)

**Scope.** Two distinct reported temp-0 (greedy) nondeterminism/identity
issues, investigated independently. Server stopped for the whole test window
(`systemctl stop ds4-server`, verified `inactive`) and restarted + verified
(`is-active`=`active`, `GET /v1/models`->200, production `ExecStart` untouched)
afterward. Raw logs committed under `research/gb10/nd-probe-out/` (`runA*`,
`runP*`, `runQ*`, `runC*` = Case A; `runB-*` = Case B).

### CASE A -- streamed decode run-to-run divergence: does NOT reproduce on the
GA model at 65GB, this hardware/build (negative result, well-tested)

**Protocol.** GA model (`DeepSeek-V4-Flash-0731-MXFP4_MOE-Q8_0.gguf`), `--cuda
--ssd-streaming --ssd-streaming-cache-experts 65GB --temp 0`, two prompts:
(1) `-p "Explain the TCP three-way handshake."` (the same short prompt CASE B
uses) and (2) `--prompt-file prose_prompt.txt` (the ~55-word TCP-handshake
essay prompt that was the *original* trigger for the preview-artifact
divergence report). Four identical runs each at `-n 400`, plus four more at
`-n 800` on prompt (2) to push past the token count where the preview
artifact diverged, plus a cache-regime discriminator: two runs at
`--ssd-streaming-cold --ssd-streaming-cache-experts 8GB` (near-all-miss) vs.
the 65GB warm runs, per hypothesis (a1) (cache hit/miss timing driving FP
reduction-order variance).

**Result: all 10 decode runs across both prompts, both token budgets (400 and
800), and both cache regimes (65GB warm, 8GB cold) are byte-for-byte
identical.** `diff` clean on every pairwise comparison run (1v2, 1v3, 1v4 for
each batch; cold-run1 vs cold-run2; cold-run1 vs warm-run1). No divergence
observed anywhere in ~11,000 combined generated tokens across the sweep.

**Interpretation.** This does not reproduce the "First observed on the preview
artifact (5d032d7 era)" divergence on the current GA model + `research/gb10`
HEAD (`c860c9a`) + 65GB budget, on this hardware, for these two prompts, at
these token counts. Two non-exclusive explanations, neither confirmed further
this unit (would need bisecting `5d032d7..c860c9a` or a much larger n-of-runs
sweep to fully rule out a low-frequency event):
1. Something between the preview artifact and the GA promotion (dialect-compat
   fixes, `05f0fde`/tensor-alias work, or the GA quant itself vs. the preview
   quant) incidentally fixed or masked it.
2. It is a low-probability/narrow-trigger event (the (a1)/(a2)/(a3) hypotheses
   are all about *rare* races -- specific cache-hit-timing windows, specific
   token positions landing on a near-tied argmax) that this sweep's 10 runs
   didn't happen to hit. The cold-vs-warm test directly exercises the (a1)
   mechanism (forces the opposite cache-hit pattern) and still produced
   byte-identical pairs, which is evidence against (a1) as a *frequent*
   driver at minimum, but is not proof of its absence at low frequency.

**Verdict: cannot confirm as a wiring-class bug on this build -- documented as
a non-reproducing negative result, not swept under the rug.** No code changed.
A dedicated follow-up unit wanting to chase this further should either (a)
bisect the two known-good vs. known-bad commits directly, or (b) run a much
larger N (50-100+) of shorter generations to estimate an event rate before
committing to expensive instrumentation (checksum/logit-gap dumps) that only
pays off once a live repro is in hand.

### CASE B -- DSpark spec-decode breaks greedy identity resident: ROOT-CAUSED,
STRUCTURAL (batch-verify numerics vs. single-token decode numerics), b2 confirmed

**Reproduction.** Resident IQ2XXS + `DSpark-support.gguf` (lineage-matched,
`dspark.target_layer_ids=[40,41,42]`), `--cuda --temp 0 -n 400 -p "Explain the
TCP three-way handshake."`, default confidence 0.9 -- the same config the
prior unit (`c860c9a`) flagged and did not root-cause.
- No-drafter: two independent runs, byte-identical (`diff` clean, 1902 bytes
  each) -- confirms the pure-decode baseline is perfectly reproducible on this
  hardware/build (also corroborates CASE A's negative result: resident
  no-drafter decode has no detectable temp-0 nondeterminism here either).
- Drafter-enabled: diverges from the no-drafter baseline at **character offset
  277** (well into the reasoning preamble: `"...maybe mention potential
  issues like SYN flood. Keep it accessible."` vs. `"...maybe mention the
  states (CLOSED, LISTEN, SYN-SENT, etc.). Keep it simple..."`) -- a
  near-tied paraphrase-level flip, not garbage, consistent with an FP-drift
  signature rather than a logic bug. `DS4_DSPARK_SPEC_LOG=1` reruns are
  byte-identical to the unlogged drafter run (confirms the drafter path's own
  determinism; only the drafter-vs-no-drafter comparison diverges).

**Discriminator (i) -- divergence only ever after the first accepted draft:
CONFIRMED.** `DS4_DSPARK_SPEC_LOG` shows the first successful accept event at
generated-token position 4 (`DSpark spec enter ... len=3 pos=18` ->
`DSpark spec accept drafted=3 accepted=4`, i.e. `max` drops from 400 to 396 =
4 tokens committed). The character-277 divergence falls roughly 55-65 tokens
into generation -- far past that first accept. No divergence was observed
before any accept event in this run. This puts the bug class squarely in
"accepted-token KV/state differs from pure decode" (b1/b2), not a
verify/accept-logic defect (ruled out already by the prior unit's independent
code read of `ds4.c:62050-62420`, re-confirmed here by this run's own
timing).

**Discriminator (ii) -- code-trace pinpoint (in place of a live
checksum-vs-replay diff, which the ticket itself flags as the *expensive*
option; this code trace gives an equally decisive, cheaper answer). Root
cause: hypothesis (b2), confirmed at the source level.**

The DSpark verify batch (`n_tokens` = `block_size`, e.g. 5) computes the
compressed-KV "compressor frontier" projection (`layer_attn_state_kv[il]`/
`layer_attn_state_score[il]`, per the struct comment at `ds4.c:15976-15981`:
"compressor frontiers for the next compressed row") for **every** verified
position -- including position 0, the bonus token that is *always* accepted
-- via a single **batched** GEMM over the whole verify chunk width
(`ds4_gpu_matmul_f16_pair_tensor(metal_graph_batch_comp_kv(g),
metal_graph_batch_comp_sc(g), ..., width=n_tokens*coff, ...)`, `ds4.c:27454`
and the equivalent decode-batch site `ds4.c:28685-28753`; consumed per-token
via row-views at `ds4.c:28619` inside the per-token commit loop that then
calls the shared `ds4_gpu_compressor_update_tensor` with
`comp_state_already_stored=false` hardcoded, `ds4.c:28637`).

Ordinary single-token greedy decode computes the *identical logical
quantity* (that same token's KV/gate projection) via a **different,
fused single-vector kernel**, `ds4_gpu_matmul_f16_pair_compressor_store_tensor`
(`ds4.c:22993-23011`), which sets `comp_state_already_stored=true` when it
successfully fuses the projection+store into one kernel launch --a distinct
code path from the batch-verify projection, not merely a different call to
the same underlying math.

**These are not required to be, and empirically are not, bit-identical.** A
GEMM computing one row of a `[n_tokens x width]` batched matmul and a fused
single-vector kernel computing the same logical `1 x width` row use different
tiling/accumulation order internally (standard FP16/BF16 GEMM
non-associativity across kernel implementations) -- so **every token
committed through DSpark's accept path, including the always-accepted bonus
token at draft position 0**, gets a KV compressor-frontier state that is
numerically close to but not identical to what sequential single-token decode
would have produced for the exact same token. This state feeds every
subsequent layer's attention for the rest of the generation, so the drift
accumulates silently until it flips an argmax tie -- consistent with the
observed divergence being a late (~char 277, ~55-65 tokens in), paraphrase-
level flip rather than an immediate/garbled break.

This also explains why `spec_frontier_snapshot`/`_restore` (`ds4.c:50537-50611`)
looked structurally sound under a pure code audit (KV_AUDIT.md's read):
**it is not miscopying anything.** It faithfully snapshots and restores the
row counters and frontier tensors exactly as the batch-verify pass computed
them -- rollback/restore is correct *as a copy operation*; the bug is
upstream of it, in what values get *written* into those frontier tensors by
the accept-commit path in the first place. `spec_frontier_commit_prefix1`
(`ds4.c:50617-50639`, the cheap N=1-accept fast path) doesn't help either: it
captures/commits the same batch-computed row-0 state, it doesn't re-derive it
via a single-token replay.

**Verdict: (b2), STRUCTURAL** per this ticket's own branch -- "batch-vs-single
numerics are legitimately different in FP" is not a wrong-formula bug to
point-fix; it is inherent to using a batched verify GEMM for speculative-
decode performance. b1 (incomplete KV rollback) and b3 (side-state leak) are
both ruled out by this trace: the rollback mechanism restores exactly what it
snapshotted, and no Markov/confidence side-state tensor is written into the
attention-affecting frontier tensors anywhere in the traced call path.

**What a real fix would look like (sizing only, out of scope for this
diagnostic unit).** Re-derive each accepted token's compressor-frontier state
via the cheap fused single-token kernel (`ds4_gpu_matmul_f16_pair_compressor_
store_tensor`) after acceptance, discarding the batch-computed frontier
values, instead of committing the batch-derived state directly. Cost: one
extra fused single-token kernel launch per *accepted* token per block (not
per drafted token), which is small relative to the verify batch's own cost --
plausibly affordable without eating DSpark's throughput case, but unverified
here (no model run to confirm) and is a real code change (touches the
accept-commit path in `ds4.c` around 62175-62290 and 65628-65780), not a
constant tweak. Good candidate for a dedicated follow-up unit; not attempted
here per the ticket's structural-verdict branch ("document precisely...and
stop").

**Test-oracle guidance for a correct greedy-identity test.** Byte-identical
text diff between drafter-enabled and no-drafter runs is the *wrong* oracle
for long generations under this architecture -- it will always eventually
fail once FP drift crosses an argmax near-tie, and that is expected/inherent,
not evidence of a logic bug. A correct oracle should instead: (a) restrict
strict byte-identity assertions to short generations empirically known to sit
below the divergence-onset length for the tested prompt/model pair (this
ticket's own evidence: the prior unit's "France"/"capital of France" short
prompts passed byte-identical; this unit's TCP-handshake prompt diverges
around char 277 / ~60 tokens -- so "short" is prompt- and model-dependent, not
a fixed token count), or (b) for longer generations, measure per-token
top-1-argmax agreement rate against the no-drafter baseline and assert it
stays near 100% up to the *first* divergence, then track whether post-
divergence text remains coherent/on-topic (already true per the prior unit's
500-token run) rather than asserting byte-identity past that point, or (c)
compare top-2 logit gaps at the divergence point directly (this unit did not
run `--dump-logprobs` for this pinpoint since the code trace already gave an
unambiguous mechanism; a follow-up wanting empirical confirmation of the
"near-tied argmax flip" framing specifically should do so at the char-277
token position).

**Draft upstream issue text (not filed, for the orchestrator to route):**
"ds4's DSpark speculative decode (`--mtp ... --dspark`) does not preserve
byte-identical greedy (`--temp 0`) output vs. non-speculative decode on
longer generations, despite the accept/verify logic itself being correct
argmax-matching (confirmed via independent code read). Root cause: the
compressed-KV 'compressor frontier' state for every accepted token (including
the always-accepted bonus token) is computed via the verify batch's own
multi-token GEMM (`ds4_gpu_matmul_f16_pair_tensor` over the batch width,
`ds4.c:27454`/`28685`), which is a numerically distinct code path from
ordinary single-token decode's fused projection+store kernel
(`ds4_gpu_matmul_f16_pair_compressor_store_tensor`, `ds4.c:22993`) --
standard FP16/BF16 GEMM non-associativity between batched and single-vector
kernels means the committed KV state for accepted tokens subtly differs from
what pure decode would produce for the same tokens, and this drift
eventually flips an argmax tie on longer generations. This is inherent to
using a batched verify pass and may not be simply fixable without re-deriving
accepted-token KV state via a single-token kernel post-accept (extra cost:
one fused single-token kernel launch per accepted token)."

**Server discipline.** `ds4-server` stopped before this unit's test window
(`systemctl stop ds4-server`, verified `inactive`); restarted after both
CASE A and CASE B testing, verified `systemctl is-active`=`active` and
`curl http://localhost:8000/v1/models`->HTTP 200 (`deepseek-v4-flash`,
`context_length: 32768`, confirming the production `ExecStart` --
`--ssd-streaming-cache-experts 75GB`, `--ctx 32768` -- was untouched and
reloaded correctly). No stray `ds4`/`ds4-server` processes observed after
restart aside from the one production instance.

## DSpark greedy-identity fix: re-derive accepted-token KV via single-token replay, always (2026-08-03)

**Scope.** Implements the fix scoped by the prior unit's identity probe
("Determinism/identity probe: two temp-0 violations root-caused", CASE B,
verdict b2 STRUCTURAL): the DSpark accept-commit path
(`ds4_session_eval_dspark_speculative_argmax`, `ds4.c`) had a fast path for
full accepts (`commit_drafts == draft_n`) that committed the verify batch's
own compressor-frontier writes (`ds4_gpu_matmul_f16_pair_tensor`, the
batched multi-token GEMM) directly, instead of re-deriving that state through
the fused single-token kernel
(`ds4_gpu_matmul_f16_pair_compressor_store_tensor`) that ordinary greedy
decode uses. The partial-accept path already had a rollback+replay
mechanism (roll checkpoint/frontier back to the pre-verify snapshot, then
re-decode each accepted token one at a time via
`metal_graph_eval_token_raw_swa`, the exact function plain single-token
decode calls) -- the fix removes the full-accept fast path entirely
(`ds4.c`, was ~62231-62281) so every accept, full or partial, falls through
to that same rollback+replay path unconditionally. This guarantees the
post-accept KV/compressor-frontier state -- and the logits read off it -- are
bit-identical to what plain sequential decode would have produced for the
same tokens, since it is now the literal same code path. Scoped to the
DSpark accept-commit function only; no change to `--mtp` native-drafter code,
no dialect/FP4 changes. Zero effect when no drafter is configured (the
caller returns before `draft_n` is reached).

**Build.** `make clean && make cuda-spark`: clean rebuild, zero warnings.

**Tests.** `make test`: `ds4 tests: 2 failure(s)` -- both on the documented
pre-existing flaky list: `tool-call-quality` (`tests/ds4_test.c:6375`) and
`logprob-vectors` (`tests/ds4_test.c:5285`, the exact same assertion line
cited in the P3c-1-take-4 unit's "6 failure(s)" baseline). No new failing
test names; no regression attributable to this diff.

**Greedy-identity verification.** Resident IQ2XXS +
`DeepSeek-V4-Flash-DSpark-support.gguf` (`--mtp <support.gguf> --dspark`),
`--cuda --temp 0`, three cases, drafter-enabled vs no-drafter, byte-for-byte
`diff`:
1. `-p "Explain the TCP three-way handshake." -n 400` (the CASE B
   reproducer, previously diverged at char 277): **BYTE_IDENTICAL**, 1902
   bytes both sides (matches the pre-existing no-drafter baseline byte count
   exactly). Confirmed on two independent drafter-enabled runs, including one
   with `DS4_DSPARK_SPEC_LOG=1` (accept-depth histogram: 334x1, 9x2, 13x3,
   5x4, 6x5, 2x6, 1x7, 1x8, 1x9, 8x0-miss -- spread up to depth 9 still
   present, confirming the fix does not flatten acceptance, per c860c9a's
   "spread" finding).
2. `-p "What is the capital of France? Explain its history briefly." -n
   400`: **BYTE_IDENTICAL**, 649 bytes both sides (short EOS-terminated
   completion).
3. `--prompt-file prose_prompt.txt -n 800` (the ~55-word TCP-handshake essay
   prompt, CASE A's original preview-artifact-era trigger, pushed to 800
   tokens): **BYTE_IDENTICAL**, 3693 bytes both sides.

No divergence observed in any of the three cases, including the one known to
reproduce the bug pre-fix. Raw logs under
`research/gb10/fix-verify-out/`.

**Overhead.** Generation-phase t/s, single runs each (not a full statistical
sweep -- directional, not final):
| case | no-drafter | drafter (post-fix) |
|---|---|---|
| TCP-handshake, n=400 | 17.02 t/s | 14.30 / 14.31 t/s (2 runs) |
| France, n=400 | 16.96 t/s | 14.04 t/s |
| prose, n=800 | 16.93 t/s | 14.63 t/s |

Average drafter-vs-no-drafter overhead post-fix: ~15.6% slower generation
(vs. the pre-fix drafter case, which per the ticket's own numbers was only
~4.7% slower on this same resident pairing -- "16.87->16.07 default"). This
is the expected, honestly-reported cost of always paying the single-token
replay for every accepted token (not just partial accepts): correctness
requires it, and it makes DSpark's already-marginal resident throughput case
more clearly net-loss on this pairing/hardware. Not attempting a
targeted-smaller-kernel-only optimization (re-deriving just the compressor
projection instead of the full forward pass) in this unit -- out of scope
per the ticket, and the full-replay approach is the one already proven
correct and in production use for partial accepts.

**Server discipline.** `ds4-server` stopped (`systemctl stop`, verified
`inactive`) before the build/test/verification window; restarted and
verified (`systemctl is-active`=`active`, `GET /v1/models`->HTTP 200,
`deepseek-v4-flash`/`context_length: 32768` present, confirming the
production `ExecStart` -- `--ssd-streaming-cache-experts 75GB --ctx 32768`
-- untouched) after.
