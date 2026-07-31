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

## Pending (updated)

- [x] CUDA hit-rate counter hook -> **done this pass**: `DS4_CUDA_STREAM_STATS=1`
      (`ds4_cuda.cu`/`ds4_gpu.h`/`ds4_cli.c`), measured hit_rate=0.000 on the real artifact
      (see above) -- confirms, doesn't just estimate, that the CUDA expert-cache budget is
      currently inert.
- [ ] **New, higher-priority than the item below**: port Metal's real per-`(layer,expert)`
      LRU expert cache to CUDA (`ds4_gpu_set_streaming_expert_cache_budget`/
      `_expert_bytes` are no-op stubs today; `cuda_stream_selected_cache_begin_load`
      unconditionally re-fetches every call) -- this is *why* the cache-size sweep was flat,
      not a compute ceiling or disk-bandwidth ceiling as such (disk bandwidth is real and
      close to saturated per the Q1 bound-discrimination numbers above, but a working cache
      would still reduce the fraction of tokens that need to hit it).
- [ ] Fold the 11/43 mixed-precision "bypass expert cache" routed layers into the cacheable
      slab class (or a secondary slab tier) -- unchanged by this pass; matters most once the
      CUDA cache above is real (currently moot on CUDA since nothing is cached regardless).
- [ ] Single blob vs sharded layout: same scattered-read benchmark, 1x81 GB vs 142x3 GB
- [ ] Expert reordering within a blob by co-activation affinity
- [ ] Session working-set size from routing traces (gates the locality-training thesis, and
      now also gates how much benefit a real CUDA expert cache would realistically deliver)
