# GB10 streaming research

Working notes for running large MoE models at usable speed on **NVIDIA GB10** hardware
(DGX Spark and OEM variants), streaming weights from NVMe when the model exceeds memory.

**Goal:** make it easier to run a capable local model at a usable speed on hardware that a
home user or small business already owns — no multi-100k infrastructure. Slow but effective
is acceptable: the target is **~1–3 tok/s for asynchronous/batch work** (overnight coding
tickets, document analysis), not interactive chat speeds.

This branch holds measurements, methodology and design notes. Code changes land on separate
`feat/*` branches so each can be proposed upstream independently.

## Test hardware

| | |
|---|---|
| Machine | GIGABYTE AI TOP ATOM (OEM GB10 variant — *not* an NVIDIA-branded DGX Spark) |
| SoC | NVIDIA GB10, `sm_121` (Blackwell, `compute_120f` family) |
| Memory | 121 GB usable **unified** LPDDR5X (CPU and GPU share one physical pool) |
| OS / toolchain | Ubuntu 24.04.4 aarch64, CUDA 13.0.88, driver 580.159.03 |
| Storage | 1 TB `ESL01TBTLCZ-27J2-TYN`, Phison **PS5027-E27T**, PCIe **Gen4 x4**, **DRAM-less** (confirmed: requests 64 MB HMB) |
| Networking | ConnectX-7, 2× QSFP (unpopulated); Realtek 10 GbE; no discrete VRAM |

Build target: `make cuda-spark` — builds clean on aarch64, `rc=0`.

## Storage findings

Measured with a synthetic expert-access pattern: scattered **19 MB** reads (one GLM-5.2
expert at int4) spread across a **429 GB** corpus, page cache dropped before every arm.

| queue depth | throughput | % of 5.0 GB/s sequential |
|---|---|---|
| QD1 | 0.80 GB/s | 16% |
| QD4 | 2.44 GB/s | 49% |
| QD8 | 3.29 GB/s | 66% |
| **QD16** | **3.73 GB/s** | **75%** |

**Queue depth matters far more than DRAM cache.** A budget DRAM-less drive delivers 75% of
its sequential figure under realistic scattered expert access, provided I/O is parallel.
This contradicts the common assumption that DRAM-less SSDs are unsuitable for weight
streaming.

What *is* disqualifying: the same class of drive **behind a USB bridge**. USB mass-storage
carries no HMB, so a DRAM-less drive must fetch FTL mapping entries from flash on every
scattered read — throughput then *falls* with parallelism (729 → 614 MB/s from QD1 to 8-way)
where native NVMe *rises*. **Recommendation for a streaming appliance: require DRAM cache
and native PCIe attachment; never USB.**

Striping is real and roughly additive: 2× Samsung 990 PRO (Gen4 x4 each) measured
**11.9 GB/s** read together, better than the sum of their singles.

## Reference: baseline engine behaviour

Measured on this machine with a 744B-parameter MoE (429 GB, int4) under an engine whose
expert cache could not populate on unified memory:

- Decode pinned at **0.21–0.27 tok/s** across every configuration tried.
- **tok/s was insensitive to cache hit rate over a 30x range (1.9% → 59.6%).** With the
  expert tier empty, expert matmul falls back to CPU (~48 s per 32 tokens) and co-limits with
  disk wait (~57 s), so storage tuning cannot move the result.
- I/O queue depth was worth **+11% tok/s and −34% disk traffic** (engine default of 1 → 8).
- `O_DIRECT` gave **no measurable benefit** on a DRAM-less drive — expected, since bypassing
  the page cache removes the only thing partly compensating for flash-resident FTL lookups.

**Lesson for engine design on GB10:** "VRAM" and "RAM" are the *same physical pool*.
A tiering design that budgets them additively will either over-commit (kernel OOM) or starve
the cache. Unified memory needs a single-pool budget:
`ram_budget = physical − vram_tier − dense − runtime`.

## Reasoning cost

At streaming speeds a reasoning token is expensive: at 0.21 tok/s each costs **4.8 s**, so a
1,000–3,000 token thinking block is **1.3–4 hours before the answer begins**. A hard
reasoning budget is therefore not a nicety — it converts unbounded tail latency into a
deterministic ceiling. ds4 exposes `--think` / `--think-max` / `--nothink`; a token-budget
cap (as in llama.cpp's `--reasoning-budget`) is the more precise control.

## Methodology

Two rules, each of which caught a false result during this work:

1. **Sample across the whole model corpus and drop caches between arms.** An early benchmark
   read one 3 GB file repeatedly and reported 20–40 GB/s at high queue depth — entirely page
   cache, not disk.
2. **Restore the engine's learned cache/heat state between arms.** Persistent pin state
   drifted from 8% to 30% pin-hits across runs; comparing run 1 to run 3 naively would have
   manufactured a 35% → 53% "improvement" that was only the cache warming up.

Additionally: measure a **noise floor** (5–10 identical runs, σ per metric) before trusting
any A/B, and report **cold-start and steady-state separately** — a cache tier judged on a run
too short to amortise its fill cost will look worse than useless.

## Attribution

Ideas for cache and prefetch design are taken from published literature, not from other
implementations: FlashMoE (arXiv:2601.17063), ProMoE (arXiv:2410.22134), Fate
(arXiv:2502.12224), Pre-gated MoE (ISCA 2024), MoE-Infinity (arXiv:2401.14361), Ripple
(arXiv:2410.19274), ExFlow (arXiv:2401.08383), LLM in a Flash (arXiv:2312.11514).
