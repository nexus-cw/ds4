# Routing telemetry (task #28)

Per-(layer,expert) routing-traffic counters on the CUDA streamed-MoE
dispatch path. One mechanism, two consumers: the product side (cache-budget
advisor, console heat, personalized repack input) and the expert-genesis
research line (entropy trigger signals; co-routing and corpus capture are
designed here as v1, not implemented).

## What is counted (v0, shipped)

| Counter | Where it ticks | Notes |
|---|---|---|
| `sel[layer][expert]` | `cuda_stream_selected_cache_begin_load` (ds4_cuda.cu) | every selection slot, duplicates included (k per token); the one choke point all streamed paths funnel through: decode, prefill batch, GLM tensor entry, async selected-load worker |
| `hit/miss[layer][expert]` | beside the LRU `cuda_stream_expert_cache_peek` hit/else branches | per unique expert per fetch call, consistent with the task-18 `DS4_CUDA_STREAM_STATS` aggregates |
| `decode_tokens` / `prefill_tokens` | the extern dispatch entries (`begin_selected_load`, `prepare_selected_batch`, GLM `_tensor` entry) | layer-wrap heuristic, same as `DS4_ROUTING_TRACE` |
| `entropy_measured/high[layer]` | CPU-router decode path (`metal_graph_decode_cpu_router`, ds4.c) | see "Router entropy" below |

Arrays are fixed `u64[80][384]` (matching the CUDA expert-cache table
bounds), ~740 KB RSS for sel+hit+miss. Increments are relaxed
`__atomic_fetch_add`: the dispatch path is single-threaded per layer step in
the common case, but the async selected-load worker
(`metal_graph_selected_async_load_*`) calls the same begin-load entry from
its own thread, so plain u64 increments are not provably race-free. Relaxed
atomics cost one LSE add each -- unmeasurable next to the device copies
beside them (verified: warm decode and cold prefill within noise, see
MEASUREMENTS.md task-28).

## Env vars

| Var | Default | Meaning |
|---|---|---|
| `DS4_ROUTING_COUNTERS` | on | `0` = kill switch, every hook returns immediately |
| `DS4_ROUTING_STATS_DIR` | `~/.ds4/routing-stats` | persistence directory |
| `DS4_ROUTING_FLUSH_SEC` | `60` | flush period; `0` = never (RAM only) |
| `DS4_ROUTING_ENTROPY_TAU` | auto (`0.85*ln(n_experts)`) | high-entropy threshold in nats; `off` disables |
| `DS4_ROUTING_TOPN` | `20` | top-N size in `/v1/routing-stats` (the HTTP parser strips query strings, so this is env, not `?n=`) |

## Persistence format (`.rstats` v1)

Model-keyed filename `<sanitized general.name>-<file_bytes>.rstats` so
multi-model boxes never mix streams. Atomic tmp+rename writes; on boot the
file is loaded and **added** into the live arrays, so all totals are
monotonic across restarts (`/v1/routing-stats` `persistence.merged_prior_state`).

Little-endian layout:

```
magic  "DS4RTCV1"                       8 B
u32    version        = 1
u32    key_form       = 0               (see lineage note)
u32    n_layer        = 80
u32    n_expert       = 384
u64    decode_tokens, prefill_tokens, total_selections, flush_unix_time
u64    entropy_measured[n_layer]
u64    entropy_high[n_layer]
u32    n_entries, u32 reserved
entry  { u16 layer; u16 expert; u64 sel; u64 hit; u64 miss }  (sparse, nonzero only)
```

**Lineage-ready note.** Keys are numeric `(u16 layer, u16 expert)` today
(`key_form 0`). In the manifest/genesis world experts get dynamic lineage
ids; `key_form 1` is reserved to prepend a string-id table mapping u16
slots to lineage ids, so grown/spawned experts keep stable identity across
repacks without breaking the record layout. Readers must skip files whose
`key_form` they do not know; v0 does exactly that (logs and starts fresh).

## /v1/routing-stats

Follows the `/v1/capabilities` / `/v1/activity` pattern: request-time
aggregation, lock-free counter reads, no GPU work, O(30720-key qsort) per
request. Sections: `totals`, `persistence`, `top_experts`,
`per_layer` (selections, unique experts, selection-distribution Shannon
entropy vs `ln(unique)`, high-entropy token counts), `router_entropy`,
`coverage` (selection share of the hottest 10/25/50% of distinct keys --
the popularity-skew figures previously derived offline from
`DS4_ROUTING_TRACE` captures), and `advisor`.

**Cache-budget advisor v0.** For budgets 50/55/60/63.6/70 GiB:
`K = budget / expert_bytes` (nominal slab expert size the engine planned,
`ds4_gpu_stream_expert_cache_expert_bytes_configured()`), and
`estimated_hit_rate_static` = share of all selections served by the top-K
keys by lifetime frequency. The field name is deliberately honest: this is
the *static popularity-skew approximation* -- an LRU with temporal locality
does better, `locality_sim.py` remains the offline reference. Good enough
for the console's "what would trimming cost" panel.

The console (`/`) has a Routing section: per-layer entropy bars, top-10
hottest experts, and the advisor table, polled every 10 s.

## Router entropy (genesis trigger feed)

Decision, from code: on the GB10 GLM production path the router
probabilities never leave the device (`ds4_gpu_glm_router_select_tensor`
writes `g->router_probs` on-GPU; only the i32 selected ids are read back),
so a host-side entropy compare is **not** free there -- it would add a
384-float D2H sync per (layer, token). v0 therefore hooks the entropy
counter where the scores are already host-resident: the CPU-router decode
path (`metal_graph_decode_cpu_router`), where the full score vector was
just computed on the host and the compare costs one normalize+log loop
next to a matvec. Counting is ON by default with auto tau
(`0.85*ln(n_experts)`) at that site; on the GLM GPU-router path the
per-layer entropy counters stay zero until v1.

**v1: GPU-router entropy.** Two options, in preference order:
1. Device-side compare: a tiny kernel fused after `glm_router_select_tensor`
   computes entropy from `g->router_probs` and bumps a device counter
   array, copied back on the routing-stats flush cadence (zero extra D2H
   per token).
2. Opt-in host readback of `g->router_probs` beside the existing selected-id
   readback in the async load job (already a D2H sync point), gated on
   `DS4_ROUTING_ENTROPY_TAU` being explicitly set.

## v1 designs (documented, not implemented)

**Co-routing.** The naive pairwise matrix is ~10.2k x 10.2k keys --
untenable. Design: reservoir-sample K token-events per hour (K~4096); each
sample stores the token's full per-layer expert sets (46 layers x 8
experts x u16 = 736 B/event, ~3 MB/hour bounded by the reservoir). Pairwise
co-occurrence, conditional routing P(e_j at l+1 | e_i at l), and clustering
for repack grouping are then computed offline from the reservoir file. Hook
point: the same begin-load choke point, tagged with the token index the
phase tracker already maintains.

**Corpus capture.** The request text and its routed experts meet in
ds4_server.c's session worker: the prompt/decode token stream lives in the
session slot while the per-token expert sets tick in the dispatch path
underneath. Hook point for v1: tag each request with a session/request id,
have the routing-stats module keep a small ring of (token_index -> expert
sets) for the current request, and let the server flush
(request_text_hash, expert-set summary) pairs on completion -- giving the
genesis corpus miner "which experts fire for which text" without storing
raw text in the telemetry stream (hash + optional opt-in text path).

## Overhead + verification

See MEASUREMENTS.md "task #28". Contract: counters default ON only because
warm decode and cold-prefill chunk t/s measured within noise; kill switch
`DS4_ROUTING_COUNTERS=0` regardless; greedy identity untouched (counters
never feed back into computation); CPU builds compile the module but no
hook fires, and the endpoint reports zeros/absent advisor.
