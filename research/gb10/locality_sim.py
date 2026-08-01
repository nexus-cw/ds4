#!/usr/bin/env python3
"""
research/gb10/locality_sim.py -- offline locality/policy study over DS4_ROUTING_TRACE
captures (ds4, branch research/gb10). No GPU, no ds4 dependency at runtime; pure Python
stdlib.

Trace binary format (little-endian), written by the DS4_ROUTING_TRACE hook in
ds4_cuda.cu (ds4_gpu_stream_expert_cache_begin_selected_load /
..._prepare_selected_batch): a flat sequence of variable-length records, each:

    u32 seq            monotonic record counter (global write order)
    u32 token_index    per-phase token index (see below)
    u8  phase          0 = prefill, 1 = decode
    u8  layer          MoE layer index (0..n_layer-1)
    u8  n_selected     number of expert ids in this record
    u8  _pad
    i16 expert_ids[n_selected]

token_index semantics: decode token_index increments once per decode step (detected by
a non-increasing layer number across consecutive decode records -- layers are visited in
increasing order within one token). prefill token_index is the token's position *within
its own prefill batch/chunk* (0..n_tokens-1); a new chunk restarts at 0 relative to the
previous chunk's own base, i.e. token_index is monotonic *within* one continuous prefill
sweep but is not a global position across multiple ds4-eval questions run in one process
(each question's prefill starts its own chunk). For this study we do not need true global
position -- we need per-phase temporal order, which `seq` gives exactly (records are
written synchronously, in real access order, no reordering) -- so the simulator always
walks records in `seq` order and only uses token_index for locality-statistic bucketing
(e.g. reuse distance is measured in trace-record steps, not by token_index arithmetic).

Real ds4 cache facts this simulator must match to be a faithful model (see
research/gb10/MEASUREMENTS.md and ds4.c's
ds4_engine_configure_streaming_cache_budget / ds4_cuda.cu's
cuda_stream_expert_cache_* family):
  - cache key is (layer, expert) -- NOT expert alone. Two different layers'
    expert 5 are two different cache slots.
  - capacity is a flat ENTRY COUNT (not a per-entry byte accounting), derived from a
    configured GiB budget via a *fixed* average bytes-per-expert-entry (confirmed
    constant across configured budgets from real ds4 startup logs on this artifact:
    12.75 MiB/entry) minus a fixed ~6.38 GiB prefill headroom reserved off the top,
    both figures read directly from real `ds4`/`ds4-eval` startup log lines on
    DeepSeek-V4-Flash-MXFP4_MOE.patched.gguf (100GB run: "93.62 GiB dynamic cache
    (7519 experts, 12.75 MiB each)"; 20GB run: "13.62 GiB dynamic cache (1094 experts,
    12.75 MiB each)" -- same 12.75 MiB/entry both times, and the 8GB-budget entry count
    this formula predicts, 130, matches MEASUREMENTS.md's independently-reported
    "8GB arm (130 entries...)" exactly).
  - eviction is a plain global LRU across all (layer,expert) entries (P3a-fix's CUDA
    expert LRU, `cuda_stream_expert_cache_peek`/`_prune_global`).
"""
import argparse
import struct
import sys
import os
from collections import defaultdict, OrderedDict

PREFILL_HEADROOM_GIB = 6.38
BYTES_PER_EXPERT_MIB = 12.75
GIB = 1024 ** 3
MIB = 1024 ** 2
DISK_BW_GBPS = 3.7  # GB/s, from prior MEASUREMENTS.md figures
# Compute floor derived from two independent real measurements in
# MEASUREMENTS.md's "P3a expert-cache sweep" table (steady-state means,
# reps 2-4), both converging on ~0.19s/token:
#   100GB arm: decode 2.957 t/s -> 0.3382s/tok; bytes_from_file/tok 0.552GB
#              -> disk time 0.552/3.7=0.1492s -> compute floor 0.3382-0.1492
#              = 0.1890s.
#   8GB arm, POOLED-ALLOCATOR-FIXED figure from the "P3b item 3" entry
#              (0.000 hit rate, so bytes/tok = full cold working set ~3.06GB):
#              decode 0.98 t/s -> 1.0204s/tok; disk time 3.06/3.7=0.8270s
#              -> compute floor 1.0204-0.8270 = 0.1934s.
#   (The PRE-pool-fix 8GB figure, 0.73 t/s, is excluded: that arm's slowdown
#   is dominated by cudaMalloc/cudaFree/memcpy churn on every 0%-hit install,
#   not compute -- using it would conflate allocator overhead with the GEMM/
#   dequant compute floor this constant is meant to isolate.)
# Mean of the two: 0.191s/token (~5.24 t/s compute-only ceiling).
DEFAULT_COMPUTE_FLOOR_S = 0.191


def capacity_entries_for_gib(gib):
    """Real ds4 GiB-budget -> LRU entry-count conversion (see module docstring)."""
    dynamic_gib = gib - PREFILL_HEADROOM_GIB
    if dynamic_gib <= 0:
        return 0
    return int((dynamic_gib * 1024) / BYTES_PER_EXPERT_MIB)


RECORD_HEAD = struct.Struct("<IIBBBB")


def iter_records(path):
    with open(path, "rb") as f:
        data = f.read()
    off = 0
    n = len(data)
    while off + RECORD_HEAD.size <= n:
        seq, token_index, phase, layer, n_sel, _pad = RECORD_HEAD.unpack_from(data, off)
        off += RECORD_HEAD.size
        ids_bytes = n_sel * 2
        if off + ids_bytes > n:
            break
        ids = struct.unpack_from("<%dh" % n_sel, data, off)
        off += ids_bytes
        yield seq, token_index, phase, layer, ids


def load_trace(paths):
    """Merge one or more trace files. Records from each file are already in
    seq order internally; files are concatenated in the given order (each file is
    one continuous session), re-numbering seq/token_index to be globally
    monotonic per phase across the concatenation so the simulator can walk one
    combined stream."""
    events = []  # (global_seq, phase, layer, ids, orig_token_index, source_idx)
    g_seq = 0
    for src_idx, p in enumerate(paths):
        local_max_seq = -1
        for seq, tok, phase, layer, ids in iter_records(p):
            events.append((g_seq, phase, layer, ids, tok, src_idx))
            g_seq += 1
            local_max_seq = max(local_max_seq, seq)
    return events


# ---------------------------------------------------------------------------
# Access-stream construction: one (layer, expert) reference per selected slot,
# in seq order. This mirrors what cuda_stream_selected_cache_begin_load sees:
# every slot in `selected_ids` (duplicates possible pre-compaction) is looked
# up against the LRU; ds4 compacts duplicates within a single call before the
# device-side fetch, but for hit-rate purposes a repeated id within the same
# call is still just as "hot" (it will hit on its own first occurrence) so we
# keep duplicates -- they cannot cause spurious misses (a dup always hits
# whatever the first occurrence in the same record just did).
# ---------------------------------------------------------------------------

def build_access_stream(events):
    stream = []  # (layer, expert)
    phase_of = []
    for g_seq, phase, layer, ids, tok, src in events:
        for e in ids:
            stream.append((layer, e))
            phase_of.append(phase)
    return stream, phase_of


# ---------------------------------------------------------------------------
# Policy: plain LRU
# ---------------------------------------------------------------------------

def sim_lru(stream, capacity):
    if capacity <= 0:
        return {"hits": 0, "misses": len(stream), "hit_rate": 0.0}
    cache = OrderedDict()  # key -> True, ordered LRU (front = LRU, back = MRU)
    hits = 0
    for key in stream:
        if key in cache:
            cache.move_to_end(key)
            hits += 1
        else:
            if len(cache) >= capacity:
                cache.popitem(last=False)
            cache[key] = True
    misses = len(stream) - hits
    return {"hits": hits, "misses": misses,
            "hit_rate": hits / len(stream) if stream else 0.0}


# ---------------------------------------------------------------------------
# Policy: LFU/heat-pinned -- pin the top-K hottest (layer,expert) keys
# (by global frequency in this trace) permanently; remaining capacity runs
# plain LRU underneath for everything else.
# ---------------------------------------------------------------------------

def sim_pinned(stream, capacity, pin_fraction):
    if capacity <= 0:
        return {"hits": 0, "misses": len(stream), "hit_rate": 0.0, "pinned": 0}
    freq = defaultdict(int)
    for key in stream:
        freq[key] += 1
    n_pin = int(capacity * pin_fraction)
    n_pin = min(n_pin, len(freq))
    hottest = sorted(freq.items(), key=lambda kv: -kv[1])[:n_pin]
    pinned = set(k for k, _ in hottest)
    lru_capacity = capacity - len(pinned)
    cache = OrderedDict()
    hits = 0
    for key in stream:
        if key in pinned:
            hits += 1
            continue
        if key in cache:
            cache.move_to_end(key)
            hits += 1
        else:
            if lru_capacity <= 0:
                continue
            if len(cache) >= lru_capacity:
                cache.popitem(last=False)
            cache[key] = True
    misses = len(stream) - hits
    return {"hits": hits, "misses": misses,
            "hit_rate": hits / len(stream) if stream else 0.0,
            "pinned": len(pinned)}


# ---------------------------------------------------------------------------
# Policy: Belady oracle (offline optimal) -- evict the resident entry whose
# next use is furthest in the future (or never used again).
# ---------------------------------------------------------------------------

def _next_use_after(key, pos, occurrences):
    import bisect
    lst = occurrences.get(key)
    if not lst:
        return None
    idx = bisect.bisect_right(lst, pos)
    if idx >= len(lst):
        return None
    return lst[idx] - pos


def sim_belady_fast(stream, capacity):
    """Belady with per-key sorted occurrence lists + bisect, O(n log n)-ish."""
    if capacity <= 0:
        return {"hits": 0, "misses": len(stream), "hit_rate": 0.0}
    occurrences = defaultdict(list)
    for i, key in enumerate(stream):
        occurrences[key].append(i)
    resident = set()
    hits = 0
    n = len(stream)
    for i, key in enumerate(stream):
        if key in resident:
            hits += 1
            continue
        if len(resident) >= capacity:
            worst_key = None
            worst_dist = -1
            for rk in resident:
                nd = _next_use_after(rk, i, occurrences)
                if nd is None:
                    worst_key = rk
                    break
                if nd > worst_dist:
                    worst_dist = nd
                    worst_key = rk
            resident.discard(worst_key)
        resident.add(key)
    misses = n - hits
    return {"hits": hits, "misses": misses, "hit_rate": hits / n if n else 0.0}


# ---------------------------------------------------------------------------
# Policy: last-token-same-layer prefetch on top of LRU.
#
# Model: decode only (prefetch "at token start" is a decode-loop notion; the
# stream is walked one full token's layers at a time). Before token t's real
# (demand) accesses are simulated, we first "warm" the cache with token
# (t-1)'s selected experts at each layer, in a single batch (as if issued
# asynchronously at the start of token t, while token t-1's output is still
# being sampled / token t's early layers are being computed). Prefetch
# insertions use the same LRU eviction as demand traffic (shared capacity,
# no separate reservation). We track which resident entries were put there
# by *this token's* prefetch step so a subsequent demand hit against one of
# them counts separately ("hits from prefetch").
#
# Bandwidth accounting: prefetch bytes are charged only for entries that
# were not already resident (a no-op prefetch of an already-cached entry
# costs nothing). Demand bytes are the post-prefetch LRU misses. Per the
# ticket's instructed model, prefetch traffic below the bandwidth budget
# available during the token's own compute window (compute_floor * BW) is
# "free" (fully hidden behind compute); only the excess, plus 100% of
# demand-miss bytes (which gate the compute itself), sits on the critical
# path.
# ---------------------------------------------------------------------------

def sim_prefetch_last_token_same_layer(events, capacity, expert_bytes,
                                        compute_floor_s):
    """events: list of (g_seq, phase, layer, ids, tok, src) as from load_trace,
    DECODE records only (already filtered by caller), assumed in seq order and
    grouped by token (all of one token's layers contiguous)."""
    cache = OrderedDict()
    prefetched_this_token = set()

    def touch(key):
        if key in cache:
            cache.move_to_end(key)
            return True
        if len(cache) >= capacity:
            cache.popitem(last=False)
        cache[key] = True
        return False

    # group decode events into per-token layer->ids maps, preserving order
    tokens = []  # list of dict layer -> ids, in first-seen layer order
    cur = OrderedDict()
    last_layer = None
    for g_seq, phase, layer, ids, tok, src in events:
        if last_layer is not None and layer <= last_layer:
            tokens.append(cur)
            cur = OrderedDict()
        cur[layer] = ids
        last_layer = layer
    if cur:
        tokens.append(cur)

    demand_hits = 0
    demand_misses = 0
    prefetch_hits = 0          # demand accesses that hit an entry this
                                # token's prefetch step installed
    prefetch_bytes_total = 0.0
    demand_bytes_total = 0.0
    hidden_prefetch_bytes = 0.0
    exposed_prefetch_bytes = 0.0
    n_tokens_simulated = 0

    prev_layers = None
    for tok_map in tokens:
        n_tokens_simulated += 1
        prefetched_this_token = set()
        if prev_layers is not None:
            for layer, prev_ids in prev_layers.items():
                for e in set(prev_ids):
                    key = (layer, e)
                    if key not in cache:
                        was_hit = touch(key)
                        assert not was_hit
                        prefetched_this_token.add(key)
                        prefetch_bytes_total += expert_bytes
        for layer, ids in tok_map.items():
            for e in ids:
                key = (layer, e)
                if key in prefetched_this_token:
                    demand_hits += 1
                    prefetch_hits += 1
                    touch(key)
                elif key in cache:
                    demand_hits += 1
                    touch(key)
                else:
                    demand_misses += 1
                    demand_bytes_total += expert_bytes
                    touch(key)
        prev_layers = tok_map

    total_accesses = demand_hits + demand_misses
    hit_rate = demand_hits / total_accesses if total_accesses else 0.0
    bandwidth_budget_per_token = compute_floor_s * DISK_BW_GBPS * (1024 ** 3) if False else None
    return {
        "hits": demand_hits,
        "misses": demand_misses,
        "hit_rate": hit_rate,
        "prefetch_hits": prefetch_hits,
        "prefetch_bytes_total_gib": prefetch_bytes_total / GIB,
        "demand_bytes_total_gib": demand_bytes_total / GIB,
        "n_tokens": n_tokens_simulated,
    }


def implied_tps(hit_rate, n_decode_layer_accesses_per_token, expert_bytes_mib,
                 compute_floor_s):
    """bytes/token = miss_rate * accesses_per_token * expert_bytes; implied
    t/s = 1 / (compute_floor + bytes_per_token_GB / DISK_BW_GBPS). This is the
    ticket's own formula (item 4), with the compute floor derived honestly
    from measured data (see DEFAULT_COMPUTE_FLOOR_S)."""
    miss_rate = 1.0 - hit_rate
    bytes_per_token_gb = miss_rate * n_decode_layer_accesses_per_token * expert_bytes_mib * MIB / 1e9
    time_per_token = compute_floor_s + bytes_per_token_gb / DISK_BW_GBPS
    return (1.0 / time_per_token if time_per_token > 0 else float("inf")), bytes_per_token_gb


# ---------------------------------------------------------------------------
# Locality statistics (item 4 deliverables)
# ---------------------------------------------------------------------------

def reuse_distance_stats(stream):
    """Per-access reuse distance (in access-stream steps since the SAME
    (layer,expert) key's previous occurrence); returns a coarse histogram."""
    last_seen = {}
    dists = []
    for i, key in enumerate(stream):
        if key in last_seen:
            dists.append(i - last_seen[key])
        last_seen[key] = i
    return dists


def cross_token_same_layer_overlap(events):
    """For consecutive decode tokens, per layer: |intersect(selected_t,
    selected_t-1)| / |selected_t| (the "last-token-same-layer" locality
    hypothesis this study's prefetch policy exploits)."""
    tokens = []
    cur = OrderedDict()
    last_layer = None
    for g_seq, phase, layer, ids, tok, src in events:
        if phase != 1:
            continue
        if last_layer is not None and layer <= last_layer:
            tokens.append(cur)
            cur = OrderedDict()
        cur[layer] = ids
        last_layer = layer
    if cur:
        tokens.append(cur)
    overlaps = []
    prev = None
    for tok_map in tokens:
        if prev is not None:
            for layer, ids in tok_map.items():
                if layer in prev:
                    a = set(ids)
                    b = set(prev[layer])
                    if a:
                        overlaps.append(len(a & b) / len(a))
        prev = tok_map
    return overlaps


def cross_layer_adjacent_overlap(events):
    """Within one token/prefill-position, overlap between layer L's and
    layer L+1's selected-expert sets (both drawn from 0..n_expert-1, so
    numeric equality is only meaningful if the router assigns a consistent
    "role" per expert id across layers -- which is architecture-dependent;
    reported here purely as a measured statistic, no assumption of meaning
    beyond "how often does the raw id set repeat", the same sense in which
    the ~70% predictability figure from the literature is normally quoted)."""
    tokens = []
    cur = OrderedDict()
    last_layer = None
    for g_seq, phase, layer, ids, tok, src in events:
        if phase != 1:
            continue
        if last_layer is not None and layer <= last_layer:
            tokens.append(cur)
            cur = OrderedDict()
        cur[layer] = ids
        last_layer = layer
    if cur:
        tokens.append(cur)
    overlaps = []
    for tok_map in tokens:
        layers = sorted(tok_map.keys())
        for a_l, b_l in zip(layers, layers[1:]):
            if b_l != a_l + 1:
                continue
            a = set(tok_map[a_l])
            b = set(tok_map[b_l])
            if a:
                overlaps.append(len(a & b) / len(a))
    return overlaps


def popularity_skew(stream):
    freq = defaultdict(int)
    for key in stream:
        freq[key] += 1
    total = sum(freq.values())
    sorted_counts = sorted(freq.values(), reverse=True)
    out = {}
    cum = 0
    n_keys = len(sorted_counts)
    thresholds = [0.5, 0.9, 0.99]
    ti = 0
    for i, c in enumerate(sorted_counts):
        cum += c
        while ti < len(thresholds) and cum >= thresholds[ti] * total:
            out[thresholds[ti]] = (i + 1) / n_keys if n_keys else 0.0
            ti += 1
        if ti >= len(thresholds):
            break
    return {"n_distinct_keys": n_keys, "total_accesses": total,
            "pct_keys_for_threshold": out}


def histogram(values, edges):
    counts = [0] * (len(edges) + 1)
    for v in values:
        placed = False
        for i, e in enumerate(edges):
            if v <= e:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    return counts


def _count_decode_tokens(decode_events):
    n = 0
    last_layer = None
    for g_seq, phase, layer, ids, tok, src in decode_events:
        if last_layer is not None and layer <= last_layer:
            n += 1
        last_layer = layer
    if decode_events:
        n += 1  # the final in-progress token
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+", help="DS4_ROUTING_TRACE binary files, in session order")
    ap.add_argument("--budgets-gib", default="8,40,60,100",
                     help="comma-separated GiB budgets to sweep for LRU/pinned/belady")
    ap.add_argument("--expert-bytes-mib", type=float, default=BYTES_PER_EXPERT_MIB)
    ap.add_argument("--compute-floor-s", type=float, default=DEFAULT_COMPUTE_FLOOR_S)
    ap.add_argument("--skip-belady", action="store_true",
                     help="Belady is O(n*capacity) worst case; skip on huge traces/large budgets")
    ap.add_argument("--pin-fractions", default="0.1,0.25,0.5,0.75,1.0")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    budgets = [float(x) for x in args.budgets_gib.split(",")]
    pin_fracs = [float(x) for x in args.pin_fractions.split(",")]

    events = load_trace(args.traces)
    print(f"# loaded {len(events)} records from {len(args.traces)} trace file(s)")
    n_decode = sum(1 for e in events if e[1] == 1)
    n_prefill = sum(1 for e in events if e[1] == 0)
    print(f"# decode records: {n_decode}  prefill records: {n_prefill}")

    stream, phase_of = build_access_stream(events)
    decode_stream = [k for k, p in zip(stream, phase_of) if p == 1]
    print(f"# total expert-selection accesses: {len(stream)} "
          f"(decode: {len(decode_stream)}, prefill: {len(stream) - len(decode_stream)})")

    decode_events = [e for e in events if e[1] == 1]
    n_decode_tokens = _count_decode_tokens(decode_events)
    accesses_per_decode_token = (len(decode_stream) / n_decode_tokens
                                  if n_decode_tokens else 0.0)
    print(f"# decode tokens observed: {n_decode_tokens}  "
          f"accesses/decode-token: {accesses_per_decode_token:.2f}")

    results = {"budgets": {},
               "n_decode_tokens": n_decode_tokens,
               "accesses_per_decode_token": accesses_per_decode_token}

    for gib in budgets:
        cap = capacity_entries_for_gib(gib)
        row = {"capacity_entries": cap}
        lru = sim_lru(stream, cap)
        row["lru"] = lru
        row["pinned"] = {}
        for pf in pin_fracs:
            row["pinned"][pf] = sim_pinned(stream, cap, pf)
        if not args.skip_belady:
            # Belady's per-eviction linear scan over resident set makes this
            # O(n * cap) worst case; only run it for feasible (trace,cap)
            # combinations.
            if len(stream) * max(cap, 1) <= 200_000_000:
                row["belady"] = sim_belady_fast(stream, cap)
            else:
                row["belady"] = {"skipped": "trace_len*cap too large for this pass"}

        # Decode-only variants -- these are what the implied-t/s formula and
        # the compute-floor derivation (both decode-specific) apply to.
        lru_d = sim_lru(decode_stream, cap)
        row["lru_decode"] = lru_d
        tps, bpt = implied_tps(lru_d["hit_rate"], accesses_per_decode_token,
                                args.expert_bytes_mib, args.compute_floor_s)
        row["lru_decode"]["implied_tps"] = tps
        row["lru_decode"]["bytes_per_token_gb"] = bpt
        belady_d = None
        if not args.skip_belady and len(decode_stream) * max(cap, 1) <= 200_000_000:
            belady_d = sim_belady_fast(decode_stream, cap)
            tps_b, bpt_b = implied_tps(belady_d["hit_rate"], accesses_per_decode_token,
                                        args.expert_bytes_mib, args.compute_floor_s)
            belady_d["implied_tps"] = tps_b
            belady_d["bytes_per_token_gb"] = bpt_b
            row["belady_decode"] = belady_d

        results["budgets"][gib] = row
        print(f"\n== budget {gib} GiB -> capacity {cap} entries ==")
        print(f"  LRU (all accesses) hit_rate={lru['hit_rate']:.4f}")
        print(f"  LRU (decode only)  hit_rate={lru_d['hit_rate']:.4f}  "
              f"bytes/tok={bpt:.3f} GB  implied t/s={tps:.3f}")
        for pf in pin_fracs:
            pr = row["pinned"][pf]
            print(f"  pinned(frac={pf}, all accesses) hit_rate={pr['hit_rate']:.4f} pinned_n={pr['pinned']}")
        if "belady" in row and "hit_rate" in row["belady"]:
            print(f"  belady (all accesses) hit_rate={row['belady']['hit_rate']:.4f}")
        if belady_d:
            print(f"  belady (decode only) hit_rate={belady_d['hit_rate']:.4f} "
                  f"bytes/tok={belady_d['bytes_per_token_gb']:.3f} GB "
                  f"implied t/s={belady_d['implied_tps']:.3f}")

    # prefetch policy (decode-only, needs the raw events not just the flat stream)
    decode_events = [e for e in events if e[1] == 1]
    print("\n== last-token-same-layer prefetch (decode) ==")
    results["prefetch"] = {}
    for gib in budgets:
        cap = capacity_entries_for_gib(gib)
        pf_res = sim_prefetch_last_token_same_layer(
                decode_events, cap, args.expert_bytes_mib * MIB, args.compute_floor_s)
        results["prefetch"][gib] = pf_res
        print(f"  budget {gib} GiB cap={cap}: hit_rate={pf_res['hit_rate']:.4f} "
              f"prefetch_hits={pf_res['prefetch_hits']} "
              f"prefetch_bytes/tok(GiB)={pf_res['prefetch_bytes_total_gib']/max(pf_res['n_tokens'],1):.4f} "
              f"demand_bytes/tok(GiB)={pf_res['demand_bytes_total_gib']/max(pf_res['n_tokens'],1):.4f}")

    # locality statistics
    print("\n== locality statistics ==")
    dists = reuse_distance_stats(decode_stream)
    edges = [1, 5, 10, 50, 100, 500, 1000, 5000]
    hist = histogram(dists, edges)
    print(f"  decode reuse-distance histogram (edges {edges} + overflow): {hist}")
    results["reuse_distance_hist_edges"] = edges
    results["reuse_distance_hist_counts"] = hist
    results["reuse_distance_median"] = sorted(dists)[len(dists) // 2] if dists else None

    overlaps = cross_token_same_layer_overlap(decode_events)
    if overlaps:
        mean_overlap = sum(overlaps) / len(overlaps)
        print(f"  cross-token same-layer overlap: mean={mean_overlap:.4f} n={len(overlaps)}")
        results["cross_token_same_layer_overlap_mean"] = mean_overlap
    else:
        print("  cross-token same-layer overlap: insufficient decode tokens")

    adj_overlaps = cross_layer_adjacent_overlap(decode_events)
    if adj_overlaps:
        mean_adj = sum(adj_overlaps) / len(adj_overlaps)
        print(f"  cross-layer adjacent overlap (raw id sets): mean={mean_adj:.4f} n={len(adj_overlaps)}")
        results["cross_layer_adjacent_overlap_mean"] = mean_adj
    else:
        print("  cross-layer adjacent overlap: insufficient data")

    skew = popularity_skew(decode_stream)
    print(f"  popularity skew: {skew['n_distinct_keys']} distinct (layer,expert) keys, "
          f"{skew['total_accesses']} accesses")
    for th, frac in sorted(skew["pct_keys_for_threshold"].items()):
        print(f"    {int(th*100)}% of selections served by {frac*100:.2f}% of keys")
    results["popularity_skew"] = skew

    if args.json_out:
        import json
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n# wrote {args.json_out}")


if __name__ == "__main__":
    main()
