/* ds4_routing_stats.h -- task#28 per-(layer,expert) routing-traffic telemetry.
 *
 * Cheap persistent counters over the routed-MoE dispatch path: per
 * (layer,expert) selection counts, expert-LRU hit/miss counts, per-layer
 * token totals (decode/prefill), and an optional per-layer high-router-entropy
 * event counter.  Serves the consumer product (cache-budget advisor, console
 * heat, /v1/routing-stats) and the expert-genesis research line (entropy
 * trigger signals; co-routing + corpus capture are documented v1, not here).
 *
 * The module is plain host-side C compiled into every build (CPU, CUDA,
 * Metal, ROCm).  The increment hooks live in the CUDA streamed-dispatch path
 * (ds4_cuda.cu) and the CPU-router decode path (ds4.c); on builds/paths that
 * never call them the counters simply stay zero and the endpoint degrades
 * gracefully.
 *
 * Concurrency: selection/lookup increments use relaxed atomics -- the CUDA
 * streamed dispatch is single-threaded per layer step in the common path,
 * but the async selected-load worker (metal_graph_selected_async_load_*)
 * calls the same begin-load entry from its own thread, so plain u64
 * increments are not provably safe.  Relaxed __atomic adds cost one LSE/lock
 * add each -- noise next to the cudaMemcpy work beside them.  Token-phase
 * tracking (layer-wrap heuristic, mirrors the DS4_ROUTING_TRACE logic) takes
 * a tiny mutex once per (token,layer) call.
 *
 * Persistence: binary file under DS4_ROUTING_STATS_DIR (default
 * ~/.ds4/routing-stats/), model-keyed filename, versioned header, sparse
 * (layer,expert) entries; loaded+merged at init so totals are monotonic
 * across restarts; flushed every DS4_ROUTING_FLUSH_SEC (default 60) from the
 * dispatch tick, plus atexit.
 *
 * Lineage-ready key note: keys are (u16 layer, u16 expert) today.  The file
 * header carries a key_form field (0 = numeric pair).  When dynamic /
 * manifest-born expert ids arrive (genesis world), key_form 1 will add a
 * string id table mapping u16 slots to lineage ids without breaking v0
 * readers, which must skip files whose key_form they do not know.
 *
 * Env:
 *   DS4_ROUTING_COUNTERS=0        kill switch (all hooks become no-ops)
 *   DS4_ROUTING_STATS_DIR=<dir>   override persistence directory
 *   DS4_ROUTING_FLUSH_SEC=<n>     flush period seconds (default 60, 0=never)
 *   DS4_ROUTING_ENTROPY_TAU=<x>   high-entropy threshold in nats;
 *                                 "off"/negative disables; unset = auto
 *                                 0.85*ln(n_experts).  Only counted where
 *                                 router probs are host-resident (CPU-router
 *                                 decode path); the GLM GPU-router path keeps
 *                                 probs on device -- v1 documents the opt-in
 *                                 readback hook.
 */
#ifndef DS4_ROUTING_STATS_H
#define DS4_ROUTING_STATS_H

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    DS4_ROUTING_STATS_MAX_LAYER  = 80,
    DS4_ROUTING_STATS_MAX_EXPERT = 384,
};

/* One-time init: loads+merges any existing persisted file for this model
 * and registers the atexit flush.  Safe to skip entirely (counters then
 * live in RAM only and the endpoint reports persistence:false). */
void ds4_routing_stats_init(const char *model_name, uint64_t model_file_bytes);

/* 0 when DS4_ROUTING_COUNTERS=0 -- hooks return immediately. */
int ds4_routing_stats_enabled(void);

/* Selection counts: one call per streamed begin-load, ids may repeat
 * (k slots per token; prefill passes n_tokens*k ids). */
void ds4_routing_stats_note_selections(uint32_t layer, const int32_t *ids,
                                       uint32_t n_ids);

/* Expert-LRU lookup outcome beside the cache peek (per unique expert). */
void ds4_routing_stats_note_lookup(uint32_t layer, uint32_t expert, int hit);

/* Token-phase accounting.  Decode: one call per (token,layer); a
 * non-increasing layer starts a new token (same heuristic as
 * DS4_ROUTING_TRACE).  Prefill: one call per (chunk,layer) with the chunk's
 * token count; token total advances once per chunk sweep. */
void ds4_routing_stats_note_decode_layer(uint32_t layer);
void ds4_routing_stats_note_prefill_layer(uint32_t layer, uint32_t n_tokens);

/* Router-entropy event counter: probs = unnormalized non-negative router
 * scores for all n experts (host-resident call sites only).  Normalizes,
 * computes Shannon entropy in nats, bumps the per-layer measured/high
 * counters against tau.  No-op when entropy counting is disabled. */
void ds4_routing_stats_note_probs(uint32_t layer, const float *probs,
                                  uint32_t n);

/* Periodic flush tick -- call from the dispatch path; cheap (one relaxed
 * counter + a time check every 4096 calls).  Actual file write at most once
 * per DS4_ROUTING_FLUSH_SEC. */
void ds4_routing_stats_tick(void);

/* Force a flush now (used at shutdown; also atexit-registered by init). */
void ds4_routing_stats_flush(void);

/* Read-only view for /v1/routing-stats.  Pointers are into the live arrays;
 * u64 reads may be torn mid-increment in theory -- monotonic counters, a
 * one-off snapshot for humans, matching the /v1/activity lock-free
 * convention. */
typedef struct {
    int             enabled;
    uint32_t        n_layer;         /* DS4_ROUTING_STATS_MAX_LAYER */
    uint32_t        n_expert;        /* DS4_ROUTING_STATS_MAX_EXPERT */
    const uint64_t *sel;             /* [n_layer*n_expert] selections */
    const uint64_t *hit;             /* [n_layer*n_expert] LRU hits */
    const uint64_t *miss;            /* [n_layer*n_expert] LRU misses */
    uint64_t        decode_tokens;
    uint64_t        prefill_tokens;
    uint64_t        total_selections;
    const uint64_t *entropy_measured; /* [n_layer] tokens with entropy computed */
    const uint64_t *entropy_high;     /* [n_layer] tokens above tau */
    double          entropy_tau;      /* nats; <0 = entropy counting disabled */
    int             persist_loaded;   /* 1 if a prior file was merged at init */
    const char     *persist_path;     /* NULL if persistence inactive */
    uint64_t        flushes;          /* successful flushes this uptime */
} ds4_routing_stats_view;

void ds4_routing_stats_get_view(ds4_routing_stats_view *v);

/* End-of-run aggregate summary line to `out` (stderr if NULL).  Called by the
 * ds4 CLI and ds4-eval at exit when DS4_CUDA_STREAM_STATS=1, alongside
 * ds4_gpu_print_cuda_stream_stats(), so long REPL/eval runs report hit rates
 * without the HTTP endpoint. */
void ds4_routing_stats_print_summary(FILE *out);

#ifdef __cplusplus
}
#endif

#endif /* DS4_ROUTING_STATS_H */
