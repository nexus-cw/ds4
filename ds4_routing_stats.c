/* ds4_routing_stats.c -- task#28 per-(layer,expert) routing telemetry.
 * See ds4_routing_stats.h for the design contract. */
#include "ds4_routing_stats.h"

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define RS_NL DS4_ROUTING_STATS_MAX_LAYER
#define RS_NE DS4_ROUTING_STATS_MAX_EXPERT
#define RS_KEYS ((uint32_t)RS_NL * (uint32_t)RS_NE)

/* Live counters.  Relaxed atomic increments; see header for why. */
static uint64_t g_sel[RS_NL][RS_NE];
static uint64_t g_hit[RS_NL][RS_NE];
static uint64_t g_miss[RS_NL][RS_NE];
static uint64_t g_total_selections;
static uint64_t g_decode_tokens;
static uint64_t g_prefill_tokens;
static uint64_t g_entropy_measured[RS_NL];
static uint64_t g_entropy_high[RS_NL];

/* -1 unknown, 0 disabled, 1 enabled. */
static int g_enabled_cache = -1;

/* Token-phase tracking (mutex-guarded; once per (token,layer)). */
static pthread_mutex_t g_phase_mu = PTHREAD_MUTEX_INITIALIZER;
static int32_t  g_decode_last_layer = -1;
static int32_t  g_prefill_last_layer = -1;

/* Entropy config: NAN = not yet resolved; <0 = disabled. */
static double g_entropy_tau = -2.0;
static int    g_entropy_tau_resolved = 0;

/* Persistence. */
static pthread_mutex_t g_flush_mu = PTHREAD_MUTEX_INITIALIZER;
static char     g_persist_path[1024];
static int      g_persist_active = 0;
static int      g_persist_loaded = 0;
static uint64_t g_flushes = 0;
static time_t   g_last_flush_time = 0;
static long     g_flush_period_sec = 60;
static uint64_t g_tick_calls = 0;

#define RS_MAGIC "DS4RTCV1"
#define RS_VERSION 1u
#define RS_KEY_FORM_NUMERIC 0u  /* (u16 layer, u16 expert); form 1 reserved
                                 * for lineage/manifest string-id tables. */

int ds4_routing_stats_enabled(void) {
    if (g_enabled_cache < 0) {
        const char *e = getenv("DS4_ROUTING_COUNTERS");
        g_enabled_cache = (e && e[0] == '0' && e[1] == '\0') ? 0 : 1;
    }
    return g_enabled_cache;
}

static inline void rs_add(uint64_t *p, uint64_t v) {
    __atomic_fetch_add(p, v, __ATOMIC_RELAXED);
}

void ds4_routing_stats_note_selections(uint32_t layer, const int32_t *ids,
                                       uint32_t n_ids) {
    if (!ds4_routing_stats_enabled() || layer >= RS_NL || !ids) return;
    for (uint32_t i = 0; i < n_ids; i++) {
        const int32_t e = ids[i];
        if (e < 0 || e >= (int32_t)RS_NE) continue;
        rs_add(&g_sel[layer][e], 1);
    }
    rs_add(&g_total_selections, n_ids);
}

void ds4_routing_stats_note_lookup(uint32_t layer, uint32_t expert, int hit) {
    if (!ds4_routing_stats_enabled() || layer >= RS_NL || expert >= RS_NE) {
        return;
    }
    rs_add(hit ? &g_hit[layer][expert] : &g_miss[layer][expert], 1);
}

void ds4_routing_stats_note_decode_layer(uint32_t layer) {
    if (!ds4_routing_stats_enabled()) return;
    pthread_mutex_lock(&g_phase_mu);
    if (g_decode_last_layer >= 0 && (int32_t)layer <= g_decode_last_layer) {
        g_decode_tokens++;
    }
    if (g_decode_last_layer < 0) g_decode_tokens++;  /* first ever token */
    g_decode_last_layer = (int32_t)layer;
    pthread_mutex_unlock(&g_phase_mu);
}

void ds4_routing_stats_note_prefill_layer(uint32_t layer, uint32_t n_tokens) {
    if (!ds4_routing_stats_enabled()) return;
    pthread_mutex_lock(&g_phase_mu);
    /* Count each chunk's tokens once, at its first layer sweep. */
    if (g_prefill_last_layer < 0 || (int32_t)layer <= g_prefill_last_layer) {
        g_prefill_tokens += n_tokens;
    }
    g_prefill_last_layer = (int32_t)layer;
    pthread_mutex_unlock(&g_phase_mu);
}

static double rs_entropy_tau(uint32_t n_experts) {
    if (!g_entropy_tau_resolved) {
        const char *e = getenv("DS4_ROUTING_ENTROPY_TAU");
        if (e && (!strcmp(e, "off") || !strcmp(e, "OFF"))) {
            g_entropy_tau = -1.0;
        } else if (e && e[0]) {
            g_entropy_tau = atof(e);
            if (g_entropy_tau < 0.0) g_entropy_tau = -1.0;
        } else {
            /* Auto: 0.85 of the max possible entropy over n experts. */
            g_entropy_tau = n_experts > 1 ? 0.85 * log((double)n_experts)
                                          : -1.0;
        }
        g_entropy_tau_resolved = 1;
    }
    return g_entropy_tau;
}

void ds4_routing_stats_note_probs(uint32_t layer, const float *probs,
                                  uint32_t n) {
    if (!ds4_routing_stats_enabled() || layer >= RS_NL || !probs || n == 0) {
        return;
    }
    const double tau = rs_entropy_tau(n);
    if (tau < 0.0) return;
    double sum = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        if (probs[i] > 0.0f) sum += (double)probs[i];
    }
    if (sum <= 0.0) return;
    double h = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        if (probs[i] <= 0.0f) continue;
        const double p = (double)probs[i] / sum;
        h -= p * log(p);
    }
    rs_add(&g_entropy_measured[layer], 1);
    if (h > tau) rs_add(&g_entropy_high[layer], 1);
}

/* ---------------- persistence ---------------- */

static void rs_put_u32(FILE *f, uint32_t v) { fwrite(&v, sizeof(v), 1, f); }
static void rs_put_u64(FILE *f, uint64_t v) { fwrite(&v, sizeof(v), 1, f); }
static int rs_get_u32(FILE *f, uint32_t *v) { return fread(v, sizeof(*v), 1, f) == 1; }
static int rs_get_u64(FILE *f, uint64_t *v) { return fread(v, sizeof(*v), 1, f) == 1; }

/* Write the full current state (cumulative, includes any merged baseline)
 * to path atomically (tmp + rename).  Caller holds g_flush_mu. */
static int rs_write_file(const char *path) {
    char tmp[1100];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) return 0;
    fwrite(RS_MAGIC, 1, 8, f);
    rs_put_u32(f, RS_VERSION);
    rs_put_u32(f, RS_KEY_FORM_NUMERIC);
    rs_put_u32(f, RS_NL);
    rs_put_u32(f, RS_NE);
    rs_put_u64(f, g_decode_tokens);
    rs_put_u64(f, g_prefill_tokens);
    rs_put_u64(f, g_total_selections);
    rs_put_u64(f, (uint64_t)time(NULL));
    for (uint32_t l = 0; l < RS_NL; l++) rs_put_u64(f, g_entropy_measured[l]);
    for (uint32_t l = 0; l < RS_NL; l++) rs_put_u64(f, g_entropy_high[l]);
    uint32_t n_entries = 0;
    for (uint32_t l = 0; l < RS_NL; l++) {
        for (uint32_t e = 0; e < RS_NE; e++) {
            if (g_sel[l][e] | g_hit[l][e] | g_miss[l][e]) n_entries++;
        }
    }
    rs_put_u32(f, n_entries);
    rs_put_u32(f, 0);  /* reserved */
    for (uint32_t l = 0; l < RS_NL; l++) {
        for (uint32_t e = 0; e < RS_NE; e++) {
            if (!(g_sel[l][e] | g_hit[l][e] | g_miss[l][e])) continue;
            const uint16_t l16 = (uint16_t)l, e16 = (uint16_t)e;
            fwrite(&l16, sizeof(l16), 1, f);
            fwrite(&e16, sizeof(e16), 1, f);
            rs_put_u64(f, g_sel[l][e]);
            rs_put_u64(f, g_hit[l][e]);
            rs_put_u64(f, g_miss[l][e]);
        }
    }
    const int ok = ferror(f) == 0;
    fclose(f);
    if (!ok || rename(tmp, path) != 0) {
        unlink(tmp);
        return 0;
    }
    return 1;
}

/* Load path and ADD into the live arrays (merge-across-restarts). */
static int rs_load_merge(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    char magic[8];
    uint32_t ver = 0, key_form = 0, nl = 0, ne = 0, n_entries = 0, res = 0;
    uint64_t dec = 0, pre = 0, tot = 0, when = 0;
    int ok = fread(magic, 1, 8, f) == 8 && memcmp(magic, RS_MAGIC, 8) == 0 &&
             rs_get_u32(f, &ver) && ver == RS_VERSION &&
             rs_get_u32(f, &key_form) && key_form == RS_KEY_FORM_NUMERIC &&
             rs_get_u32(f, &nl) && nl <= RS_NL &&
             rs_get_u32(f, &ne) && ne <= RS_NE &&
             rs_get_u64(f, &dec) && rs_get_u64(f, &pre) &&
             rs_get_u64(f, &tot) && rs_get_u64(f, &when);
    if (ok) {
        for (uint32_t l = 0; ok && l < nl; l++) {
            uint64_t v; ok = rs_get_u64(f, &v); g_entropy_measured[l] += v;
        }
        for (uint32_t l = 0; ok && l < nl; l++) {
            uint64_t v; ok = rs_get_u64(f, &v); g_entropy_high[l] += v;
        }
    }
    ok = ok && rs_get_u32(f, &n_entries) && rs_get_u32(f, &res) &&
         n_entries <= RS_KEYS;
    for (uint32_t i = 0; ok && i < n_entries; i++) {
        uint16_t l16, e16;
        uint64_t sel, hit, miss;
        ok = fread(&l16, sizeof(l16), 1, f) == 1 &&
             fread(&e16, sizeof(e16), 1, f) == 1 &&
             rs_get_u64(f, &sel) && rs_get_u64(f, &hit) && rs_get_u64(f, &miss);
        if (ok && l16 < RS_NL && e16 < RS_NE) {
            g_sel[l16][e16] += sel;
            g_hit[l16][e16] += hit;
            g_miss[l16][e16] += miss;
        }
    }
    fclose(f);
    if (ok) {
        g_decode_tokens += dec;
        g_prefill_tokens += pre;
        g_total_selections += tot;
    } else {
        fprintf(stderr,
                "ds4: routing-stats: %s unreadable or wrong version; "
                "starting counters fresh (file will be overwritten)\n", path);
    }
    return ok;
}

static void rs_sanitize(const char *in, char *out, size_t cap) {
    size_t j = 0;
    for (size_t i = 0; in && in[i] && j + 1 < cap && j < 64; i++) {
        const char c = in[i];
        out[j++] = ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                    (c >= '0' && c <= '9') || c == '.' || c == '-')
                       ? c : '_';
    }
    if (j == 0) out[j++] = 'm';
    out[j] = '\0';
}

static void rs_atexit_flush(void) { ds4_routing_stats_flush(); }

void ds4_routing_stats_init(const char *model_name, uint64_t model_file_bytes) {
    if (!ds4_routing_stats_enabled()) return;
    pthread_mutex_lock(&g_flush_mu);
    if (!g_persist_active) {
        const char *env_period = getenv("DS4_ROUTING_FLUSH_SEC");
        if (env_period && env_period[0]) g_flush_period_sec = atol(env_period);
        char dir[768];
        const char *env_dir = getenv("DS4_ROUTING_STATS_DIR");
        if (env_dir && env_dir[0]) {
            snprintf(dir, sizeof(dir), "%s", env_dir);
        } else {
            const char *home = getenv("HOME");
            snprintf(dir, sizeof(dir), "%s/.ds4/routing-stats",
                     home && home[0] ? home : ".");
        }
        /* mkdir -p (two levels max for the default layout). */
        char parent[768];
        snprintf(parent, sizeof(parent), "%s", dir);
        char *slash = strrchr(parent, '/');
        if (slash && slash != parent) { *slash = '\0'; (void)mkdir(parent, 0755); }
        (void)mkdir(dir, 0755);
        char name[80];
        rs_sanitize(model_name, name, sizeof(name));
        snprintf(g_persist_path, sizeof(g_persist_path),
                 "%s/%s-%llu.rstats", dir, name,
                 (unsigned long long)model_file_bytes);
        g_persist_loaded = rs_load_merge(g_persist_path);
        g_persist_active = 1;
        g_last_flush_time = time(NULL);
        atexit(rs_atexit_flush);
        fprintf(stderr,
                "ds4: routing-stats: persisting to %s (%s prior state, "
                "flush every %lds)\n",
                g_persist_path, g_persist_loaded ? "merged" : "no",
                g_flush_period_sec);
    }
    pthread_mutex_unlock(&g_flush_mu);
}

void ds4_routing_stats_flush(void) {
    if (!ds4_routing_stats_enabled()) return;
    pthread_mutex_lock(&g_flush_mu);
    if (g_persist_active && rs_write_file(g_persist_path)) {
        g_flushes++;
        g_last_flush_time = time(NULL);
    }
    pthread_mutex_unlock(&g_flush_mu);
}

void ds4_routing_stats_tick(void) {
    if (!ds4_routing_stats_enabled()) return;
    /* Cheap gate: only look at the clock every 4096 dispatch calls. */
    if ((__atomic_add_fetch(&g_tick_calls, 1, __ATOMIC_RELAXED) & 4095ull) !=
        0ull) {
        return;
    }
    if (!g_persist_active || g_flush_period_sec <= 0) return;
    const time_t now = time(NULL);
    if (now - g_last_flush_time < g_flush_period_sec) return;
    ds4_routing_stats_flush();
}

void ds4_routing_stats_get_view(ds4_routing_stats_view *v) {
    if (!v) return;
    memset(v, 0, sizeof(*v));
    v->enabled = ds4_routing_stats_enabled();
    v->n_layer = RS_NL;
    v->n_expert = RS_NE;
    v->sel = &g_sel[0][0];
    v->hit = &g_hit[0][0];
    v->miss = &g_miss[0][0];
    v->decode_tokens = g_decode_tokens;
    v->prefill_tokens = g_prefill_tokens;
    v->total_selections = g_total_selections;
    v->entropy_measured = g_entropy_measured;
    v->entropy_high = g_entropy_high;
    v->entropy_tau = g_entropy_tau_resolved ? g_entropy_tau : -2.0;
    v->persist_loaded = g_persist_loaded;
    v->persist_path = g_persist_active ? g_persist_path : NULL;
    v->flushes = g_flushes;
}

/* End-of-run one-line summary for the CLI tools (task#15 item 1).  The server
 * self-reports via /v1/routing-stats; the ds4 CLI and ds4-eval call this at
 * exit when DS4_CUDA_STREAM_STATS=1 so long REPL/eval sessions emit the same
 * aggregate hit-rate figures the A/B tables need.  Aggregates only -- the
 * per-(layer,expert) matrix stays behind the HTTP endpoint. */
void ds4_routing_stats_print_summary(FILE *out) {
    if (!out) out = stderr;
    ds4_routing_stats_view v;
    ds4_routing_stats_get_view(&v);
    if (!v.enabled) {
        fprintf(out, "ds4: routing-stats: disabled (DS4_ROUTING_COUNTERS=0)\n");
        return;
    }
    uint64_t hits = 0, misses = 0, layers_active = 0;
    for (uint32_t l = 0; l < v.n_layer; l++) {
        uint64_t layer_sel = 0;
        for (uint32_t e = 0; e < v.n_expert; e++) {
            const size_t i = (size_t)l * v.n_expert + e;
            layer_sel += v.sel[i];
            hits += v.hit[i];
            misses += v.miss[i];
        }
        if (layer_sel) layers_active++;
    }
    const uint64_t lookups = hits + misses;
    fprintf(out,
            "ds4: routing-stats summary: decode_tokens=%llu prefill_tokens=%llu "
            "selections=%llu lru_hits=%llu lru_misses=%llu hit_rate=%.3f "
            "layers_active=%llu flushes=%llu persist=%s\n",
            (unsigned long long)v.decode_tokens,
            (unsigned long long)v.prefill_tokens,
            (unsigned long long)v.total_selections,
            (unsigned long long)hits,
            (unsigned long long)misses,
            lookups ? (double)hits / (double)lookups : 0.0,
            (unsigned long long)layers_active,
            (unsigned long long)v.flushes,
            v.persist_path ? v.persist_path : "none");
}
