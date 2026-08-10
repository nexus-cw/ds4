/* ds4_inkling_server.c -- console-compatible OpenAI-compatible HTTP server
 * for the "inkling" engine (ds4_inkling.c).  Sibling of ds4_server.c but
 * scoped to the much smaller inkling surface: SSE streaming chat
 * completions, capability/activity introspection and a swap-back model
 * select so an accretion console can leave inkling mode.  No external
 * libraries: plain POSIX sockets + pthreads and a small hand-written JSON
 * reader/writer, in the same spirit as the rest of ds4.
 *
 * Threading model (v1): the inkling engine (ink_model / ink_forward_batch)
 * keeps exactly one KV/shortconv state and is NOT reentrant, so all engine
 * calls happen on ONE dedicated worker thread.  Every accepted connection
 * gets its own short-lived "network thread" that reads + parses the HTTP
 * request and, for /v1/chat/completions, pushes a job onto a FIFO work
 * queue (mutex + cond) and blocks until the worker signals completion; the
 * worker thread is the only one that touches g_model's mutable state and it
 * writes the HTTP/SSE response directly to the job's fd.  Network threads
 * never touch the engine.
 *
 * The N session slots (default 2, -s / DS4_SESSIONS) bound queue depth --
 * they are a fairness/backpressure construct, NOT concurrent decode slots.
 * Because the engine has a single KV state, every queued request still does
 * a full ink_state_reset + prefill on the worker; slots only decide how
 * many requests may wait in line before new ones are rejected with a 503.
 * Fair scheduling is plain FIFO.
 *
 * Endpoints:
 *   GET  /v1/capabilities (+ /capabilities) -> engine/model capability blob
 *   GET  /v1/activity     (+ /activity)     -> live per-slot snapshot
 *   GET  /v1/models                          -> OpenAI-shaped model list
 *   GET  /v1/models/available                -> scan for swappable *.gguf
 *   POST /v1/models/select                   -> admin-gated swap-back
 *   POST /v1/chat/completions                -> OpenAI chat completion,
 *                                                streaming (SSE) or not
 */

#include <arpa/inet.h>
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "ds4_inkling.h"

#define MAX_HEADER_BYTES  (64 * 1024)
#define MAX_BODY_BYTES    (1 * 1024 * 1024)
#define PREFILL_CHUNK     32   /* CPU engine batch; GPU uses INK_GPU_PREFILL_CHUNK */

/* Backend selection.  A CUDA build defaults to the GPU engine (that is the
 * whole point of it: the CPU engine runs the same math ~25x slower and is
 * kept as the correctness reference).  --cpu or DS4_INKLING_BACKEND=cpu
 * forces the reference path; on a CPU-only build the GPU is unavailable and
 * the flag is a no-op. */
static bool g_use_gpu = false;

static const char *backend_name(void) { return g_use_gpu ? "cuda" : "cpu"; }

/* Both engines advance the SAME ink_model state (KV + shortconv), so they
 * are interchangeable per call; the server picks one at startup and keeps
 * it for the process lifetime. */
static void engine_forward_batch(ink_model *m, const int *tokens, uint32_t n_tok,
                                 uint32_t pos0, float *out_logits) {
#ifdef DS4_INKLING_CUDA
    if (g_use_gpu) { ink_forward_gpu(m, tokens, n_tok, pos0, out_logits); return; }
#endif
    ink_forward_batch(m, tokens, n_tok, pos0, out_logits);
}

static int engine_prefill_chunk(void) {
#ifdef DS4_INKLING_CUDA
    if (g_use_gpu) return INK_GPU_PREFILL_CHUNK;
#endif
    return PREFILL_CHUNK;
}

/* POST /v1/models/select: a successful select drains and exits with this
 * code so a Restart=on-failure supervisor restarts the process reading the
 * updated env file (same constant/convention as ds4_server.c). */
#define DS4_SERVER_SWAP_EXIT_CODE 42

/* ============================== small utils =========================== */

static void diesys(const char *msg) {
    perror(msg);
    exit(1);
}

/* growable byte buffer, used both for request accumulation and for
 * building JSON responses / decoded strings. */
typedef struct { char *p; size_t len, cap; } sbuf;

static void sbuf_init(sbuf *b) { b->p = NULL; b->len = 0; b->cap = 0; }

static void sbuf_reserve(sbuf *b, size_t extra) {
    if (b->len + extra <= b->cap) return;
    size_t ncap = b->cap ? b->cap * 2 : 256;
    while (ncap < b->len + extra) ncap *= 2;
    b->p = realloc(b->p, ncap);
    if (!b->p) diesys("realloc");
    b->cap = ncap;
}

static void sbuf_append(sbuf *b, const char *data, size_t n) {
    sbuf_reserve(b, n + 1);
    memcpy(b->p + b->len, data, n);
    b->len += n;
    b->p[b->len] = 0;
}

static void sbuf_appendz(sbuf *b, const char *s) { sbuf_append(b, s, strlen(s)); }

static void sbuf_appendf(sbuf *b, const char *fmt, ...) {
    char tmp[512];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if ((size_t)n < sizeof(tmp)) {
        sbuf_append(b, tmp, (size_t)n);
    } else {
        /* rare: fell back to a heap buffer for the overflow case */
        char *big = malloc((size_t)n + 1);
        if (!big) diesys("malloc");
        va_start(ap, fmt);
        vsnprintf(big, (size_t)n + 1, fmt, ap);
        va_end(ap);
        sbuf_append(b, big, (size_t)n);
        free(big);
    }
}

static void sbuf_free(sbuf *b) { free(b->p); b->p = NULL; b->len = b->cap = 0; }

/* JSON-escape `s` (n bytes) into `out` (control chars, quote, backslash;
 * raw UTF-8 bytes pass through untouched). */
static void json_escape_append(sbuf *out, const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '"':  sbuf_appendz(out, "\\\""); break;
        case '\\': sbuf_appendz(out, "\\\\"); break;
        case '\n': sbuf_appendz(out, "\\n"); break;
        case '\r': sbuf_appendz(out, "\\r"); break;
        case '\t': sbuf_appendz(out, "\\t"); break;
        case '\b': sbuf_appendz(out, "\\b"); break;
        case '\f': sbuf_appendz(out, "\\f"); break;
        default:
            if (c < 0x20) sbuf_appendf(out, "\\u%04x", c);
            else sbuf_append(out, (const char *)&c, 1);
        }
    }
}

/* ============================ tiny JSON reader ========================= */

typedef struct { const char *p, *end; bool err; } jcur;

static void j_skip_ws(jcur *c) {
    while (c->p < c->end && (*c->p == ' ' || *c->p == '\t' || *c->p == '\n' || *c->p == '\r')) c->p++;
}

static bool j_eof(jcur *c) { return c->p >= c->end; }

static bool j_expect(jcur *c, char ch) {
    j_skip_ws(c);
    if (j_eof(c) || *c->p != ch) { c->err = true; return false; }
    c->p++;
    return true;
}

static int hexval(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static void utf8_encode(sbuf *out, uint32_t cp) {
    char b[4];
    int n;
    if (cp < 0x80) { b[0] = (char)cp; n = 1; }
    else if (cp < 0x800) {
        b[0] = (char)(0xC0 | (cp >> 6));
        b[1] = (char)(0x80 | (cp & 0x3F));
        n = 2;
    } else if (cp < 0x10000) {
        b[0] = (char)(0xE0 | (cp >> 12));
        b[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        b[2] = (char)(0x80 | (cp & 0x3F));
        n = 3;
    } else {
        b[0] = (char)(0xF0 | (cp >> 18));
        b[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        b[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        b[3] = (char)(0x80 | (cp & 0x3F));
        n = 4;
    }
    sbuf_append(out, b, (size_t)n);
}

/* Parse a JSON string literal (cursor at the opening quote) into `out`,
 * decoding \" \\ \/ \n \r \t \b \f \uXXXX (incl. surrogate pairs) to raw
 * UTF-8 bytes. */
static bool j_parse_string(jcur *c, sbuf *out) {
    j_skip_ws(c);
    if (j_eof(c) || *c->p != '"') { c->err = true; return false; }
    c->p++;
    while (true) {
        if (j_eof(c)) { c->err = true; return false; }
        unsigned char ch = (unsigned char)*c->p;
        if (ch == '"') { c->p++; return true; }
        if (ch == '\\') {
            c->p++;
            if (j_eof(c)) { c->err = true; return false; }
            char e = *c->p++;
            switch (e) {
            case '"': sbuf_append(out, "\"", 1); break;
            case '\\': sbuf_append(out, "\\", 1); break;
            case '/': sbuf_append(out, "/", 1); break;
            case 'n': sbuf_append(out, "\n", 1); break;
            case 'r': sbuf_append(out, "\r", 1); break;
            case 't': sbuf_append(out, "\t", 1); break;
            case 'b': sbuf_append(out, "\b", 1); break;
            case 'f': sbuf_append(out, "\f", 1); break;
            case 'u': {
                if (c->end - c->p < 4) { c->err = true; return false; }
                int h0 = hexval(c->p[0]), h1 = hexval(c->p[1]), h2 = hexval(c->p[2]), h3 = hexval(c->p[3]);
                if (h0 < 0 || h1 < 0 || h2 < 0 || h3 < 0) { c->err = true; return false; }
                uint32_t cp = (uint32_t)(h0 << 12 | h1 << 8 | h2 << 4 | h3);
                c->p += 4;
                if (cp >= 0xD800 && cp <= 0xDBFF && c->end - c->p >= 6 &&
                    c->p[0] == '\\' && c->p[1] == 'u') {
                    int g0 = hexval(c->p[2]), g1 = hexval(c->p[3]), g2 = hexval(c->p[4]), g3 = hexval(c->p[5]);
                    if (g0 >= 0 && g1 >= 0 && g2 >= 0 && g3 >= 0) {
                        uint32_t lo = (uint32_t)(g0 << 12 | g1 << 8 | g2 << 4 | g3);
                        if (lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            c->p += 6;
                        }
                    }
                }
                utf8_encode(out, cp);
                break;
            }
            default: c->err = true; return false;
            }
        } else {
            c->p++;
            sbuf_append(out, (const char *)&ch, 1);
        }
    }
}

/* Skip any JSON value (used to ignore fields we don't care about). */
static bool j_skip_value(jcur *c) {
    j_skip_ws(c);
    if (j_eof(c)) { c->err = true; return false; }
    char ch = *c->p;
    if (ch == '"') {
        sbuf tmp; sbuf_init(&tmp);
        bool ok = j_parse_string(c, &tmp);
        sbuf_free(&tmp);
        return ok;
    }
    if (ch == '{') {
        c->p++;
        j_skip_ws(c);
        if (!j_eof(c) && *c->p == '}') { c->p++; return true; }
        while (true) {
            sbuf key; sbuf_init(&key);
            bool ok = j_parse_string(c, &key);
            sbuf_free(&key);
            if (!ok) return false;
            if (!j_expect(c, ':')) return false;
            if (!j_skip_value(c)) return false;
            j_skip_ws(c);
            if (j_eof(c)) { c->err = true; return false; }
            if (*c->p == ',') { c->p++; continue; }
            if (*c->p == '}') { c->p++; return true; }
            c->err = true; return false;
        }
    }
    if (ch == '[') {
        c->p++;
        j_skip_ws(c);
        if (!j_eof(c) && *c->p == ']') { c->p++; return true; }
        while (true) {
            if (!j_skip_value(c)) return false;
            j_skip_ws(c);
            if (j_eof(c)) { c->err = true; return false; }
            if (*c->p == ',') { c->p++; continue; }
            if (*c->p == ']') { c->p++; return true; }
            c->err = true; return false;
        }
    }
    if (ch == 't') { if (c->end - c->p >= 4 && !memcmp(c->p, "true", 4)) { c->p += 4; return true; } c->err = true; return false; }
    if (ch == 'f') { if (c->end - c->p >= 5 && !memcmp(c->p, "false", 5)) { c->p += 5; return true; } c->err = true; return false; }
    if (ch == 'n') { if (c->end - c->p >= 4 && !memcmp(c->p, "null", 4)) { c->p += 4; return true; } c->err = true; return false; }
    /* number */
    const char *start = c->p;
    if (*c->p == '-') c->p++;
    while (!j_eof(c) && (isdigit((unsigned char)*c->p) || *c->p == '.' || *c->p == 'e' || *c->p == 'E' || *c->p == '+' || *c->p == '-')) c->p++;
    if (c->p == start) { c->err = true; return false; }
    return true;
}

static bool j_parse_number(jcur *c, double *out) {
    j_skip_ws(c);
    if (j_eof(c)) { c->err = true; return false; }
    char *endp;
    /* strtod needs a NUL-terminated-ish region; our buffer is NUL
     * terminated by sbuf_append, and body is copied into a sbuf too. */
    double v = strtod(c->p, &endp);
    if (endp == c->p) { c->err = true; return false; }
    c->p = endp;
    *out = v;
    return true;
}

static bool j_parse_bool(jcur *c, bool *out) {
    j_skip_ws(c);
    if (c->end - c->p >= 4 && !memcmp(c->p, "true", 4)) { c->p += 4; *out = true; return true; }
    if (c->end - c->p >= 5 && !memcmp(c->p, "false", 5)) { c->p += 5; *out = false; return true; }
    c->err = true;
    return false;
}

/* ========================= chat request model ========================== */

typedef struct { char *role, *content; } chat_msg;
typedef struct { chat_msg *items; int len, cap; } chat_msgs;

static void msgs_push(chat_msgs *v, char *role, char *content) {
    if (v->len == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 8;
        v->items = realloc(v->items, (size_t)v->cap * sizeof(chat_msg));
        if (!v->items) diesys("realloc");
    }
    v->items[v->len].role = role;
    v->items[v->len].content = content;
    v->len++;
}

static void msgs_free(chat_msgs *v) {
    for (int i = 0; i < v->len; i++) { free(v->items[i].role); free(v->items[i].content); }
    free(v->items);
    v->items = NULL; v->len = v->cap = 0;
}

typedef struct {
    chat_msgs messages;
    long max_tokens;    /* -1 if unset */
    double temperature; /* NAN if unset */
    bool stream;
    bool stream_include_usage;
    bool have_max_tokens, have_temperature, have_stream, have_stream_include_usage;
    bool content_not_string; /* a message content was non-string (e.g. array) */
    bool parse_error;
} chat_request;

/* Parse one message object {"role":"...", "content":"..."}. Unknown keys
 * are ignored. If "content" is present but not a JSON string, sets
 * req->content_not_string and aborts parsing that value (still consumes
 * it via skip so the cursor stays in sync, then bails out). */
static bool parse_message(jcur *c, chat_request *req, char **role_out, char **content_out) {
    *role_out = NULL;
    *content_out = NULL;
    if (!j_expect(c, '{')) return false;
    j_skip_ws(c);
    if (!j_eof(c) && *c->p == '}') { c->p++; return true; }
    while (true) {
        sbuf key; sbuf_init(&key);
        if (!j_parse_string(c, &key)) { sbuf_free(&key); return false; }
        if (!j_expect(c, ':')) { sbuf_free(&key); return false; }
        if (strcmp(key.p, "role") == 0) {
            sbuf val; sbuf_init(&val);
            if (!j_parse_string(c, &val)) { sbuf_free(&key); sbuf_free(&val); return false; }
            free(*role_out);
            *role_out = strdup(val.p ? val.p : "");
            sbuf_free(&val);
        } else if (strcmp(key.p, "content") == 0) {
            j_skip_ws(c);
            if (!j_eof(c) && *c->p == '"') {
                sbuf val; sbuf_init(&val);
                if (!j_parse_string(c, &val)) { sbuf_free(&key); sbuf_free(&val); return false; }
                free(*content_out);
                *content_out = strdup(val.p ? val.p : "");
                sbuf_free(&val);
            } else {
                /* content is present but not a plain string (e.g. an
                 * array of content parts): reject per spec. */
                req->content_not_string = true;
                if (!j_skip_value(c)) { sbuf_free(&key); return false; }
            }
        } else {
            if (!j_skip_value(c)) { sbuf_free(&key); return false; }
        }
        sbuf_free(&key);
        j_skip_ws(c);
        if (j_eof(c)) { c->err = true; return false; }
        if (*c->p == ',') { c->p++; continue; }
        if (*c->p == '}') { c->p++; return true; }
        c->err = true;
        return false;
    }
}

/* Parse {"include_usage": bool} (the only stream_options field we honor). */
static bool parse_stream_options(jcur *c, bool *include_usage, bool *have) {
    if (!j_expect(c, '{')) return false;
    j_skip_ws(c);
    if (!j_eof(c) && *c->p == '}') { c->p++; return true; }
    while (true) {
        sbuf key; sbuf_init(&key);
        if (!j_parse_string(c, &key)) { sbuf_free(&key); return false; }
        if (!j_expect(c, ':')) { sbuf_free(&key); return false; }
        if (strcmp(key.p, "include_usage") == 0) {
            bool v;
            if (!j_parse_bool(c, &v)) { sbuf_free(&key); return false; }
            *include_usage = v;
            *have = true;
        } else {
            if (!j_skip_value(c)) { sbuf_free(&key); return false; }
        }
        sbuf_free(&key);
        j_skip_ws(c);
        if (j_eof(c)) { c->err = true; return false; }
        if (*c->p == ',') { c->p++; continue; }
        if (*c->p == '}') { c->p++; return true; }
        c->err = true; return false;
    }
}

/* Parse the top-level chat completion request body. */
static bool parse_chat_request(const char *body, size_t body_len, chat_request *req) {
    memset(req, 0, sizeof(*req));
    req->max_tokens = -1;
    req->temperature = NAN;
    jcur c = { body, body + body_len, false };
    if (!j_expect(&c, '{')) return false;
    j_skip_ws(&c);
    if (!j_eof(&c) && *c.p == '}') { c.p++; return true; }
    while (true) {
        sbuf key; sbuf_init(&key);
        if (!j_parse_string(&c, &key)) { sbuf_free(&key); return false; }
        if (!j_expect(&c, ':')) { sbuf_free(&key); return false; }
        if (strcmp(key.p, "messages") == 0) {
            j_skip_ws(&c);
            if (!j_expect(&c, '[')) { sbuf_free(&key); return false; }
            j_skip_ws(&c);
            if (!j_eof(&c) && *c.p == ']') {
                c.p++;
            } else {
                while (true) {
                    char *role, *content;
                    if (!parse_message(&c, req, &role, &content)) { sbuf_free(&key); return false; }
                    msgs_push(&req->messages, role, content ? content : strdup(""));
                    j_skip_ws(&c);
                    if (j_eof(&c)) { c.err = true; sbuf_free(&key); return false; }
                    if (*c.p == ',') { c.p++; continue; }
                    if (*c.p == ']') { c.p++; break; }
                    c.err = true; sbuf_free(&key); return false;
                }
            }
        } else if (strcmp(key.p, "max_tokens") == 0) {
            double v;
            if (!j_parse_number(&c, &v)) { sbuf_free(&key); return false; }
            req->max_tokens = (long)v;
            req->have_max_tokens = true;
        } else if (strcmp(key.p, "temperature") == 0) {
            double v;
            if (!j_parse_number(&c, &v)) { sbuf_free(&key); return false; }
            req->temperature = v;
            req->have_temperature = true;
        } else if (strcmp(key.p, "stream") == 0) {
            bool v;
            if (!j_parse_bool(&c, &v)) { sbuf_free(&key); return false; }
            req->stream = v;
            req->have_stream = true;
        } else if (strcmp(key.p, "stream_options") == 0) {
            j_skip_ws(&c);
            if (!j_eof(&c) && *c.p == 'n') {
                if (!j_skip_value(&c)) { sbuf_free(&key); return false; } /* null */
            } else if (!parse_stream_options(&c, &req->stream_include_usage, &req->have_stream_include_usage)) {
                sbuf_free(&key); return false;
            }
        } else {
            if (!j_skip_value(&c)) { sbuf_free(&key); return false; }
        }
        sbuf_free(&key);
        j_skip_ws(&c);
        if (j_eof(&c)) { c.err = true; return false; }
        if (*c.p == ',') { c.p++; continue; }
        if (*c.p == '}') { c.p++; break; }
        c.err = true; return false;
    }
    req->parse_error = c.err;
    return !c.err;
}

/* ============================== HTTP layer ============================= */

typedef struct {
    char method[16];
    char path[512];
    char auth[256];   /* raw Authorization header value, "" if absent */
    sbuf raw;         /* full accumulated request bytes (headers + body) */
    const char *body;
    size_t body_len;
    bool ok;
    int status_on_fail; /* HTTP status to use if !ok */
    const char *fail_msg;
} http_request;

/* Read a full HTTP/1.1 request off `fd`: headers up to \r\n\r\n (capped),
 * then the body per Content-Length (capped).  Fills *req.  On any
 * violation, sets req->ok=false with a status/message to send back. */
static void http_read_request(int fd, http_request *req) {
    memset(req, 0, sizeof(*req));
    sbuf_init(&req->raw);
    req->ok = false;
    req->status_on_fail = 400;
    req->fail_msg = "bad request";

    char chunk[4096];
    const char *hdr_end = NULL;
    while (true) {
        if (req->raw.len > MAX_HEADER_BYTES && !hdr_end) {
            req->fail_msg = "headers too large";
            return;
        }
        ssize_t n = recv(fd, chunk, sizeof(chunk), 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            req->fail_msg = "read error";
            return;
        }
        if (n == 0) {
            req->fail_msg = "connection closed before headers complete";
            return;
        }
        sbuf_append(&req->raw, chunk, (size_t)n);
        hdr_end = strstr(req->raw.p, "\r\n\r\n");
        if (hdr_end) break;
    }

    size_t hdr_len = (size_t)(hdr_end - req->raw.p) + 4;

    /* request line */
    const char *line_end = strstr(req->raw.p, "\r\n");
    if (!line_end || (size_t)(line_end - req->raw.p) >= sizeof(req->method) + sizeof(req->path)) {
        req->fail_msg = "bad request line";
        return;
    }
    char method[16] = {0}, path[512] = {0}, version[16] = {0};
    if (sscanf(req->raw.p, "%15s %511s %15s", method, path, version) != 3) {
        req->fail_msg = "bad request line";
        return;
    }
    snprintf(req->method, sizeof(req->method), "%s", method);
    snprintf(req->path, sizeof(req->path), "%s", path);

    /* Content-Length / Authorization headers (case-insensitive), if any */
    long content_length = 0;
    {
        const char *p = req->raw.p;
        const char *headers_stop = req->raw.p + hdr_len;
        while (p < headers_stop) {
            const char *eol = strstr(p, "\r\n");
            if (!eol || eol > headers_stop) break;
            if (eol == p) break; /* blank line: end of headers */
            if ((size_t)(eol - p) > 15 && strncasecmp(p, "Content-Length:", 15) == 0) {
                content_length = strtol(p + 15, NULL, 10);
            } else if ((size_t)(eol - p) > 14 && strncasecmp(p, "Authorization:", 14) == 0) {
                const char *v = p + 14;
                while (v < eol && (*v == ' ' || *v == '\t')) v++;
                size_t vlen = (size_t)(eol - v);
                if (vlen >= sizeof(req->auth)) vlen = sizeof(req->auth) - 1;
                memcpy(req->auth, v, vlen);
                req->auth[vlen] = '\0';
            }
            p = eol + 2;
        }
    }

    if (content_length < 0) { req->fail_msg = "bad content-length"; return; }
    if ((size_t)content_length > MAX_BODY_BYTES) {
        req->status_on_fail = 400;
        req->fail_msg = "request body too large";
        return;
    }

    size_t have_body = req->raw.len - hdr_len;
    while (have_body < (size_t)content_length) {
        ssize_t n = recv(fd, chunk, sizeof(chunk), 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            req->fail_msg = "read error";
            return;
        }
        if (n == 0) { req->fail_msg = "connection closed before body complete"; return; }
        sbuf_append(&req->raw, chunk, (size_t)n);
        have_body += (size_t)n;
    }

    req->body = req->raw.p + hdr_len;
    req->body_len = (size_t)content_length;
    req->ok = true;
}

static bool send_all(int fd, const char *data, size_t len) {
    size_t off = 0;
    while (off < len) {
        ssize_t n = send(fd, data + off, len - off, 0);
        if (n < 0) { if (errno == EINTR) continue; return false; }
        if (n == 0) return false;
        off += (size_t)n;
    }
    return true;
}

static void http_send_status(int fd, int status, const char *status_text,
                              const char *content_type, const char *body, size_t body_len) {
    sbuf out; sbuf_init(&out);
    sbuf_appendf(&out, "HTTP/1.1 %d %s\r\n", status, status_text);
    sbuf_appendf(&out, "Content-Type: %s\r\n", content_type);
    sbuf_appendf(&out, "Content-Length: %zu\r\n", body_len);
    sbuf_appendz(&out, "Connection: close\r\n\r\n");
    sbuf_append(&out, body, body_len);
    send_all(fd, out.p, out.len);
    sbuf_free(&out);
}

static void http_send_json(int fd, int status, const char *status_text, const sbuf *body) {
    http_send_status(fd, status, status_text, "application/json", body->p ? body->p : "", body->len);
}

/* Plain {"error":{"message":"..."}} -- used for raw HTTP/JSON protocol
 * violations that predate any endpoint-specific error taxonomy. */
static void http_send_error(int fd, int status, const char *status_text, const char *err_msg) {
    sbuf out; sbuf_init(&out);
    sbuf_appendz(&out, "{\"error\":{\"message\":\"");
    json_escape_append(&out, err_msg, strlen(err_msg));
    sbuf_appendz(&out, "\"}}");
    http_send_json(fd, status, status_text, &out);
    sbuf_free(&out);
}

/* {"error":{"type":"...","message":"..."}} -- used for the specific error
 * shapes the admin/model-management endpoints promise (overloaded,
 * admin_disabled, unauthorized, not_env_file_managed, bad_request,
 * unknown_model, not_loadable, env_write_failed). */
static void http_send_error_typed(int fd, int status, const char *status_text,
                                  const char *type, const char *msg) {
    sbuf out; sbuf_init(&out);
    sbuf_appendz(&out, "{\"error\":{\"type\":\"");
    json_escape_append(&out, type, strlen(type));
    sbuf_appendz(&out, "\",\"message\":\"");
    json_escape_append(&out, msg, strlen(msg));
    sbuf_appendz(&out, "\"}}");
    http_send_json(fd, status, status_text, &out);
    sbuf_free(&out);
}

/* ============================ engine wiring ============================ */

static ink_model g_model;
static int g_end_message_id = -1;
static uint32_t g_max_tokens_cap = 512;
static int g_sessions = 2;
static bool g_resident = false;
static const char *g_model_path;
static char g_model_realpath[PATH_MAX];
static const char *g_admin_token;   /* ACCRETION_ADMIN_TOKEN, NULL = select disabled */
static const char *g_env_file;      /* env file a select rewrites, NULL = hand-managed */
static const char *g_model_dirs;    /* DS4_MODEL_DIRS */
static long g_req_counter = 0;      /* worker-thread only: no lock needed */

/* xorshift64* PRNG for temperature sampling (worker-thread only). */
static uint64_t g_rng_state;

static uint64_t xorshift64star(void) {
    uint64_t x = g_rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    g_rng_state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static double rng_uniform(void) {
    return (double)(xorshift64star() >> 11) * (1.0 / 9007199254740992.0); /* [0,1) */
}

static int argmax_logits(const float *logits, uint32_t n) {
    int best = 0;
    float bestv = -INFINITY;
    for (uint32_t i = 0; i < n; i++) if (logits[i] > bestv) { bestv = logits[i]; best = (int)i; }
    return best;
}

static int sample_logits(const float *logits, uint32_t n, double temperature) {
    float maxv = -INFINITY;
    for (uint32_t i = 0; i < n; i++) if (logits[i] > maxv) maxv = logits[i];
    double sum = 0.0;
    float *probs = malloc(n * sizeof(float));
    if (!probs) diesys("malloc");
    for (uint32_t i = 0; i < n; i++) {
        double p = exp(((double)logits[i] - (double)maxv) / temperature);
        probs[i] = (float)p;
        sum += p;
    }
    double r = rng_uniform() * sum;
    double acc = 0.0;
    int chosen = (int)n - 1;
    for (uint32_t i = 0; i < n; i++) {
        acc += probs[i];
        if (r < acc) { chosen = (int)i; break; }
    }
    free(probs);
    return chosen;
}

static bool token_is_special_bracket(const char *s, int n) {
    return n >= 4 && s[0] == '<' && s[1] == '|' && s[n-2] == '|' && s[n-1] == '>';
}

/* Tolerant u64 KV read (u32/i32/u64/i64) -- ink_get_u32 truncates to 32
 * bits, which is not enough for inkling.context_length on some models. */
static bool read_kv_u64(const ink_gguf *g, const char *key, uint64_t *out) {
    const ink_kv *kv = ink_kv_find(g, key);
    if (!kv) return false;
    switch (kv->type) {
    case 4: { uint32_t v; memcpy(&v, kv->val, 4); *out = v; return true; }              /* u32 */
    case 5: { int32_t v; memcpy(&v, kv->val, 4); *out = (uint64_t)(int64_t)v; return true; } /* i32 */
    case 10: { uint64_t v; memcpy(&v, kv->val, 8); *out = v; return true; }             /* u64 */
    case 11: { int64_t v; memcpy(&v, kv->val, 8); *out = (uint64_t)v; return true; }    /* i64 */
    default: return false;
    }
}

static uint64_t model_param_count(const ink_model *m) {
    uint64_t total = 0;
    for (uint64_t i = 0; i < m->gg.n_tensors; i++) {
        const ink_tensor *t = &m->gg.tensors[i];
        uint64_t elems = 1;
        for (uint32_t d = 0; d < t->ndim; d++) elems *= t->dims[d];
        total += elems;
    }
    return total;
}

/* =========================== activity snapshot ========================= */

/* Lock-free per-slot activity, written only by the worker thread and read
 * without synchronization by GET /v1/activity (a beat-stale read is fine
 * and documented in the response, matching ds4-server's convention). */
typedef struct {
    volatile int state;          /* 0=idle 1=prefill 2=decode */
    volatile int prefill_done;
    volatile int prefill_total;
    volatile double prefill_tps;
    volatile int gen_tokens;
} act_slot;

static act_slot *g_slots;   /* [g_sessions] */
static bool *g_slot_busy;   /* [g_sessions]: guarded by g_mu */

/* ========================== FIFO work queue ============================ */

typedef struct job {
    int fd;
    int slot_id;
    int *tokens;      /* prompt token ids; owned by the connection thread */
    int n_tokens;
    long max_tokens;
    double temperature;
    bool stream;
    bool stream_include_usage;
    char id[40];             /* "chatcmpl-<n>", assigned by the worker */
    int prompt_tokens;
    int completion_tokens;
    const char *finish_reason;
    int status;               /* HTTP status, for the access log line */
    double t0;
    bool done;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    struct job *next;
} job;

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_work_cv = PTHREAD_COND_INITIALIZER;   /* worker waits here */
static pthread_cond_t g_drain_cv = PTHREAD_COND_INITIALIZER;  /* main waits for clients==0 */
static job *g_head = NULL, *g_tail = NULL;
static int g_clients = 0;         /* active connection ("network") threads */
static bool g_worker_stop = false;

static volatile sig_atomic_t g_stop = 0;
static volatile sig_atomic_t g_listen_fd = -1;
static volatile sig_atomic_t g_swap_requested = 0;

static void stop_signal_handler(int sig) {
    (void)sig;
    if (g_stop) _exit(130);
    g_stop = 1;
    if (g_listen_fd >= 0) {
        int fd = (int)g_listen_fd;
        g_listen_fd = -1;
        close(fd);
    }
}

typedef enum { ENQ_OK, ENQ_SHUTTING_DOWN, ENQ_OVERLOADED } enq_result;

/* Bounded (never blocks): queue depth is capped at g_sessions slots.  A
 * full queue is rejected immediately with 503, matching "slots bound queue
 * depth" -- there is no unbounded backlog. */
static enq_result try_enqueue(job *j) {
    pthread_mutex_lock(&g_mu);
    if (g_stop) { pthread_mutex_unlock(&g_mu); return ENQ_SHUTTING_DOWN; }
    int slot_id = -1;
    for (int i = 0; i < g_sessions; i++) if (!g_slot_busy[i]) { slot_id = i; break; }
    if (slot_id < 0) { pthread_mutex_unlock(&g_mu); return ENQ_OVERLOADED; }
    g_slot_busy[slot_id] = true;
    j->slot_id = slot_id;
    j->next = NULL;
    if (g_tail) g_tail->next = j; else g_head = j;
    g_tail = j;
    pthread_cond_signal(&g_work_cv);
    pthread_mutex_unlock(&g_mu);
    return ENQ_OK;
}

/* ============================== SSE frames ============================= */

static bool sse_send_headers(int fd) {
    static const char hdr[] =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n\r\n";
    return send_all(fd, hdr, sizeof(hdr) - 1);
}

static bool sse_first_frame(int fd, const char *id, long now) {
    sbuf b; sbuf_init(&b);
    sbuf_appendf(&b,
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,"
        "\"model\":\"inkling-small\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},"
        "\"finish_reason\":null}]}\n\n", id, now);
    bool ok = send_all(fd, b.p, b.len);
    sbuf_free(&b);
    return ok;
}

static bool sse_content_frame(int fd, const char *id, long now, const char *text, size_t n) {
    sbuf b; sbuf_init(&b);
    sbuf_appendf(&b,
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,"
        "\"model\":\"inkling-small\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"",
        id, now);
    json_escape_append(&b, text, n);
    sbuf_appendz(&b, "\"},\"finish_reason\":null}]}\n\n");
    bool ok = send_all(fd, b.p, b.len);
    sbuf_free(&b);
    return ok;
}

static bool sse_finish_frame(int fd, const char *id, long now, const char *finish_reason) {
    sbuf b; sbuf_init(&b);
    sbuf_appendf(&b,
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,"
        "\"model\":\"inkling-small\",\"choices\":[{\"index\":0,\"delta\":{},"
        "\"finish_reason\":\"%s\"}]}\n\n", id, now, finish_reason);
    bool ok = send_all(fd, b.p, b.len);
    sbuf_free(&b);
    return ok;
}

static bool sse_usage_frame(int fd, const char *id, long now, int prompt_tokens, int completion_tokens) {
    sbuf b; sbuf_init(&b);
    sbuf_appendf(&b,
        "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,"
        "\"model\":\"inkling-small\",\"choices\":[],\"usage\":{\"prompt_tokens\":%d,"
        "\"completion_tokens\":%d,\"total_tokens\":%d}}\n\n",
        id, now, prompt_tokens, completion_tokens, prompt_tokens + completion_tokens);
    bool ok = send_all(fd, b.p, b.len);
    sbuf_free(&b);
    return ok;
}

/* "event: error\ndata: {...}\n\n" -- defined per spec for completeness.
 * Nothing in the current inkling decode path returns a recoverable
 * mid-stream error (corruption is a hard ink_logits_guard() die(), per
 * requirement #9): kept available for the one defensive call site in
 * run_job() and for future engine error returns. */
static bool sse_error_event(int fd, const char *msg) {
    sbuf b; sbuf_init(&b);
    sbuf_appendz(&b, "event: error\ndata: {\"error\":{\"message\":\"");
    json_escape_append(&b, msg, strlen(msg));
    sbuf_appendz(&b, "\",\"type\":\"server_error\"}}\n\n");
    bool ok = send_all(fd, b.p, b.len);
    sbuf_free(&b);
    return ok;
}

/* ============================ request handlers ========================= */

static void handle_capabilities(int fd) {
    ink_str name_s;
    char name[256];
    if (ink_get_str(&g_model.gg, "general.name", &name_s) && name_s.len > 0) {
        size_t n = name_s.len < sizeof(name) - 1 ? name_s.len : sizeof(name) - 1;
        memcpy(name, name_s.ptr, n);
        name[n] = '\0';
    } else {
        snprintf(name, sizeof(name), "inkling-small");
    }
    uint64_t trained;
    if (!read_kv_u64(&g_model.gg, "inkling.context_length", &trained)) trained = 1048576;
    uint64_t params = model_param_count(&g_model);

    sbuf out; sbuf_init(&out);
    sbuf_appendz(&out, "{\"schema_version\":1,\"model\":{\"name\":\"");
    json_escape_append(&out, name, strlen(name));
    sbuf_appendf(&out,
        "\",\"architecture\":\"inkling\",\"parameters\":%llu,\"file_bytes\":%llu},"
        "\"context\":{\"configured\":%u,\"trained\":%llu},"
        "\"serving\":{\"resident\":%s,\"sessions\":%d,\"backend\":\"%s\"},"
        "\"apis\":{\"openai_chat_completions\":true,\"sse_streaming\":true}",
        (unsigned long long)params, (unsigned long long)g_model.gg.map_len,
        g_model.n_ctx, (unsigned long long)trained,
        g_resident ? "true" : "false", g_sessions, backend_name());
    const char *tp = getenv("DS4_CAPS_THROUGHPUT_JSON");
    if (tp && tp[0] == '{') {
        sbuf_appendz(&out, ",\"throughput\":");
        sbuf_appendz(&out, tp);
    }
    sbuf_appendz(&out, "}");
    http_send_json(fd, 200, "OK", &out);
    sbuf_free(&out);
    fprintf(stderr, "GET /v1/capabilities 200 - - -\n");
}

static void handle_activity(int fd) {
    sbuf out; sbuf_init(&out);
    sbuf_appendz(&out, "{\"schema_version\":1,\"note\":\"lock-free snapshot; "
                       "values may be slightly stale\",\"slots\":[");
    for (int i = 0; i < g_sessions; i++) {
        act_slot *s = &g_slots[i];
        int state = s->state;
        if (i) sbuf_appendz(&out, ",");
        sbuf_appendf(&out, "{\"id\":%d,\"state\":\"%s\"", i,
                    state == 1 ? "prefill" : state == 2 ? "decode" : "idle");
        if (state == 1) {
            int done = s->prefill_done, total = s->prefill_total;
            double tps = s->prefill_tps;
            sbuf_appendf(&out,
                ",\"prefill\":{\"tokens_done\":%d,\"tokens_total\":%d,\"tokens_per_second\":%.2f",
                done, total, tps);
            if (tps > 0.0 && total > done) {
                sbuf_appendf(&out, ",\"eta_seconds\":%.1f", (double)(total - done) / tps);
            }
            sbuf_appendz(&out, "}");
        } else if (state == 2) {
            sbuf_appendf(&out, ",\"generated_tokens\":%d", s->gen_tokens);
        }
        sbuf_appendz(&out, "}");
    }
    sbuf_appendz(&out, "]}");
    http_send_json(fd, 200, "OK", &out);
    sbuf_free(&out);
    fprintf(stderr, "GET /v1/activity 200 - - -\n");
}

static void handle_models(int fd) {
    sbuf out; sbuf_init(&out);
    sbuf_appendz(&out,
        "{\"object\":\"list\",\"data\":[{\"id\":\"inkling-small\",\"object\":\"model\","
        "\"created\":0,\"owned_by\":\"ds4-inkling\"}]}");
    http_send_json(fd, 200, "OK", &out);
    sbuf_free(&out);
    fprintf(stderr, "GET /v1/models 200 - - -\n");
}

/* ===================== model management (available/select) ============= */

typedef struct {
    char path[PATH_MAX];
    char name[256];
    uint64_t size;
    char architecture[64];
    bool active;
    bool has_sidecar;
    const char *loadable;   /* "yes" | "no" | "unknown" */
    const char *mode;       /* "interactive" | "batch" | "unknown" */
    char mode_reason[160];
    char decode_tps_reference[24]; /* numeric string, "" if none */
} avail_model;

#define AVAIL_MODELS_MAX 256
#define AVAIL_DIRS_MAX 16
#define GGUF_PEEK_MAX_BYTES (4u * 1024u * 1024u)

/* Per-model sidecar env convention (same as ds4-server): "<model>.gguf.env"
 * next to the gguf, KEY=VALUE lines (no quoting, no export). */
static bool sidecar_path_for(const char *gguf, char *out, size_t out_sz) {
    return snprintf(out, out_sz, "%s.env", gguf) < (int)out_sz;
}

static bool sidecar_get(const char *sidecar, const char *key, char *out, size_t out_sz) {
    FILE *f = fopen(sidecar, "r");
    if (!f) return false;
    const size_t kl = strlen(key);
    char line[2048];
    bool found = false;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, key, kl) != 0 || line[kl] != '=') continue;
        size_t vl = strlen(line + kl + 1);
        while (vl > 0 && (line[kl + 1 + vl - 1] == '\n' || line[kl + 1 + vl - 1] == '\r')) vl--;
        if (vl >= out_sz) vl = out_sz - 1;
        memcpy(out, line + kl + 1, vl);
        out[vl] = '\0';
        found = true; /* last occurrence wins, like the shell */
    }
    fclose(f);
    return found;
}

static const char *model_loadable(const char *arch) {
    if (!arch[0]) return "unknown";
    if (!strcmp(arch, "inkling")) return "yes";
    if (!strcmp(arch, "deepseek4") || !strcmp(arch, "glm-dsa"))
        return getenv("ACCRETION_ARCH_WRAPPER") ? "yes" : "no";
    return "no";
}

static void model_mode_classify(avail_model *m) {
    m->mode = "unknown";
    m->mode_reason[0] = '\0';
    m->decode_tps_reference[0] = '\0';
    char sc[PATH_MAX + 8];
    if (!sidecar_path_for(m->path, sc, sizeof(sc))) return;
    m->has_sidecar = access(sc, R_OK) == 0;
    if (!m->has_sidecar) {
        snprintf(m->mode_reason, sizeof(m->mode_reason),
                "no sidecar (.gguf.env absent): serving profile unknown");
        return;
    }
    char v[1024];
    const bool have_flags = sidecar_get(sc, "DS4_EXTRA_FLAGS", v, sizeof(v));
    const bool streamed = have_flags && strstr(v, "--ssd-streaming") != NULL;
    if (streamed) {
        m->mode = "batch";
        snprintf(m->mode_reason, sizeof(m->mode_reason),
                "sidecar flags use --ssd-streaming: experts stream from disk "
                "(big-model capability, slower decode)");
    } else {
        m->mode = "interactive";
        snprintf(m->mode_reason, sizeof(m->mode_reason),
                "sidecar flags are resident (no --ssd-streaming): weights fit "
                "in memory, fast decode");
    }
    if (sidecar_get(sc, "DS4_DECODE_TPS_REFERENCE", v, sizeof(v))) {
        char *end = NULL;
        const double tps = strtod(v, &end);
        if (end && end != v && *end == '\0' && tps > 0) {
            snprintf(m->decode_tps_reference, sizeof(m->decode_tps_reference), "%.20s", v);
        }
    }
}

/* Bounded (<=4MB) GGUF header peek for general.architecture only -- quant
 * and manifest presence are intentionally not reported here (acceptable
 * reduced subset vs. ds4-server's richer peek; noted in the ticket). */
static bool peek_read(FILE *f, void *out, size_t n, long *budget) {
    if (*budget < (long)n) return false;
    if (fread(out, 1, n, f) != n) return false;
    *budget -= (long)n;
    return true;
}

static bool peek_seek(FILE *f, long n, long *budget) {
    if (*budget < n) return false;
    if (fseek(f, n, SEEK_CUR) != 0) return false;
    *budget -= n;
    return true;
}

static bool gguf_peek_arch(const char *path, char *arch, size_t arch_sz) {
    arch[0] = '\0';
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    bool ok = false;
    long budget = (long)GGUF_PEEK_MAX_BYTES;
    uint32_t magic = 0, version = 0;
    uint64_t n_tensors = 0, n_kv = 0;
    if (!peek_read(f, &magic, 4, &budget) || magic != 0x46554747u) goto out;
    if (!peek_read(f, &version, 4, &budget) || version < 2 || version > 3) goto out;
    if (!peek_read(f, &n_tensors, 8, &budget) || !peek_read(f, &n_kv, 8, &budget)) goto out;
    if (n_kv > 4096) goto out;
    for (uint64_t i = 0; i < n_kv && !ok && budget > 0; i++) {
        uint64_t klen = 0;
        if (!peek_read(f, &klen, 8, &budget) || klen > 1024) goto out;
        char key[1025];
        if (!peek_read(f, key, (size_t)klen, &budget)) goto out;
        key[klen] = '\0';
        uint32_t type = 0;
        if (!peek_read(f, &type, 4, &budget)) goto out;
        static const int fixed_sz[] = {1,1,2,2,4,4,4,1,-1,-2,8,8,8};
        if (type == 8) { /* string */
            uint64_t slen = 0;
            if (!peek_read(f, &slen, 8, &budget) || slen > (64u << 20)) goto out;
            if (!strcmp(key, "general.architecture") && slen < arch_sz) {
                if (!peek_read(f, arch, (size_t)slen, &budget)) goto out;
                arch[slen] = '\0';
                ok = true;
            } else if (!peek_seek(f, (long)slen, &budget)) goto out;
        } else if (type == 9) { /* array */
            uint32_t etype = 0;
            uint64_t count = 0;
            if (!peek_read(f, &etype, 4, &budget) || !peek_read(f, &count, 8, &budget)) goto out;
            if (etype == 8) {
                for (uint64_t j = 0; j < count; j++) {
                    uint64_t slen = 0;
                    if (!peek_read(f, &slen, 8, &budget) || slen > (64u << 20)) goto out;
                    if (!peek_seek(f, (long)slen, &budget)) goto out;
                }
            } else if (etype <= 12 && etype != 9 && fixed_sz[etype] > 0) {
                if (!peek_seek(f, (long)fixed_sz[etype] * (long)count, &budget)) goto out;
            } else goto out;
        } else if (type <= 12 && fixed_sz[type] > 0) {
            uint8_t vbuf[8];
            if (!peek_read(f, vbuf, (size_t)fixed_sz[type], &budget)) goto out;
        } else goto out;
    }
out:
    fclose(f);
    return ok;
}

static int scan_dir_for_models(const char *dir, avail_model *out, int len) {
    DIR *d = opendir(dir);
    if (!d) return len;
    struct dirent *de;
    while ((de = readdir(d)) && len < AVAIL_MODELS_MAX) {
        const size_t nl = strlen(de->d_name);
        if (nl < 6 || strcmp(de->d_name + nl - 5, ".gguf") != 0) continue;
        char path[PATH_MAX];
        if (snprintf(path, sizeof(path), "%s/%s", dir, de->d_name) >= (int)sizeof(path)) continue;
        struct stat st;
        if (stat(path, &st) != 0 || !S_ISREG(st.st_mode)) continue;
        char rp[PATH_MAX];
        const char *canon = realpath(path, rp) ? rp : path;
        bool dup = false;
        for (int i = 0; i < len; i++) if (!strcmp(out[i].path, canon)) { dup = true; break; }
        if (dup) continue;
        avail_model *m = &out[len++];
        memset(m, 0, sizeof(*m));
        snprintf(m->path, sizeof(m->path), "%s", canon);
        snprintf(m->name, sizeof(m->name), "%s", de->d_name);
        m->size = (uint64_t)st.st_size;
        gguf_peek_arch(canon, m->architecture, sizeof(m->architecture));
        m->loadable = model_loadable(m->architecture);
        m->active = g_model_realpath[0] && !strcmp(canon, g_model_realpath);
        model_mode_classify(m);
    }
    closedir(d);
    return len;
}

/* Scan the active model's directory + DS4_MODEL_DIRS.  Returns model count. */
static int scan_available_models(avail_model *out) {
    char dirs[AVAIL_DIRS_MAX][PATH_MAX];
    int ndirs = 0;
    if (g_model_realpath[0]) {
        snprintf(dirs[ndirs], PATH_MAX, "%s", g_model_realpath);
        char *slash = strrchr(dirs[ndirs], '/');
        if (slash && slash != dirs[ndirs]) *slash = '\0';
        else snprintf(dirs[ndirs], PATH_MAX, "/");
        ndirs++;
    }
    if (g_model_dirs && g_model_dirs[0]) {
        const char *p = g_model_dirs;
        while (*p && ndirs < AVAIL_DIRS_MAX) {
            const char *colon = strchr(p, ':');
            size_t dl = colon ? (size_t)(colon - p) : strlen(p);
            if (dl > 0 && dl < PATH_MAX) {
                memcpy(dirs[ndirs], p, dl);
                dirs[ndirs][dl] = '\0';
                bool dup = false;
                for (int i = 0; i < ndirs; i++) if (!strcmp(dirs[i], dirs[ndirs])) { dup = true; break; }
                if (!dup) ndirs++;
            }
            if (!colon) break;
            p = colon + 1;
        }
    }
    int len = 0;
    for (int i = 0; i < ndirs; i++) len = scan_dir_for_models(dirs[i], out, len);
    return len;
}

static void handle_models_available(int fd) {
    static avail_model models[AVAIL_MODELS_MAX]; /* large; not stack-safe */
    static pthread_mutex_t scan_mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&scan_mu);
    const int n = scan_available_models(models);
    const char *active = g_model_realpath[0] ? g_model_realpath : (g_model_path ? g_model_path : "");

    sbuf out; sbuf_init(&out);
    sbuf_appendz(&out, "{\"schema_version\":1,\"active_path\":\"");
    json_escape_append(&out, active, strlen(active));
    sbuf_appendf(&out, "\",\"select_enabled\":%s,\"models\":[",
                (g_admin_token && g_env_file) ? "true" : "false");
    for (int i = 0; i < n; i++) {
        const avail_model *m = &models[i];
        if (i) sbuf_appendz(&out, ",");
        sbuf_appendz(&out, "{\"name\":\"");
        json_escape_append(&out, m->name, strlen(m->name));
        sbuf_appendz(&out, "\",\"path\":\"");
        json_escape_append(&out, m->path, strlen(m->path));
        sbuf_appendf(&out, "\",\"size_bytes\":%llu,\"architecture\":\"",
                    (unsigned long long)m->size);
        const char *arch = m->architecture[0] ? m->architecture : "unknown";
        json_escape_append(&out, arch, strlen(arch));
        sbuf_appendf(&out,
            "\",\"active\":%s,\"has_sidecar\":%s,\"loadable\":\"%s\","
            "\"mode\":\"%s\",\"mode_reason\":\"",
            m->active ? "true" : "false", m->has_sidecar ? "true" : "false",
            m->loadable, m->mode);
        json_escape_append(&out, m->mode_reason, strlen(m->mode_reason));
        sbuf_appendz(&out, "\"");
        if (m->decode_tps_reference[0]) {
            sbuf_appendf(&out, ",\"decode_tps_reference\":%s", m->decode_tps_reference);
        }
        sbuf_appendz(&out, "}");
    }
    sbuf_appendz(&out, "]}");
    pthread_mutex_unlock(&scan_mu);
    http_send_json(fd, 200, "OK", &out);
    sbuf_free(&out);
    fprintf(stderr, "GET /v1/models/available 200 - - -\n");
}

/* Minimal "path" extraction from the select body.  Paths are validated
 * against the scanned list afterwards, so only \" \\ \/ escapes matter. */
static bool select_body_path(const char *body, size_t body_len, char *out, size_t out_sz) {
    if (!body || body_len == 0) return false;
    const char *k = strstr(body, "\"path\"");
    if (!k || (size_t)(k - body) >= body_len) return false;
    k = strchr(k + 6, ':');
    if (!k) return false;
    k++;
    while (*k && isspace((unsigned char)*k)) k++;
    if (*k != '"') return false;
    k++;
    size_t o = 0;
    while (*k && *k != '"' && o + 1 < out_sz) {
        if (*k == '\\' && (k[1] == '"' || k[1] == '\\' || k[1] == '/')) k++;
        out[o++] = *k++;
    }
    if (*k != '"') return false;
    out[o] = '\0';
    return o > 0;
}

/* Rewrite (or append) KEY= lines in the env file; tmp + rename.  Keys with
 * a NULL value are left untouched. */
static bool env_file_set_kvs(const char *env_file, const char *const *keys,
                             const char *const *vals, int nkv,
                             char *err, size_t err_sz) {
    FILE *f = fopen(env_file, "r");
    if (!f) {
        snprintf(err, err_sz, "cannot read %s: %s", env_file, strerror(errno));
        return false;
    }
    sbuf nb; sbuf_init(&nb);
    char line[4096];
    bool replaced[8] = {false};
    while (fgets(line, sizeof(line), f)) {
        int hit = -1;
        for (int i = 0; i < nkv; i++) {
            const size_t kl = strlen(keys[i]);
            if (vals[i] && !strncmp(line, keys[i], kl) && line[kl] == '=') { hit = i; break; }
        }
        if (hit >= 0) {
            sbuf_appendf(&nb, "%s=%s\n", keys[hit], vals[hit]);
            replaced[hit] = true;
        } else {
            sbuf_appendz(&nb, line);
        }
    }
    fclose(f);
    for (int i = 0; i < nkv; i++) {
        if (!vals[i] || replaced[i]) continue;
        if (nb.len && nb.p[nb.len - 1] != '\n') sbuf_appendz(&nb, "\n");
        sbuf_appendf(&nb, "%s=%s\n", keys[i], vals[i]);
    }
    char tmp[PATH_MAX + 8];
    snprintf(tmp, sizeof(tmp), "%s.tmp", env_file);
    FILE *o = fopen(tmp, "w");
    if (!o) {
        snprintf(err, err_sz, "cannot write temp env file: %s", strerror(errno));
        sbuf_free(&nb);
        return false;
    }
    bool ok = fwrite(nb.p, 1, nb.len, o) == nb.len;
    ok = (fclose(o) == 0) && ok;
    sbuf_free(&nb);
    if (!ok || rename(tmp, env_file) != 0) {
        snprintf(err, err_sz, "failed to update %s: %s", env_file, strerror(errno));
        unlink(tmp);
        return false;
    }
    return true;
}

static void handle_models_select(int fd, const http_request *hr) {
    if (!g_admin_token) {
        http_send_error_typed(fd, 405, "Method Not Allowed", "admin_disabled",
            "model switching is disabled: set ACCRETION_ADMIN_TOKEN in the "
            "server environment and restart to enable POST /v1/models/select");
        fprintf(stderr, "POST /v1/models/select 405 - - -\n");
        return;
    }
    const char *tok = hr->auth;
    if (!strncasecmp(tok, "Bearer ", 7)) tok += 7;
    if (strcmp(tok, g_admin_token) != 0) {
        http_send_error_typed(fd, 401, "Unauthorized", "unauthorized",
            "missing or wrong admin token (Authorization: Bearer <token>)");
        fprintf(stderr, "POST /v1/models/select 401 - - -\n");
        return;
    }
    if (!g_env_file) {
        http_send_error_typed(fd, 409, "Conflict", "not_env_file_managed",
            "this ds4-inkling-server install is not env-file managed (no "
            "DS4_ENV_FILE and no /opt/accretion/etc/ds4-server.env); switch "
            "models by hand");
        fprintf(stderr, "POST /v1/models/select 409 - - -\n");
        return;
    }
    char want[PATH_MAX];
    if (!select_body_path(hr->body, hr->body_len, want, sizeof(want))) {
        http_send_error_typed(fd, 400, "Bad Request", "bad_request",
            "body must be {\"path\": \"/abs/path/model.gguf\"}");
        fprintf(stderr, "POST /v1/models/select 400 - - -\n");
        return;
    }
    char wrp[PATH_MAX];
    const char *canon = realpath(want, wrp) ? wrp : want;

    static avail_model models[AVAIL_MODELS_MAX];
    static pthread_mutex_t sel_mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&sel_mu);
    const int n = scan_available_models(models);
    const avail_model *hit = NULL;
    for (int i = 0; i < n; i++) if (!strcmp(models[i].path, canon)) { hit = &models[i]; break; }
    if (!hit) {
        pthread_mutex_unlock(&sel_mu);
        http_send_error_typed(fd, 404, "Not Found", "unknown_model",
            "path is not in the scanned model list (GET /v1/models/available)");
        fprintf(stderr, "POST /v1/models/select 404 - - -\n");
        return;
    }
    if (hit->active) {
        pthread_mutex_unlock(&sel_mu);
        static const char body[] = "{\"status\":\"already_active\"}\n";
        http_send_status(fd, 200, "OK", "application/json", body, sizeof(body) - 1);
        fprintf(stderr, "POST /v1/models/select 200 - - -\n");
        return;
    }
    if (!strcmp(hit->loadable, "no")) {
        pthread_mutex_unlock(&sel_mu);
        http_send_error_typed(fd, 409, "Conflict", "not_loadable",
            "this binary cannot load that model's architecture");
        fprintf(stderr, "POST /v1/models/select 409 - - -\n");
        return;
    }

    /* Compose the new unit environment: DS4_MODEL + DS4_ARCH always; if the
     * target model has a sidecar (<gguf>.env), its launch keys override the
     * base env file so the restarted server (ds4-server or ds4-inkling-server,
     * per DS4_ARCH) runs that model's proven flags. */
    char sc[PATH_MAX + 8];
    char sc_ctx[64] = "", sc_cache[64] = "", sc_flags[1024] = "";
    const char *ctx_val = NULL, *cache_val = NULL, *flags_val = NULL;
    if (sidecar_path_for(hit->path, sc, sizeof(sc)) && access(sc, R_OK) == 0) {
        if (sidecar_get(sc, "DS4_CTX", sc_ctx, sizeof(sc_ctx)) && sc_ctx[0]) ctx_val = sc_ctx;
        cache_val = sidecar_get(sc, "DS4_CACHE_BUDGET", sc_cache, sizeof(sc_cache)) ? sc_cache : "";
        flags_val = sidecar_get(sc, "DS4_EXTRA_FLAGS", sc_flags, sizeof(sc_flags)) ? sc_flags : "";
    }
    const char *arch_val = !strcmp(hit->architecture, "inkling") ? "inkling" : "deepseek4";
    const char *keys[5] = {"DS4_MODEL", "DS4_CTX", "DS4_CACHE_BUDGET", "DS4_EXTRA_FLAGS", "DS4_ARCH"};
    const char *vals[5] = {hit->path, ctx_val, cache_val, flags_val, arch_val};
    char err[512];
    if (!env_file_set_kvs(g_env_file, keys, vals, 5, err, sizeof(err))) {
        pthread_mutex_unlock(&sel_mu);
        http_send_error_typed(fd, 500, "Internal Server Error", "env_write_failed", err);
        fprintf(stderr, "POST /v1/models/select 500 - - -\n");
        return;
    }

    sbuf out; sbuf_init(&out);
    sbuf_appendz(&out, "{\"status\":\"swapping\",\"path\":\"");
    json_escape_append(&out, hit->path, strlen(hit->path));
    sbuf_appendz(&out, "\",\"note\":\"draining sessions, then restarting on the "
                       "new model; poll /v1/capabilities\"}");
    http_send_json(fd, 200, "OK", &out);
    sbuf_free(&out);
    fprintf(stderr,
            "ds4-inkling-server: model select accepted path=%s (was %s); draining for swap restart\n",
            hit->path, g_model_realpath);
    fprintf(stderr, "POST /v1/models/select 200 - - -\n");
    pthread_mutex_unlock(&sel_mu);

    /* Trigger the existing graceful-shutdown drain: stop accepting, finish
     * in-flight/queued work, then main() exits with the swap code. */
    g_swap_requested = 1;
    if (!g_stop) kill(getpid(), SIGTERM);
}

/* ========================= chat completions (job) ======================= */

/* Runs entirely on the single worker thread: the only code in this file
 * that touches g_model's mutable engine state (KV cache, shortconv state).
 * Writes the full HTTP/SSE response directly to j->fd. */
static void run_job(job *j) {
    act_slot *slot = &g_slots[j->slot_id];
    slot->prefill_done = 0;
    slot->prefill_total = j->n_tokens;
    slot->prefill_tps = 0.0;
    slot->gen_tokens = 0;
    slot->state = 1; /* prefill */

    long id_num = ++g_req_counter;
    snprintf(j->id, sizeof(j->id), "chatcmpl-%ld", id_num);
    long now = (long)time(NULL);

    /* Defensive re-check: the connection thread already validated this
     * before enqueueing (ctx never changes at runtime), so this should be
     * unreachable; kept as the one genuine call site for the mid-stream
     * SSE error frame the spec requires the wire format for. */
    if ((long)j->n_tokens + j->max_tokens > (long)g_model.n_ctx) {
        slot->state = 0;
        j->prompt_tokens = j->n_tokens;
        j->completion_tokens = 0;
        j->finish_reason = "error";
        if (j->stream) {
            sse_send_headers(j->fd);
            sse_error_event(j->fd, "context_overflow: prompt + max_tokens exceeds context length");
            j->status = 200;
        } else {
            http_send_error(j->fd, 400, "Bad Request",
                            "context_overflow: prompt + max_tokens exceeds context length");
            j->status = 400;
        }
        return;
    }

    ink_state_reset(&g_model);
    float *logits = malloc((size_t)g_model.n_vocab * sizeof(float));
    if (!logits) diesys("malloc");

    double prefill_t0 = ink_now_sec();
    const int chunk = engine_prefill_chunk();
    for (int i = 0; i < j->n_tokens; i += chunk) {
        int n = j->n_tokens - i < chunk ? j->n_tokens - i : chunk;
        bool last = (i + n == j->n_tokens);
        engine_forward_batch(&g_model, j->tokens + i, (uint32_t)n, (uint32_t)i, last ? logits : NULL);
        slot->prefill_done = i + n;
        double elapsed = ink_now_sec() - prefill_t0;
        slot->prefill_tps = elapsed > 0.0 ? (double)slot->prefill_done / elapsed : 0.0;
    }

    slot->state = 2; /* decode */

    /* Corruption check BEFORE any bytes go out (streaming headers included),
     * so the common case fails the request cleanly instead of the process. */
    if (!ink_logits_ok(logits, g_model.n_vocab, g_model.n_vocab_unpadded,
                       "server prefill")) {
        free(logits);
        slot->state = 0;
        http_send_error(j->fd, 500, "Internal Server Error",
                        "logits corruption detected: refusing to sample "
                        "(see server log; engine state has been reset)");
        j->status = 500;
        return;
    }

    bool ok = true;
    bool corrupt = false;
    if (j->stream) {
        ok = sse_send_headers(j->fd);
        if (ok) ok = sse_first_frame(j->fd, j->id, now);
    }

    sbuf text; sbuf_init(&text); /* accumulated only for the non-stream path */
    int completion_tokens = 0;
    const char *finish_reason = "length";
    uint32_t pos = (uint32_t)j->n_tokens;

    for (long t = 0; ok && t < j->max_tokens; t++) {
        /* never emit token 0 off a NaN-poisoned vector: stop this request */
        if (!ink_logits_ok(logits, g_model.n_vocab, g_model.n_vocab_unpadded,
                           "server decode")) { corrupt = true; break; }
        int tok = (j->temperature <= 0.0) ? argmax_logits(logits, g_model.n_vocab)
                                            : sample_logits(logits, g_model.n_vocab, j->temperature);
        bool is_stop = (tok == g_model.tk.eos) || (g_end_message_id >= 0 && tok == g_end_message_id);
        if (!is_stop) completion_tokens++;   /* usage counts SAMPLED tokens */
        slot->gen_tokens = completion_tokens;
        if (!is_stop) {
            char piece[512];
            int n = ink_detokenize(&g_model.tk, tok, piece, sizeof(piece));
            if (!token_is_special_bracket(piece, n)) {
                if (j->stream) {
                    ok = sse_content_frame(j->fd, j->id, now, piece, (size_t)n);
                } else {
                    sbuf_append(&text, piece, (size_t)n);
                }
            }
        }
        if (is_stop) { finish_reason = "stop"; break; }
        if (t + 1 == j->max_tokens) { finish_reason = "length"; break; }
        engine_forward_batch(&g_model, &tok, 1, pos++, logits);
    }
    free(logits);
    slot->state = 0;

    int prompt_tokens = j->n_tokens;
    j->prompt_tokens = prompt_tokens;
    j->completion_tokens = completion_tokens;
    j->finish_reason = finish_reason;

    if (corrupt) {
        /* Mid-generation corruption: streaming already sent frames, so the
         * only honest close is an SSE error event; non-streaming can still
         * return a real 500. */
        if (j->stream) {
            sse_error_event(j->fd, "logits corruption detected mid-generation: "
                                   "generation aborted");
            send_all(j->fd, "data: [DONE]\n\n", 14);
            j->status = 200;
        } else {
            http_send_error(j->fd, 500, "Internal Server Error",
                            "logits corruption detected mid-generation: "
                            "generation aborted");
            j->status = 500;
        }
        sbuf_free(&text);
        return;
    }

    if (j->stream) {
        if (ok) ok = sse_finish_frame(j->fd, j->id, now, finish_reason);
        if (ok && j->stream_include_usage) ok = sse_usage_frame(j->fd, j->id, now, prompt_tokens, completion_tokens);
        if (ok) send_all(j->fd, "data: [DONE]\n\n", 14);
        j->status = 200;
    } else {
        sbuf out; sbuf_init(&out);
        sbuf_appendf(&out,
            "{\"id\":\"%s\",\"object\":\"chat.completion\",\"created\":%ld,"
            "\"model\":\"inkling-small\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"",
            j->id, now);
        json_escape_append(&out, text.p ? text.p : "", text.len);
        sbuf_appendf(&out,
            "\"},\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
            finish_reason, prompt_tokens, completion_tokens, prompt_tokens + completion_tokens);
        http_send_json(j->fd, 200, "OK", &out);
        sbuf_free(&out);
        j->status = 200;
    }
    sbuf_free(&text);
}

/* Network-thread side: parses + validates (no engine access), then enqueues
 * for the worker and blocks until it signals completion. */
static void handle_chat_completions(int fd, const char *body, size_t body_len) {
    double t0 = ink_now_sec();
    chat_request req;
    if (!parse_chat_request(body, body_len, &req)) {
        msgs_free(&req.messages);
        http_send_error(fd, 400, "Bad Request", "invalid JSON request body");
        fprintf(stderr, "POST /v1/chat/completions 400 - - %.3f\n", ink_now_sec() - t0);
        return;
    }
    if (req.content_not_string) {
        msgs_free(&req.messages);
        http_send_error(fd, 400, "Bad Request", "message content must be a string");
        fprintf(stderr, "POST /v1/chat/completions 400 - - %.3f\n", ink_now_sec() - t0);
        return;
    }
    if (req.messages.len == 0) {
        msgs_free(&req.messages);
        http_send_error(fd, 400, "Bad Request", "messages must be a non-empty array");
        fprintf(stderr, "POST /v1/chat/completions 400 - - %.3f\n", ink_now_sec() - t0);
        return;
    }

    long max_tokens = req.have_max_tokens ? req.max_tokens : 256;
    if (max_tokens < 1) max_tokens = 1;
    if ((uint32_t)max_tokens > g_max_tokens_cap) max_tokens = (long)g_max_tokens_cap;
    double temperature = req.have_temperature ? req.temperature : 0.0;
    bool stream = req.have_stream && req.stream;
    bool stream_include_usage = req.have_stream_include_usage && req.stream_include_usage;

    ink_ids ids = {0};
    for (int i = 0; i < req.messages.len; i++) {
        ink_chat_append(&g_model, &ids, req.messages.items[i].role, req.messages.items[i].content);
    }
    ink_chat_append_model_prefix(&g_model, &ids);
    msgs_free(&req.messages);

    if ((long)ids.len + max_tokens > (long)g_model.n_ctx) {
        free(ids.ids);
        http_send_error(fd, 400, "Bad Request", "context_overflow: prompt + max_tokens exceeds context length");
        fprintf(stderr, "POST /v1/chat/completions 400 %d - %.3f\n", ids.len, ink_now_sec() - t0);
        return;
    }

    job j;
    memset(&j, 0, sizeof(j));
    j.fd = fd;
    j.tokens = ids.ids;
    j.n_tokens = ids.len;
    j.max_tokens = max_tokens;
    j.temperature = temperature;
    j.stream = stream;
    j.stream_include_usage = stream_include_usage;
    j.t0 = t0;
    pthread_mutex_init(&j.mu, NULL);
    pthread_cond_init(&j.cv, NULL);

    enq_result er = try_enqueue(&j);
    if (er != ENQ_OK) {
        free(ids.ids);
        pthread_mutex_destroy(&j.mu);
        pthread_cond_destroy(&j.cv);
        if (er == ENQ_SHUTTING_DOWN) {
            http_send_error(fd, 503, "Service Unavailable", "server shutting down");
        } else {
            http_send_error_typed(fd, 503, "Service Unavailable", "overloaded", "all slots busy");
        }
        fprintf(stderr, "POST /v1/chat/completions 503 %d - %.3f\n", ids.len, ink_now_sec() - t0);
        return;
    }

    pthread_mutex_lock(&j.mu);
    while (!j.done) pthread_cond_wait(&j.cv, &j.mu);
    pthread_mutex_unlock(&j.mu);

    pthread_mutex_lock(&g_mu);
    g_slot_busy[j.slot_id] = false;
    pthread_mutex_unlock(&g_mu);

    free(j.tokens);
    pthread_mutex_destroy(&j.mu);
    pthread_cond_destroy(&j.cv);

    fprintf(stderr, "POST /v1/chat/completions %d %d %d %.3f\n",
            j.status, j.prompt_tokens, j.completion_tokens, ink_now_sec() - t0);
}

/* ============================== dispatch ================================ */

static void handle_connection(int fd) {
    http_request req;
    http_read_request(fd, &req);
    if (!req.ok) {
        http_send_error(fd, req.status_on_fail, req.status_on_fail == 400 ? "Bad Request" : "Error", req.fail_msg);
        fprintf(stderr, "- - %d - - -\n", req.status_on_fail);
        sbuf_free(&req.raw);
        return;
    }

    /* Draining: reject new work uniformly, matching the ds4-server drain
     * convention (accept() itself stops once the listen fd is closed; this
     * covers connections that raced in just before that). */
    if (g_stop) {
        http_send_error(fd, 503, "Service Unavailable", "server shutting down");
        fprintf(stderr, "%s %s 503 - - -\n", req.method, req.path);
        sbuf_free(&req.raw);
        return;
    }

    if (!strcmp(req.method, "GET") &&
        (!strcmp(req.path, "/v1/capabilities") || !strcmp(req.path, "/capabilities"))) {
        handle_capabilities(fd);
    } else if (!strcmp(req.method, "GET") && !strcmp(req.path, "/v1/models")) {
        handle_models(fd);
    } else if (!strcmp(req.method, "GET") &&
               (!strcmp(req.path, "/v1/activity") || !strcmp(req.path, "/activity"))) {
        handle_activity(fd);
    } else if (!strcmp(req.method, "GET") && !strcmp(req.path, "/v1/models/available")) {
        handle_models_available(fd);
    } else if (!strcmp(req.method, "POST") && !strcmp(req.path, "/v1/models/select")) {
        handle_models_select(fd, &req);
    } else if (!strcmp(req.method, "POST") && !strcmp(req.path, "/v1/chat/completions")) {
        handle_chat_completions(fd, req.body, req.body_len);
    } else if (!strcmp(req.path, "/v1/capabilities") || !strcmp(req.path, "/capabilities") ||
               !strcmp(req.path, "/v1/models") ||
               !strcmp(req.path, "/v1/activity") || !strcmp(req.path, "/activity") ||
               !strcmp(req.path, "/v1/models/available") ||
               !strcmp(req.path, "/v1/models/select") ||
               !strcmp(req.path, "/v1/chat/completions")) {
        http_send_error(fd, 405, "Method Not Allowed", "method not allowed for this endpoint");
        fprintf(stderr, "%s %s 405 - - -\n", req.method, req.path);
    } else {
        http_send_error(fd, 404, "Not Found", "no such endpoint");
        fprintf(stderr, "%s %s 404 - - -\n", req.method, req.path);
    }

    sbuf_free(&req.raw);
}

/* ============================ threads + main ============================ */

static void *worker_main(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&g_mu);
        while (!g_head && !g_worker_stop) pthread_cond_wait(&g_work_cv, &g_mu);
        if (!g_head && g_worker_stop) { pthread_mutex_unlock(&g_mu); break; }
        job *j = g_head;
        g_head = j->next;
        if (!g_head) g_tail = NULL;
        pthread_mutex_unlock(&g_mu);

        run_job(j);

        pthread_mutex_lock(&j->mu);
        j->done = true;
        pthread_cond_signal(&j->cv);
        pthread_mutex_unlock(&j->mu);
    }
    return NULL;
}

static void *client_main(void *arg) {
    int fd = *(int *)arg;
    free(arg);
    handle_connection(fd);
    close(fd);
    pthread_mutex_lock(&g_mu);
    g_clients--;
    pthread_cond_broadcast(&g_drain_cv);
    pthread_mutex_unlock(&g_mu);
    return NULL;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    uint32_t n_ctx = 4096;
    bool have_ctx_flag = false;
    int port = 8090;
    const char *host = "0.0.0.0";
    uint32_t max_tokens_cap = 512;
    int sessions = 2;
    bool have_sessions_flag = false;
    bool resident = false;
    bool force_cpu = false;
    uint64_t resident_budget_bytes = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) { n_ctx = (uint32_t)atoi(argv[++i]); have_ctx_flag = true; }
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-H") && i + 1 < argc) host = argv[++i];
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) max_tokens_cap = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) { sessions = atoi(argv[++i]); have_sessions_flag = true; }
        else if (!strcmp(argv[i], "--cpu")) force_cpu = true;
        else if (!strcmp(argv[i], "--resident")) resident = true;
        else if (!strcmp(argv[i], "--resident-budget") && i + 1 < argc) {
            resident = true;
            resident_budget_bytes = (uint64_t)(atof(argv[++i]) * 1073741824.0);
        } else {
            fprintf(stderr,
                "usage: ds4-inkling-server -m model.gguf [-c CTX] [-p PORT] [-H HOST] "
                "[-t MAX_TOKENS_CAP] [-s SESSIONS] [--resident] [--resident-budget GiB] [--cpu]\n");
            return 1;
        }
    }
    if (!model_path) {
        fprintf(stderr,
            "usage: ds4-inkling-server -m model.gguf [-c CTX] [-p PORT] [-H HOST] "
            "[-t MAX_TOKENS_CAP] [-s SESSIONS] [--resident] [--resident-budget GiB] [--cpu]\n");
        return 1;
    }
    if (!have_ctx_flag) {
        const char *e = getenv("DS4_CTX");
        if (e && e[0]) n_ctx = (uint32_t)atoi(e);
    }
    if (!have_sessions_flag) {
        const char *e = getenv("DS4_SESSIONS");
        if (e && e[0]) sessions = atoi(e);
    }
    if (sessions < 1) sessions = 1;
    g_max_tokens_cap = max_tokens_cap;
    g_sessions = sessions;
    g_resident = resident;
    g_model_path = model_path;

    {
        const char *at = getenv("ACCRETION_ADMIN_TOKEN");
        g_admin_token = (at && at[0]) ? at : NULL;
        const char *ef = getenv("DS4_ENV_FILE");
        if (ef && ef[0]) g_env_file = ef;
        else if (access("/opt/accretion/etc/ds4-server.env", W_OK) == 0) g_env_file = "/opt/accretion/etc/ds4-server.env";
        else g_env_file = NULL;
    }
    g_model_dirs = getenv("DS4_MODEL_DIRS");

    signal(SIGPIPE, SIG_IGN);
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = stop_signal_handler;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    g_rng_state = (uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ 0x9E3779B97F4A7C15ULL;
    if (g_rng_state == 0) g_rng_state = 1;

    double t0 = ink_now_sec();
    {
        const char *be = getenv("DS4_INKLING_BACKEND");
        if (be && !strcmp(be, "cpu")) force_cpu = true;
#ifdef DS4_INKLING_CUDA
        if (!force_cpu) {
            ink_cuda_init();     /* dies with a clear message if unusable */
            g_use_gpu = true;
        }
#else
        (void)force_cpu;
#endif
        fprintf(stderr, "ds4-inkling-server: backend %s\n", backend_name());
    }

    ink_model_open(&g_model, model_path, n_ctx);
    if (!realpath(model_path, g_model_realpath)) {
        snprintf(g_model_realpath, sizeof(g_model_realpath), "%s", model_path);
    }
    if (resident) {
#ifdef DS4_INKLING_CUDA
        if (g_use_gpu) {
            /* Split arena: quantized weights into device memory, host-read
             * tensors on the host -- identical to what ds4-inkling-cuda
             * proves out, and it prints its own GiB split line. */
            ink_cuda_make_resident(&g_model, resident_budget_bytes);
        } else
#endif
        {
            uint64_t n_res = 0;
            uint64_t nb = ink_model_make_resident(&g_model, resident_budget_bytes, malloc, &n_res);
            fprintf(stderr, "ds4-inkling-server: resident %.1f GiB in %llu tensors (host)\n",
                    nb / 1073741824.0, (unsigned long long)n_res);
        }
    }
    fprintf(stderr, "ds4-inkling-server: loaded %s (%u layers, vocab %u, ctx %u) in %.1fs\n",
            model_path, g_model.n_layer, g_model.n_vocab, g_model.n_ctx, ink_now_sec() - t0);

    g_end_message_id = ink_token_lookup(&g_model.tk, "<|end_message|>");

    g_slots = calloc((size_t)g_sessions, sizeof(*g_slots));
    g_slot_busy = calloc((size_t)g_sessions, sizeof(*g_slot_busy));
    if (!g_slots || !g_slot_busy) diesys("calloc");

    pthread_t worker;
    if (pthread_create(&worker, NULL, worker_main, NULL) != 0) diesys("pthread_create");

    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    if (lfd < 0) diesys("socket");
    int one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    const char *bind_host = !strcmp(host, "localhost") ? "127.0.0.1" : host;
    if (inet_pton(AF_INET, bind_host, &addr.sin_addr) != 1) {
        fprintf(stderr, "ds4-inkling-server: bad host %s\n", host);
        return 1;
    }
    addr.sin_port = htons((uint16_t)port);
    if (bind(lfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) diesys("bind");
    if (listen(lfd, 64) < 0) diesys("listen");
    g_listen_fd = lfd;

    fprintf(stderr, "ds4-inkling-server: listening on http://%s:%d (sessions=%d, max_tokens cap %u)\n",
            host, port, g_sessions, g_max_tokens_cap);

    while (!g_stop) {
        int fd = accept(lfd, NULL, NULL);
        if (fd < 0) {
            if (g_stop) break;
            if (errno == EINTR) continue;
            continue;
        }
        if (g_stop) { close(fd); break; }

        int *pfd = malloc(sizeof(*pfd));
        if (!pfd) { close(fd); continue; }
        *pfd = fd;
        pthread_mutex_lock(&g_mu);
        g_clients++;
        pthread_mutex_unlock(&g_mu);
        pthread_t th;
        if (pthread_create(&th, NULL, client_main, pfd) != 0) {
            pthread_mutex_lock(&g_mu);
            g_clients--;
            pthread_cond_broadcast(&g_drain_cv);
            pthread_mutex_unlock(&g_mu);
            free(pfd);
            close(fd);
            continue;
        }
        pthread_detach(th);
    }
    if (g_listen_fd >= 0) { close(lfd); g_listen_fd = -1; }

    fprintf(stderr, "ds4-inkling-server: shutdown requested, draining requests\n");
    pthread_mutex_lock(&g_mu);
    while (g_clients > 0) pthread_cond_wait(&g_drain_cv, &g_mu);
    g_worker_stop = true;
    pthread_cond_broadcast(&g_work_cv);
    pthread_mutex_unlock(&g_mu);
    pthread_join(worker, NULL);

    fprintf(stderr, "ds4-inkling-server: shutdown complete\n");
    if (g_swap_requested) {
        fprintf(stderr,
                "ds4-inkling-server: model swap drain complete; exiting %d for supervisor restart on the new selection\n",
                DS4_SERVER_SWAP_EXIT_CODE);
        return DS4_SERVER_SWAP_EXIT_CODE;
    }
    return 0;
}
