/* ds4_inkling_server.c -- thin standalone OpenAI-compatible HTTP server for
 * the "inkling" engine (ds4_inkling.c).  Single-threaded, single-session:
 * the underlying engine keeps exactly one KV/shortconv state, so this
 * server accepts and serves one request at a time (accept loop, blocking
 * I/O, no worker pool).  No external libraries: plain POSIX sockets and a
 * small hand-written JSON reader/writer, in the same spirit as the rest of
 * ds4 (see ds4_inkling.c, ds4_server.c).
 *
 * Endpoints:
 *   GET  /v1/capabilities   -> engine/model capability blob
 *   GET  /v1/models         -> OpenAI-shaped model list (one model)
 *   POST /v1/chat/completions -> OpenAI-shaped non-streaming chat completion
 *
 * Known limitation: "stream":true is rejected with 400 (documented below);
 * there is no SSE/chunked-transfer support in this v1 server. */

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "ds4_inkling.h"

#define MAX_HEADER_BYTES  (64 * 1024)
#define MAX_BODY_BYTES    (1 * 1024 * 1024)
#define PREFILL_CHUNK     32

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
    bool have_max_tokens, have_temperature, have_stream;
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
    sbuf raw;      /* full accumulated request bytes (headers + body) */
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

    /* Content-Length header (case-insensitive), if any */
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

static void http_send_status(int fd, int status, const char *status_text,
                              const char *content_type, const char *body, size_t body_len) {
    sbuf out; sbuf_init(&out);
    sbuf_appendf(&out, "HTTP/1.1 %d %s\r\n", status, status_text);
    sbuf_appendf(&out, "Content-Type: %s\r\n", content_type);
    sbuf_appendf(&out, "Content-Length: %zu\r\n", body_len);
    sbuf_appendz(&out, "Connection: close\r\n\r\n");
    sbuf_append(&out, body, body_len);
    size_t off = 0;
    while (off < out.len) {
        ssize_t n = send(fd, out.p + off, out.len - off, 0);
        if (n < 0) { if (errno == EINTR) continue; break; }
        off += (size_t)n;
    }
    sbuf_free(&out);
}

static void http_send_json(int fd, int status, const char *status_text, const sbuf *body) {
    http_send_status(fd, status, status_text, "application/json", body->p ? body->p : "", body->len);
}

static void http_send_error(int fd, int status, const char *status_text, const char *err_msg) {
    sbuf out; sbuf_init(&out);
    sbuf_appendz(&out, "{\"error\":{\"message\":\"");
    json_escape_append(&out, err_msg, strlen(err_msg));
    sbuf_appendz(&out, "\"}}");
    http_send_json(fd, status, status_text, &out);
    sbuf_free(&out);
}

/* ============================ engine wiring ============================ */

static ink_model g_model;
static int g_end_message_id = -1;
static uint32_t g_max_tokens_cap = 512;
static long g_req_counter = 0;

/* xorshift64* PRNG for temperature sampling. */
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

/* Build the JSON error body for a bad chat request. */
static void send_chat_error(int fd, int status, const char *status_text, const char *msg) {
    http_send_error(fd, status, status_text, msg);
}

static void handle_capabilities(int fd) {
    sbuf out; sbuf_init(&out);
    sbuf_appendf(&out,
        "{\"model_name\":\"inkling-small\",\"architecture\":\"inkling\","
        "\"context_length\":%u,\"vocab_size\":%u,\"engine\":\"ds4-inkling\",\"backend\":\"cpu\"}",
        g_model.n_ctx, g_model.n_vocab);
    http_send_json(fd, 200, "OK", &out);
    sbuf_free(&out);
    fprintf(stderr, "GET /v1/capabilities 200 - - -\n");
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

static void handle_chat_completions(int fd, const char *body, size_t body_len) {
    double t0 = ink_now_sec();
    chat_request req;
    if (!parse_chat_request(body, body_len, &req)) {
        msgs_free(&req.messages);
        send_chat_error(fd, 400, "Bad Request", "invalid JSON request body");
        fprintf(stderr, "POST /v1/chat/completions 400 - - %.3f\n", ink_now_sec() - t0);
        return;
    }
    if (req.content_not_string) {
        msgs_free(&req.messages);
        send_chat_error(fd, 400, "Bad Request", "message content must be a string");
        fprintf(stderr, "POST /v1/chat/completions 400 - - %.3f\n", ink_now_sec() - t0);
        return;
    }
    if (req.messages.len == 0) {
        msgs_free(&req.messages);
        send_chat_error(fd, 400, "Bad Request", "messages must be a non-empty array");
        fprintf(stderr, "POST /v1/chat/completions 400 - - %.3f\n", ink_now_sec() - t0);
        return;
    }
    if (req.stream) {
        msgs_free(&req.messages);
        send_chat_error(fd, 400, "Bad Request", "streaming not supported in v1");
        fprintf(stderr, "POST /v1/chat/completions 400 - - %.3f\n", ink_now_sec() - t0);
        return;
    }

    long max_tokens = req.have_max_tokens ? req.max_tokens : 256;
    if (max_tokens < 1) max_tokens = 1;
    if ((uint32_t)max_tokens > g_max_tokens_cap) max_tokens = (long)g_max_tokens_cap;
    double temperature = req.have_temperature ? req.temperature : 0.0;

    ink_ids ids = {0};
    for (int i = 0; i < req.messages.len; i++) {
        ink_chat_append(&g_model, &ids, req.messages.items[i].role, req.messages.items[i].content);
    }
    ink_chat_append_model_prefix(&g_model, &ids);

    if ((long)ids.len + max_tokens > (long)g_model.n_ctx) {
        free(ids.ids);
        msgs_free(&req.messages);
        send_chat_error(fd, 400, "Bad Request", "context_overflow: prompt + max_tokens exceeds context length");
        fprintf(stderr, "POST /v1/chat/completions 400 %d - %.3f\n", ids.len, ink_now_sec() - t0);
        return;
    }

    int prompt_tokens = ids.len;

    ink_state_reset(&g_model);
    float *logits = malloc((size_t)g_model.n_vocab * sizeof(float));
    if (!logits) diesys("malloc");

    for (int i = 0; i < ids.len; i += PREFILL_CHUNK) {
        int n = ids.len - i < PREFILL_CHUNK ? ids.len - i : PREFILL_CHUNK;
        bool last = (i + n == ids.len);
        ink_forward_batch(&g_model, ids.ids + i, (uint32_t)n, (uint32_t)i, last ? logits : NULL);
    }

    sbuf text; sbuf_init(&text);
    int completion_tokens = 0;
    const char *finish_reason = "length";
    uint32_t pos = (uint32_t)ids.len;

    for (long t = 0; t < max_tokens; t++) {
        int tok = (temperature <= 0.0) ? argmax_logits(logits, g_model.n_vocab)
                                        : sample_logits(logits, g_model.n_vocab, temperature);
        completion_tokens++;
        bool is_stop = (tok == g_model.tk.eos) || (g_end_message_id >= 0 && tok == g_end_message_id);
        if (!is_stop) {
            char piece[512];
            int n = ink_detokenize(&g_model.tk, tok, piece, sizeof(piece));
            if (!token_is_special_bracket(piece, n)) sbuf_append(&text, piece, (size_t)n);
        }
        if (is_stop) { finish_reason = "stop"; break; }
        if (t + 1 == max_tokens) { finish_reason = "length"; break; }
        ink_forward(&g_model, tok, pos++, logits);
    }

    free(logits);
    free(ids.ids);
    msgs_free(&req.messages);

    long id_num = ++g_req_counter;
    sbuf out; sbuf_init(&out);
    sbuf_appendf(&out,
        "{\"id\":\"chatcmpl-%ld\",\"object\":\"chat.completion\",\"created\":%ld,"
        "\"model\":\"inkling-small\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"",
        id_num, (long)time(NULL));
    json_escape_append(&out, text.p ? text.p : "", text.len);
    sbuf_appendf(&out,
        "\"},\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
        finish_reason, prompt_tokens, completion_tokens, prompt_tokens + completion_tokens);

    http_send_json(fd, 200, "OK", &out);
    sbuf_free(&out);
    sbuf_free(&text);

    fprintf(stderr, "POST /v1/chat/completions 200 %d %d %.3f\n",
            prompt_tokens, completion_tokens, ink_now_sec() - t0);
}

static void handle_connection(int fd) {
    http_request req;
    http_read_request(fd, &req);
    if (!req.ok) {
        http_send_error(fd, req.status_on_fail, req.status_on_fail == 400 ? "Bad Request" : "Error", req.fail_msg);
        fprintf(stderr, "- - %d - - -\n", req.status_on_fail);
        sbuf_free(&req.raw);
        return;
    }

    if (strcmp(req.method, "GET") == 0 && strcmp(req.path, "/v1/capabilities") == 0) {
        handle_capabilities(fd);
    } else if (strcmp(req.method, "GET") == 0 && strcmp(req.path, "/v1/models") == 0) {
        handle_models(fd);
    } else if (strcmp(req.method, "POST") == 0 && strcmp(req.path, "/v1/chat/completions") == 0) {
        handle_chat_completions(fd, req.body, req.body_len);
    } else if (strcmp(req.path, "/v1/capabilities") == 0 || strcmp(req.path, "/v1/models") == 0 ||
               strcmp(req.path, "/v1/chat/completions") == 0) {
        http_send_error(fd, 405, "Method Not Allowed", "method not allowed for this endpoint");
        fprintf(stderr, "%s %s 405 - - -\n", req.method, req.path);
    } else {
        http_send_error(fd, 404, "Not Found", "no such endpoint");
        fprintf(stderr, "%s %s 404 - - -\n", req.method, req.path);
    }

    sbuf_free(&req.raw);
}

/* ================================ main ================================= */

static volatile sig_atomic_t g_stop = 0;

static void on_sigint(int sig) { (void)sig; g_stop = 1; }

int main(int argc, char **argv) {
    const char *model_path = NULL;
    uint32_t n_ctx = 4096;
    int port = 8090;
    uint32_t max_tokens_cap = 512;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) n_ctx = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) max_tokens_cap = (uint32_t)atoi(argv[++i]);
        else {
            fprintf(stderr, "usage: ds4-inkling-server -m model.gguf [-c CTX] [-p PORT] [-t MAX_TOKENS_CAP]\n");
            return 1;
        }
    }
    if (!model_path) {
        fprintf(stderr, "usage: ds4-inkling-server -m model.gguf [-c CTX] [-p PORT] [-t MAX_TOKENS_CAP]\n");
        return 1;
    }
    g_max_tokens_cap = max_tokens_cap;

    signal(SIGPIPE, SIG_IGN);
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_sigint;
    sigaction(SIGINT, &sa, NULL);

    g_rng_state = (uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ 0x9E3779B97F4A7C15ULL;
    if (g_rng_state == 0) g_rng_state = 1;

    double t0 = ink_now_sec();
    ink_model_open(&g_model, model_path, n_ctx);
    fprintf(stderr, "ds4-inkling-server: loaded %s (%u layers, vocab %u, ctx %u) in %.1fs\n",
            model_path, g_model.n_layer, g_model.n_vocab, g_model.n_ctx, ink_now_sec() - t0);

    g_end_message_id = ink_token_lookup(&g_model.tk, "<|end_message|>");

    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    if (lfd < 0) diesys("socket");
    int one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((uint16_t)port);
    if (bind(lfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) diesys("bind");
    if (listen(lfd, 16) < 0) diesys("listen");

    fprintf(stderr, "ds4-inkling-server: listening on 0.0.0.0:%d (max_tokens cap %u)\n", port, g_max_tokens_cap);

    while (!g_stop) {
        struct sockaddr_in caddr;
        socklen_t clen = sizeof(caddr);
        int cfd = accept(lfd, (struct sockaddr *)&caddr, &clen);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            if (g_stop) break;
            continue;
        }
        handle_connection(cfd);
        close(cfd);
    }

    fprintf(stderr, "ds4-inkling-server: shutting down\n");
    close(lfd);
    return 0;
}
