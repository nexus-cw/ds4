# ds4 KV cache machinery audit (read-only)

Repo: `~/src/ds4` on robo-dog, branch `research/gb10`.
Scope: cross-request KV reuse, disk KV persistence, fork/rollback potential,
batched-session slot/KV relationship, `--ssd-streaming` interaction, and a
gap list against three roadmap directions (prefix-reuse economics,
checkpoint/fork/restore "time travel", KV transfer/persistence).

All claims below are grounded in file:line citations from this repo. No
model runs, builds, or server operations were performed for this audit.

---

## 1. Live KV reuse (the log line and its matching)

The log line comes from `generate_job()`:

- `ds4_server.c:11066-11071` — emits `"ds4-server: live kv cache miss ... live=%d prompt=%d common=%d reason=%s"` when every live-continuation match attempt has failed (`cached == 0`) and the session already has live state (`old_pos > 0`).

**Matching semantics.** `common` is computed by `ds4_session_common_prefix()`:

- `ds4.c:59968-59974` — a **linear, exact token-id scan**: `while (i < n && checkpoint.v[i] == prompt.v[i]) i++`. No fuzzy/edit-distance matching, no threshold — it's longest-common-prefix by exact token equality, capped at `min(checkpoint.len, prompt.len)`.

**Scope: per-session, not global.** The checkpoint (`s->checkpoint`, a `token_vec`) lives inside `struct ds4_session` (`ds4.c:48599-48666`), and each `server_slot` owns exactly one `ds4_session` created at startup (`ds4_server.c:12980-12988`). There is one live token history per slot; no cross-slot/global KV pool or dedup — reuse is strictly "does *this* slot's live state extend into the new request."

**"Live" vs "disk" vocabulary.** "Live" = the resident GPU/CPU session state for a slot (`slot->session`, `ds4_session.checkpoint` + the actual attention-cache tensors in `s->graph`/`s->glm_graph`/`s->cpu_cache`). "Disk" = a serialized checkpoint file under `--kv-disk-dir`, looked up and loaded via `ds4_kvstore_*` (`ds4_kvstore.c`), which *replaces* live state when loaded — disk is not a passive extension of live, it's a swap-in (see §2).

**Order of matching before the raw-token check that produced the log.** `generate_job()` tries, in order, before falling back to raw-token `common`:
1. `responses_live_visible_prefix_prompt` — Responses API, matches the **rendered visible transcript text** (not tokens) against the live checkpoint's remembered visible-text key (`ds4_server.c:9568-9591`).
2. `responses_live_continuation_prompt` — tool-output-only continuation keyed by `call_id`s (`ds4_server.c:9505-9520`).
3. `anthropic_live_continuation_prompt` — same idea for `/v1/messages` `tool_use_id` (`ds4_server.c:9533-9552`).
4. raw exact-token `common == old_pos` check (`ds4_server.c:11036-11037`, `cache_source = "memory-token"`).
5. `thinking_live_visible_prefix_prompt` — non-Responses APIs, matches visible transcript text remembered after a tool-less "thinking" turn (`ds4_server.c:9611-9633`).
6. `live_text_prefix_prompt` — general **rendered-text byte-prefix** match: detokenizes the live checkpoint back to text (`render_tokens_text`) and does a byte-prefix compare against the raw incoming request text, then retokenizes only the *new suffix bytes* (never slices `req->prompt` tokens, because BPE can merge across the boundary) (`ds4_server.c:9471-9497`).
7. only if **all of the above fail** does the code fall through to the `kv cache miss` log (`ds4_server.c:11066`), then disk lookup (§2).

So template-aware, text-level (not raw-token) prefix matching **already exists** as step 6 — this is exactly the mechanism that would be needed to survive chat-template header churn between requests. The design comment at `ds4_server.c:10949-10965` documents this staging explicitly ("Clients resend full prompts as text ... old exact token-prefix hit, then rendered-text prefix hit ... ").

**Why `common=2` in the observed log.** Because the log is only reached when text-level matching (step 6) *also* failed (`byte_prefix_match` returned false), a `common=2` value means the raw token-id prefix diverges almost immediately **and** the rendered text also diverges early — i.e., the incoming prompt is not a byte-level continuation of the live checkpoint at all past ~2 tokens. Likely causes consistent with the code: a different `slot` was picked for this request than served the prior turn (no session/slot affinity is visible in the audited code — `slot_count` sessions are independent, and nothing in `generate_job` pins a client to "its" prior slot; that routing lives outside the audited window), a cold slot (`old_pos>0` but only from unrelated prior traffic), or a system prompt with per-request-varying content (e.g., a timestamp) breaking the byte-prefix at the first divergent byte. `--trace` output (`trace_write_cache_diag`, `ds4_server.c:9750-9825`) records the exact token window around the first mismatch and would confirm which; this was not run (read-only, no server touch). `trace_cache_miss_reason()` (`ds4_server.c:9703-9709`) classifies `common=2, old_pos>2` as `"token-mismatch"`, matching the observed log's `reason=token-mismatch`.

**What would it take for prefix match to hit on a short shared system prompt.** Nothing structural — the byte-prefix path (step 6) already is template-aware in the sense that it compares *rendered* bytes, not raw tokens. What's missing is (a) session/slot affinity so the "right" slot's live state is the one being compared (a routing concern, not a KV concern, and outside the audited files), and (b) if the divergence is because the system prompt itself is not byte-identical across requests (e.g. embeds a clock), no matching scheme can help — the bytes genuinely differ. There is no "multi-slot live cache" / N-best live-checkpoint search: each slot checks only its own single checkpoint.

---

## 2. Disk KV (`--kv-disk-dir`)

**What persists.** Full per-slot session checkpoints: exact token sequence + logits + the full live GPU/CPU attention-cache tensors for every layer, via `ds4_session_save_payload()` (`ds4.c:50564` and on, GLM path `ds4.c:50564-50658`, DeepSeek4/CPU path continuing after `ds4.c:50658`) and its counterpart `ds4_session_load_payload()` (`ds4.c:50873`). This is a **full KV state snapshot**, not just tokens.

**Granularity / triggers** — three store reasons, all writing via the same `ds4_kvstore_store_live_prefix[_text]()` path (`ds4_kvstore.c:923`):
- `"cold"` — when a request's stable prefix is shorter than the full canonical prompt, the server prefills to that boundary and stores it before continuing generation, so future identical-prefix requests can reuse it cold (`ds4_server.c:11178-11219`; gated by `kv.opt.cold_max_tokens`, default 30000 tokens, `ds4_kvstore.c:34`).
- `"continued"` — periodically **during generation**, every `continued_interval_tokens` new tokens (default 10000, `ds4_kvstore.c:42`), via `kv_cache_maybe_store_continued()` (`ds4_server.c:9414-9422`), called from the decode loop (`ds4_server.c:10452, 10493, 11251, 11379`).
- `"evict"` — right before a slot's live state is about to be replaced by a different disk checkpoint (i.e. on a live-cache-miss where a disk hit will be tried), so the newer conversation state is not silently lost (`ds4_server.c:11072-11078`, comment explains the rationale directly above).
- `"shutdown"` — on clean server shutdown, one store per slot (`ds4_server.c:13138`).

Reason codes are enumerated at `ds4_kvstore.h:20-27` (`DS4_KVSTORE_REASON_COLD/CONTINUED/EVICT/SHUTDOWN`, plus agent-specific `AGENT_SYSTEM`/`AGENT_SESSION` reasons not traced further here).

**Format.** Not raw — a versioned custom binary payload: fixed 48-byte header (`DS4_KVSTORE_FIXED_HEADER`, `ds4_kvstore.h:11`) containing model id, quant bits, reason, ext flags, token count, hit count, ctx size, timestamps, payload/text/file byte sizes (`ds4_kvstore_fill_header`, `ds4_kvstore.c:393-416`; read side `ds4_kvstore_read_header`, `ds4_kvstore.c:417`); the payload body itself starts with a magic+version u32 header (`DS4_SESSION_PAYLOAD_MAGIC`/`_VERSION`, `ds4.c:50573-50580` region) followed by shape metadata, the exact token vector, logits, and then per-layer KV spans (compact MLA-latent cache + RoPE cache + indexer cache for GLM; raw + compressed spans for DeepSeek4/CPU — `ds4.c:50564-50710` and continuation). Files are named by **SHA1 of the rendered prompt text**, truncated to the checkpoint's text length (`ds4_kvstore.c:350-351`, `ds4_kvstore_sha1_bytes_hex`), i.e. content-addressed by rendered text, not by session id.

**Restore path and gating.** `ds4_kvstore_try_load_text()` (`ds4_kvstore.c:1215`) calls `ds4_kvstore_find_text_prefix()` (`ds4_kvstore.c:1189-1211`), which scans **all** disk entries and picks the one with the **longest matching text byte-prefix** (re-hashing the incoming prompt text truncated to each candidate's stored length and comparing SHA1 — `ds4_kvstore.c:1205-1208`), subject to `min_tokens`, matching `model_id`, `ctx_size <= incoming ctx_size`, and (optionally) matching `quant_bits` if `reject_different_quant`. So: **yes, disk restore is also byte-prefix-gated**, and it is a longest-prefix search across the whole disk store, unlike the single-slot live check. On load it also gates on the file header's own `quant_bits` (`ds4_kvstore.c:1226-1227`) and invalidates/discards the session if the payload fails to load (`ds4_kvstore.c:1277, 1295, 1305`).

Eviction of old disk entries uses a scored LRU-like policy (`ds4_kvstore_entry_eviction_score`, declared `ds4_kvstore.h:143-146`, referencing `hits`/`last_used`/half-life `DS4_KVSTORE_HIT_HALF_LIFE_SECONDS` = 6h, `ds4_kvstore.h:13`) under a configured `budget_bytes` (`--kv-disk-space`).

---

## 3. Fork/snapshot potential

**Contiguous per-layer buffers, not paged.** Each `ds4_session` owns its own fixed-size per-layer tensors sized at session-creation time to `ctx_size` (GPU path: `ds4_gpu_graph g->layer_attn_state_kv[il]` etc., sized via `metal_graph_kv_cache_bytes_for_context()`, `ds4.c:16838-16852`; CPU path: `ds4_kv_cache cpu_cache` with `ds4_layer_cache layer[DS4_MAX_LAYER]`, `ds4.c:12902-12903` region). This is a **contiguous, single-buffer-per-layer** design (no block/paged-attention indirection layer like vLLM's PagedAttention) — sized once for the whole context, with a "raw" window capped to the SWA size (`ds4_default_raw_cap`, `ds4.c:12904-12909`, = `min(DS4_N_SWA, ctx_size)`, e.g. 128 for the Flash shape at `ds4.c:551`) and a separate "compressed" (MLA latent) span that scales with `ctx_size / compression_ratio` (`ds4.c:16838-16852`).

**Cost of a full copy/fork today, estimated.** Using the sizing formula at `ds4.c:16838` (`metal_graph_kv_cache_bytes_for_context`) with the DeepSeek V4 Flash shape resident on this box (`ds4flash.gguf` present in repo root; shape at `ds4.c:536-568`: `n_layer=43`, `n_head_dim=512`, `n_indexer_head_dim=128`, `n_swa=128`; per-layer compression ratios from `ds4_expected_layer_compress_ratio`, `ds4.c:1096-1109`: layers 0-1 uncompressed, then alternating ratio-4 / ratio-128 for layers 2-42) at `ctx_size=16384` with the default f16 compressed-cache format (`DS4_GPU_ATTN_COMP_CACHE_F16=1`, `ds4.c:15708`):
- raw SWA span (all 43 layers, capped at 128 rows regardless of ctx): `43 * 128 * 512 * 4B ≈ 10.7 MiB`.
- ratio-4 layers (21 of them, `comp_cap ≈ 4098`): latent `4098*512*2B` + indexer `4098*128*4B` ≈ `6.0 MiB`/layer → `≈126 MiB`.
- ratio-128 layers (20 of them, `comp_cap ≈ 130`): `130*512*2B ≈ 0.13 MiB`/layer → `≈2.6 MiB`.
- **Total ≈ 139 MiB per session at 16k context** (plus a small fixed vocab-sized logits buffer, `DS4_N_VOCAB=129280` floats ≈ 0.5 MiB).

This is a computed estimate from the repo's own sizing formula (not a live measurement — no server was touched); it should be treated as order-of-magnitude, not exact, since I did not confirm at runtime which shape/quant flags are actually active for the currently-loaded model. It does show MLA compression keeps a full-context fork cheap relative to naive multi-head KV (the dominant term is the compressed latent span, not a per-token full-width K/V cache).

**Existing rollback machinery — DSpark verify-reject path.** `spec_frontier_snapshot()`/`spec_frontier_restore()` (`ds4.c:50318-50395`) is exactly a GPU-side per-layer tensor-copy checkpoint/restore: it does `ds4_gpu_tensor_copy()` of each layer's `attn_state_kv`/`attn_state_score` (and, for ratio-4 layers, `index_state_kv`/`index_state_score`) into/from parallel `spec_*` scratch tensors, plus the small scalar frontier counters (`mtp_n_raw`, `dspark_cache_*`, per-layer `n_comp`/`n_index_comp`). It's used around the DSpark speculative-verify pass and on partial-accept to roll back rejected draft tokens (call sites `ds4.c:51745, 51949, 52135-52323, 61949-62282, 65402-65735`). Also `ds4_session_rewind(s, pos)` (`ds4.c:65858-65869`) does a cheap **logical** truncation — sets `checkpoint.len = pos` — relying on the fact that KV rows beyond `pos` are simply overwritten by the next append rather than physically erased; this is O(1), not a copy.

**Is the rollback primitive generalizable to arbitrary-point restore?** Partially, and it clarifies the gap precisely:
- `spec_frontier_snapshot/_restore` is scoped to a **narrow, bounded window** (only the compressor "spec window" state used during multi-token verify, not the full context), and it snapshots into a single scratch slot reused across calls (one outstanding checkpoint at a time), not an arbitrary set of named forkpoints.
- `ds4_session_rewind` gives free instant *linear* rollback (move the frontier backward) but only within the *same* live buffer — it cannot keep two divergent futures alive simultaneously (advancing past `pos` again overwrites what was there before).
- A true fork (keep both branches live) would need to either (a) generalize the `spec_frontier_snapshot` tensor-copy pattern from the narrow spec window to the *full* per-layer compressed+raw span (structurally the same `ds4_gpu_tensor_copy` per-layer loop, just over more bytes — the same code shape, no new abstraction), or (b) reuse the disk save/load payload round-trip (`ds4_session_save_payload`/`_load_payload`, §2) as an in-memory-equivalent fork by writing to a second `ds4_session`'s buffers directly instead of a file. Both routes reuse existing primitives; there is no existing "N live forkpoints" bookkeeping (ids, refcounts, GC) anywhere in the audited files.

---

## 4. Batched sessions and KV

`--batched-session N` sets `slot_count = N` (`ds4_server.c:12944`), and the server allocates exactly `N` `server_slot`s, each with its own **fixed, pre-allocated** `ds4_session` created once at startup via `ds4_session_create(&slot->session, engine, cfg.ctx_size)` (`ds4_server.c:12980-12988`) — i.e., fixed per-slot allocation sized to full `ctx_size`, not a dynamic pool.

**Can a slot's KV be swapped out/in?** Yes, but only opportunistically, per the same mechanism as §2/§1: when a request arrives that misses the slot's live checkpoint (`cached==0`) and `old_pos >= kv.opt.min_tokens`, the slot's current live state is stored to disk first (`kv_cache_store_current(s, slot, "evict")`, `ds4_server.c:11072-11078`) before a disk-hit lookup potentially loads a *different* checkpoint into that same slot (`kv_cache_try_load`, `ds4_server.c:11081-11090`). So it is genuinely a "more sessions than slots" tiering mechanism — an idle/stale conversation's KV can live only on disk while its slot is reused for other traffic, and later be paged back in — but it is driven purely by cache-miss-on-arrival, not by any explicit scheduler, LRU-slot-eviction policy, or session affinity/pinning logic in the audited files. Nothing here proactively swaps a slot on session idle; it only reacts when a *new request* for a stale session arrives at a slot whose live state doesn't match.

---

## 5. `--ssd-streaming` interaction

Confirmed: `--ssd-streaming` and its supporting code (`ds4_ssd.c`, `ds4_ssd.h`) are entirely about **routed-expert MoE weight** caching/streaming (`ds4_ssd_cache_plan`, `cache_experts`, `per_expert_bytes`, `ds4_ssd.h:7-31`) — there is no KV-related field anywhere in `ds4_ssd.h`/`ds4_ssd.c`. KV attention-cache tensors are separate GPU buffers always resident regardless of this flag, as also stated explicitly in a design comment: "KV rollback on a verify miss (`spec_frontier_snapshot/_restore`) operates on the raw SWA cache, which is **always fully resident regardless of `--ssd-streaming`** (only routed-expert weights are streamed)" (`ds4.c:56843-56847`).

**Prefill chunking vs decode, re: KV.** The same comment block documents that streamed batch-verify prefill (`n_tokens > 1`, e.g. DSpark's draft-batch verify) and single-token decode (`n_tokens == 1`) both route routed-expert fetches through the *same* selected-cache/LRU protocol (`cuda_stream_selected_cache_*`, `ds4.c:56830-56845`) — i.e. chunked prefill is "not a new code path, just a smaller n_tokens call into the existing one" for the *expert-weight* streaming machinery. This says nothing changes structurally for *KV* between chunked prefill and decode either — both write into the same resident per-layer KV buffers, just prefill writes many rows per call and decode writes one.

---

## 6. Gap list vs the three roadmap directions

**(a) Prefix reuse economics (agentic self-prompting).**
- Exists: per-slot exact-token and text-byte-prefix live matching (§1, multiple fallback layers for tool-call/reasoning continuation shapes); disk-side longest-text-prefix search across all stored checkpoints (§2); usage accounting already separates `cache_read_tokens` vs `cache_write_tokens` per request (`ds4_server.c:11169-11170`).
- Missing: (i) no cross-slot / global live-state search — a request that would hit a *different* slot's live checkpoint (or any slot for a fresh conversation routed to a currently-idle slot) falls straight to disk, even though a cheaper live-to-live copy might be possible; this needs slot-selection/affinity logic, which was not found in the audited files (may live in the request-routing code not reached from `generate_job`, worth a follow-up grep for "pick slot" / dispatch logic). (ii) no template/normalization-aware matching beyond raw-byte prefix — if the roadmap wants prefix hits when only a system prompt is shared but rendered text legitimately differs (e.g. dynamic content), that is a template-diff feature, not present.
- Size estimate: **wiring-class**. The mechanism (exact-token, then text-byte-prefix, then disk longest-prefix) is already end-to-end; a smarter slot-affinity router to raise the hit rate is additive routing logic, not a new KV primitive.

**(b) Fork/checkpoint/restore ("time travel" for tree search).**
- Exists: full-fidelity save/restore to disk (`ds4_session_save_payload`/`_load_payload`, exact tokens + logits + full per-layer KV, §2); O(1) same-buffer linear rewind (`ds4_session_rewind`, §3); a proven small-window GPU tensor-copy snapshot/restore primitive exercised continuously in production for DSpark verify-reject (`spec_frontier_snapshot/_restore`, §3).
- Missing: (i) no full-context in-memory fork (keep 2+ live divergent branches concurrently) — would need generalizing the `spec_frontier_snapshot` per-layer `ds4_gpu_tensor_copy` loop from the narrow spec window to the whole per-layer span, and a second `ds4_session`'s buffers (or an unused slot) as the copy target; (ii) no forkpoint bookkeeping (named/id'd checkpoints with refcounting/GC) — today "checkpoints" are either "the one live state per slot" or "a disk file addressed by exact rendered-text SHA1," neither of which models a tree of speculative branches sharing a common ancestor prefix; (iii) disk round-trip is a viable *today* fork substitute (write to file, load into an idle slot) but pays full serialize+deserialize I/O cost rather than a device-side memcpy.
- Size estimate: **structural-adjacent but small** — the low-level copy primitive (`ds4_gpu_tensor_copy` per layer) already exists and is battle-tested at small scale; extending it to full-context copies plus adding a forkpoint table (id → per-layer tensor snapshot, refcount, LRU/GC) is a bounded, well-scoped addition, not a rearchitecture. The CPU-backend path would need the equivalent for `ds4_kv_cache cpu_cache` (not audited in as much depth — flagged as a follow-up).

(2026-08-03 follow-up, MEASUREMENTS.md "Determinism/identity probe" unit: this narrow-scope observation above turned out not to be where the CASE B greedy-identity bug lives; direct code trace confirmed snapshot/restore faithfully copies exactly what it captured. The actual root cause is upstream of the copy: the batch-verify pass computes accepted tokens compressed-KV compressor-frontier values via a different GEMM/kernel code path than ordinary single-token decode (ds4_gpu_matmul_f16_pair_tensor batched vs. ds4_gpu_matmul_f16_pair_compressor_store_tensor fused-single-token), so the values being snapshotted/restored are themselves not bit-identical to what pure decode would have written. See MEASUREMENTS.md CASE B for the full trace and file:line citations.)


**(c) KV transfer/persistence (disaggregation handoff, session durability).**
- Exists: full binary payload format with magic/version header, engine-portable enough that it round-trips through a file already (§2) — this is most of what's needed for "durability" and even a same-model cross-process handoff (write payload, ship the file, load into a fresh process's session). Distributed/tensor-parallel sessions already have a save/load variant (`ds4_dist_session_save_payload`, referenced `ds4.c:50572`), suggesting some cross-process awareness already exists for TP, worth a deeper follow-up read.
- Missing: (i) no network transport for the payload — today it's file-based only (`ds4_kvstore` writes/reads local files under `--kv-disk-dir`); shipping a payload to a different physical node for disaggregated prefill/decode handoff would need a wire protocol wrapping the existing serializer, not a new serializer. (ii) versioning is present (`DS4_SESSION_PAYLOAD_VERSION`) but no visible compatibility/migration logic was found for reading an older version — a persistence roadmap spanning ds4 upgrades would need that audited further. (iii) `model_id`/`quant_bits` gating on load (§2) means a snapshot is not portable across model variants or quant configs by design — expected for correctness, but a "transfer" story needs to define what's portable (same-model-only) up front.
- Size estimate: **wiring-class for same-node persistence** (the hard part — serialization format — already exists and works); **structural for true cross-node transfer** (needs a transport layer, and probably explicit compatibility contracts) but builds directly on top of the existing payload format rather than replacing it.

---

## Notes / follow-ups not covered here (out of scope for this read-only pass)

- Request-to-slot routing/dispatch logic (which slot a given client's request lands on) was not located in the audited window of `ds4_server.c`; it materially affects the live-hit-rate story in §1/§6(a) and is worth a targeted follow-up grep.
- CPU-backend (`ds4_kv_cache cpu_cache`) fork/rollback equivalents were not traced in the same depth as the GPU path in §3.
- `ds4_dist_session_save_payload` (TP/distributed session save) was noted but not read in detail — potentially directly relevant to (c).
