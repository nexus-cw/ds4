# KV pinning + fresh-session prefix restore (task #16)

Status 2026-08-03, commit e83d95d on research/gb10. Measurements in
MEASUREMENTS.md ("Task #16" section).

## What was built vs what already existed

The ratified design had three parts. Audit of the current tree showed (a) and
(c) already implemented upstream; only (b) needed code.

- **(a) Restore-before-prefill — already present, verified working.**
  `generate_job` (ds4_server.c:11081) consults the disk store via
  `kv_cache_try_load` whenever every live-cache probe misses, restores the
  longest stored byte-prefix (`ds4_kvstore_find_text_prefix`,
  ds4_kvstore.c:1190) into the slot, and prefills only the tail. Cold
  checkpoints are anchored at the chat task boundary
  (`ds4_kvstore_chat_anchor_pos`: last user marker before the first assistant
  marker), which is precisely the system-prompt + tool-schema prefix shared by
  independent claude CLI sessions. No new code, no new env var; it is default
  behavior gated only by --kv-disk-dir being set. Deviation from the task
  text: no DS4_KV_PREFIX_RESTORE_MIN was added — the existing `min=512`
  (--kv-cache-min-tokens / kv min_tokens option) already plays that role.

- **(b) Pinning — NEW, env-gated, default off.**
  `DS4_KV_PIN_MIN_HITS=N` (positive integer): an *anchor* checkpoint
  (reason cold/evict/shutdown) whose raw on-disk hit count has reached N is
  never selected as a budget-eviction victim. If the store is over budget and
  only pinned entries remain, the store keeps them and logs a warning instead
  of deleting a hot restore anchor (persistence beats budget, per operator
  decree). Raw hits are used deliberately — no half-life decay — so a prefix
  pinned today still restores next week. Continued waypoints are never pinned.
  Code: ds4_kvstore.c (`ds4_kvstore_entry_pinned`, evict loop, env parse in
  `ds4_kvstore_open`), ds4_kvstore.h. Unit test:
  `test_kv_cache_pinning_predicate` in ds4_server.c. Hit counts are already
  persisted in the file header and refreshed on every hit
  (`ds4_kvstore_touch_file`), so no new metadata was added.

- **(c) Slot affinity — already present.** `job_slot_score`
  (ds4_server.c:12070) routes each queued job to the idle slot with the
  longest live common token prefix; live tool-state bindings force their slot.
  Nothing added.

## Persistence / expiry audit (operator requirement)

The disk store applies **no TTL, no LRU sweep, no startup purge**. Entries are
deleted only by:
1. Budget eviction (`ds4_kvstore_evict`, --kv-disk-space-mb, default
   4096 MiB) by score; the 6 h hit half-life decays the *score*, never deletes.
   Pinning (above) exempts hot anchors from this path.
2. Consume-on-hit for entries with tokens > cold_max (30000) — does not apply
   to anchored prefixes at the current ~24k scale.
3. Corrupt/failed payloads on load or prefill failure.

So long-lived prefixes are already the default; a stored prefix restores after
arbitrary idle time and across service restarts (verified: session 3's hit was
served by a freshly restarted process; live slots additionally write
reason=shutdown checkpoints on clean stop).

## Known limits / deferred

- The <60 s fresh-turn target was missed honestly: restore is ~0.2-1 s for a
  340 MiB / 24k-token checkpoint; the residual ~2 min is the ~1.1k-token tail
  (per-session CLAUDE.md system-reminder + user message, after the anchor)
  prefilling at ~8 t/s under expert streaming. Faster short-prefill is
  task #14/#22 work, not KV work.
- Byte-exact prefix matching means a prefix stored by one client config does
  not serve another whose rendering diverges early (observed: foreign-config
  entries missed at ~token 1067). Each distinct config pays one cold session.
- No mid-live-state rewind: hybrid-Mamba state cannot roll back
  (`ds4_session_rewrite_requires_rebuild`), so near-identical prompts that
  diverge before the live frontier restore from the anchor rather than
  trimming live KV. Cost is one tail prefill; accepted.
- Production has pinning OFF (env unset) until the operator opts in; suggested
  starting point `DS4_KV_PIN_MIN_HITS=2`, ideally with a larger
  --kv-disk-space-mb (store is at 3.7 GiB of 4 GiB).
