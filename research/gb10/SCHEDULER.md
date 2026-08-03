# ds4-server concurrent-session scheduler (task 12 design note)

Status as of 33e2b3d. Scope: `--batched-session N` mode; without the flag the
server is single-slot and fully serialized (one worker, `inference_mu`).

## What exists (mostly predates task 12 -- the old "fully serialized" audit drifted)

### Request -> slot dispatch
- `enqueue`/`dispatch_jobs_locked` (ds4_server.c ~12079): FIFO among runnable
  jobs; slot chosen by longest common token prefix (session affinity, task 16),
  with live tool-state bindings forcing a specific slot. One worker thread per
  slot (`slot_worker_main`).

### Prefill-quantum interleaving
- `server_session_sync` (~10301) prefill runs in quanta: 2048 tokens when no
  generation is active, 128 when one is (`DS4_SERVER_PREFILL_QUANTUM`,
  `DS4_SERVER_MIXED_PREFILL_QUANTUM`).
- Between quanta the model is released; `server_prefill_enter` (~10237)
  round-robins among waiting slots (`server_next_prefill_slot_locked`) and
  yields to any pending decode (`decode_pending > 0` blocks prefill entry).
  A 25k prefill therefore stalls another session's decode by at most one
  128-token quantum. No starvation: strict round-robin + FIFO dispatch.

### Batched decode
- `server_eval_token` (~10820) posts the token to a dedicated decode worker.
- `decode_worker_main` (~10874) coalesces pending decodes for up to
  `DS4_SERVER_DECODE_COALESCE_US` (default 2000) while fewer than
  min(slot_count, active_generations) are pending, then calls
  `ds4_sessions_eval_batch` (ds4.c ~61911) under `inference_mu`.
- CUDA path (`ds4_sessions_eval_batch_cuda`, ds4.c ~65071): all sessions'
  EXACT one-token kernels encoded in one command submission (pipeline-encoded
  when `graph.placement` is set; `DS4_CUDA_SESSION_BATCH_INTERLEAVE=0` forces
  sequential encode). Because the per-session kernels and KV writes are the
  same as solo decode, greedy identity is preserved by construction
  (verified byte-identical on robo-dog, 3 sessions, temp 0).
- Preconditions for the CUDA batch path: DeepSeek (non-GLM), no drafter
  (`support_kind == NONE`), GPU sessions. GLM / drafter configs fall back to a
  serial per-session loop (logically all-or-nothing via invalidation).

## Added by task 12
- `DS4_SCHED_BATCH_DECODE=0` (33e2b3d): forces the decode worker to take one
  session per model pass and skips the coalesce wait -- interleave-only mode
  for A/B measurement and operator opt-out. Default on (unchanged behavior).
  Startup line now reports `batch_decode=`.

## Measured result (see MEASUREMENTS.md, task 12 section)
Aggregate tokens/hour is FLAT (~12.6-13.1k) across serialized / interleave /
interleave+batch with 3 concurrent sessions; solo is ~13.2k. Batch-3 decode
steps cost ~3.16x a solo step: the batch shares command submission but not
weight reads. The 2x aggregate bar is not met; the win today is fairness
(simultaneous progress, all sessions ~equal wall) and bounded prefill stalls,
at unchanged aggregate throughput.

## Deferred
- Fused multi-sequence decode kernels: batched expert GEMV over all sessions'
  tokens per (layer,expert) with per-sequence KV, deduplicating weight reads
  (grouped-prefill a627e80 approach applied to decode). This is the unit that
  could actually move aggregate tok/hr.
- Wiring `ds4_sessions_eval_batch_with_prefill` (library-only today) into the
  server for mixed prefill+decode passes.
- Watch #658-class numerics if fused batched decode kernels are built: batched
  and single-token paths must stay byte-identical or the greedy-identity
  guarantee breaks.

## Production
robo-dog service runs `--batched-session 2` with defaults (interleave + batched
decode active -- this was already HEAD behavior before task 12). Opt-out:
`Environment=DS4_SCHED_BATCH_DECODE=0` in the unit.
