# research/gb10/traces/

Routing traces captured via `DS4_ROUTING_TRACE=<path>` (see `ds4_cuda.cu`,
`ds4_gpu_stream_expert_cache_begin_selected_load` / `..._prepare_selected_batch`),
used by `research/gb10/locality_sim.py` and written up in
`research/gb10/LOCALITY_STUDY.md`.

## Format

Flat sequence of variable-length little-endian binary records:

```
u32 seq            monotonic record counter (global write order)
u32 token_index    per-phase token index (decode: increments once per decode
                    step; prefill: position within the current prefill chunk)
u8  phase           0 = prefill, 1 = decode
u8  layer           MoE layer index
u8  n_selected      number of expert ids that follow
u8  _pad
i16 expert_ids[n_selected]
```

No routing weights are recorded (out of scope for the hit-rate/locality study
this trace format was built for; only expert identity matters for cache
simulation). Buffered writes, no per-record fsync -- negligible overhead
relative to a streamed-MoE decode/prefill step (see the doc comment at the
hook site in `ds4_cuda.cu` for the exact overhead argument).

## Files (raw traces gitignored above 10MB; see `.gitignore`)

- `eval12.bin` -- the 12-item `ds4-eval` subset (GPQA/SuperGPQA/AIME2025),
  `--cuda --ssd-streaming --ssd-streaming-cache-experts 100GB`, same
  invocation as MEASUREMENTS.md's "MXFP4-streaming quality eval" entry.
- `long_essay.bin` -- a single ~1000-token `--nothink` essay-style generation.
- `multiturn.bin` -- a several-prompt interactive session (`./ds4` REPL,
  piped stdin, one process / one KV+expert-cache session across all turns).
- `sample_*.bin` -- a small, git-committed excerpt of one of the above (first
  N records) for anyone who wants to sanity-check the parser/simulator
  without pulling the full raw traces.

## Regenerating

```
sudo systemctl stop ds4-server
DS4_ROUTING_TRACE=research/gb10/traces/<name>.bin ./ds4 --cuda -m gguf/DeepSeek-V4-Flash-MXFP4_MOE.patched.gguf \
    --ssd-streaming --ssd-streaming-cache-experts 100GB --nothink -p "..." -n <N>
sudo systemctl start ds4-server
```
