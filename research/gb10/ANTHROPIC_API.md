# Anthropic API surface — verify-and-complete for claude CLI (2026-08-03)

Goal: real `claude` CLI (Claude Code) session working end-to-end against ds4-server
on robo-dog (DeepSeek-V4-Flash MXFP4, --cuda --ssd-streaming, ctx 32768).
Result: ACCEPTED with zero code changes — the surface in ds4_server.c at 42ecea7
already implements everything the CLI needs. This file is the audit + evidence.

## Audit inventory (ds4_server.c @ 42ecea7)

| Requirement | Status | Where |
|---|---|---|
| POST /v1/messages routing | WORKS | route at ~12411, parse_anthropic_request ~3121 |
| system as string or content blocks | WORKS | parse_anthropic_system ~2017 (x-anthropic-* parts filtered) |
| content blocks: text / tool_use / tool_result / thinking | WORKS | parse_anthropic_content_block ~1744 |
| tools array with input_schema | WORKS | parse_tools_value (shared, handles both wrappings) ~1601 |
| SSE sequence message_start → content_block_* → message_delta → message_stop | WORKS | anthropic_sse_* ~7504-8124; `: prefill` keepalive comments during prefill |
| text_delta + input_json_delta streaming | WORKS | ~7621, ~7646 (tool args streamed as real JSON fragments) |
| stop_reason mapping | WORKS | anthropic_stop_reason ~7354: tool_calls→tool_use, length→max_tokens, else end_turn. `stop_sequence` value never emitted (always null) — CLI does not depend on it |
| usage tokens | WORKS | append_anthropic_usage_json ~7415 — proper Anthropic semantics: input_tokens excludes cache_read/creation portions |
| x-api-key / Authorization bearer / anthropic-version | WORKS | server enforces no auth; all headers tolerated |
| unknown fields (thinking, metadata, cache_control, context_management, output_config) | WORKS | json_skip_value fallback in the key loop; `thinking` explicitly parsed and honored |
| tool_result live continuation | WORKS | anthropic_prepare_live_continuation ~2907 / anthropic_live_* ~8575 — matching tool_use_ids resume the live KV instead of re-prefilling |
| POST /v1/messages/count_tokens | MISSING (tolerable) | 404 `{"error":{...}}`; claude CLI degrades gracefully — no observed impact |
| Anthropic-shaped error bodies | PARTIAL | context-length + SSE errors are Anthropic-shaped (`{"type":"error",...}`); generic http_error (400/404) uses OpenAI shape. CLI displays the message fine (observed: "API Error: 400 Prompt has ... tokens") |

No gaps required code. No commits to server code; this doc is the only change.

## Live smokes (robo-dog, production, litellm untouched)

- Non-stream /v1/messages with system + metadata + thinking: 200, correct message shape.
- Stream with tools: exact event sequence observed incl. thinking_delta blocks,
  tool_use content_block_start with id/name, input_json_delta fragments
  (`{`, `"city":"`, `Paris`, `"`, `}`), content_block_stop, message_delta
  stop_reason=tool_use, message_stop.
- tool_result round-trip (assistant tool_use replay + user tool_result): correct
  final text; KV cache hit on the replayed prefix (cache_read_input_tokens=413).
- /v1/chat/completions still 200 with usage (litellm path intact).
- /v1/models 200, advertises deepseek-v4-flash + deepseek-v4-pro.

## claude CLI acceptance (croft → 100.92.111.3:8000)

    env ANTHROPIC_BASE_URL=http://100.92.111.3:8000 ANTHROPIC_AUTH_TOKEN=dummy \
        ANTHROPIC_MODEL=deepseek-v4-flash ANTHROPIC_SMALL_FAST_MODEL=deepseek-v4-flash \
        claude -p "list the files in the current directory, then tell me which is largest" \
        --allowedTools "Bash(ls:*)" --strict-mcp-config --mcp-config '{"mcpServers":{}}'

Passed: CLI streamed, emitted a Bash tool_use, ran `ls`, returned the tool_result,
and answered correctly ("The largest is big.bin at 5,000 bytes"). Second probe
`claude -p --continue "total size of both files combined"` also passed
("Total combined size: 5,006 bytes") — the follow-up turn restored 24213 tokens
from the KV disk cache and only prefilled the delta; tool_result turns used the
live-KV continuation (17-token delta prefill observed in the journal).

- Model naming: CLI accepts non-claude model ids via ANTHROPIC_MODEL — no alias
  route needed. ANTHROPIC_SMALL_FAST_MODEL must also be set or the CLI calls a
  haiku id for background turns.
- Rates observed: bulk prefill ~29 t/s (2048-token chunks), small-delta prefill
  ~2 t/s, decode ~4.4 t/s. First cold turn on a Claude-Code-sized prompt
  (~25k tokens) ≈ 15 min wall; later turns ride the KV cache.

## Caveats / recommendations

1. ctx 32768 is the binding constraint, not the API. shadow's full croft config
   (75 built-in+MCP tools ≈ 200 KB body ≈ 50k tokens) 400s with
   "Prompt has 46292/50741 tokens, but the configured context size is 32768".
   Works when MCP servers are stripped (--strict-mcp-config, prompt ≈ 25k).
   For daily-driver use raise --ctx (model supports more) or keep MCP light.
2. count_tokens: implement later if something depends on it; a clean 404 is fine
   for claude CLI today.
3. Cosmetic: make generic http_error emit Anthropic-shaped bodies when the
   request targeted /v1/messages.
