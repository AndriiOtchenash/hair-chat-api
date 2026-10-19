# 0003 — Chat answers stream over SSE; one generation per conversation; only full answers saved

Date: 2026-10-19 · Status: accepted

## Context
`POST /conversations/{id}/messages` returns an LLM answer that takes seconds to generate. The
client should see tokens as they come, survive a provider switch mid-answer (ADR 0002 `Restart`)
and get REST-like errors. A user may double-submit or close the tab mid-answer.

## Decision
- **SSE, not WebSocket.** One request → one stream of server events (`token`, `restart`,
  `done`, `error`; contract in CLAUDE.md). It is plain HTTP: the same Auth0 bearer token,
  consent gate, status codes (404/409/422 before the stream starts) and error body as REST,
  works through proxies and CDNs. WebSocket would add a second protocol with its own auth,
  reconnect and sticky-session concerns for a two-way channel we do not need.
- FastAPI's built-in `EventSourceResponse`: `: ping` comment every 15 s while idle,
  `Cache-Control: no-cache`, `X-Accel-Buffering: no` (nginx/fly proxies do not buffer).
- Checks that must be real HTTP statuses run in a yield dependency (`start_reply`) before the
  first byte; its teardown runs after the stream, including on client disconnect.
- **Only a complete answer is saved**, when `done` is reached. A client disconnect cancels the
  generation (Starlette cancels the stream task; the provider HTTP stream is closed at once,
  not left to the GC) and leaves only the user's message. A partial answer would enter the
  context of later turns and can lose a caveat mid-sentence. `llm_calls` of the generation are
  linked to the saved assistant message by the ids ResilientLLM generated for them.
- **One generation per conversation**: a second POST while an answer streams → 409
  `conversation_busy`. The lock is an in-memory set per process (`ConversationLocks`).

## Consequences
- **The lock only works with a single app instance.** With several instances (or several
  uvicorn workers) two POSTs can land on different processes. Then it needs a shared
  mechanism: a Postgres advisory lock keyed by the conversation id held for the stream (costs
  one connection per active stream), or a `generating_since` column claimed with a
  conditional UPDATE plus a timeout for crashed holders.
- A disconnect wastes the tokens generated so far; the user re-sends (their message is kept).
- `last_message_at` moves on the user's message and on the answer, so a conversation whose
  generation failed still rises in the list.
