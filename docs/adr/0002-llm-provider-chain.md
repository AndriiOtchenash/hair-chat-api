# 0002 — LLM provider chain: Gemini + Groq via OpenAI-compatible API, fallback with cooldown

Date: 2026-10-19 · Status: accepted

## Context
Answers must keep streaming when one provider is slow, rate-limited or down. Free/cheap tiers
(Gemini, Groq) fail often (429, 5xx). Providers must stay stateless (CLAUDE.md rule 2), so any
link can take over with the same full context. Each SDK per provider would multiply code paths.

## Decision
- **One implementation**, `OpenAICompatibleProvider` (openai SDK, async), parametrized by
  base_url / key / model. Gemini: `https://generativelanguage.googleapis.com/v1beta/openai/`,
  Groq: `https://api.groq.com/openai/v1`. Groq joins the stack as the non-Google link.
- **`LLM_CHAIN`** = `provider:model,...` in priority order. Every model is listed in
  `app/llm/registry.py` with its `max_context_tokens` (from the providers' `GET /models`);
  unknown model or missing key fails at startup. Context budget = min over the chain (rule 3).
  Initial chain: `gemini:gemini-3.8-flash, gemini:gemini-3.5-flash-lite, groq:openai/gpt-oss-120b`
  (pinned versions, not `*-latest` aliases). Thinking models run with `reasoning_effort=low`.
- **Fallback (ResilientLLM)** on connect / first-token / read timeouts, dropped connections,
  408/429/5xx, 401/403/404 (link misconfigured, logged as error; includes Gemini's bad-key 400,
  recognised by its message). **No fallback** on 400/413/422: our request is wrong, every
  provider would reject it. A failure after the first token emits `Restart`, then the next link
  generates the full answer.
- SDK retries are off (`max_retries=0`): the chain owns the retry policy.
- **Cooldown**, not a circuit breaker: 3 consecutive failures -> link skipped 60 s, in-memory
  per process; 400s do not count.
- Every attempt goes to `llm_calls` (status ok/fallback/error, error code only, no text);
  best effort with a 2 s cap, a DB outage never breaks the answer.

## Consequences
- Adding a provider = registry entry + key in Settings, if it speaks the OpenAI protocol.
- openai SDK v3 runs on `httpx2`, which we import directly (it is a transitive dependency for now).
- Each worker learns about a failing link separately (no shared cooldown state); acceptable
  without Redis (CLAUDE.md: no Celery/Redis).
- Free-tier Gemini may use prompts for training: prod must use a paid-tier key.
- Groq's free-tier TPM is far below its 131k window: under load expect 429 -> fallback.
