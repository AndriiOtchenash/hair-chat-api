# 0004 — Input moderation with OpenAI omni-moderation; fail-open when it is down

Date: 2026-10-19 · Status: accepted

## Context
Every user message goes to the chat chain and is stored. Messages that express self-harm need
a different answer than hair advice (help lines in the user's language), and abusive content
should not reach the model or the history. A moderation call adds a network dependency in
front of every reply, so we have to decide what happens when that dependency is down.

## Decision
- **OpenAI `omni-moderation-latest`** (free, multilingual, covers ru/uk/pl/en), key
  `OPENAI_API_KEY`. US-hosted, so CLAUDE.md's China rule holds; OpenAI is one more processor
  of user messages and must be named in the privacy policy and the AI consent text.
- **Before anything is saved or streamed**, in the `start_reply` dependency like 404/409:
  flagged `self-harm*` → 422 `message_blocked_self_harm` (the client shows help lines);
  any other flagged category → 422 `message_blocked`. A blocked message is not saved; the log
  gets the code and the categories only, no text and no user id.
- **Fail-open**: on timeout (3 s), 429, 5xx or a network error the message goes through and
  is stored with `messages.moderation = {"status": "unavailable"}` (warning in the log).
  401/403/400 mean our key or request is wrong: also fail-open, but logged as error.
  A checked message stores `{provider, model, flagged: false}`; scores are never stored.
- No SDK retries (`max_retries=0`): a retry would delay the start of every answer by seconds.
- No key: `NoopModerator`, messages stored with `{"status": "disabled"}`, a startup warning.
  `ENV=prod` refuses to start without the key.

## Why fail-open
Fail-closed (block everything while moderation is down) turns any OpenAI outage into a full
outage of our chat, and a person in crisis would get an error instead of a reply. For us the
moderation is an extra layer, not the only one:
- the product gives cosmetic advice only; the system prompt forbids diagnoses, dosages and
  treatment plans and escalates to a trichologist/doctor (CLAUDE.md rule 8);
- the chat models have their own safety filters;
- an answer is seen only by the user who asked; nothing is published to other users.
Outages are short, and every unchecked message is marked `unavailable`, so it can be reviewed
later. Fail-closed is the right choice where unchecked content is published to others or
triggers real-world actions; neither is the case here.

## Consequences
- During an OpenAI outage self-harm messages reach the chat model without the help-line
  screen; the system prompt's escalation rules are the remaining guard.
- Moderation adds ~100–300 ms before the first token (up to 3 s when it times out).
- The default per-category thresholds of the endpoint give false positives on hair topics
  (a scalp burn from bleach is flagged as self-harm/violence). Tuning with our own
  per-category score thresholds is a follow-up decision, not part of this ADR.
- Text only. If image upload arrives, omni-moderation also takes images (same call).
