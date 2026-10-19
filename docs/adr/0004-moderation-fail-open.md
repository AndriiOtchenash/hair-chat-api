# 0004 — Input moderation: allow / soft / block, fail-open when moderation is down

Date: 2026-10-19 · Status: accepted

## Context
Every user message goes to the chat chain and is stored. Messages that express self-harm need
care, not hair advice alone; explicit threats and abuse should not reach the model or the
history. A moderation call adds a network dependency in front of every reply, so we also have
to decide what happens when it is down.

A first version blocked every flagged message and gave self-harm its own 422 code. A live
check showed the problem: hair-care talk is full of burns, cuts and bleeding scalps.
"I burned my scalp with bleach" was flagged self-harm (0.70) and violence (0.52), so a user
with a chemical burn would get an error and a crisis card instead of first-aid advice.

## Decision
- **OpenAI `omni-moderation-latest`** (free, multilingual, covers ru/uk/pl/en), key
  `OPENAI_API_KEY`, 3 s timeout, no SDK retries. US-hosted, so CLAUDE.md's China rule holds;
  OpenAI is one more processor of user messages and must be named in the privacy policy and
  the AI consent text.
- Runs in `ChatService.reply` after the 404/409 checks, before the message is saved.
  Policy in `app.services.moderation_policy.decide`, three actions:
  - **soft**: any `self-harm*` category flagged. The message is saved and answered. The model
    gets a `<moderation_signal>` block right after the system prompt (answer gently, in the
    user's language, no moralizing, suggest support, still answer the hair question), and the
    SSE stream starts with `safety_notice {kind: "self_harm"}` so the client shows help lines.
    Soft wins over every other category: blocking a person in crisis leaves them with an error.
  - **block**: `violence` / `violence/graphic` only at score >= `MODERATION_VIOLENCE_THRESHOLD`
    (0.9) or together with another blocking category such as `harassment/threatening`; any
    other flagged category blocks as flagged by OpenAI. 422 `message_blocked`, not saved.
  - **allow**: everything else, including violence below the threshold.
- Logs carry the action and categories only: no text, no scores, no user id.
- `messages.moderation`: `{provider, model, flagged, action, categories?}` (`flagged` is
  OpenAI's flag, `action` is ours), `{status: "unavailable"}` or `{status: "disabled"}`.
  Scores are read for the threshold but never stored.
- **Fail-open**: on timeout, 429, 5xx or a network error the message goes through, stored as
  `unavailable` (warning). 401/403/400 mean our key or request is wrong: also fail-open, but
  logged as error. No key: `NoopModerator` in dev (startup warning); prod refuses to start.
- **Eval set**: `tests/moderation_eval/phrases.yaml` (our own phrases, ru/uk/pl/en, expected
  action each), run against the real API with `uv run python -m scripts.moderation_eval`.
  Not in CI. Re-run after any policy or threshold change and when OpenAI updates the model.
  A phrase may carry `known_gap: "<reason>"`: it is listed apart and does not fail the run;
  only new mismatches give exit code 1.

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
- During an OpenAI outage there is no safety notice and no signal block; the system prompt's
  escalation rules are the remaining guard.
- Moderation adds ~100–300 ms before the first token (up to 3 s when it times out).
- The model is told about the signal in `system_v3`: unlike data blocks, its guidance is
  followed; a `<moderation_signal>` typed inside a user message is not genuine.
- Text only. If image upload arrives, omni-moderation also takes images (same call).

## Known gaps
First eval run (2026-10-19, threshold 0.9): 39/46 as expected, the 7 misses are marked
`known_gap` in phrases.yaml and accepted for now:
- **Accidental injuries and self-harm hyperbole → soft** (4): "I burned my scalp with bleach",
  "I cut myself with the razor…", "я себя убью, лол". Decided: soft stays on any self-harm
  category. Missing a crisis costs more than a care card shown to someone who does not need it;
  gating soft on `self-harm/intent` would have fixed only 2 of the 4 anyway.
- **Hyperbole vs real threats.** Violence scores do not separate them: real threats scored
  0.69 and 0.85, hyperbole 0.64 and 0.77, so no threshold fits both. Decided: keep 0.9; the
  two threats pass to the chat model, which refuses harmful requests on its own, and the answer
  is seen by nobody else. "I could murder my hairdresser" is blocked as `harassment`, which
  blocks as flagged; left as is.
- **Planned for stage 3: an LLM classifier for the grey zone.** Messages with a violence or
  harassment score in a grey band (about 0.6–0.9) go to a cheap model of the chain with a
  narrow prompt ("real threat to a person, or hyperbole about hair?"). Costs one more call and
  some latency for a small share of messages; needs its own eval run on this set.
