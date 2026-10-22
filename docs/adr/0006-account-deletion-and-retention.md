# 0006 — Data rights: export, hard account deletion, 30-day retention

Date: 2026-10-22 · Status: accepted

## Context
Users tell the chat about their hair, scalp and sometimes their health (GDPR art. 9 data,
covered by the `health_data` consent). They need to get a copy of what we store (access and
portability, art. 15/20), have it erased (art. 17), and we should not keep conversations
longer than needed (storage limitation, art. 5(1)(e)). The login itself lives in Auth0, a
separate processor that we can reach only through its Management API.

## Decision
- **Export**: `GET /me/export` returns one JSON file (`Content-Disposition: attachment;
  hair-chat-export-<UTC date>.json`, `Cache-Control: no-store`): the users row, every consent
  row (all versions, revoked ones too), every conversation with its summary and messages
  (role, content, created_at, provider/model of answers, moderation action or status, never
  categories or scores). Every query filters by the current user's id (rule 5). Built in
  memory: with 30-day retention an account is small; streaming can come later if needed.
- **Delete**: `DELETE /me` with `{"confirm": "DELETE"}` (anything else = 422, nothing done)
  hard-deletes the users row; Postgres `ON DELETE CASCADE` removes consents, conversations,
  messages and, through messages, the linked llm_calls. No soft delete: there is no reason
  to keep the data and a soft-deleted row is still personal data. `users.deleted_at` stays in
  the schema, unused by this flow.
- **Auth0 after our commit, best effort**: a background task deletes the Auth0 user via the
  Management API (client credentials, `AUTH0_MGMT_CLIENT_ID/SECRET`, scope `delete:users`;
  404 counts as done). Not configured -> warning, our deletion stands. Auth0 down or
  rejecting -> our deletion is *not* rolled back; an error with `user_id` and `auth_sub` is
  logged so the Auth0 account can be removed by hand. Rolling back would leave the user
  unable to erase their health data because a third party is down.
- **Both endpoints are protected, not consented**: access and erasure rights do not depend
  on the processing consents, so they work after the user revoked them.
- **Retention**: `python -m app.cli.retention [--dry-run]` deletes conversations whose
  `last_message_at` is older than `RETENTION_DAYS` (30), cascading to messages and their
  llm_calls, plus llm_calls never linked to a message (summaries, failed or cancelled
  generations) that are as old. Batches of 500 locked `FOR UPDATE SKIP LOCKED`, one
  transaction each: parallel runs split the work and never count a row twice, re-runs find
  nothing. Prints counts only. No HTTP endpoint: a scheduled Fly machine runs it daily
  (docs/deploy.md), so there is no API surface that deletes other users' data.

## Consequences
- An access token stays valid until it expires (and the Auth0 login too when the Management
  API is not configured or failed): the next request creates a new, empty account. It holds
  none of the old data; we keep no tombstone, since the `auth_sub` would itself be personal
  data.
- An answer still streaming while its user is deleted fails to save (FK) and ends with
  `internal_error`; a summary in flight updates nothing.
- Unlinked llm_calls have no user reference and no text (provider, model, latency, status,
  error code), so they cannot be exported or deleted per user; retention removes them after
  30 days. Logs carry ids only (rule 6).
- A failed Auth0 deletion needs a manual retry from the logs; an automatic retry queue would
  need persistent state that outlives the user row, which we avoid for now.
- Changing `RETENTION_DAYS` changes what users were told: update the privacy policy together.
