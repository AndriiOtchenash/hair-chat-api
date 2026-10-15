# Consent versions

The consent texts (18+, health data, AI processing) live in the frontend repo `hair-chat-web`.
The backend stores only **which version** of those texts a user accepted, in `consents.version`.

## What a version is

`CONSENT_VERSION` (env, default in `app/core/config.py`) names one exact revision of the texts:
a git tag in `hair-chat-web`, e.g. `consents-2026-10-v1`, whose value without the prefix is the
version string (`2026-10-v1`). If there is no tag yet, use the short commit hash of the texts.
The tag lets anyone answer "what exactly did this user agree to on that date?".

## Rule: changed a text -> bump the version

Any change to a consent text that changes its meaning (new data use, new provider, new
purpose) needs a new version:

1. Change the texts in `hair-chat-web`, tag the commit (`consents-YYYY-MM-vN`).
2. Set the frontend's version constant to the same value.
3. Deploy the backend with `CONSENT_VERSION=YYYY-MM-vN`.

After step 3 every older consent stops counting: `GET /consents` shows `granted: false`,
chat endpoints answer `403 consent_required`, and the frontend asks the user to accept again.
`POST /consents` with any other version is rejected with `400 consent_version_mismatch`, so a
stale frontend cannot record consent to a text the backend no longer considers current.

Typos and formatting fixes that do not change meaning may keep the version; when in doubt, bump.

## Revoking is not deleting

`DELETE /consents/{type}` sets `revoked_at`; the row stays as evidence that earlier processing
was lawful. Deleting the user's data is a separate flow (retention / account deletion).
