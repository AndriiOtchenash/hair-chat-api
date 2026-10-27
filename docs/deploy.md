# Deploy notes

## Staging on Fly.io (`hair-chat-api-stg`)

`fly.toml` in the repo root describes staging: `ENV=staging` (prod checks: Auth0, moderation,
every LLM_CHAIN key and GEMINI_API_KEY are required, `app.cli.chat` refuses to run) plus
`KB_INCLUDE_DRAFTS=true`. One `shared-cpu-1x`/512 MB machine in `ams` that suspends when idle
and wakes on the next request. Migrations run as the `release_command` before each release.

### Private only: Flycast

The app has no public IP, only a private Flycast IPv6. It is reachable from the org's private
network (WireGuard, other Fly apps), never from the internet. Flycast is plain HTTP, so
`force_https` stays false, and Fly Proxy reaches the machine over its private IPv4: uvicorn must
listen on `0.0.0.0:8000` (the Dockerfile CMD does; `fly-local-6pn` would not work for Flycast).

```sh
fly apps create hair-chat-api-stg --org personal
fly ips allocate-v6 --private -a hair-chat-api-stg
fly ips list -a hair-chat-api-stg   # only "private ingress" v6, nothing public
```

`fly deploy` allocates public IPs on the first deploy of an app with `[http_service]`, unless
`--no-public-ips` is passed. If a public address shows up in `fly ips list`, remove it with
`fly ips release <address> -a hair-chat-api-stg`.

Reach it from the codespace:

```sh
# Through Flycast (port 80 -> internal 8000; wakes a suspended machine): http://localhost:8000/health
fly proxy 8000:80 hair-chat-api-stg.flycast -a hair-chat-api-stg
# or, with a WireGuard peer (fly wireguard create): http://hair-chat-api-stg.flycast/health
```

### Secrets

Values come from the codespace env, through stdin, so they are neither printed nor in argv:

```sh
{ for k in GEMINI_API_KEY GROQ_API_KEY OPENAI_API_KEY AUTH0_DOMAIN AUTH0_AUDIENCE; do
    printf '%s=%s\n' "$k" "${!k}"; done; } | fly secrets import --stage -a hair-chat-api-stg
```

`DATABASE_URL` is the Neon connection string as Neon shows it
(`postgresql://...?sslmode=require&channel_binding=require`); Settings turns it into
`postgresql+asyncpg://...?ssl=require` for the app and Alembic. `read -rs` keeps it off the
screen and out of the shell history:

```sh
read -rs NEON_URL && printf 'DATABASE_URL=%s\n' "$NEON_URL" | fly secrets import --stage -a hair-chat-api-stg; unset NEON_URL
fly secrets list -a hair-chat-api-stg   # names and digests only
```

`--stage` only stores them; the next `fly deploy` (or `fly secrets deploy`) applies them.

### Deploy

```sh
fly deploy -a hair-chat-api-stg --ha=false --no-public-ips
```

The image is built remotely, `alembic upgrade head` runs in a temporary machine with the app's
secrets (non-zero exit = deploy aborted, the old release keeps serving), then the machine is
replaced. If a required key is missing the app fails at startup, the `/health` check never
passes and the deploy fails.

### Knowledge base ingest

After a deploy that changed `docs/kb` (the image carries the articles):

```sh
fly machine list -a hair-chat-api-stg          # wake a suspended one: fly machine start <id>
fly ssh console -a hair-chat-api-stg -C "python -m app.rag.ingest docs/kb"
```

Idempotent: unchanged articles cost no embeddings.

### Rollback

```sh
fly releases -a hair-chat-api-stg --image      # versions with their image references
fly deploy -a hair-chat-api-stg --image registry.fly.io/hair-chat-api-stg:<tag> --no-public-ips
```

The release command runs again, but `alembic upgrade head` of an older image does not undo
newer migrations. If the old code cannot work with the new schema, downgrade first:
`fly ssh console -a hair-chat-api-stg -C "alembic downgrade <revision>"`.

### Logs

```sh
fly logs -a hair-chat-api-stg                  # live; --no-tail for recent lines only
fly status -a hair-chat-api-stg                # machines, health checks, release
```

## Known issues

### Gemini: "User location is not supported" from Fly `ams` (2026-10-27)

From the staging machine in `ams` every Gemini call (chat and embeddings) answers
`400 FAILED_PRECONDITION: User location is not supported for the API use.` The key is fine (the
same key works from the codespace); Google rejects the Fly shared egress IP
(`216.246.119.90`, AS30081, which ipinfo places in Amsterdam, NL), so its own geolocation of that
range evidently differs. Shared egress IPs belong to the region and may change without notice.

- Effect: the `gemini:*` links fail -> since `fix: provider-side 400s`, ResilientLLM falls back
  to `groq:openai/gpt-oss-120b` and logs `llm attempt failed ... code=BadRequestError:400:location_unsupported`
  at error level (then the link cools down). Embeddings fail the same way: answers go without
  `<knowledge>` (error log), and `app.rag.ingest` exits with
  `embeddings unavailable (BadRequestError:400:location_unsupported)`.
- Checked: a temporary machine in `fra` (egress `204.93.236.11`, DE) gets answers for chat
  (`gemini-3.5-flash-lite`) and embeddings. Probe a region without redeploying:

  ```sh
  fly machine run registry.fly.io/hair-chat-api-stg:<tag> -a hair-chat-api-stg \
    --region fra --vm-size shared-cpu-1x --rm --restart no -- python -c "<probe>"
  fly logs -a hair-chat-api-stg --machine <id> --no-tail
  ```

- Workarounds for staging: run in a region whose egress Google accepts (`fra` today), or a static
  egress IP (`fly machine egress-ip allocate`, paid) checked the same way. Neither is guaranteed
  to last: the egress geolocation is outside our control.
- Prod: plan Gemini via **Vertex AI in an EU region** (`europe-west1/3/4`): regional endpoints,
  no AI Studio location check, data processed in the EU, service-account auth instead of an API
  key. That changes the LLM and embedding providers (endpoint, auth, possibly the embedding
  model name -> re-ingest), so it needs its own ADR before prod. Not started.

## Daily retention cleanup on Fly.io (ADR 0006)

`python -m app.cli.retention` deletes conversations idle for longer than `RETENTION_DAYS`
(default 30). It runs as a separate **scheduled Machine** in the API's Fly app, from the same
image, and exits when done. There is deliberately no HTTP endpoint for it.

### Create the machine

```sh
# The image the app runs now (look for the image reference in the output)
fly image show -a hair-chat-api

# Check first: counts only, deletes nothing; --rm destroys the machine after it exits
fly machine run registry.fly.io/hair-chat-api:<tag> \
  -a hair-chat-api --region <app region> --vm-size shared-cpu-1x --rm \
  -- python -m app.cli.retention --dry-run

# The daily job
fly machine run registry.fly.io/hair-chat-api:<tag> \
  -a hair-chat-api --region <app region> --vm-size shared-cpu-1x \
  --schedule daily --name retention-daily \
  -- python -m app.cli.retention
```

Flags go before `--`: everything after it is the command (otherwise flyctl would try to
parse `--dry-run` as its own flag).

- The machine gets the app's secrets (`DATABASE_URL`, `ENV`, ...) as env vars, like the API
  machines. A secret set *after* the machine was created reaches it only once the machine is
  updated (`fly machine update`).
- `--schedule` accepts `hourly`, `daily`, `weekly`, `monthly` and is fuzzy (an approximate
  interval, no fixed time of day). That is fine here: being a few hours late is harmless.
- The first run starts right after `fly machine run`. Scheduled machines restart only
  `on-failure` by default, so a clean exit (code 0) waits for the next slot.
- Do not set `fly_process_group` metadata on it: then `fly deploy` would treat it as an app
  machine.

### After each deploy

`fly deploy` ignores machines created with `fly machine run`, so the job keeps the old image.
Usually harmless (the CLI reads the DB only), but after a release that changes the schema or
the retention code, move it to the new image:

```sh
fly machine list -a hair-chat-api            # id of retention-daily
fly machine update <id> -a hair-chat-api --image registry.fly.io/hair-chat-api:<new tag>
```

### Check it ran

```sh
fly logs -a hair-chat-api --machine <id>
# retention (30 days, before 2026-09-22T04:10:00Z): deleted conversations=12 messages=230 orphan_llm_calls=40
```

Output is counts only, never message text. Running two copies at once is safe (rows are
taken with `FOR UPDATE SKIP LOCKED`), so a manual run next to the schedule needs no
coordination.
