# Deploy notes

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
