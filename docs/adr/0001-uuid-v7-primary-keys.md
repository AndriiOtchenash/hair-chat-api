# 0001 — UUID v7 primary keys generated in the application

Date: 2026-10-15 · Status: accepted

## Context
Every table needs a primary key. Ids are exposed in URLs (`/conversations/{id}`) and must not be
guessable or leak row counts, so sequential integers are out. Random UUID v4 fragments B-tree
indexes because inserts land on random pages. We want ids known before the INSERT (to link rows
in one unit of work and log ids without a round trip).

## Decision
- PK type: Postgres `uuid`, value is **UUID v7** (RFC 9562): 48-bit Unix-ms timestamp prefix +
  random bits. Time-ordered, so inserts append to the right edge of the index like a sequence.
- Generated in Python, not by the DB (`app/db/base.py::new_id`, used as the column `default`).
- Library: **`uuid-utils`** (`uuid_utils.compat.uuid7`). Python 3.12 has no `uuid.uuid7`
  (added in 3.14). uuid-utils is actively maintained, Rust-backed, typed, and its `compat`
  module returns the stdlib `uuid.UUID`, so SQLAlchemy/asyncpg/Pydantic need no adapters.
  Rejected: `uuid6` (pure Python, slower, less active); Postgres-side generation (PG 16 has no
  native uuidv7 — arrives in PG 18 — and would need an extension).

## Consequences
- Ids leak creation time (ms). Acceptable: created_at is not secret here.
- Moving to Python 3.14: replace the import in `new_id()` with `uuid.uuid7` and drop the dependency.
- Rows inserted with raw SQL must supply an id (no DB default).
