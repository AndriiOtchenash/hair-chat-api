# hair-chat-api — CLAUDE.md

Backend for a hair-care chatbot (cosmetic advice, NOT medical). FastAPI + PostgreSQL/pgvector.
Sibling repos: `hair-chat-web` (React SPA), `hair-chat-infra` (docker-compose, Portkey config, fly.toml).
Owner background: senior-ish Java/Spring dev, new to Python and LLM apps — explain Python/LLM-specific
idioms briefly in PR descriptions when they differ from Spring conventions.

## Stack (do not change without an ADR in docs/adr/)
- Python 3.12, uv, FastAPI, Pydantic v2, pydantic-settings
- SQLAlchemy 2.0 async (asyncpg), Alembic, PostgreSQL 16 + pgvector
- httpx, openai SDK (used against OpenAI-compatible endpoints: Gemini, Mistral, Bedrock, Portkey)
- Provider chain from env `LLM_CHAIN`: dev = gemini-free,mistral-experiment;
  prod = gemini,mistral[,bedrock]. Tests use FakeLLMProvider only.
- Dev environment: GitHub Codespaces via .devcontainer/ (Python 3.12, uv, Node LTS for Claude Code,
  docker-in-docker for testcontainers). The owner's local PC cannot run Docker — never assume it.
- Never call China-hosted APIs (DeepSeek, Zhipu, Moonshot, Qwen Singapore) with real user data.
- Auth: Auth0 JWT validated via JWKS; roles stored in our DB (user / specialist / admin)
- Background work: FastAPI BackgroundTasks only (no Celery/Redis)
- Tests: pytest, pytest-asyncio, testcontainers (pgvector/pgvector image)
- Lint/types: ruff (lint + format), mypy --strict on app/

## Layout
app/core      config, logging, security (JWT/JWKS, current_user, require_role)
app/api       routers only: request/response models, no business logic
app/services  chat_service, context_builder, profile_extractor, retention
app/llm       LLMProvider protocol, gemini.py, mistral.py, resilient.py, moderation.py
app/rag       embeddings, chunker, retriever, ingest CLI (python -m app.rag.ingest)
app/db        SQLAlchemy models, repositories, session
migrations/   Alembic (async env); URL comes from DATABASE_URL via Settings, not alembic.ini
tests/unit    bulk of tests, fakes instead of mocks where possible
tests/integration  only what Postgres decides: pgvector/tsvector queries, cascades, migrations, SSE e2e

## Non-negotiable rules
1. No test calls a real LLM, embedding or moderation API. Use FakeLLMProvider with scripted scenarios
   (success, fail-before-first-token, fail-mid-stream, timeout).
2. LLM providers are stateless. Never use provider-side conversation state (stored responses,
   interactions/session APIs, context caching). Full context is rebuilt from the DB on every call.
3. Token budget = min(max_context_tokens) across the fallback chain.
4. One embedding model for the whole index; every vector row stores `embedding_model`.
5. Every query on messages/profile/conversations filters by `user_id` of the current user.
   A missing filter is a security bug.
6. Never log message content or profile facts. Log ids, token counts, provider, latency.
7. Endpoints that touch chat return 403 `consent_required` until consents (18+, health data, AI) exist.
8. System prompt forbids diagnoses, dosages and treatment plans; escalate to a trichologist/doctor.
9. Tests assert behaviour through public interfaces, not implementation details.
   Coverage gate 80% on app/services, app/llm, app/rag.
10. Secrets only via env (see .env.example). Never commit keys.

## Commands
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy app
uv run pytest tests/unit
uv run pytest tests/integration        # needs Docker
uv run alembic upgrade head
uv run uvicorn app.main:app --reload

## Workflow for the agent
- Branching: GitFlow. Branch names: `feature/*`, `fix/*`, `chore/*`, always cut from `develop`.
  - Work branches → PR into `develop`, **squash** merge.
  - Release: PR `develop` → `main`, **merge commit** (never squash), so main keeps develop's history.
  - `main` deploys to prod; `develop` will deploy to staging (later).
- One checklist item per branch/PR (items 1.1–1.12 in the plan doc).
- Before coding: restate the item, list files to touch, list tests to add. Wait for "go" if the
  change touches DB schema, auth or the LLM layer.
- After coding: run lint, mypy, unit tests; paste results; update this file if a convention changed.
- Record decisions that are hard to reverse as ADRs in docs/adr/NNNN-title.md.

## SSE contract (POST /conversations/{id}/messages)
event: token    data: {"text": "..."}
event: restart  data: {"reason": "..."}    # provider failed mid-stream; client clears partial answer
event: done     data: {"message_id": "...", "provider": "...", "model": "..."}
event: error    data: {"code": "...", "message": "..."}
