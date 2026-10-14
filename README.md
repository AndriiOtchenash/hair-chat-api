# hair-chat-api
Backend for a multilingual hair-care chatbot: FastAPI, PostgreSQL/pgvector, RAG, LLM fallback

## Run locally (Codespaces / devcontainer)
```bash
uv sync                                    # install deps from uv.lock into .venv
cp .env.example .env                       # fill in DATABASE_URL, API keys
docker run -d --name hair-db -p 5432:5432 -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=hair_chat pgvector/pgvector:pg16
uv run alembic upgrade head                # apply migrations
uv run uvicorn app.main:app --reload       # http://localhost:8000/health
uv run pytest tests/unit                   # integration: uv run pytest tests/integration (Docker)
```
