from enum import StrEnum
from functools import lru_cache
from typing import Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url


class Env(StrEnum):
    DEV = "dev"
    # Behaves like prod (keys required, moderation on, no dev CLI) but may search draft articles.
    STAGING = "staging"
    PROD = "prod"


def normalize_database_url(url: str) -> str:
    """A libpq URL (Neon: postgresql://...?sslmode=require&channel_binding=require) for asyncpg.

    asyncpg rejects libpq-only query parameters, so only `ssl` survives (from `sslmode`).
    """
    parsed = make_url(url)
    query = dict(parsed.query)
    ssl = query.get("ssl", query.get("sslmode"))
    parsed = parsed.set(
        drivername="postgresql+asyncpg", query={"ssl": ssl} if isinstance(ssl, str) else {}
    )
    return parsed.render_as_string(hide_password=False)


class Settings(BaseSettings):
    # env_ignore_empty: a blank `KEY=` copied from .env.example falls back to the default.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_ignore_empty=True, extra="ignore"
    )

    env: Env = Env.DEV
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/hair_chat"
    # Per statement and per connect: a pooled connection that died during a Fly suspend hangs
    # instead of failing; past this it is dropped and the pool reconnects (app/db/session.py).
    db_command_timeout_s: float = Field(default=10.0, gt=0)
    # Pooled connections older than this are replaced on checkout (Neon/proxies drop idle ones).
    db_pool_recycle_s: int = Field(default=240, gt=0)
    # "provider:model,provider:model,...", priority order; models are listed in app/llm/registry.py.
    llm_chain: str = "gemini:gemini-3.8-flash,gemini:gemini-3.5-flash-lite,groq:openai/gpt-oss-120b"
    llm_connect_timeout_s: float = 5.0
    # Until the first content token; thinking models need a few seconds before they answer.
    llm_first_token_timeout_s: float = 20.0
    # Max silence between chunks once streaming; longer counts as a broken stream.
    llm_read_timeout_s: float = 30.0
    # Answer length cap per chat reply; reasoning tokens of thinking models count against it too.
    chat_max_answer_tokens: int = 2048
    # Input tokens per chat call, far below the models' windows for cost and latency; the
    # effective budget is min(this, chain window - chat_max_answer_tokens).
    context_budget_tokens: int = 6000
    gemini_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    mistral_api_key: SecretStr | None = None
    # Input moderation (OpenAI omni-moderation); without it messages go unchecked: dev only.
    openai_api_key: SecretStr | None = None
    # Whole moderation call; on timeout the message goes through unchecked (ADR 0004).
    moderation_timeout_s: float = 3.0
    # violence / violence/graphic block only at this score (or with harassment/threatening):
    # burns and bleeding scalps are hair-care topics, not violence (ADR 0004).
    moderation_violence_threshold: float = 0.9
    # Auth0 tenant domain (e.g. hair-chat.eu.auth0.com) and API identifier; required outside dev.
    auth0_domain: str | None = None
    auth0_audience: str | None = None
    # Auth0 Management API (machine-to-machine app with the delete:users scope): DELETE /me also
    # removes the Auth0 account. Unset = our data is deleted, the Auth0 account stays (warning).
    auth0_mgmt_client_id: str | None = None
    auth0_mgmt_client_secret: SecretStr | None = None
    # python -m app.cli.retention deletes conversations idle (last_message_at) for longer.
    retention_days: int = Field(default=30, gt=0)
    # Version of the consent texts shipped by the frontend; see docs/consents/README.md.
    consent_version: str = "2026-10-v1"
    # Knowledge base (RAG). One embedding model for the whole index (rule 4, ADR 0005): changing
    # it means re-running the ingest; rows of another model are never searched.
    embedding_model: str = "gemini-embedding-2"
    # Must match the kb_chunks.embedding column (vector(768)).
    embedding_dimensions: int = 768
    # Query embedding in a chat reply; on timeout the answer goes without the knowledge base.
    embedding_timeout_s: float = 3.0
    kb_top_k: int = 4
    # Cosine similarity a chunk needs to reach the context. scripts/rag_eval.py (2026-10-21,
    # gemini-embedding-2/768): off-topic best 0.53, weakest right hit 0.64 -> 0.60.
    kb_min_similarity: float = 0.60
    # Tokens of the <knowledge> block at most.
    kb_max_tokens: int = 1500
    # Search draft articles too; empty = yes in dev only (prod refuses true, staging allows it).
    kb_include_drafts: bool | None = None

    @property
    def is_deployed(self) -> bool:
        """staging or prod: real users' keys and checks are mandatory."""
        return self.env is not Env.DEV

    @property
    def kb_drafts_searched(self) -> bool:
        if self.kb_include_drafts is None:
            return self.env is Env.DEV
        return self.kb_include_drafts

    @field_validator("database_url")
    @classmethod
    def _asyncpg_database_url(cls, value: str) -> str:
        return normalize_database_url(value)

    @model_validator(mode="after")
    def _require_auth0_when_deployed(self) -> Self:
        if self.is_deployed and not (self.auth0_domain and self.auth0_audience):
            raise ValueError(
                "AUTH0_DOMAIN and AUTH0_AUDIENCE are required when ENV is staging or prod"
            )
        return self

    @model_validator(mode="after")
    def _no_drafts_in_prod(self) -> Self:
        if self.env is Env.PROD and self.kb_include_drafts:
            raise ValueError("KB_INCLUDE_DRAFTS must not be true when ENV=prod (reviewed only)")
        return self

    @model_validator(mode="after")
    def _require_moderation_when_deployed(self) -> Self:
        if self.is_deployed and self.openai_api_key is None:
            raise ValueError(
                "OPENAI_API_KEY (input moderation) is required when ENV is staging or prod"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
