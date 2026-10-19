from enum import StrEnum
from functools import lru_cache
from typing import Self

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Env(StrEnum):
    DEV = "dev"
    PROD = "prod"


class Settings(BaseSettings):
    # env_ignore_empty: a blank `KEY=` copied from .env.example falls back to the default.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_ignore_empty=True, extra="ignore"
    )

    env: Env = Env.DEV
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/hair_chat"
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
    # Auth0 tenant domain (e.g. hair-chat.eu.auth0.com) and API identifier; required in prod.
    auth0_domain: str | None = None
    auth0_audience: str | None = None
    # Version of the consent texts shipped by the frontend; see docs/consents/README.md.
    consent_version: str = "2026-10-v1"

    @model_validator(mode="after")
    def _require_auth0_in_prod(self) -> Self:
        if self.env is Env.PROD and not (self.auth0_domain and self.auth0_audience):
            raise ValueError("AUTH0_DOMAIN and AUTH0_AUDIENCE are required when ENV=prod")
        return self

    @model_validator(mode="after")
    def _require_moderation_in_prod(self) -> Self:
        if self.env is Env.PROD and self.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY (input moderation) is required when ENV=prod")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
