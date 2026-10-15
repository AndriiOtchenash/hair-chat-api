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
    llm_chain: str = "gemini-free,mistral-experiment"
    gemini_api_key: SecretStr | None = None
    mistral_api_key: SecretStr | None = None
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

    @property
    def llm_chain_list(self) -> list[str]:
        return [name.strip() for name in self.llm_chain.split(",") if name.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
