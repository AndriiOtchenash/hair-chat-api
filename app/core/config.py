from enum import StrEnum
from functools import lru_cache

from pydantic import SecretStr
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

    @property
    def llm_chain_list(self) -> list[str]:
        return [name.strip() for name in self.llm_chain.split(",") if name.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
