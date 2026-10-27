import pytest
from pydantic import ValidationError

from app.core.chat import build_embedder, build_moderator, check_deployed_config
from app.core.config import Settings


@pytest.fixture(autouse=True)
def no_auth0_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ENV",
        "AUTH0_DOMAIN",
        "AUTH0_AUDIENCE",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "LLM_CHAIN",
        "KB_INCLUDE_DRAFTS",
        "DATABASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_prod_requires_auth0_settings() -> None:
    with pytest.raises(ValidationError, match="AUTH0_DOMAIN and AUTH0_AUDIENCE"):
        Settings(_env_file=None, env="prod", auth0_domain="t.auth0.com", openai_api_key="k")


def test_prod_with_auth0_and_moderation_settings_is_valid() -> None:
    settings = Settings(
        _env_file=None,
        env="prod",
        auth0_domain="t.auth0.com",
        auth0_audience="a",
        openai_api_key="k",
    )

    assert settings.auth0_domain == "t.auth0.com"


def test_prod_requires_moderation_key() -> None:
    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        Settings(_env_file=None, env="prod", auth0_domain="t.auth0.com", auth0_audience="a")


def test_dev_without_moderation_key_gets_noop_moderator() -> None:
    assert not build_moderator(Settings(_env_file=None)).enabled


def test_dev_allows_missing_auth0_settings() -> None:
    assert Settings(_env_file=None).auth0_domain is None


PROD = {"env": "prod", "auth0_domain": "t.auth0.com", "auth0_audience": "a", "openai_api_key": "k"}


def test_drafts_are_searched_in_dev_and_not_in_prod_by_default() -> None:
    assert Settings(_env_file=None).kb_drafts_searched
    assert not Settings(_env_file=None, **PROD).kb_drafts_searched
    assert not Settings(_env_file=None, kb_include_drafts=False).kb_drafts_searched


def test_prod_refuses_draft_articles() -> None:
    with pytest.raises(ValidationError, match="KB_INCLUDE_DRAFTS"):
        Settings(_env_file=None, kb_include_drafts=True, **PROD)


def test_without_gemini_key_there_is_no_embedder() -> None:
    assert build_embedder(Settings(_env_file=None)) is None
    assert build_embedder(Settings(_env_file=None, gemini_api_key="k")) is not None


STAGING = {**PROD, "env": "staging"}


def test_staging_requires_auth0_settings() -> None:
    with pytest.raises(ValidationError, match="AUTH0_DOMAIN and AUTH0_AUDIENCE"):
        Settings(_env_file=None, env="staging", auth0_domain="t.auth0.com", openai_api_key="k")


def test_staging_requires_moderation_key() -> None:
    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        Settings(_env_file=None, env="staging", auth0_domain="t.auth0.com", auth0_audience="a")


def test_staging_moderates_messages() -> None:
    assert build_moderator(Settings(_env_file=None, **STAGING)).enabled


def test_staging_may_search_drafts_but_does_not_by_default() -> None:
    assert not Settings(_env_file=None, **STAGING).kb_drafts_searched
    assert Settings(_env_file=None, kb_include_drafts=True, **STAGING).kb_drafts_searched


async def test_staging_refuses_to_start_without_a_key_of_the_chain() -> None:
    settings = Settings(
        _env_file=None,
        llm_chain="gemini:gemini-3.8-flash,groq:openai/gpt-oss-120b",
        gemini_api_key="g",
        **STAGING,
    )

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        await check_deployed_config(settings)


async def test_staging_refuses_to_start_without_the_embeddings_key() -> None:
    settings = Settings(
        _env_file=None, llm_chain="groq:openai/gpt-oss-120b", groq_api_key="q", **STAGING
    )

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        await check_deployed_config(settings)


async def test_staging_with_all_keys_starts_and_dev_starts_without_any() -> None:
    await check_deployed_config(
        Settings(_env_file=None, gemini_api_key="g", groq_api_key="q", **STAGING)
    )
    await check_deployed_config(Settings(_env_file=None))


def test_neon_url_is_converted_for_asyncpg() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql://u:p%40ss@ep-x-pooler.eu-central-1.aws.neon.tech/db"
        "?sslmode=require&channel_binding=require",
    )

    assert settings.database_url == (
        "postgresql+asyncpg://u:p%40ss@ep-x-pooler.eu-central-1.aws.neon.tech/db?ssl=require"
    )


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("postgres://u:p@h:5432/db", "postgresql+asyncpg://u:p@h:5432/db"),
        ("postgresql+asyncpg://u:p@h/db?ssl=require", "postgresql+asyncpg://u:p@h/db?ssl=require"),
        ("postgresql://u:p@h/db?connect_timeout=5", "postgresql+asyncpg://u:p@h/db"),
    ],
)
def test_database_url_normalization_is_idempotent(url: str, expected: str) -> None:
    once = Settings(_env_file=None, database_url=url).database_url

    assert once == expected
    assert Settings(_env_file=None, database_url=once).database_url == expected
