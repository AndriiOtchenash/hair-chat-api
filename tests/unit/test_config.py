import pytest
from pydantic import ValidationError

from app.core.chat import build_moderator
from app.core.config import Settings


@pytest.fixture(autouse=True)
def no_auth0_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ENV", "AUTH0_DOMAIN", "AUTH0_AUDIENCE", "OPENAI_API_KEY"):
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
