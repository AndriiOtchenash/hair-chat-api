import pytest
from pydantic import ValidationError

from app.core.config import Settings


@pytest.fixture(autouse=True)
def no_auth0_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ENV", "AUTH0_DOMAIN", "AUTH0_AUDIENCE"):
        monkeypatch.delenv(name, raising=False)


def test_prod_requires_auth0_settings() -> None:
    with pytest.raises(ValidationError, match="AUTH0_DOMAIN and AUTH0_AUDIENCE"):
        Settings(_env_file=None, env="prod", auth0_domain="t.auth0.com")


def test_prod_with_auth0_settings_is_valid() -> None:
    settings = Settings(_env_file=None, env="prod", auth0_domain="t.auth0.com", auth0_audience="a")

    assert settings.auth0_domain == "t.auth0.com"


def test_dev_allows_missing_auth0_settings() -> None:
    assert Settings(_env_file=None).auth0_domain is None
