import logging
from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.core.logging import configure_logging
from app.main import create_app


def own_handlers() -> list[logging.Handler]:
    """pytest (9+) also attaches its capture handlers to non-propagating loggers: skip those."""
    return [
        h for h in logging.getLogger("app").handlers if not type(h).__module__.startswith("_pytest")
    ]


@pytest.fixture
def fresh_settings() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_app_logger_has_one_handler_and_settings_level(
    fresh_settings: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "debug")
    create_app()
    logger = logging.getLogger("app")
    assert len(own_handlers()) == 1
    assert logger.level == logging.DEBUG
    assert logger.propagate is False


def test_configure_logging_twice_keeps_one_handler() -> None:
    configure_logging("INFO")
    configure_logging("WARNING")
    assert len(own_handlers()) == 1
    assert logging.getLogger("app").level == logging.WARNING


def test_info_goes_to_stdout_with_level_and_name(
    fresh_settings: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    create_app()
    logging.getLogger("app.services.demo").info("reply saved: tokens_out=%d", 7)
    assert "INFO app.services.demo: reply saved: tokens_out=7" in capsys.readouterr().out


def test_log_level_warning_silences_info(
    fresh_settings: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    create_app()
    logger = logging.getLogger("app.services.demo")
    logger.info("hidden info")
    logger.warning("shown warning")
    out = capsys.readouterr().out
    assert "hidden info" not in out
    assert "WARNING app.services.demo: shown warning" in out


@pytest.mark.parametrize("value", ["VERBOSE", "NOTSET", "10", ""])
def test_log_level_rejects_non_standard_levels(value: str) -> None:
    with pytest.raises(ValidationError):
        Settings(log_level=value)
