"""Logging of the API process: uvicorn configures only its own loggers, so `app.*` needs this.

Without it only logging.lastResort prints (WARNING+, no level or logger name) and INFO is lost.
"""

import logging
import sys

APP_LOGGER = "app"
FORMAT = "%(levelname)s %(name)s: %(message)s"


class _AppHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Marks the handler we own, so a repeated call replaces it instead of adding another."""


def configure_logging(level: str) -> None:
    """One stdout handler on the `app` logger; idempotent, the latest call wins.

    propagate=False: records do not also reach root (or uvicorn's handlers) and print twice.
    The handler is replaced on every call, so it writes to the current sys.stdout.
    """
    logger = logging.getLogger(APP_LOGGER)
    for handler in [h for h in logger.handlers if isinstance(h, _AppHandler)]:
        logger.removeHandler(handler)
    handler = _AppHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(FORMAT))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
