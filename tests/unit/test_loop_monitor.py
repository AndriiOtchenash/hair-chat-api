import asyncio
import logging
import time

import pytest

from app.core.loop_monitor import monitor_loop_lag, start_loop_monitor, stop_loop_monitor


async def test_a_blocked_loop_is_logged_as_lag(caplog: pytest.LogCaptureFixture) -> None:
    task = asyncio.create_task(monitor_loop_lag(interval=0.01, threshold=0.05))
    await asyncio.sleep(0)  # let the monitor start its sleep
    with caplog.at_level(logging.WARNING, logger="app.core.loop_monitor"):
        time.sleep(0.2)  # noqa: ASYNC251 - blocking the loop is the point
        await asyncio.sleep(0.05)
    await stop_loop_monitor(task)

    assert any("event loop lag: lag_ms=" in r.getMessage() for r in caplog.records)


async def test_an_idle_loop_logs_nothing(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="app.core.loop_monitor"):
        task = asyncio.create_task(monitor_loop_lag(interval=0.01, threshold=0.5))
        await asyncio.sleep(0.05)
        await stop_loop_monitor(task)

    assert caplog.records == []


async def test_stop_cancels_the_monitor_cleanly() -> None:
    task = start_loop_monitor()
    await asyncio.sleep(0)

    await stop_loop_monitor(task)

    assert task.cancelled()
