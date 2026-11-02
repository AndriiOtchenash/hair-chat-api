"""Event-loop lag monitor: a diagnostic for slow replies (staging embedding timeouts).

A background task sleeps INTERVAL_S at a time and logs a WARNING when the sleep overran by more
than THRESHOLD_S: something blocked the event loop (sync work, CPU) for about that long.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)

INTERVAL_S = 0.5
THRESHOLD_S = 0.25


async def monitor_loop_lag(interval: float = INTERVAL_S, threshold: float = THRESHOLD_S) -> None:
    """Runs until cancelled."""
    loop = asyncio.get_running_loop()
    while True:
        start = loop.time()
        await asyncio.sleep(interval)
        lag = loop.time() - start - interval
        if lag > threshold:
            logger.warning("event loop lag: lag_ms=%d", lag * 1000)


def start_loop_monitor() -> asyncio.Task[None]:
    return asyncio.create_task(monitor_loop_lag(), name="loop-lag-monitor")


async def stop_loop_monitor(task: asyncio.Task[None]) -> None:
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if current is not None and current.cancelling():
            raise  # we are being cancelled ourselves, not just the monitor
