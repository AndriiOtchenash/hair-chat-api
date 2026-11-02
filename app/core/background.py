"""Fire-and-forget work that must not hold the request.

FastAPI's BackgroundTasks run inside the request, before its yield-dependencies exit: after a
chat answer that means while `start_reply` still holds the conversation's generation slot, so
the user's next message would get 409 conversation_busy until the task ends. Work that needs
no slot (the conversation index, the summarizer) is spawned here instead: an asyncio task with
a strong reference (the loop keeps only weak ones), drained on shutdown.
"""

import asyncio
import logging
from collections.abc import Coroutine
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

SHUTDOWN_GRACE_S = 5.0


class BackgroundJobs:
    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[None]] = set()

    def spawn(self, job: Coroutine[Any, Any, None]) -> None:
        """`job` must handle its own errors (it is never awaited by a caller)."""
        task = asyncio.create_task(job)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def join(self) -> None:
        """Wait until every job spawned so far has ended (tests: no polling)."""
        if self._tasks:
            await asyncio.wait(set(self._tasks))

    async def aclose(self, grace: float = SHUTDOWN_GRACE_S) -> None:
        """Let running jobs finish for up to `grace` seconds, then cancel the rest."""
        if not self._tasks:
            return
        _, pending = await asyncio.wait(set(self._tasks), timeout=grace)
        for task in pending:
            task.cancel()
        if pending:
            logger.warning("background jobs cancelled at shutdown: count=%d", len(pending))
            await asyncio.gather(*pending, return_exceptions=True)


@lru_cache
def get_background_jobs() -> BackgroundJobs:
    return BackgroundJobs()
