import asyncio

from app.core.background import BackgroundJobs


async def test_spawn_returns_at_once_and_join_waits_for_the_job() -> None:
    jobs, release, done = BackgroundJobs(), asyncio.Event(), asyncio.Event()

    async def job() -> None:
        await release.wait()
        done.set()

    jobs.spawn(job())  # not awaited by the caller
    assert not done.is_set()
    release.set()
    await jobs.join()

    assert done.is_set()


async def test_close_cancels_jobs_still_running_after_the_grace_period() -> None:
    jobs, started, cancelled = BackgroundJobs(), asyncio.Event(), asyncio.Event()

    async def stuck() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    jobs.spawn(stuck())
    await started.wait()
    await jobs.aclose(grace=0.01)

    assert cancelled.is_set()
