import io
from datetime import UTC, datetime, timedelta

from app.cli.retention import format_result, run
from app.db.repositories.retention import PurgeCounts
from app.services.retention import purge

NOW = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)


class FakeRetentionRepository:
    """Conversations as (last_message_at, message_count); orphan llm_calls as created_at."""

    def __init__(self, conversations: list[tuple[datetime, int]], orphans: list[datetime]) -> None:
        self.conversations = list(conversations)
        self.orphans = list(orphans)
        self.batches: list[int] = []

    def _expired(self, cutoff: datetime) -> list[tuple[datetime, int]]:
        return [c for c in self.conversations if c[0] < cutoff]

    async def count_expired(self, cutoff: datetime) -> PurgeCounts:
        expired = self._expired(cutoff)
        return PurgeCounts(len(expired), sum(n for _, n in expired))

    async def delete_expired_batch(self, cutoff: datetime, limit: int) -> PurgeCounts:
        batch = self._expired(cutoff)[:limit]
        for c in batch:
            self.conversations.remove(c)
        self.batches.append(len(batch))
        return PurgeCounts(len(batch), sum(n for _, n in batch))

    async def count_orphan_llm_calls(self, cutoff: datetime) -> int:
        return sum(1 for o in self.orphans if o < cutoff)

    async def delete_orphan_llm_calls(self, cutoff: datetime) -> int:
        old = [o for o in self.orphans if o < cutoff]
        self.orphans = [o for o in self.orphans if o >= cutoff]
        return len(old)


def _repo() -> FakeRetentionRepository:
    day = timedelta(days=1)
    return FakeRetentionRepository(
        [(NOW - 31 * day, 4), (NOW - 40 * day, 2), (NOW - 30 * day + timedelta(seconds=1), 9)]
        + [(NOW - 100 * day, 1)] * 3,
        [NOW - 31 * day, NOW - day],
    )


async def test_purge_deletes_only_older_than_days_in_batches() -> None:
    repo = _repo()

    result = await purge(repo, now=NOW, days=30, dry_run=False, batch_size=2)

    assert result.cutoff == NOW - timedelta(days=30)
    assert (result.conversations, result.messages, result.orphan_llm_calls) == (5, 9, 1)
    assert repo.batches == [2, 2, 1, 0]
    assert repo.conversations == [(NOW - timedelta(days=30) + timedelta(seconds=1), 9)]
    assert repo.orphans == [NOW - timedelta(days=1)]


async def test_purge_is_idempotent() -> None:
    repo = _repo()
    await purge(repo, now=NOW, days=30, dry_run=False)

    again = await purge(repo, now=NOW, days=30, dry_run=False)

    assert (again.conversations, again.messages, again.orphan_llm_calls) == (0, 0, 0)


async def test_dry_run_counts_and_deletes_nothing() -> None:
    repo = _repo()

    result = await purge(repo, now=NOW, days=30, dry_run=True)

    assert (result.conversations, result.messages, result.orphan_llm_calls) == (5, 9, 1)
    assert result.dry_run
    assert repo.batches == []
    assert len(repo.conversations) == 6
    assert len(repo.orphans) == 2


async def test_cli_prints_counts_only() -> None:
    out = io.StringIO()

    await run(_repo(), days=30, dry_run=True, out=out, now=NOW)

    line = out.getvalue()
    assert "would delete conversations=5 messages=9 orphan_llm_calls=1" in line
    assert "(30 days, before " in line


async def test_format_result_for_a_real_run() -> None:
    result = await purge(_repo(), now=NOW, days=30, dry_run=False)

    assert format_result(result, 30) == (
        "retention (30 days, before 2026-09-22T12:00:00Z): deleted "
        "conversations=5 messages=9 orphan_llm_calls=1"
    )
