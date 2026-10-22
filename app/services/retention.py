"""Retention: conversations idle for longer than RETENTION_DAYS are deleted (ADR 0006)."""

from dataclasses import dataclass
from datetime import datetime, timedelta

from app.db.repositories.retention import RetentionRepository

DEFAULT_BATCH_SIZE = 500


@dataclass(frozen=True, slots=True)
class RetentionResult:
    cutoff: datetime
    conversations: int
    messages: int
    orphan_llm_calls: int
    dry_run: bool


async def purge(
    repo: RetentionRepository,
    *,
    now: datetime,
    days: int,
    dry_run: bool,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> RetentionResult:
    """Deletes (or with dry_run only counts) what is older than `days`; safe to re-run."""
    cutoff = now - timedelta(days=days)
    if dry_run:
        counts = await repo.count_expired(cutoff)
        return RetentionResult(
            cutoff,
            counts.conversations,
            counts.messages,
            await repo.count_orphan_llm_calls(cutoff),
            dry_run=True,
        )
    conversations = messages = 0
    while True:
        batch = await repo.delete_expired_batch(cutoff, batch_size)
        if batch.conversations == 0:
            break
        conversations += batch.conversations
        messages += batch.messages
    return RetentionResult(
        cutoff,
        conversations,
        messages,
        await repo.delete_orphan_llm_calls(cutoff),
        dry_run=False,
    )
