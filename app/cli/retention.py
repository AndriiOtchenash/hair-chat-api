"""Retention cleanup: python -m app.cli.retention [--dry-run] [--days N]

Deletes conversations whose last message is older than RETENTION_DAYS (cascading to their
messages and llm_calls), plus llm_calls never linked to a message that are as old. Prints
counts only, never texts. Idempotent and safe to run in parallel (ADR 0006); scheduled daily on
Fly.io, see docs/deploy.md. Deliberately no HTTP endpoint.
"""

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TextIO

from app.core.config import get_settings
from app.db.repositories.retention import RetentionRepository, SqlRetentionRepository
from app.db.session import get_engine, get_sessionmaker
from app.services.retention import RetentionResult, purge


def format_result(result: RetentionResult, days: int) -> str:
    verb = "would delete" if result.dry_run else "deleted"
    return (
        f"retention ({days} days, before {result.cutoff:%Y-%m-%dT%H:%M:%SZ}): {verb} "
        f"conversations={result.conversations} messages={result.messages} "
        f"orphan_llm_calls={result.orphan_llm_calls}"
    )


async def run(
    repo: RetentionRepository,
    *,
    days: int,
    dry_run: bool,
    out: TextIO,
    now: datetime | None = None,
) -> None:
    result = await purge(repo, now=now or datetime.now(UTC), days=days, dry_run=dry_run)
    out.write(format_result(result, days) + "\n")


async def _main(days: int | None, dry_run: bool) -> int:
    try:
        settings = get_settings()
    except ValueError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    try:
        await run(
            SqlRetentionRepository(get_sessionmaker()),
            days=days or settings.retention_days,
            dry_run=dry_run,
            out=sys.stdout,
        )
    finally:
        await get_engine().dispose()
    return 0


def _positive(value: str) -> int:
    days = int(value)
    if days <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return days


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli.retention", description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="only count, delete nothing")
    parser.add_argument("--days", type=_positive, help="override RETENTION_DAYS")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return asyncio.run(_main(args.days, args.dry_run))


if __name__ == "__main__":
    sys.exit(main())
