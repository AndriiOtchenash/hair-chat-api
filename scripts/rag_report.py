"""Knowledge-base use of the answers of the last N days, from messages.retrieval:

    uv run python -m scripts.rag_report [--days 7]

Shares of retrieval ok / empty / error (disabled = dev without an embeddings key; unknown =
answers saved before messages.retrieval existed), the share of answers whose <kb_used> tag named
a passage, the error codes, and the same per chain link (provider:model that answered).
Reads DATABASE_URL like the app. Aggregates only, across all users on purpose (like
app.cli.retention): no message text, no user ids.
"""

import argparse
import asyncio
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.models import Message, MessageRole
from app.db.session import get_engine, get_sessionmaker

STATUSES = ("ok", "empty", "error", "disabled", "unknown")


@dataclass(frozen=True, slots=True)
class Group:
    """Answers of one link with one retrieval outcome (a row of `fetch`)."""

    link: str  # provider:model
    status: str  # RetrievalStatus value, or "unknown" (no messages.retrieval)
    kb_used: bool  # the <kb_used> tag named at least one passage
    error: str | None
    count: int


@dataclass
class Tally:
    statuses: Counter[str] = field(default_factory=Counter)
    kb_used: int = 0

    @property
    def total(self) -> int:
        return sum(self.statuses.values())

    def add(self, group: Group) -> None:
        self.statuses[group.status] += group.count
        self.kb_used += group.count if group.kb_used else 0


async def fetch(session: AsyncSession, since: datetime) -> list[Group]:
    status = func.coalesce(Message.retrieval["status"].astext, "unknown")
    used = func.coalesce(func.jsonb_array_length(Message.retrieval["kb_used"]), 0) > 0
    error = Message.retrieval["error"].astext
    link = func.concat(Message.provider, ":", Message.model)
    rows = await session.execute(
        select(link, status, used, error, func.count())
        .where(Message.role == MessageRole.ASSISTANT, Message.created_at >= since)
        .group_by(link, status, used, error)
    )
    return [Group(*row) for row in rows]


def _share(part: int, whole: int) -> str:
    return f"{part / whole:.1%}" if whole else "-"


def format_report(groups: Iterable[Group], days: int, since: datetime) -> str:
    groups = list(groups)
    overall = Tally()
    by_link: dict[str, Tally] = {}
    errors: Counter[str] = Counter()
    for group in groups:
        overall.add(group)
        by_link.setdefault(group.link, Tally()).add(group)
        if group.error:
            errors[group.error] += group.count
    total = overall.total
    lines = [f"RAG report: last {days} days (since {since:%Y-%m-%d %H:%M} UTC), {total} answers"]
    if not total:
        return "\n".join(lines)
    lines.append(
        "retrieval: "
        + "  ".join(
            f"{s} {overall.statuses[s]} ({_share(overall.statuses[s], total)})"
            for s in STATUSES
            if overall.statuses[s] or s in ("ok", "empty", "error")
        )
    )
    ok = overall.statuses["ok"]
    lines.append(
        f"kb_used: {overall.kb_used} of {total} answers ({_share(overall.kb_used, total)}); "
        f"of retrieval ok: {_share(overall.kb_used, ok)}"
    )
    if errors:
        lines.append("errors: " + ", ".join(f"{code} {n}" for code, n in errors.most_common()))
    width = max(len("link"), *(len(link) for link in by_link))
    lines += ["", f"{'link':<{width}}  answers      ok   empty   error  kb_used"]
    for link, tally in sorted(by_link.items(), key=lambda item: -item[1].total):
        n = tally.total
        lines.append(
            f"{link:<{width}}  {n:>7}  "
            + "  ".join(f"{_share(tally.statuses[s], n):>6}" for s in ("ok", "empty", "error"))
            + f"  {_share(tally.kb_used, n):>7}"
        )
    return "\n".join(lines)


async def run(days: int) -> int:
    try:
        get_settings()
    except ValueError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    since = datetime.now(UTC) - timedelta(days=days)
    try:
        async with get_sessionmaker()() as session:
            groups = await fetch(session, since)
    finally:
        await get_engine().dispose()
    print(format_report(groups, days, since))
    return 0


def _positive(value: str) -> int:
    days = int(value)
    if days <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return days


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.rag_report", description=__doc__)
    parser.add_argument("--days", type=_positive, default=7, help="look back N days (7)")
    args = parser.parse_args(argv)
    return asyncio.run(run(args.days))


if __name__ == "__main__":
    sys.exit(main())
