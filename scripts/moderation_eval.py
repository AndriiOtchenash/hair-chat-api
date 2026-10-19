"""Moderation eval against the real API: tests/moderation_eval/phrases.yaml -> table + mismatches.

    uv run python -m scripts.moderation_eval [--threshold 0.9] [--file PATH] [--only-mismatches]

Sends only our own phrases from the YAML file, never user data. Not run in CI (real API, and
the provider's scores drift between model versions). Exit code 1 when any phrase gets an
action other than `expected`. Uses OPENAI_API_KEY and MODERATION_VIOLENCE_THRESHOLD from
Settings, i.e. exactly the moderator and policy the chat uses.
"""

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.core.config import get_settings
from app.llm.moderation import ModerationResult, OpenAIModerator
from app.services.moderation_policy import ModerationAction, decide

DEFAULT_FILE = Path(__file__).resolve().parents[1] / "tests/moderation_eval/phrases.yaml"
# The free endpoint rate-limits bursts; a few at a time is quick enough for ~50 phrases.
CONCURRENCY = 4
TEXT_WIDTH = 60


@dataclass(frozen=True)
class Phrase:
    lang: str
    group: str
    expected: ModerationAction
    text: str


@dataclass(frozen=True)
class Outcome:
    phrase: Phrase
    result: ModerationResult
    action: ModerationAction

    @property
    def ok(self) -> bool:
        return self.action is self.phrase.expected


def load(path: Path) -> list[Phrase]:
    rows: list[dict[str, Any]] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [Phrase(r["lang"], r["group"], ModerationAction(r["expected"]), r["text"]) for r in rows]


async def run(phrases: list[Phrase], threshold: float) -> list[Outcome]:
    settings = get_settings()
    if settings.openai_api_key is None:
        raise SystemExit("OPENAI_API_KEY is not set")
    moderator = OpenAIModerator(api_key=settings.openai_api_key.get_secret_value(), timeout=15)
    limit = asyncio.Semaphore(CONCURRENCY)

    async def one(phrase: Phrase) -> Outcome:
        async with limit:
            result = await moderator.check(phrase.text)
        decision = decide(result, violence_threshold=threshold)
        return Outcome(phrase, result, decision.action)

    try:
        return await asyncio.gather(*(one(p) for p in phrases))
    finally:
        await moderator.aclose()


def _top_scores(result: ModerationResult, count: int = 2) -> str:
    top = sorted(result.scores.items(), key=lambda kv: -kv[1])[:count]
    return " ".join(f"{name}={score:.2f}" for name, score in top)


def _short(text: str) -> str:
    return text if len(text) <= TEXT_WIDTH else text[: TEXT_WIDTH - 1] + "…"


def report(outcomes: list[Outcome], only_mismatches: bool) -> str:
    lines = ["ok  lang group      expected got    flagged categories / top scores | phrase"]
    for o in outcomes:
        if only_mismatches and o.ok:
            continue
        mark = "  " if o.ok else "XX"
        categories = ",".join(o.result.categories) or "-"
        lines.append(
            f"{mark}  {o.phrase.lang:<4} {o.phrase.group:<10} {o.phrase.expected:<8} "
            f"{o.action:<6} {categories} / {_top_scores(o.result)} | {_short(o.phrase.text)}"
        )
    misses = [o for o in outcomes if not o.ok]
    lines.append("")
    lines.append(f"{len(outcomes) - len(misses)}/{len(outcomes)} as expected")
    for o in misses:
        lines.append(f"  expected {o.phrase.expected}, got {o.action}: {o.phrase.text}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.moderation_eval", description=__doc__)
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="violence threshold; default MODERATION_VIOLENCE_THRESHOLD",
    )
    parser.add_argument("--only-mismatches", action="store_true")
    args = parser.parse_args(argv)
    threshold = (
        args.threshold
        if args.threshold is not None
        else get_settings().moderation_violence_threshold
    )
    outcomes = asyncio.run(run(load(args.file), threshold))
    print(f"violence threshold {threshold}, model {outcomes[0].result.model if outcomes else '-'}")
    print(report(outcomes, args.only_mismatches))
    return 0 if all(o.ok for o in outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
