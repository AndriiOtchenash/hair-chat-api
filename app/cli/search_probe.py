"""Experiment: calibrate the conversation search. Reads only, writes nothing.

    uv run python -m app.cli.search_probe USER_ID "query" ["query" ...] [--top N]
        [--all-pairs | --pairs]

Compares, in memory, cosine scores of one user's indexed answers (message_embeddings) in two
modes, the query in the index's query format ("task: search result | query: ..."):

    A  the stored vectors as they are: what the search does now (rows of an older format, see
       the version counts printed first, may score differently);
    B  the pairs re-embedded now in the index's current format (app.rag.conversation_index),
       without the small talk it no longer indexes: what the search does after the backfill.

Formats 2 and 3 of the index came out of this probe (asymmetric prefixes; then no title,
which filled short pairs so that a greeting beat the real question in its conversation). Two
embeddings calls in parallel (batches of 100).
Like the API, the best hit per conversation is shown unless --all-pairs; --pairs adds, under
each conversation, its best 3 pairs (question[:40] | score): which pair wins inside it. The
output prints conversation titles, questions and the queries: a dev tool for the operator's own
data, not for logs.
"""

import argparse
import asyncio
import math
import sys
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import select, true, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.core.config import get_settings
from app.db.models import Conversation, Message, MessageEmbedding, MessageRole
from app.db.session import get_engine, get_sessionmaker
from app.rag.conversation_index import (
    CONVERSATION_INDEX_FORMAT_VERSION,
    index_text,
    is_small_talk,
    query_text,
)
from app.rag.embeddings import Embedder, EmbeddingUnavailable, OpenAICompatibleEmbedder
from app.rag.ingest import INGEST_RETRIES, INGEST_TIMEOUT_S

MODES = ("A", "B")
PAIRS_PER_CONVERSATION = 3
SNIPPET_CHARS = 40


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


@dataclass(frozen=True, slots=True)
class Pair:
    message_id: uuid.UUID
    conversation_id: uuid.UUID
    title: str | None
    question: str
    answer: str
    vector: list[float]  # stored, mode A
    index_format: str | None = None  # of the stored vector; None = stale or format 1


@dataclass(frozen=True, slots=True)
class Hit:
    pair: Pair
    score: float


def rank(
    pairs: Sequence[Pair],
    vectors: Sequence[Sequence[float]],
    query_vector: Sequence[float],
    *,
    per_conversation: bool = True,
) -> list[Hit]:
    """Best first; with per_conversation only each conversation's best pair (like the API)."""
    hits = sorted(
        (Hit(p, cosine(v, query_vector)) for p, v in zip(pairs, vectors, strict=True)),
        key=lambda h: h.score,
        reverse=True,
    )
    return best_per_conversation(hits) if per_conversation else hits


def best_per_conversation(hits: Sequence[Hit]) -> list[Hit]:
    """Each conversation's best hit, in the order of `hits` (best first)."""
    seen: set[uuid.UUID] = set()
    best: list[Hit] = []
    for hit in hits:
        if hit.pair.conversation_id not in seen:
            seen.add(hit.pair.conversation_id)
            best.append(hit)
    return best


def gap(hits: Sequence[Hit]) -> float | None:
    return hits[0].score - hits[1].score if len(hits) >= 2 else None


async def load_pairs(session: AsyncSession, user_id: uuid.UUID, model: str) -> list[Pair]:
    """The user's indexed answers of `model`, with the user message before each (as indexed)."""
    question = aliased(Message)
    previous = (
        select(question.content)
        .where(
            question.conversation_id == Message.conversation_id,
            question.user_id == Message.user_id,
            question.role == MessageRole.USER,
            tuple_(question.created_at, question.id) < tuple_(Message.created_at, Message.id),
        )
        .order_by(question.created_at.desc(), question.id.desc())
        .limit(1)
        .lateral("previous")
    )
    rows = await session.execute(
        select(
            Message.id,
            Message.conversation_id,
            Conversation.title,
            previous.c.content,
            Message.content,
            MessageEmbedding.embedding,
            MessageEmbedding.index_format,
        )
        .join(MessageEmbedding, MessageEmbedding.message_id == Message.id)
        .join(Conversation, Conversation.id == Message.conversation_id)
        .join(previous, true())
        .where(
            Message.user_id == user_id,
            MessageEmbedding.user_id == user_id,
            Conversation.user_id == user_id,
            MessageEmbedding.embedding_model == model,
        )
        .order_by(Message.id)
    )
    return [Pair(mid, cid, title, q, a, list(vec), fmt) for mid, cid, title, q, a, vec, fmt in rows]


async def probe(
    pairs: Sequence[Pair],
    queries: Sequence[str],
    embedder: Embedder,
    *,
    max_chars: int,
    min_answer_chars: int = 0,
) -> dict[tuple[str, str], list[Hit]]:
    """(query, mode) -> every pair, best first (render picks the best per conversation); B
    without small talk. Two embeddings calls in parallel: queries and B documents (each split
    into batches of 100 by the embedder)."""
    current = [p for p in pairs if not is_small_talk(p.answer, min_answer_chars)]
    query_vectors, docs_b = await asyncio.gather(
        embedder.embed([query_text(q) for q in queries]),
        embedder.embed([index_text(p.question, p.answer, max_chars=max_chars) for p in current]),
    )
    stored = [p.vector for p in pairs]
    results: dict[tuple[str, str], list[Hit]] = {}
    for query, qv in zip(queries, query_vectors, strict=True):
        results[(query, "A")] = rank(pairs, stored, qv, per_conversation=False)
        results[(query, "B")] = rank(current, docs_b, qv, per_conversation=False)
    return results


def snippet(text: str) -> str:
    return " ".join(text.split())[:SNIPPET_CHARS]


def render(
    results: dict[tuple[str, str], list[Hit]],
    queries: Sequence[str],
    *,
    top: int,
    per_conversation: bool = True,
    pairs: bool = False,
) -> str:
    """`results` as probe returns them (every pair). per_conversation: one row per
    conversation, its best pair; pairs (with per_conversation): its best 3 pairs under it."""
    qw = min(max((len(q) for q in queries), default=5), 30)
    lines = [f"{'query':<{qw}} | mode | {'title':<30} | score", "-" * (qw + 48)]
    gaps: list[str] = []
    for query in queries:
        for mode in MODES:
            ranked = results[(query, mode)]
            hits = best_per_conversation(ranked) if per_conversation else ranked
            for hit in hits[:top]:
                title = (hit.pair.title or "(no title)")[:30]
                lines.append(f"{query[:qw]:<{qw}} | {mode:<4} | {title:<30} | {hit.score:.4f}")
                if pairs and per_conversation:
                    inside = [
                        h for h in ranked if h.pair.conversation_id == hit.pair.conversation_id
                    ]
                    for h in inside[:PAIRS_PER_CONVERSATION]:
                        lines.append(
                            f"{'':<{qw}} |      |   {snippet(h.pair.question):<{SNIPPET_CHARS}}"
                            f" | {h.score:.4f}"
                        )
            g = gap(hits)
            gaps.append(f"{query[:qw]:<{qw}} | {mode:<4} | " + ("n/a" if g is None else f"{g:.4f}"))
        lines.append("")
    lines += [f"{'query':<{qw}} | mode | gap top1-top2", "-" * (qw + 23), *gaps]
    return "\n".join(lines)


async def run(
    user_id: uuid.UUID,
    queries: Sequence[str],
    *,
    top: int,
    per_conversation: bool,
    show_pairs: bool = False,
) -> int:
    try:
        settings = get_settings()
    except ValueError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    if settings.gemini_api_key is None:
        print("GEMINI_API_KEY is not set", file=sys.stderr)
        return 2
    embedder = OpenAICompatibleEmbedder(
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        api_key=settings.gemini_api_key.get_secret_value(),
        timeout=INGEST_TIMEOUT_S,
        max_retries=INGEST_RETRIES,
    )
    try:
        async with get_sessionmaker()() as session:
            pairs = await load_pairs(session, user_id, settings.embedding_model)
        if not pairs:
            print(f"no indexed answers of model {settings.embedding_model} for this user")
            return 1
        results = await probe(
            pairs,
            queries,
            embedder,
            max_chars=settings.conversation_index_max_chars,
            min_answer_chars=settings.conversation_index_min_answer_chars,
        )
    except EmbeddingUnavailable as exc:
        print(f"embeddings unavailable ({exc.code})", file=sys.stderr)
        return 1
    finally:
        await embedder.aclose()
        await get_engine().dispose()
    unit = "conversation (best pair)" if per_conversation else "pair"
    formats = Counter(p.index_format or "stale/1" for p in pairs)
    min_chars = settings.conversation_index_min_answer_chars
    small_talk = sum(is_small_talk(p.answer, min_chars) for p in pairs)
    versions = ", ".join(f"{fmt}: {n}" for fmt, n in sorted(formats.items()))
    print(
        f"{len(pairs)} pairs, model {settings.embedding_model}, stored formats {versions} "
        f"(current {CONVERSATION_INDEX_FORMAT_VERSION}), "
        f"small talk not in B {small_talk}, one row per {unit}\n"
    )
    print(render(results, queries, top=top, per_conversation=per_conversation, pairs=show_pairs))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli.search_probe",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("user_id", type=uuid.UUID)
    parser.add_argument("queries", nargs="+")
    parser.add_argument("--top", type=int, default=10, help="rows per query x mode (10)")
    unit = parser.add_mutually_exclusive_group()
    unit.add_argument("--all-pairs", action="store_true", help="rank pairs, not conversations")
    unit.add_argument(
        "--pairs",
        action="store_true",
        help=f"under each conversation, its best {PAIRS_PER_CONVERSATION} pairs",
    )
    args = parser.parse_args(argv)
    queries = [q.strip() for q in args.queries if q.strip()]
    return asyncio.run(
        run(
            args.user_id,
            queries,
            top=args.top,
            per_conversation=not args.all_pairs,
            show_pairs=args.pairs,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
