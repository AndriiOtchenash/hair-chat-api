"""Experiment: calibrate the conversation search. Reads only, writes nothing.

    uv run python -m app.cli.search_probe USER_ID "query" ["query" ...] [--top N] [--all-pairs]

Compares, in memory, cosine scores of one user's indexed answers (message_embeddings) in three
modes:

    A  the stored vectors as they are, the query embedded as is (what the search does today);
    B  the pairs re-embedded in gemini-embedding-2's document format
       ("title: <conversation title> | text: Вопрос: ...\\nОтвет: ..."), the query in its
       query format ("task: search result | query: ...");
    C  the user's question only in the document format, the query in the query format.

Hypotheses: the model wants asymmetric prefixes (ai.google.dev/gemini-api/docs/embeddings), and
a long boilerplate answer blurs the pair's vector. One embeddings call per mode (batches of 100).
Like the API, the best hit per conversation is shown unless --all-pairs. The output prints
conversation titles and the queries: a dev tool for the operator's own data, not for logs.
"""

import argparse
import asyncio
import math
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import select, true, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.core.config import get_settings
from app.db.models import Conversation, Message, MessageEmbedding, MessageRole
from app.db.session import get_engine, get_sessionmaker
from app.rag.conversation_index import pair_text
from app.rag.embeddings import Embedder, EmbeddingUnavailable, OpenAICompatibleEmbedder
from app.rag.ingest import INGEST_RETRIES, INGEST_TIMEOUT_S

MODES = ("A", "B", "C")
NO_TITLE = "none"  # Google's placeholder for a document without a title


def query_text(query: str) -> str:
    """gemini-embedding-2 query format for search (asymmetric retrieval)."""
    return f"task: search result | query: {query.strip()}"


def document_text(title: str | None, content: str) -> str:
    """gemini-embedding-2 document format; a missing or blank title becomes "none"."""
    title = (title or "").strip() or NO_TITLE
    return f"title: {title} | text: {content.strip()}"


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
    if not per_conversation:
        return hits
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
    return [Pair(mid, cid, title, q, a, list(vec)) for mid, cid, title, q, a, vec in rows]


async def probe(
    pairs: Sequence[Pair],
    queries: Sequence[str],
    embedder: Embedder,
    *,
    max_chars: int,
    per_conversation: bool = True,
) -> dict[tuple[str, str], list[Hit]]:
    """(query, mode) -> hits. Four embeddings calls in parallel: raw queries, prefixed queries,
    B documents, C documents (each split into batches of 100 by the embedder)."""
    raw_q, pref_q, docs_b, docs_c = await asyncio.gather(
        embedder.embed(list(queries)),
        embedder.embed([query_text(q) for q in queries]),
        embedder.embed(
            [
                document_text(p.title, pair_text(p.question, p.answer, max_chars=max_chars))
                for p in pairs
            ]
        ),
        embedder.embed([document_text(p.title, p.question) for p in pairs]),
    )
    stored = [p.vector for p in pairs]
    results: dict[tuple[str, str], list[Hit]] = {}
    for i, query in enumerate(queries):
        for mode, docs, qv in (
            ("A", stored, raw_q[i]),
            ("B", docs_b, pref_q[i]),
            ("C", docs_c, pref_q[i]),
        ):
            results[(query, mode)] = rank(pairs, docs, qv, per_conversation=per_conversation)
    return results


def render(results: dict[tuple[str, str], list[Hit]], queries: Sequence[str], *, top: int) -> str:
    qw = min(max((len(q) for q in queries), default=5), 30)
    lines = [f"{'query':<{qw}} | mode | {'title':<30} | score", "-" * (qw + 48)]
    gaps: list[str] = []
    for query in queries:
        for mode in MODES:
            hits = results[(query, mode)]
            for hit in hits[:top]:
                title = (hit.pair.title or "(no title)")[:30]
                lines.append(f"{query[:qw]:<{qw}} | {mode:<4} | {title:<30} | {hit.score:.4f}")
            g = gap(hits)
            gaps.append(f"{query[:qw]:<{qw}} | {mode:<4} | " + ("n/a" if g is None else f"{g:.4f}"))
        lines.append("")
    lines += [f"{'query':<{qw}} | mode | gap top1-top2", "-" * (qw + 23), *gaps]
    return "\n".join(lines)


async def run(
    user_id: uuid.UUID, queries: Sequence[str], *, top: int, per_conversation: bool
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
            per_conversation=per_conversation,
        )
    except EmbeddingUnavailable as exc:
        print(f"embeddings unavailable ({exc.code})", file=sys.stderr)
        return 1
    finally:
        await embedder.aclose()
        await get_engine().dispose()
    unit = "conversation (best pair)" if per_conversation else "pair"
    print(f"{len(pairs)} pairs, model {settings.embedding_model}, one row per {unit}\n")
    print(render(results, queries, top=top))
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
    parser.add_argument("--all-pairs", action="store_true", help="rank pairs, not conversations")
    args = parser.parse_args(argv)
    queries = [q.strip() for q in args.queries if q.strip()]
    return asyncio.run(
        run(args.user_id, queries, top=args.top, per_conversation=not args.all_pairs)
    )


if __name__ == "__main__":
    sys.exit(main())
