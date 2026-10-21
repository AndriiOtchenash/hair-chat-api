"""RAG eval against the real embeddings API: tests/rag_eval/queries.yaml -> hit@4, false finds.

    uv run python -m scripts.rag_eval [--threshold 0.65] [--kb docs/kb] [--file PATH]

Embeds only our own texts (docs/kb articles and the YAML questions), never user data. Not run in
CI (real API). No database either: chunks come from the same loader and chunker as the ingest and
are searched in memory by cosine similarity (what pgvector's <=> computes), through the same
RagKnowledge (top-k, threshold, token budget) the chat uses.

hit@4: the expected article is among the sources that reach the context. False find: an off-topic
question (expected: null) gets any source. Exit code 1 only on a new miss (no `known_gap`).
A sweep over thresholds helps to pick KB_MIN_SIMILARITY: the highest hit@4 with no false finds.
"""

import argparse
import asyncio
import math
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.core.config import get_settings
from app.db.repositories.kb import KBHit
from app.rag.chunker import chunk_markdown
from app.rag.documents import load_articles
from app.rag.embeddings import OpenAICompatibleEmbedder
from app.rag.retriever import QueryEmbedding, RagKnowledge

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILE = ROOT / "tests/rag_eval/queries.yaml"
DEFAULT_KB = ROOT / "docs/kb"
SWEEP = [round(0.50 + 0.05 * i, 2) for i in range(7)]  # 0.50 .. 0.80
TEXT_WIDTH = 60
EVAL_USER = uuid.UUID(int=0)


@dataclass(frozen=True)
class Query:
    lang: str
    expected: str | None
    text: str
    known_gap: str | None = None


@dataclass(frozen=True)
class Outcome:
    query: Query
    found: tuple[str, ...]  # slugs that reached the context
    ranked: list[KBHit]  # top-k before the threshold

    @property
    def ok(self) -> bool:
        if self.query.expected is None:
            return not self.found
        return self.query.expected in self.found

    @property
    def new_miss(self) -> bool:
        return not self.ok and self.query.known_gap is None


class MemoryKB:
    """KBRepository over chunks held in memory; cosine similarity like pgvector's <=>."""

    def __init__(self, rows: Sequence[tuple[KBHit, Sequence[float]]]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    async def search(
        self, vector: Sequence[float], *, model: str, include_drafts: bool, limit: int
    ) -> list[KBHit]:
        scored = [
            KBHit(hit.content, hit.title, hit.slug, hit.url, _cosine(vector, v))
            for hit, v in self._rows
        ]
        return sorted(scored, key=lambda h: -h.similarity)[:limit]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def load(path: Path) -> list[Query]:
    rows: list[dict[str, Any]] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [Query(r["lang"], r["expected"], r["text"], r.get("known_gap")) for r in rows]


async def embed_all(kb_dir: Path, queries: list[Query]) -> tuple[MemoryKB, list[QueryEmbedding]]:
    settings = get_settings()
    if settings.gemini_api_key is None:
        raise SystemExit("GEMINI_API_KEY is not set")
    embedder = OpenAICompatibleEmbedder(
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        api_key=settings.gemini_api_key.get_secret_value(),
        timeout=30,
        max_retries=3,
    )
    try:
        chunks = [
            (article, chunk)
            for article in load_articles(kb_dir)
            for chunk in chunk_markdown(article.meta.title, article.body)
        ]
        chunk_vectors = await embedder.embed([c.text for _, c in chunks])
        query_vectors = await embedder.embed([q.text for q in queries])
    finally:
        await embedder.aclose()
    rows = [
        (KBHit(chunk.text, article.meta.title, article.slug, article.meta.url, 0.0), vector)
        for (article, chunk), vector in zip(chunks, chunk_vectors, strict=True)
    ]
    embeddings = [QueryEmbedding(v, embedder.model) for v in query_vectors]
    return MemoryKB(rows), embeddings


async def evaluate(
    kb: MemoryKB, queries: list[Query], embeddings: list[QueryEmbedding], threshold: float
) -> list[Outcome]:
    settings = get_settings()
    knowledge = RagKnowledge(
        kb,
        top_k=settings.kb_top_k,
        min_similarity=threshold,
        max_tokens=settings.kb_max_tokens,
        include_drafts=True,
    )
    outcomes: list[Outcome] = []
    for query, embedding in zip(queries, embeddings, strict=True):
        result = await knowledge.knowledge(EVAL_USER, query.text, embedding)
        found = tuple(s.slug for s in result.sources) if result else ()
        ranked = await kb.search(
            embedding.vector, model=embedding.model, include_drafts=True, limit=settings.kb_top_k
        )
        outcomes.append(Outcome(query, found, ranked))
    return outcomes


def _short(text: str) -> str:
    return text if len(text) <= TEXT_WIDTH else text[: TEXT_WIDTH - 1] + "…"


def _scores(outcome: Outcome) -> str:
    """Top-k before the threshold: slug=similarity, expected one marked with *."""
    return " ".join(
        f"{'*' if h.slug == outcome.query.expected else ''}{h.slug}={h.similarity:.3f}"
        for h in outcome.ranked
    )


def stats(outcomes: list[Outcome]) -> tuple[int, int, int, int]:
    on_topic = [o for o in outcomes if o.query.expected is not None]
    off_topic = [o for o in outcomes if o.query.expected is None]
    hits = sum(o.ok for o in on_topic)
    false_finds = sum(not o.ok for o in off_topic)
    return hits, len(on_topic), false_finds, len(off_topic)


def report(outcomes: list[Outcome], threshold: float) -> str:
    lines = [f"threshold {threshold}", "ok  lang expected                     | top-k | question"]
    for o in outcomes:
        mark = "  " if o.ok else "XX" if o.new_miss else "kg"
        expected = o.query.expected or "-"
        lines.append(
            f"{mark}  {o.query.lang:<4} {expected:<28} | {_scores(o)} | {_short(o.query.text)}"
        )
    hits, on_topic, false_finds, off_topic = stats(outcomes)
    new = [o for o in outcomes if o.new_miss]
    gaps = [o for o in outcomes if not o.ok and not o.new_miss]
    closed = [o for o in outcomes if o.ok and o.query.known_gap is not None]
    lines += [
        "",
        f"hit@4 {hits}/{on_topic}, false finds {false_finds}/{off_topic}, "
        f"{len(gaps)} known gaps, {len(new)} new misses",
    ]
    for title, items in (
        ("New misses (fail the run):", new),
        ("Known gaps:", gaps),
        ("Known gaps that now pass (remove known_gap):", closed),
    ):
        if items:
            lines.append(title)
            for o in items:
                reason = f"  [{o.query.known_gap}]" if o.query.known_gap else ""
                found = ", ".join(o.found) or "-"
                lines.append(
                    f"  expected {o.query.expected}, found {found}: {o.query.text}{reason}"
                )
    return "\n".join(lines)


async def run(kb_dir: Path, path: Path, threshold: float) -> int:
    queries = load(path)
    kb, embeddings = await embed_all(kb_dir, queries)
    print(f"model {get_settings().embedding_model}, {len(kb)} chunks, {len(queries)} queries")
    print("\nthreshold sweep: hit@4 / false finds")
    for value in SWEEP:
        hits, on_topic, false_finds, off_topic = stats(
            await evaluate(kb, queries, embeddings, value)
        )
        print(f"  {value:.2f}  {hits}/{on_topic}  {false_finds}/{off_topic}")
    outcomes = await evaluate(kb, queries, embeddings, threshold)
    print()
    print(report(outcomes, threshold))
    return 1 if any(o.new_miss for o in outcomes) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scripts.rag_eval", description=__doc__)
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--kb", type=Path, default=DEFAULT_KB)
    parser.add_argument("--threshold", type=float, default=None, help="default KB_MIN_SIMILARITY")
    args = parser.parse_args(argv)
    threshold = args.threshold if args.threshold is not None else get_settings().kb_min_similarity
    return asyncio.run(run(args.kb, args.file, threshold))


if __name__ == "__main__":
    sys.exit(main())
