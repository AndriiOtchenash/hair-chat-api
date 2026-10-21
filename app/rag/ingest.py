"""Load docs/kb into the knowledge base: python -m app.rag.ingest docs/kb [--dry-run]

Idempotent, the directory is the source of truth:
- unchanged article (same content hash, vectors of the current EMBEDDING_MODEL) -> metadata only,
  no embedding call;
- new or changed text, chunker or model -> re-chunked, re-embedded, chunks replaced;
- article whose file is gone -> deleted with its chunks.
Each article is embedded first and then written in its own transaction, so a failure leaves
every article either old or new, never half-written; re-running continues where it stopped.
Logs slugs and counts only.
"""

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from app.core.config import get_settings
from app.db.repositories.kb import SqlKBRepository, StoredDocument
from app.db.session import get_engine, get_sessionmaker
from app.rag.chunker import Chunk, chunk_markdown
from app.rag.documents import Article, InvalidArticle, load_articles
from app.rag.embeddings import Embedder, EmbeddingUnavailable, OpenAICompatibleEmbedder

logger = logging.getLogger(__name__)

# The CLI is not latency-bound: a longer timeout and a few SDK retries for 429/5xx.
INGEST_TIMEOUT_S = 30.0
INGEST_RETRIES = 3


class IngestRepository(Protocol):
    async def stored_documents(self) -> dict[str, StoredDocument]: ...

    async def update_metadata(self, article: Article) -> None: ...

    async def replace(
        self,
        article: Article,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        *,
        model: str,
    ) -> None: ...

    async def delete_except(self, slugs: Sequence[str]) -> list[str]: ...


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    chunks_embedded: int = 0

    def summary(self) -> str:
        return (
            f"added {len(self.added)}, updated {len(self.updated)}, "
            f"unchanged {len(self.unchanged)}, deleted {len(self.deleted)}, "
            f"chunks embedded {self.chunks_embedded}"
        )


async def ingest(
    articles: Sequence[Article],
    repo: IngestRepository,
    embedder: Embedder,
    *,
    dry_run: bool = False,
) -> IngestReport:
    report = IngestReport()
    stored = await repo.stored_documents()
    for article in articles:
        current = stored.get(article.slug)
        fresh = (
            current is not None
            and current.content_hash == article.content_hash
            and current.embedding_models == {embedder.model}
        )
        if fresh:
            report.unchanged.append(article.slug)
            if not dry_run:
                await repo.update_metadata(article)
            continue
        (report.updated if current else report.added).append(article.slug)
        chunks = chunk_markdown(article.meta.title, article.body)
        report.chunks_embedded += len(chunks)
        if dry_run:
            continue
        vectors = await embedder.embed([c.text for c in chunks])
        await repo.replace(article, chunks, vectors, model=embedder.model)
        logger.info("kb article indexed: slug=%s chunks=%d", article.slug, len(chunks))
    gone = sorted(set(stored) - {a.slug for a in articles})
    if dry_run:
        report.deleted = gone
    elif gone:
        report.deleted = await repo.delete_except([a.slug for a in articles])
    return report


async def _main(directory: Path, dry_run: bool) -> int:
    settings = get_settings()
    try:
        articles = load_articles(directory)
    except InvalidArticle as exc:
        print(f"invalid article: {exc}", file=sys.stderr)
        return 2
    if not articles:
        # Would delete the whole index; almost certainly a wrong path.
        print(f"no articles in {directory}, nothing done", file=sys.stderr)
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
            report = await ingest(articles, SqlKBRepository(session), embedder, dry_run=dry_run)
    except EmbeddingUnavailable as exc:
        print(f"embeddings unavailable ({exc.code}); re-run to continue", file=sys.stderr)
        return 1
    finally:
        await embedder.aclose()
        await get_engine().dispose()
    prefix = "dry run: " if dry_run else ""
    print(f"{prefix}{report.summary()} (model {settings.embedding_model})")
    for title, slugs in (
        ("added", report.added),
        ("updated", report.updated),
        ("deleted", report.deleted),
    ):
        if slugs:
            print(f"  {title}: {', '.join(slugs)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.rag.ingest", description=__doc__)
    parser.add_argument("directory", type=Path, help="e.g. docs/kb")
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return asyncio.run(_main(args.directory, args.dry_run))


if __name__ == "__main__":
    sys.exit(main())
