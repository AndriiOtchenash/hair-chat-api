"""Backfill the conversation search index:

    uv run python -m app.cli.index_conversations [--dry-run] [--limit N]

Embeds every answer that has no row in message_embeddings, or one of another EMBEDDING_MODEL
or CONVERSATION_INDEX_FORMAT_VERSION (or NULL), in batches (one embeddings call and one
transaction each), with the ingest CLI's longer timeout and retries,
not the chat's. A failure keeps what is written; re-run to continue. Users without every valid
consent (CONSENT_VERSION) are skipped and counted: nothing of theirs is sent to the embeddings
API. Small talk (a short answer to a short question, SmallTalk) is not indexed: rows such
pairs still have are deleted and counted. Reads DATABASE_URL and GEMINI_API_KEY
like the app. Counts only in the output and the log: no message text, no user ids (like
app.cli.rag_report).
"""

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence

from app.core.config import get_settings
from app.db.repositories.message_embeddings import SmallTalk, SqlMessageEmbeddingRepository
from app.db.session import get_engine, get_sessionmaker
from app.rag.conversation_index import index_conversations
from app.rag.embeddings import EmbeddingUnavailable, OpenAICompatibleEmbedder
from app.rag.ingest import INGEST_RETRIES, INGEST_TIMEOUT_S


async def run(*, dry_run: bool, limit: int | None) -> int:
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
            report = await index_conversations(
                SqlMessageEmbeddingRepository(session, consent_version=settings.consent_version),
                embedder,
                max_chars=settings.conversation_index_max_chars,
                small_talk=SmallTalk(
                    settings.conversation_index_min_answer_chars,
                    settings.conversation_index_max_small_talk_question_chars,
                ),
                limit=limit,
                dry_run=dry_run,
            )
    except EmbeddingUnavailable as exc:
        print(f"embeddings unavailable ({exc.code}); re-run to continue", file=sys.stderr)
        return 1
    finally:
        await embedder.aclose()
        await get_engine().dispose()
    prefix = "dry run: " if dry_run else ""
    print(f"{prefix}{report.summary()} (model {settings.embedding_model})")
    return 0


def _positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return number


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli.index_conversations", description=__doc__
    )
    parser.add_argument("--dry-run", action="store_true", help="count only, embed nothing")
    parser.add_argument("--limit", type=_positive, help="look at N answers at most")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return asyncio.run(run(dry_run=args.dry_run, limit=args.limit))


if __name__ == "__main__":
    sys.exit(main())
