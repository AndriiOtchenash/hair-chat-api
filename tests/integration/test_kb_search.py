"""pgvector search: cosine order, similarity values, status and embedding-model filters."""

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import delete

from app.db.models import KBDocument, KBDocumentStatus
from app.db.repositories.kb import SqlKBRepository
from app.db.session import get_sessionmaker
from app.rag.chunker import Chunk
from app.rag.documents import Article, parse_article
from tests.llm_fakes import unit_vector

MODEL = "fake-embedding"


def article(slug: str, status: KBDocumentStatus = KBDocumentStatus.REVIEWED) -> Article:
    return parse_article(
        slug,
        f"---\ntitle: {slug.title()}\nlanguage: ru\nstatus: {status.value}\ntopics: [t]\n"
        f"license: internal\n---\n\n{slug} body\n",
    )


async def seed(
    slug: str,
    vector: list[float],
    *,
    status: KBDocumentStatus = KBDocumentStatus.REVIEWED,
    model: str = MODEL,
) -> None:
    async with get_sessionmaker()() as session:
        await SqlKBRepository(session).replace(
            article(slug, status), [Chunk(0, f"{slug} chunk")], [vector], model=model
        )


@pytest.fixture
async def repo(app_settings_env: None) -> AsyncIterator[SqlKBRepository]:
    async with get_sessionmaker()() as session:
        await session.execute(delete(KBDocument))
        await session.commit()
        yield SqlKBRepository(session)


async def test_nearest_chunks_first_with_cosine_similarity(repo: SqlKBRepository) -> None:
    await seed("far", unit_vector(0.0, 1.0))
    await seed("near", unit_vector(1.0, 0.1))
    await seed("middle", unit_vector(1.0, 1.0))

    hits = await repo.search(unit_vector(1.0), model=MODEL, include_drafts=False, limit=2)

    assert [h.slug for h in hits] == ["near", "middle"]
    assert hits[0].similarity == pytest.approx(1 / (1 + 0.01) ** 0.5, abs=1e-4)
    assert hits[1].similarity == pytest.approx(2**-0.5, abs=1e-4)
    assert (hits[0].title, hits[0].content, hits[0].url) == ("Near", "near chunk", None)


async def test_drafts_only_when_asked(repo: SqlKBRepository) -> None:
    await seed("reviewed", unit_vector(1.0, 0.5))
    await seed("draft", unit_vector(1.0), status=KBDocumentStatus.DRAFT)

    reviewed_only = await repo.search(unit_vector(1.0), model=MODEL, include_drafts=False, limit=4)
    with_drafts = await repo.search(unit_vector(1.0), model=MODEL, include_drafts=True, limit=4)

    assert [h.slug for h in reviewed_only] == ["reviewed"]
    assert [h.slug for h in with_drafts] == ["draft", "reviewed"]


async def test_rows_of_another_embedding_model_are_never_returned(repo: SqlKBRepository) -> None:
    await seed("current", unit_vector(0.5, 1.0))
    await seed("old-model", unit_vector(1.0), model="old-embedding")

    hits = await repo.search(unit_vector(1.0), model=MODEL, include_drafts=True, limit=4)

    assert [h.slug for h in hits] == ["current"]
