"""Ingest against real Postgres: idempotent re-runs, edits, deletions, model change.

Fake embeddings only (rule 1): FakeEmbedder counts what it was asked to embed.
"""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy import delete, select

from app.db.models import KBChunk, KBDocument, KBDocumentStatus
from app.db.repositories.kb import SqlKBRepository
from app.db.session import get_sessionmaker
from app.rag.documents import load_articles
from app.rag.ingest import IngestReport, ingest
from tests.llm_fakes import FakeEmbedder

LONG_SECTION = "\n\n".join(
    f"Абзац {i}: мягкий шампунь, тёплая вода, бережное расчёсывание." for i in range(60)
)


def write(kb: Path, slug: str, title: str, body: str, status: str = "draft") -> None:
    (kb / f"{slug}.md").write_text(
        f"---\ntitle: {title}\nlanguage: ru\nstatus: {status}\ntopics: [test]\n"
        f"sources: []\nlicense: internal\n---\n\n{body}\n",
        encoding="utf-8",
    )


@pytest.fixture
async def kb_dir(app_settings_env: None, tmp_path: Path) -> AsyncIterator[Path]:
    async with get_sessionmaker()() as session:
        await session.execute(delete(KBDocument))
        await session.commit()
    write(tmp_path, "dandruff", "Перхоть", f"## Уход\n\n{LONG_SECTION}\n\n## Врач\n\nЕсли больно.")
    write(tmp_path, "washing", "Мытьё", "## Как часто\n\nПо мере загрязнения.")
    (tmp_path / "README.md").write_text("# not an article", encoding="utf-8")
    yield tmp_path


async def run(kb: Path, embedder: FakeEmbedder) -> IngestReport:
    async with get_sessionmaker()() as session:
        return await ingest(load_articles(kb), SqlKBRepository(session), embedder)


async def stored() -> dict[str, tuple[KBDocument, list[KBChunk]]]:
    async with get_sessionmaker()() as session:
        docs = (await session.scalars(select(KBDocument))).all()
        result = {}
        for doc in docs:
            chunks = await session.scalars(
                select(KBChunk).where(KBChunk.document_id == doc.id).order_by(KBChunk.chunk_index)
            )
            result[doc.slug] = (doc, list(chunks))
        return result


async def test_first_run_indexes_every_article_with_the_model(kb_dir: Path) -> None:
    embedder = FakeEmbedder()

    report = await run(kb_dir, embedder)

    assert sorted(report.added) == ["dandruff", "washing"]
    rows = await stored()
    assert set(rows) == {"dandruff", "washing"}
    doc, chunks = rows["dandruff"]
    assert (doc.title, doc.status, doc.language) == ("Перхоть", KBDocumentStatus.DRAFT, "ru")
    assert len(chunks) > 1  # the long section is split
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert {c.embedding_model for c in chunks} == {"fake-embedding"}
    assert report.chunks_embedded == len(embedder.embedded) == sum(len(c) for _, c in rows.values())


async def test_second_run_embeds_nothing(kb_dir: Path) -> None:
    await run(kb_dir, FakeEmbedder())
    before = {slug: [c.id for c in chunks] for slug, (_, chunks) in (await stored()).items()}
    embedder = FakeEmbedder()

    report = await run(kb_dir, embedder)

    assert sorted(report.unchanged) == ["dandruff", "washing"]
    assert (report.added, report.updated, report.deleted) == ([], [], [])
    assert embedder.calls == 0
    after = {slug: [c.id for c in chunks] for slug, (_, chunks) in (await stored()).items()}
    assert after == before


async def test_edited_article_gets_new_chunks_others_stay(kb_dir: Path) -> None:
    await run(kb_dir, FakeEmbedder())
    old = await stored()
    write(kb_dir, "washing", "Мытьё", "## Как часто\n\nКогда корни загрязнились.")
    embedder = FakeEmbedder()

    report = await run(kb_dir, embedder)

    assert report.updated == ["washing"]
    new = await stored()
    edited = "Мытьё › Как часто\n\nКогда корни загрязнились."
    assert [c.content for c in new["washing"][1]] == [edited]
    assert new["washing"][0].id == old["washing"][0].id  # same document row
    assert [c.id for c in new["dandruff"][1]] == [c.id for c in old["dandruff"][1]]
    assert embedder.embedded == [edited]


async def test_review_updates_the_status_without_embedding(kb_dir: Path) -> None:
    await run(kb_dir, FakeEmbedder())
    write(kb_dir, "washing", "Мытьё", "## Как часто\n\nПо мере загрязнения.", status="reviewed")
    embedder = FakeEmbedder()

    report = await run(kb_dir, embedder)

    assert "washing" in report.unchanged
    assert embedder.calls == 0
    assert (await stored())["washing"][0].status is KBDocumentStatus.REVIEWED


async def test_removed_file_deletes_the_article_and_its_chunks(kb_dir: Path) -> None:
    await run(kb_dir, FakeEmbedder())
    (kb_dir / "washing.md").unlink()

    report = await run(kb_dir, FakeEmbedder())

    assert report.deleted == ["washing"]
    assert set(await stored()) == {"dandruff"}
    async with get_sessionmaker()() as session:
        orphans = await session.scalars(select(KBChunk.id).where(KBChunk.content.like("Мытьё%")))
        assert orphans.all() == []


async def test_new_embedding_model_reindexes_everything(kb_dir: Path) -> None:
    await run(kb_dir, FakeEmbedder())
    embedder = FakeEmbedder(model="fake-embedding-v2")

    report = await run(kb_dir, embedder)

    assert sorted(report.updated) == ["dandruff", "washing"]
    models = {c.embedding_model for _, chunks in (await stored()).values() for c in chunks}
    assert models == {"fake-embedding-v2"}


async def test_dry_run_writes_nothing(kb_dir: Path) -> None:
    embedder = FakeEmbedder()

    async with get_sessionmaker()() as session:
        report = await ingest(
            load_articles(kb_dir), SqlKBRepository(session), embedder, dry_run=True
        )

    assert sorted(report.added) == ["dandruff", "washing"]
    assert embedder.calls == 0
    assert await stored() == {}
