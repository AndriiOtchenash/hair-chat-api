"""Ingest decisions over an in-memory repository; the SQL side is in tests/integration."""

import uuid
from collections.abc import Sequence
from pathlib import Path

import pytest

from app.core.config import Settings
from app.db.repositories.kb import StoredDocument
from app.rag import ingest as ingest_module
from app.rag.chunker import Chunk
from app.rag.documents import Article, parse_article
from app.rag.ingest import ingest, main
from tests.llm_fakes import FakeEmbedder


def article(slug: str, body: str = "## Раздел\n\nТекст.", status: str = "draft") -> Article:
    return parse_article(
        slug,
        f"---\ntitle: {slug}\nlanguage: ru\nstatus: {status}\ntopics: [t]\n"
        f"license: internal\n---\n\n{body}\n",
    )


class MemoryRepo:
    def __init__(self) -> None:
        self.docs: dict[str, tuple[Article, list[Chunk], str]] = {}
        self.metadata_updates: list[str] = []

    async def stored_documents(self) -> dict[str, StoredDocument]:
        return {
            slug: StoredDocument(uuid.uuid4(), a.content_hash, frozenset({model}))
            for slug, (a, _, model) in self.docs.items()
        }

    async def update_metadata(self, article: Article) -> None:
        self.metadata_updates.append(article.slug)
        _, chunks, model = self.docs[article.slug]
        self.docs[article.slug] = (article, chunks, model)

    async def replace(
        self,
        article: Article,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        *,
        model: str,
    ) -> None:
        assert len(chunks) == len(vectors)
        self.docs[article.slug] = (article, list(chunks), model)

    async def delete_except(self, slugs: Sequence[str]) -> list[str]:
        gone = sorted(set(self.docs) - set(slugs))
        for slug in gone:
            del self.docs[slug]
        return gone


async def test_new_changed_unchanged_and_removed_articles() -> None:
    repo = MemoryRepo()
    await ingest([article("a"), article("b"), article("c")], repo, FakeEmbedder())
    embedder = FakeEmbedder()

    report = await ingest(
        [article("a"), article("b", "## Раздел\n\nНовый текст."), article("d")], repo, embedder
    )

    assert (report.added, report.updated, report.unchanged, report.deleted) == (
        ["d"],
        ["b"],
        ["a"],
        ["c"],
    )
    assert report.chunks_embedded == len(embedder.embedded) == 2
    assert repo.metadata_updates == ["a"]
    assert set(repo.docs) == {"a", "b", "d"}


async def test_dry_run_reports_without_writing_or_embedding() -> None:
    repo = MemoryRepo()
    await ingest([article("old")], repo, FakeEmbedder())
    embedder = FakeEmbedder()

    report = await ingest([article("new")], repo, embedder, dry_run=True)

    assert (report.added, report.deleted) == (["new"], ["old"])
    assert embedder.calls == 0
    assert set(repo.docs) == {"old"}
    assert "added 1" in report.summary() and "deleted 1" in report.summary()


async def test_another_embedding_model_reembeds_unchanged_text() -> None:
    repo = MemoryRepo()
    await ingest([article("a")], repo, FakeEmbedder(model="old"))

    report = await ingest([article("a")], repo, FakeEmbedder(model="new"))

    assert report.updated == ["a"]
    assert repo.docs["a"][2] == "new"


def test_cli_refuses_an_invalid_article(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "broken.md").write_text("no frontmatter", encoding="utf-8")

    assert main([str(tmp_path)]) == 2
    assert "broken.md" in capsys.readouterr().err


def test_cli_refuses_an_empty_directory_instead_of_deleting_the_index(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([str(tmp_path)]) == 2
    assert "no articles" in capsys.readouterr().err


def test_cli_needs_the_gemini_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.md").write_text(
        "---\ntitle: A\nlanguage: ru\nstatus: draft\ntopics: [t]\nlicense: internal\n---\n\nx\n",
        encoding="utf-8",
    )
    settings = Settings(_env_file=None, gemini_api_key=None)
    monkeypatch.setattr(ingest_module, "get_settings", lambda: settings)

    assert main([str(tmp_path)]) == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().err
