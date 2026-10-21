"""Knowledge base tables: search for the chat, writes for the ingest CLI.

Not user data: kb_documents/kb_chunks are shared by all users, so there is no user_id here.
"""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import delete, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import KBChunk, KBDocument, KBDocumentStatus
from app.rag.chunker import Chunk
from app.rag.documents import Article


@dataclass(frozen=True, slots=True)
class KBHit:
    content: str
    title: str
    slug: str
    url: str | None
    similarity: float  # cosine similarity, 1 = same direction


@dataclass(frozen=True, slots=True)
class StoredDocument:
    id: uuid.UUID
    content_hash: str
    # Models of its chunks; anything but {current model} means the vectors must be rebuilt.
    embedding_models: frozenset[str]


class KBRepository(Protocol):
    async def search(
        self, vector: Sequence[float], *, model: str, include_drafts: bool, limit: int
    ) -> list[KBHit]:
        """Nearest chunks by cosine distance, closest first; only rows of `model` (rule 4)."""
        ...


class SqlKBRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def search(
        self, vector: Sequence[float], *, model: str, include_drafts: bool, limit: int
    ) -> list[KBHit]:
        # The threshold is applied by the caller: a WHERE on the distance would keep the
        # planner from walking the HNSW index in distance order.
        distance = KBChunk.embedding.cosine_distance(vector)
        query = (
            select(
                KBChunk.content,
                KBDocument.title,
                KBDocument.slug,
                KBDocument.source_url,
                distance.label("distance"),
            )
            .join(KBDocument, KBDocument.id == KBChunk.document_id)
            .where(KBChunk.embedding_model == model)
            .order_by(distance)
            .limit(limit)
        )
        if not include_drafts:
            query = query.where(KBDocument.status == KBDocumentStatus.REVIEWED)
        rows = (await self._session.execute(query)).all()
        return [
            KBHit(content, title, slug, url, 1.0 - float(dist))
            for content, title, slug, url, dist in rows
        ]

    # --- ingest ---

    async def stored_documents(self) -> dict[str, StoredDocument]:
        rows = (
            await self._session.execute(
                select(
                    KBDocument.slug,
                    KBDocument.id,
                    KBDocument.content_hash,
                    KBChunk.embedding_model,
                ).outerjoin(KBChunk, KBChunk.document_id == KBDocument.id)
            )
        ).all()
        found: dict[str, tuple[uuid.UUID, str, set[str]]] = {}
        for slug, doc_id, content_hash, model in rows:
            _, _, models = found.setdefault(slug, (doc_id, content_hash, set()))
            if model is not None:
                models.add(model)
        return {
            slug: StoredDocument(doc_id, content_hash, frozenset(models))
            for slug, (doc_id, content_hash, models) in found.items()
        }

    async def update_metadata(self, article: Article) -> None:
        """Status, title, license... without touching the chunks."""
        await self._session.execute(
            update(KBDocument).where(KBDocument.slug == article.slug).values(**_fields(article))
        )
        await self._session.commit()

    async def replace(
        self,
        article: Article,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
        *,
        model: str,
    ) -> None:
        """Upsert the document and swap all its chunks, in one transaction."""
        doc_id = await self._session.scalar(
            select(KBDocument.id).where(KBDocument.slug == article.slug)
        )
        if doc_id is None:
            doc_id = await self._session.scalar(
                insert(KBDocument)
                .values(slug=article.slug, **_fields(article))
                .returning(KBDocument.id)
            )
        else:
            await self._session.execute(
                update(KBDocument).where(KBDocument.id == doc_id).values(**_fields(article))
            )
            await self._session.execute(delete(KBChunk).where(KBChunk.document_id == doc_id))
        if chunks:
            await self._session.execute(
                insert(KBChunk),
                [
                    {
                        "document_id": doc_id,
                        "chunk_index": chunk.index,
                        "content": chunk.text,
                        "embedding": list(vector),
                        "embedding_model": model,
                    }
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ],
            )
        await self._session.commit()

    async def delete_except(self, slugs: Sequence[str]) -> list[str]:
        """Remove documents (and, by cascade, chunks) whose file is gone; returns their slugs."""
        deleted = await self._session.scalars(
            delete(KBDocument).where(KBDocument.slug.not_in(slugs)).returning(KBDocument.slug)
        )
        result = sorted(deleted.all())
        await self._session.commit()
        return result


def _fields(article: Article) -> dict[str, object]:
    meta = article.meta
    return {
        "title": meta.title,
        "status": meta.status,
        "language": meta.language,
        "license": meta.license,
        "source_url": meta.url,
        "content_hash": article.content_hash,
    }
