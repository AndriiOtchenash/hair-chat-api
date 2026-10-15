import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Enum, ForeignKey, Index, Text, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPkMixin

EMBEDDING_DIM = 768


class Locale(StrEnum):
    UK = "uk"
    PL = "pl"
    EN = "en"
    RU = "ru"


class UserRole(StrEnum):
    USER = "user"
    SPECIALIST = "specialist"
    ADMIN = "admin"


class ConsentType(StrEnum):
    AGE_18 = "age_18"
    HEALTH_DATA = "health_data"
    AI_PROCESSING = "ai_processing"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class LLMCallStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    FALLBACK = "fallback"


def _pg_enum(enum_cls: type[StrEnum], name: str) -> Enum:
    """Native Postgres ENUM named `name`, storing the lowercase values (not member names)."""
    return Enum(enum_cls, name=name, values_callable=lambda cls: [m.value for m in cls])


class User(UUIDPkMixin, TimestampMixin, Base):
    __tablename__ = "users"

    auth_sub: Mapped[str] = mapped_column(Text, unique=True)
    email: Mapped[str | None] = mapped_column(Text)
    display_name: Mapped[str | None] = mapped_column(Text)
    locale: Mapped[Locale] = mapped_column(
        _pg_enum(Locale, "user_locale"), default=Locale.EN, server_default=Locale.EN.value
    )
    role: Mapped[UserRole] = mapped_column(
        _pg_enum(UserRole, "user_role"), default=UserRole.USER, server_default=UserRole.USER.value
    )
    age_confirmed_at: Mapped[datetime | None]
    deleted_at: Mapped[datetime | None]


class Consent(UUIDPkMixin, TimestampMixin, Base):
    __tablename__ = "consents"
    __table_args__ = (
        # At most one active (non-revoked) row per user/type/version: makes grant idempotent via
        # ON CONFLICT DO NOTHING, while revoked rows stay as history.
        Index(
            "ix_consents_user_id_type_version_active",
            "user_id",
            "type",
            "version",
            unique=True,
            postgresql_where=text("revoked_at IS NULL"),
        ),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    type: Mapped[ConsentType] = mapped_column(_pg_enum(ConsentType, "consent_type"))
    version: Mapped[str] = mapped_column(Text)
    granted_at: Mapped[datetime] = mapped_column(server_default=func.now())
    revoked_at: Mapped[datetime | None]


class Conversation(UUIDPkMixin, TimestampMixin, Base):
    __tablename__ = "conversations"

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    title: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    # conversations <-> messages is a cycle; use_alter adds this FK after both tables exist.
    summary_upto_message_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("messages.id", ondelete="SET NULL", use_alter=True)
    )
    last_message_at: Mapped[datetime] = mapped_column(server_default=func.now())


Index(
    "ix_conversations_user_id_last_message_at",
    Conversation.user_id,
    Conversation.last_message_at.desc(),
)


class Message(UUIDPkMixin, TimestampMixin, Base):
    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_conversation_id_created_at", "conversation_id", "created_at"),
    )

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    # Denormalized from conversations so every access query can filter by user_id directly.
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    role: Mapped[MessageRole] = mapped_column(_pg_enum(MessageRole, "message_role"))
    content: Mapped[str] = mapped_column(Text)
    provider: Mapped[str | None] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(Text)
    tokens_in: Mapped[int | None]
    tokens_out: Mapped[int | None]
    moderation: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class LLMCall(UUIDPkMixin, TimestampMixin, Base):
    __tablename__ = "llm_calls"

    message_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE")
    )
    provider: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(Text)
    latency_ms: Mapped[int | None]
    status: Mapped[LLMCallStatus] = mapped_column(_pg_enum(LLMCallStatus, "llm_call_status"))
    error: Mapped[str | None] = mapped_column(Text)
    fallback_from: Mapped[str | None] = mapped_column(Text)


class KBDocument(UUIDPkMixin, TimestampMixin, Base):
    __tablename__ = "kb_documents"

    title: Mapped[str] = mapped_column(Text)
    source_url: Mapped[str | None] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(Text)
    license: Mapped[str | None] = mapped_column(Text)


class KBChunk(UUIDPkMixin, TimestampMixin, Base):
    __tablename__ = "kb_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index"),
        Index(
            "ix_kb_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("kb_documents.id", ondelete="CASCADE")
    )
    chunk_index: Mapped[int]
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    embedding_model: Mapped[str] = mapped_column(Text)
