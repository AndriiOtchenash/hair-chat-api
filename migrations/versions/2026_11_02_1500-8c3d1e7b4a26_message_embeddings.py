"""message_embeddings: search index over the user's conversations (question + answer pairs)

Revision ID: 8c3d1e7b4a26
Revises: f2a5c5e9222d
Create Date: 2026-11-02 15:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "8c3d1e7b4a26"
down_revision: str | Sequence[str] | None = "f2a5c5e9222d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "message_embeddings",
        sa.Column("message_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("conversation_id", sa.Uuid(), nullable=False),
        sa.Column("embedding", Vector(768), nullable=False),
        sa.Column("embedding_model", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["conversations.id"],
            name=op.f("fk_message_embeddings_conversation_id_conversations"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["messages.id"],
            name=op.f("fk_message_embeddings_message_id_messages"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_message_embeddings_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("message_id", name=op.f("pk_message_embeddings")),
    )
    # No HNSW index: the search is exact within one user_id.
    op.create_index(
        op.f("ix_message_embeddings_user_id"), "message_embeddings", ["user_id"], unique=False
    )
    op.create_index(
        op.f("ix_message_embeddings_conversation_id"),
        "message_embeddings",
        ["conversation_id"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_message_embeddings_conversation_id"), table_name="message_embeddings")
    op.drop_index(op.f("ix_message_embeddings_user_id"), table_name="message_embeddings")
    op.drop_table("message_embeddings")
