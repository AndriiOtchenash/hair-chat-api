"""kb_documents status, content_hash, slug

Revision ID: 5b2e9c4d7a10
Revises: 10479a2d982a
Create Date: 2026-10-21 13:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5b2e9c4d7a10"
down_revision: str | Sequence[str] | None = "10479a2d982a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

kb_document_status = sa.Enum("draft", "reviewed", name="kb_document_status")


def upgrade() -> None:
    """Upgrade schema."""
    kb_document_status.create(op.get_bind(), checkfirst=False)
    # NOT NULL without defaults: kb_documents is empty until the first ingest (1.9), and a
    # failure here on a non-empty table is better than rows with a fake slug or hash.
    op.add_column("kb_documents", sa.Column("slug", sa.Text(), nullable=False))
    op.add_column(
        "kb_documents",
        sa.Column("status", kb_document_status, server_default="draft", nullable=False),
    )
    op.add_column("kb_documents", sa.Column("content_hash", sa.Text(), nullable=False))
    op.create_unique_constraint(op.f("uq_kb_documents_slug"), "kb_documents", ["slug"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(op.f("uq_kb_documents_slug"), "kb_documents", type_="unique")
    op.drop_column("kb_documents", "content_hash")
    op.drop_column("kb_documents", "status")
    op.drop_column("kb_documents", "slug")
    # drop_column leaves the Postgres ENUM type behind; drop it explicitly.
    kb_document_status.drop(op.get_bind(), checkfirst=False)
