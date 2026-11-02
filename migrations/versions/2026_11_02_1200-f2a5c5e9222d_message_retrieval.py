"""messages.retrieval: knowledge-base search outcome of each answer

Revision ID: f2a5c5e9222d
Revises: 5b2e9c4d7a10
Create Date: 2026-11-02 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "f2a5c5e9222d"
down_revision: str | Sequence[str] | None = "5b2e9c4d7a10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Nullable, no default: a metadata-only change, older answers stay null ("unknown").
    op.add_column(
        "messages",
        sa.Column("retrieval", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("messages", "retrieval")
