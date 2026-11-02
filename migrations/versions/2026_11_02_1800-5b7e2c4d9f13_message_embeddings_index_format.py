"""message_embeddings.index_format: version of the embedded text format (NULL = stale)

Revision ID: 5b7e2c4d9f13
Revises: 8c3d1e7b4a26
Create Date: 2026-11-02 18:00:00.000000

Existing rows get NULL: the next `python -m app.cli.index_conversations` re-embeds them in
format 2 (app.rag.conversation_index.CONVERSATION_INDEX_FORMAT_VERSION).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5b7e2c4d9f13"
down_revision: str | Sequence[str] | None = "8c3d1e7b4a26"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("message_embeddings", sa.Column("index_format", sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("message_embeddings", "index_format")
