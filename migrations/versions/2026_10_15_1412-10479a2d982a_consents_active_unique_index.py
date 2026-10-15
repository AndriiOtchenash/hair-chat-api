"""consents active unique index

Revision ID: 10479a2d982a
Revises: faa8ce85530b
Create Date: 2026-10-15 14:12:30.803093

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "10479a2d982a"
down_revision: str | Sequence[str] | None = "faa8ce85530b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create the new index first so consents(user_id, type) lookups are never unindexed.
    # It also covers what ix_consents_user_id_type did (same leading columns) for active rows.
    op.create_index(
        "ix_consents_user_id_type_version_active",
        "consents",
        ["user_id", "type", "version"],
        unique=True,
        postgresql_where=sa.text("revoked_at IS NULL"),
    )
    op.drop_index(op.f("ix_consents_user_id_type"), table_name="consents")


def downgrade() -> None:
    """Downgrade schema."""
    op.create_index(op.f("ix_consents_user_id_type"), "consents", ["user_id", "type"], unique=False)
    op.drop_index("ix_consents_user_id_type_version_active", table_name="consents")
