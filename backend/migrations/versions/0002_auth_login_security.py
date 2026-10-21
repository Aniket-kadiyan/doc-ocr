"""Add login lockout and audit timestamps.

Revision ID: 0002_auth_login_security
Revises: 0001_auth_foundation
"""

from alembic import op
import sqlalchemy as sa


revision = "0002_auth_login_security"
down_revision = "0001_auth_foundation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "qtracx_users",
        sa.Column(
            "failed_login_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )
    op.add_column(
        "qtracx_users",
        sa.Column("locked_until", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "qtracx_users",
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("qtracx_users", "last_login_at")
    op.drop_column("qtracx_users", "locked_until")
    op.drop_column(
        "qtracx_users", "failed_login_count", mssql_drop_default=True
    )
