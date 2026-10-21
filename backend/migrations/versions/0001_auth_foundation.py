"""Create the QTracX authentication foundation.

Revision ID: 0001_auth_foundation
Revises: None
"""

from alembic import op
import sqlalchemy as sa


revision = "0001_auth_foundation"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "qtracx_roles",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("description", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_qtracx_roles"),
        sa.UniqueConstraint("name", name="uq_qtracx_roles_name"),
    )
    op.create_table(
        "qtracx_users",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("normalized_username", sa.String(length=64), nullable=False),
        sa.Column("display_name", sa.String(length=120), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_qtracx_users"),
        sa.UniqueConstraint(
            "normalized_username",
            name="uq_qtracx_users_normalized_username",
        ),
    )
    op.create_table(
        "qtracx_auth_sessions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("csrf_token_hash", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["qtracx_users.id"],
            name="fk_qtracx_auth_sessions_user_id_qtracx_users",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_qtracx_auth_sessions"),
        sa.UniqueConstraint(
            "token_hash",
            name="uq_qtracx_auth_sessions_token_hash",
        ),
    )
    op.create_index(
        "ix_qtracx_auth_sessions_expires_at",
        "qtracx_auth_sessions",
        ["expires_at"],
    )
    op.create_index(
        "ix_qtracx_auth_sessions_user_id",
        "qtracx_auth_sessions",
        ["user_id"],
    )
    op.create_table(
        "qtracx_user_roles",
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("role_id", sa.String(length=36), nullable=False),
        sa.Column("assigned_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["role_id"],
            ["qtracx_roles.id"],
            name="fk_qtracx_user_roles_role_id_qtracx_roles",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["qtracx_users.id"],
            name="fk_qtracx_user_roles_user_id_qtracx_users",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "user_id",
            "role_id",
            name="pk_qtracx_user_roles",
        ),
    )


def downgrade() -> None:
    op.drop_table("qtracx_user_roles")
    op.drop_index(
        "ix_qtracx_auth_sessions_user_id",
        table_name="qtracx_auth_sessions",
    )
    op.drop_index(
        "ix_qtracx_auth_sessions_expires_at",
        table_name="qtracx_auth_sessions",
    )
    op.drop_table("qtracx_auth_sessions")
    op.drop_table("qtracx_users")
    op.drop_table("qtracx_roles")
