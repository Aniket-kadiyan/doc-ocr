"""Add role defaults and per-user permission overrides.

Revision ID: 0003_granular_permissions
Revises: 0002_auth_login_security
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from alembic import op
import sqlalchemy as sa


revision = "0003_granular_permissions"
down_revision = "0002_auth_login_security"
branch_labels = None
depends_on = None


PERMISSIONS = (
    (
        "drawing.use",
        "Drawing",
        "Open drawings and use OCR, scanning, ballooning, GD&T and review workflows",
    ),
    (
        "drawing.export",
        "Drawing",
        "Create drawing, Excel, PDF and checksheet exports",
    ),
    (
        "checksheet.view",
        "Checksheet",
        "View saved checksheets, inspection runs and source documents",
    ),
    (
        "checksheet.manage",
        "Checksheet",
        "Create, revise, rename, archive, duplicate and delete checksheets",
    ),
    (
        "checksheet.execute",
        "Checksheet",
        "Start, fill and complete checksheet inspection runs",
    ),
    (
        "training.manage",
        "Training",
        "Record GD&T corrections and export training labels",
    ),
    (
        "users.manage",
        "Administration",
        "Manage existing QTraX users, roles and permissions",
    ),
)

ROLE_PERMISSIONS = {
    "user": tuple(name for name, _, _ in PERMISSIONS if name != "users.manage"),
    "admin": ("users.manage",),
}


def upgrade() -> None:
    op.create_table(
        "qtracx_permissions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("description", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_qtracx_permissions"),
        sa.UniqueConstraint("name", name="uq_qtracx_permissions_name"),
    )
    op.create_table(
        "qtracx_role_permissions",
        sa.Column("role_id", sa.String(length=36), nullable=False),
        sa.Column("permission_id", sa.String(length=36), nullable=False),
        sa.Column("assigned_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["permission_id"],
            ["qtracx_permissions.id"],
            name="fk_qtracx_role_permissions_permission_id_qtracx_permissions",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["role_id"],
            ["qtracx_roles.id"],
            name="fk_qtracx_role_permissions_role_id_qtracx_roles",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "role_id",
            "permission_id",
            name="pk_qtracx_role_permissions",
        ),
    )
    op.create_table(
        "qtracx_user_permission_overrides",
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("permission_id", sa.String(length=36), nullable=False),
        sa.Column("is_allowed", sa.Boolean(), nullable=False),
        sa.Column("assigned_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["permission_id"],
            ["qtracx_permissions.id"],
            name=(
                "fk_qtracx_user_permission_overrides_permission_id_"
                "qtracx_permissions"
            ),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["qtracx_users.id"],
            name="fk_qtracx_user_permission_overrides_user_id_qtracx_users",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "user_id",
            "permission_id",
            name="pk_qtracx_user_permission_overrides",
        ),
    )

    connection = op.get_bind()
    now = datetime.now(timezone.utc)
    permission_table = sa.table(
        "qtracx_permissions",
        sa.column("id", sa.String),
        sa.column("name", sa.String),
        sa.column("category", sa.String),
        sa.column("description", sa.String),
        sa.column("created_at", sa.DateTime(timezone=True)),
    )
    permission_ids = {name: str(uuid4()) for name, _, _ in PERMISSIONS}
    connection.execute(
        permission_table.insert(),
        [
            {
                "id": permission_ids[name],
                "name": name,
                "category": category,
                "description": description,
                "created_at": now,
            }
            for name, category, description in PERMISSIONS
        ],
    )

    role_table = sa.table(
        "qtracx_roles",
        sa.column("id", sa.String),
        sa.column("name", sa.String),
    )
    role_permission_table = sa.table(
        "qtracx_role_permissions",
        sa.column("role_id", sa.String),
        sa.column("permission_id", sa.String),
        sa.column("assigned_at", sa.DateTime(timezone=True)),
    )
    role_rows = connection.execute(
        sa.select(role_table.c.id, role_table.c.name).where(
            role_table.c.name.in_(tuple(ROLE_PERMISSIONS))
        )
    ).all()
    assignments = [
        {
            "role_id": role_id,
            "permission_id": permission_ids[permission_name],
            "assigned_at": now,
        }
        for role_id, role_name in role_rows
        for permission_name in ROLE_PERMISSIONS[role_name]
    ]
    if assignments:
        connection.execute(role_permission_table.insert(), assignments)


def downgrade() -> None:
    op.drop_table("qtracx_user_permission_overrides")
    op.drop_table("qtracx_role_permissions")
    op.drop_table("qtracx_permissions")
