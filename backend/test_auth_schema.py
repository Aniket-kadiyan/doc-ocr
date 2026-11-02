from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from alembic import command
from sqlalchemy import create_engine, func, inspect, select, text

from auth_models import Permission

from manage_auth import _alembic_config


class AuthenticationSchemaTests(unittest.TestCase):
    def test_initial_migration_creates_portable_auth_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database_path = Path(temporary) / "auth-schema.db"
            database_url = f"sqlite:///{database_path.as_posix()}"
            config = _alembic_config(database_url)
            config.attributes["allow_test_sqlite"] = True
            command.upgrade(config, "head")

            engine = create_engine(database_url)
            try:
                inspector = inspect(engine)
                self.assertTrue({
                    "alembic_version",
                    "qtracx_users",
                    "qtracx_roles",
                    "qtracx_user_roles",
                    "qtracx_auth_sessions",
                    "qtracx_permissions",
                    "qtracx_role_permissions",
                    "qtracx_user_permission_overrides",
                }.issubset(set(inspector.get_table_names())))
                session_columns = {
                    column["name"]
                    for column in inspector.get_columns("qtracx_auth_sessions")
                }
                self.assertIn("token_hash", session_columns)
                self.assertIn("csrf_token_hash", session_columns)
                self.assertNotIn("token", session_columns)
                user_columns = {
                    column["name"]
                    for column in inspector.get_columns("qtracx_users")
                }
                self.assertTrue({
                    "failed_login_count",
                    "locked_until",
                    "last_login_at",
                }.issubset(user_columns))
                with engine.connect() as connection:
                    permission_count = connection.scalar(
                        select(func.count(Permission.id))
                    )
                self.assertEqual(permission_count, 7)
            finally:
                engine.dispose()

    def test_permission_migration_maps_existing_default_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database_path = Path(temporary) / "existing-auth-schema.db"
            database_url = f"sqlite:///{database_path.as_posix()}"
            config = _alembic_config(database_url)
            config.attributes["allow_test_sqlite"] = True
            command.upgrade(config, "0002_auth_login_security")

            engine = create_engine(database_url)
            try:
                with engine.begin() as connection:
                    for role_name in ("user", "admin"):
                        connection.execute(
                            text(
                                "INSERT INTO qtracx_roles "
                                "(id, name, description, created_at) "
                                "VALUES (:id, :name, :description, :created_at)"
                            ),
                            {
                                "id": str(uuid4()),
                                "name": role_name,
                                "description": role_name,
                                "created_at": datetime.now(timezone.utc),
                            },
                        )

                command.upgrade(config, "head")
                with engine.connect() as connection:
                    assignments = dict(
                        connection.execute(
                            text(
                                "SELECT r.name, COUNT(*) "
                                "FROM qtracx_role_permissions rp "
                                "JOIN qtracx_roles r ON r.id = rp.role_id "
                                "GROUP BY r.name"
                            )
                        ).all()
                    )
                self.assertEqual(assignments, {"admin": 1, "user": 6})
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
