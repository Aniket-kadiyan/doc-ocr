from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from alembic import command
from sqlalchemy import create_engine, inspect

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
                }.issubset(set(inspector.get_table_names())))
                session_columns = {
                    column["name"]
                    for column in inspector.get_columns("qtracx_auth_sessions")
                }
                self.assertIn("token_hash", session_columns)
                self.assertIn("csrf_token_hash", session_columns)
                self.assertNotIn("token", session_columns)
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
