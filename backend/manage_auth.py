"""Administrative database commands for the QTracX authentication schema.

Run from ``backend`` using the same Python environment as the OCR service:

    python manage_auth.py db upgrade
    python manage_auth.py db check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect

from config_env import load_env_files, load_runtime_config
from database import DatabaseConfigurationError, DatabaseManager, database_backend


_BACKEND_DIR = Path(__file__).resolve().parent
_REQUIRED_TABLES = {
    "alembic_version",
    "qtracx_users",
    "qtracx_roles",
    "qtracx_user_roles",
    "qtracx_auth_sessions",
}


def _alembic_config(database_url: str) -> Config:
    config = Config(str(_BACKEND_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(_BACKEND_DIR / "migrations"))
    # Config attributes avoid ConfigParser interpolation and keep credentials
    # out of command output.
    config.attributes["database_url"] = database_url
    return config


def _database_url() -> str:
    load_env_files()
    config = load_runtime_config()
    if not config.database_url:
        raise DatabaseConfigurationError("DATABASE_URL is required")
    return config.database_url


def _run_db_command(action: str) -> int:
    database_url = _database_url()
    if action == "upgrade":
        command.upgrade(_alembic_config(database_url), "head")
        print(json.dumps({
            "status": "ok",
            "database_backend": database_backend(database_url),
            "schema": "up-to-date",
        }))
        return 0
    if action == "current":
        command.current(_alembic_config(database_url), verbose=False)
        return 0

    manager = DatabaseManager(database_url)
    try:
        manager.ping()
        tables = set(inspect(manager.engine).get_table_names())
        missing = sorted(_REQUIRED_TABLES - tables)
        print(json.dumps({
            "status": "ok" if not missing else "schema-upgrade-required",
            "database_backend": manager.engine.url.get_backend_name(),
            "missing_tables": missing,
        }))
        return 0 if not missing else 2
    finally:
        manager.dispose()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QTracX authentication administration")
    commands = parser.add_subparsers(dest="command", required=True)
    database = commands.add_parser("db", help="manage the authentication database")
    database.add_argument("action", choices=("upgrade", "current", "check"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "db":
            return _run_db_command(arguments.action)
    except DatabaseConfigurationError as error:
        print(f"Configuration error: {error}")
        return 2
    except Exception as error:
        # Connection exceptions can echo credentials. Expose only the type;
        # operational details remain in controlled backend diagnostics.
        print(f"Database operation failed: {error.__class__.__name__}")
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
