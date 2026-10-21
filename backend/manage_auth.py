"""Administrative database commands for the QTracX authentication schema.

Run from ``backend`` using the same Python environment as the OCR service:

    python manage_auth.py db upgrade
    python manage_auth.py db check
    python manage_auth.py user create-user --username operator.one --display-name "Operator One"
    python manage_auth.py user grant-admin operator.one
    python manage_auth.py user revoke-admin operator.one

Creation and password-reset commands collect the password through an
interactive prompt. Passwords are deliberately not accepted as command-line
arguments because command lines can be retained in shell history or process
inspection output.
"""

from __future__ import annotations

import argparse
import json
from getpass import getpass
from pathlib import Path
from typing import Sequence

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect

from auth_repository import (
    AuthenticationRepositoryError,
    create_user,
    list_users,
    set_user_administrator,
    set_user_active,
    set_user_password,
)
from auth_security import AuthenticationInputError
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


def _prompt_password() -> str:
    password = getpass("Password: ")
    confirmation = getpass("Confirm password: ")
    if password != confirmation:
        raise AuthenticationInputError("Passwords do not match")
    return password


def _run_user_command(arguments: argparse.Namespace) -> int:
    manager = DatabaseManager(_database_url())
    try:
        with manager.session() as session:
            if arguments.user_action in {"create-user", "create-admin"}:
                record = create_user(
                    session,
                    username=arguments.username,
                    display_name=arguments.display_name or arguments.username,
                    password=_prompt_password(),
                    administrator=arguments.user_action == "create-admin",
                )
                output: object = record.as_public_dict()
            elif arguments.user_action == "list":
                output = [record.as_public_dict() for record in list_users(session)]
            elif arguments.user_action == "enable":
                output = set_user_active(
                    session,
                    username=arguments.username,
                    active=True,
                ).as_public_dict()
            elif arguments.user_action == "disable":
                output = set_user_active(
                    session,
                    username=arguments.username,
                    active=False,
                ).as_public_dict()
            elif arguments.user_action == "set-password":
                output = set_user_password(
                    session,
                    username=arguments.username,
                    password=_prompt_password(),
                ).as_public_dict()
            elif arguments.user_action == "grant-admin":
                output = set_user_administrator(
                    session,
                    username=arguments.username,
                    administrator=True,
                ).as_public_dict()
            elif arguments.user_action == "revoke-admin":
                output = set_user_administrator(
                    session,
                    username=arguments.username,
                    administrator=False,
                ).as_public_dict()
            else:
                return 2
        print(json.dumps({"status": "ok", "result": output}))
        return 0
    finally:
        manager.dispose()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QTracX authentication administration")
    commands = parser.add_subparsers(dest="command", required=True)
    database = commands.add_parser("db", help="manage the authentication database")
    database.add_argument("action", choices=("upgrade", "current", "check"))

    user = commands.add_parser("user", help="manage QTracX users")
    user_actions = user.add_subparsers(dest="user_action", required=True)
    for action, help_text in (
        ("create-user", "create a regular user using an interactive password prompt"),
        ("create-admin", "create an administrator using an interactive password prompt"),
    ):
        create_parser = user_actions.add_parser(action, help=help_text)
        create_parser.add_argument("--username", required=True)
        create_parser.add_argument("--display-name")
    user_actions.add_parser("list", help="list users without password data")
    for action in (
        "enable",
        "disable",
        "set-password",
        "grant-admin",
        "revoke-admin",
    ):
        action_parser = user_actions.add_parser(action)
        action_parser.add_argument("username")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "db":
            return _run_db_command(arguments.action)
        if arguments.command == "user":
            return _run_user_command(arguments)
    except DatabaseConfigurationError as error:
        print(f"Configuration error: {error}")
        return 2
    except (AuthenticationInputError, AuthenticationRepositoryError) as error:
        print(f"User operation rejected: {error}")
        return 2
    except Exception as error:
        # Connection exceptions can echo credentials. Expose only the type;
        # operational details remain in controlled backend diagnostics.
        print(f"Database operation failed: {error.__class__.__name__}")
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
