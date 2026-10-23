"""Create and initialize the QTracX MSSQL or PostgreSQL database.

Secrets enter through a hidden interactive prompt or JSON on stdin. They are
never accepted as command-line arguments, emitted in results, or written by
this module. The graphical installer will call ``bootstrap_database`` and then
use ``setup_config`` to persist only the application connection string.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from getpass import getpass
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Sequence


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from installer.setup_config import (  # noqa: E402
    DatabaseSetup,
    SetupConfigurationError,
)


_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,62}$")
_REQUIRED_TABLES = {
    "alembic_version",
    "qtracx_users",
    "qtracx_roles",
    "qtracx_user_roles",
    "qtracx_auth_sessions",
}
ProgressCallback = Callable[[str, str], None]


class DatabaseBootstrapError(RuntimeError):
    """A sanitized database-setup failure safe to show in the installer."""

    def __init__(self, stage: str, error_type: str) -> None:
        self.stage = stage
        self.error_type = error_type
        super().__init__(f"{stage} failed ({error_type})")


def _required_string(payload: Mapping[str, Any], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise SetupConfigurationError(f"{label} is required")
    return value


def _integer(payload: Mapping[str, Any], key: str, default: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool):
        raise SetupConfigurationError(f"{key} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise SetupConfigurationError(f"{key} must be an integer") from error


def _boolean(payload: Mapping[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise SetupConfigurationError(f"{key} must be true or false")
    return value


@dataclass(frozen=True)
class InitialApplicationAdministrator:
    username: str = "admin"
    display_name: str = "QTracX Administrator"
    password: str = field(default="", repr=False)

    def validate(self) -> None:
        if not self.username:
            raise SetupConfigurationError(
                "initial application administrator username is required"
            )
        if not self.display_name.strip():
            raise SetupConfigurationError(
                "initial application administrator display name is required"
            )
        if len(self.password) < 4:
            raise SetupConfigurationError(
                "initial application administrator password must contain at least 4 characters"
            )


@dataclass(frozen=True)
class DatabaseBootstrapRequest:
    database: DatabaseSetup
    provisioning_mode: str = "installer"
    administrator_user: str = ""
    administrator_password: str = field(default="", repr=False)
    initial_administrator: InitialApplicationAdministrator = field(
        default_factory=InitialApplicationAdministrator
    )
    reset_existing_application_password: bool = False

    def validate(self) -> None:
        self.database.validate()
        if self.provisioning_mode not in {"installer", "preconfigured"}:
            raise SetupConfigurationError(
                "database provisioning mode must be installer or preconfigured"
            )
        if self.provisioning_mode == "installer":
            if not _IDENTIFIER.fullmatch(self.administrator_user):
                raise SetupConfigurationError(
                    "database administrator user must start with a letter and "
                    "contain only letters, numbers, or underscores"
                )
            if not self.administrator_password:
                raise SetupConfigurationError(
                    "database administrator password is required"
                )
            if (
                self.administrator_user.casefold()
                == self.database.application_user.casefold()
            ):
                raise SetupConfigurationError(
                    "database administrator and application user must be different"
                )
        elif self.reset_existing_application_password:
            raise SetupConfigurationError(
                "the installer cannot reset a preconfigured database password"
            )
        self.initial_administrator.validate()

    def safe_summary(self) -> dict[str, object]:
        return {
            "provisioning_mode": self.provisioning_mode,
            "engine": self.database.engine,
            "host": self.database.host,
            "port": self.database.port,
            "database": self.database.name,
            "application_user": self.database.application_user,
            "initial_administrator": self.initial_administrator.username,
            "reset_existing_application_password": (
                self.reset_existing_application_password
            ),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DatabaseBootstrapRequest":
        database_payload = payload.get("database")
        admin_payload = payload.get("initialAdministrator")
        if not isinstance(database_payload, Mapping):
            raise SetupConfigurationError("database setup is required")
        if not isinstance(admin_payload, Mapping):
            raise SetupConfigurationError(
                "initial application administrator setup is required"
            )

        engine = str(database_payload.get("engine", "mssql")).strip().lower()
        database = DatabaseSetup(
            engine=engine,
            host=str(database_payload.get("host", "localhost")).strip(),
            port=_integer(
                database_payload,
                "port",
                1433 if engine == "mssql" else 5432,
            ),
            name=str(database_payload.get("name", "QTracX")).strip(),
            application_user=str(
                database_payload.get("applicationUser", "qtracx_app")
            ).strip(),
            application_password=_required_string(
                database_payload,
                "applicationPassword",
                "application database password",
            ),
            odbc_driver=str(
                database_payload.get(
                    "odbcDriver",
                    "ODBC Driver 18 for SQL Server",
                )
            ).strip(),
            encrypt=_boolean(database_payload, "encrypt", True),
            trust_server_certificate=_boolean(
                database_payload,
                "trustServerCertificate",
                True,
            ),
        )
        provisioning_mode = str(
            database_payload.get("provisioningMode", "installer")
        ).strip().lower()
        if provisioning_mode == "installer":
            administrator_user = _required_string(
                database_payload,
                "administratorUser",
                "database administrator user",
            ).strip()
            administrator_password = _required_string(
                database_payload,
                "administratorPassword",
                "database administrator password",
            )
        else:
            administrator_user = ""
            administrator_password = ""
        request = cls(
            database=database,
            provisioning_mode=provisioning_mode,
            administrator_user=administrator_user,
            administrator_password=administrator_password,
            reset_existing_application_password=_boolean(
                database_payload,
                "resetExistingApplicationPassword",
                False,
            ),
            initial_administrator=InitialApplicationAdministrator(
                username=str(admin_payload.get("username", "admin")).strip(),
                display_name=str(
                    admin_payload.get("displayName", "QTracX Administrator")
                ).strip(),
                password=_required_string(
                    admin_payload,
                    "password",
                    "initial application administrator password",
                ),
            ),
        )
        request.validate()
        return request


def _quote_mssql_identifier(value: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise SetupConfigurationError("invalid MSSQL identifier")
    return f"[{value}]"


def _quote_postgresql_identifier(value: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise SetupConfigurationError("invalid PostgreSQL identifier")
    return f'"{value}"'


def _sql_string(value: str) -> str:
    return "N'" + value.replace("'", "''") + "'"


def _postgresql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _odbc_value(value: str) -> str:
    return "{" + value.replace("}", "}}") + "}"


def _mssql_connection_string(
    request: DatabaseBootstrapRequest,
    *,
    database: str,
    application_credentials: bool = False,
) -> str:
    setup = request.database
    user = (
        setup.application_user
        if application_credentials
        else request.administrator_user
    )
    password = (
        setup.application_password
        if application_credentials
        else request.administrator_password
    )
    encrypt = "yes" if setup.encrypt else "no"
    trust = "yes" if setup.trust_server_certificate else "no"
    return ";".join([
        f"DRIVER={_odbc_value(setup.odbc_driver)}",
        f"SERVER={_odbc_value(f'{setup.host},{setup.port}')}",
        f"DATABASE={_odbc_value(database)}",
        f"UID={_odbc_value(user)}",
        f"PWD={_odbc_value(password)}",
        f"Encrypt={encrypt}",
        f"TrustServerCertificate={trust}",
    ])


def _test_mssql_administrator(request: DatabaseBootstrapRequest) -> None:
    import pyodbc

    connection = pyodbc.connect(
        _mssql_connection_string(request, database="master"),
        timeout=15,
        autocommit=True,
    )
    try:
        connection.cursor().execute("SELECT 1").fetchone()
    finally:
        connection.close()


def _test_mssql_application_database(request: DatabaseBootstrapRequest) -> None:
    import pyodbc

    connection = pyodbc.connect(
        _mssql_connection_string(
            request,
            database=request.database.name,
            application_credentials=True,
        ),
        timeout=15,
        autocommit=True,
    )
    try:
        permissions = connection.cursor().execute(
            "SELECT "
            "HAS_PERMS_BY_NAME(DB_NAME(), 'DATABASE', 'CREATE TABLE'), "
            "HAS_PERMS_BY_NAME(DB_NAME(), 'DATABASE', 'ALTER ANY TABLE')"
        ).fetchone()
        if not permissions or any(value != 1 for value in permissions):
            raise PermissionError(
                "The preconfigured MSSQL account lacks migration permissions"
            )
    finally:
        connection.close()


def _bootstrap_mssql(request: DatabaseBootstrapRequest) -> str:
    import pyodbc

    setup = request.database
    database_id = _quote_mssql_identifier(setup.name)
    login_id = _quote_mssql_identifier(setup.application_user)
    password = _sql_string(setup.application_password)
    admin_connection = pyodbc.connect(
        _mssql_connection_string(request, database="master"),
        timeout=15,
        autocommit=True,
    )
    try:
        cursor = admin_connection.cursor()
        cursor.execute("SELECT 1")
        database_exists = cursor.execute(
            "SELECT 1 WHERE DB_ID(?) IS NOT NULL",
            setup.name,
        ).fetchone() is not None
        if not database_exists:
            cursor.execute(f"CREATE DATABASE {database_id}")

        login_exists = cursor.execute(
            "SELECT 1 WHERE SUSER_ID(?) IS NOT NULL",
            setup.application_user,
        ).fetchone() is not None
        if not login_exists:
            cursor.execute(
                f"CREATE LOGIN {login_id} WITH PASSWORD = {password}, "
                "CHECK_POLICY = ON, CHECK_EXPIRATION = OFF"
            )
            login_action = "created"
        elif request.reset_existing_application_password:
            cursor.execute(
                f"ALTER LOGIN {login_id} WITH PASSWORD = {password}"
            )
            login_action = "password-reset"
        else:
            login_action = "preserved"
    finally:
        admin_connection.close()

    database_connection = pyodbc.connect(
        _mssql_connection_string(request, database=setup.name),
        timeout=15,
        autocommit=True,
    )
    try:
        cursor = database_connection.cursor()
        user_literal = _sql_string(setup.application_user)
        cursor.execute(
            f"IF DATABASE_PRINCIPAL_ID({user_literal}) IS NULL "
            f"CREATE USER {login_id} FOR LOGIN {login_id} "
            "WITH DEFAULT_SCHEMA = [dbo]"
        )
        for role in ("db_datareader", "db_datawriter", "db_ddladmin"):
            role_id = _quote_mssql_identifier(role)
            role_literal = _sql_string(role)
            cursor.execute(
                f"IF IS_ROLEMEMBER({role_literal}, {user_literal}) <> 1 "
                f"ALTER ROLE {role_id} ADD MEMBER {login_id}"
            )
    finally:
        database_connection.close()

    application_connection = pyodbc.connect(
        _mssql_connection_string(
            request,
            database=setup.name,
            application_credentials=True,
        ),
        timeout=15,
        autocommit=True,
    )
    try:
        application_connection.cursor().execute("SELECT 1").fetchone()
    finally:
        application_connection.close()
    return login_action


def _postgresql_connect(
    request: DatabaseBootstrapRequest,
    *,
    database: str,
    application_credentials: bool = False,
):
    import pg8000.dbapi

    setup = request.database
    connection = pg8000.dbapi.connect(
        user=(
            setup.application_user
            if application_credentials
            else request.administrator_user
        ),
        password=(
            setup.application_password
            if application_credentials
            else request.administrator_password
        ),
        host=setup.host,
        port=setup.port,
        database=database,
        timeout=15,
    )
    connection.autocommit = True
    return connection


def _bootstrap_postgresql(request: DatabaseBootstrapRequest) -> str:
    setup = request.database
    database_id = _quote_postgresql_identifier(setup.name)
    role_id = _quote_postgresql_identifier(setup.application_user)
    password = _postgresql_string(setup.application_password)
    admin_connection = _postgresql_connect(request, database="postgres")
    try:
        cursor = admin_connection.cursor()
        cursor.execute("SELECT 1")
        cursor.execute(
            "SELECT 1 FROM pg_roles WHERE rolname = %s",
            (setup.application_user,),
        )
        role_exists = cursor.fetchone() is not None
        if not role_exists:
            cursor.execute(
                f"CREATE ROLE {role_id} WITH LOGIN PASSWORD {password}"
            )
            role_action = "created"
        elif request.reset_existing_application_password:
            cursor.execute(
                f"ALTER ROLE {role_id} WITH LOGIN PASSWORD {password}"
            )
            role_action = "password-reset"
        else:
            role_action = "preserved"

        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (setup.name,),
        )
        if cursor.fetchone() is None:
            cursor.execute(f"CREATE DATABASE {database_id} OWNER {role_id}")
        else:
            cursor.execute(f"ALTER DATABASE {database_id} OWNER TO {role_id}")
    finally:
        admin_connection.close()

    database_connection = _postgresql_connect(request, database=setup.name)
    try:
        cursor = database_connection.cursor()
        cursor.execute(f"GRANT CONNECT ON DATABASE {database_id} TO {role_id}")
        cursor.execute(f"GRANT USAGE, CREATE ON SCHEMA public TO {role_id}")
    finally:
        database_connection.close()

    application_connection = _postgresql_connect(
        request,
        database=setup.name,
        application_credentials=True,
    )
    try:
        cursor = application_connection.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
    finally:
        application_connection.close()
    return role_action


def _test_postgresql_administrator(request: DatabaseBootstrapRequest) -> None:
    connection = _postgresql_connect(request, database="postgres")
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
    finally:
        connection.close()


def _test_postgresql_application_database(
    request: DatabaseBootstrapRequest,
) -> None:
    connection = _postgresql_connect(
        request,
        database=request.database.name,
        application_credentials=True,
    )
    try:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT "
            "has_schema_privilege(current_user, 'public', 'USAGE'), "
            "has_schema_privilege(current_user, 'public', 'CREATE')"
        )
        permissions = cursor.fetchone()
        if not permissions or not all(bool(value) for value in permissions):
            raise PermissionError(
                "The preconfigured PostgreSQL account lacks migration permissions"
            )
    finally:
        connection.close()


def _upgrade_schema(database_url: str, backend_dir: Path) -> None:
    from alembic import command
    from alembic.config import Config

    backend_path = str(backend_dir)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    config = Config(str(backend_dir / "alembic.ini"))
    config.set_main_option("script_location", str(backend_dir / "migrations"))
    config.attributes["database_url"] = database_url
    command.upgrade(config, "head")


def _missing_schema_tables(database_url: str, backend_dir: Path) -> list[str]:
    backend_path = str(backend_dir)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    from sqlalchemy import inspect
    from database import DatabaseManager

    manager = DatabaseManager(database_url)
    try:
        manager.ping()
        tables = set(inspect(manager.engine).get_table_names())
        return sorted(_REQUIRED_TABLES - tables)
    finally:
        manager.dispose()


def _ensure_initial_admin(
    database_url: str,
    backend_dir: Path,
    administrator: InitialApplicationAdministrator,
) -> str:
    backend_path = str(backend_dir)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    from auth_repository import (
        UserNotFoundError,
        create_user,
        get_user,
        set_user_active,
        set_user_administrator,
    )
    from database import DatabaseManager

    manager = DatabaseManager(database_url)
    try:
        with manager.session() as session:
            try:
                user = get_user(session, administrator.username)
            except UserNotFoundError:
                create_user(
                    session,
                    username=administrator.username,
                    display_name=administrator.display_name,
                    password=administrator.password,
                    administrator=True,
                )
                return "created"

            changed = False
            if not user.is_active:
                set_user_active(
                    session,
                    username=administrator.username,
                    active=True,
                )
                changed = True
            if not any(role.name == "admin" for role in user.roles):
                set_user_administrator(
                    session,
                    username=administrator.username,
                    administrator=True,
                )
                changed = True
            return "updated" if changed else "preserved"
    finally:
        manager.dispose()


def _run_stage(stage: str, operation: Callable[[], Any]) -> Any:
    try:
        return operation()
    except (SetupConfigurationError, DatabaseBootstrapError):
        raise
    except Exception as error:
        raise DatabaseBootstrapError(stage, error.__class__.__name__) from error


def check_database_connection(
    request: DatabaseBootstrapRequest,
) -> dict[str, object]:
    """Validate the selected setup connection without changing anything."""

    request.validate()
    if request.provisioning_mode == "installer":
        if request.database.engine == "mssql":
            operation = lambda: _test_mssql_administrator(request)
        else:
            operation = lambda: _test_postgresql_administrator(request)
        result = "administrator-verified"
    else:
        if request.database.engine == "mssql":
            operation = lambda: _test_mssql_application_database(request)
        else:
            operation = lambda: _test_postgresql_application_database(request)
        result = "preconfigured-application-database-verified"
    _run_stage("database-connection", operation)
    return {
        "status": "ok",
        "connection": result,
        **request.safe_summary(),
    }


def bootstrap_database(
    request: DatabaseBootstrapRequest,
    *,
    repo_root: Path = _REPO_ROOT,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    """Create the DB/login, migrate the schema, and ensure the first admin."""

    request.validate()
    backend_dir = repo_root.resolve() / "backend"
    if not (backend_dir / "alembic.ini").is_file():
        raise SetupConfigurationError("QTracX backend migrations were not found")

    def notify(stage: str, message: str) -> None:
        if progress is not None:
            progress(stage, message)

    notify("database-connection", "Testing the selected database connection")
    check_database_connection(request)
    if request.provisioning_mode == "preconfigured":
        notify("database", "Using the preconfigured QTracX database account")
        account_action = "preconfigured"
    else:
        notify("database", "Creating or validating the QTracX database")
        if request.database.engine == "mssql":
            account_action = _run_stage(
                "mssql-database-setup",
                lambda: _bootstrap_mssql(request),
            )
        else:
            account_action = _run_stage(
                "postgresql-database-setup",
                lambda: _bootstrap_postgresql(request),
            )

    database_url = request.database.url()
    notify("migrations", "Applying QTracX database migrations")
    _run_stage(
        "database-migrations",
        lambda: _upgrade_schema(database_url, backend_dir),
    )
    missing = _run_stage(
        "schema-verification",
        lambda: _missing_schema_tables(database_url, backend_dir),
    )
    if missing:
        raise DatabaseBootstrapError("schema-verification", "MissingTables")

    notify("application-admin", "Creating the initial QTracX administrator")
    administrator_action = _run_stage(
        "initial-administrator",
        lambda: _ensure_initial_admin(
            database_url,
            backend_dir,
            request.initial_administrator,
        ),
    )
    notify("complete", "Database setup completed")
    return {
        "status": "ok",
        **request.safe_summary(),
        "application_database_account": account_action,
        "schema": "up-to-date",
        "application_administrator": administrator_action,
    }


def _prompt(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def _confirmed_password(label: str) -> str:
    first = getpass(f"{label}: ")
    second = getpass(f"Confirm {label.lower()}: ")
    if first != second:
        raise SetupConfigurationError(f"{label} values do not match")
    return first


def _interactive_request() -> DatabaseBootstrapRequest:
    engine = _prompt("Database engine (mssql/postgresql)", "mssql").lower()
    if engine not in {"mssql", "postgresql"}:
        raise SetupConfigurationError("database engine must be mssql or postgresql")
    port_default = "1433" if engine == "mssql" else "5432"
    admin_default = "sa" if engine == "mssql" else "postgres"
    provisioning_mode = _prompt(
        "Database setup mode (installer/preconfigured)",
        "installer",
    ).lower()
    database_payload: dict[str, object] = {
        "engine": engine,
        "provisioningMode": provisioning_mode,
        "host": _prompt("Database server", "localhost"),
        "port": int(_prompt("Database port", port_default)),
        "name": _prompt("Database name", "QTracX"),
        "applicationUser": _prompt(
            "QTracX database user",
            "qtracx_app",
        ),
        "applicationPassword": _confirmed_password(
            "QTracX database-user password"
        ),
        "resetExistingApplicationPassword": False,
    }
    if provisioning_mode == "installer":
        database_payload.update({
            "administratorUser": _prompt(
                "Database administrator user",
                admin_default,
            ),
            "administratorPassword": getpass(
                "Database administrator password: "
            ),
            "resetExistingApplicationPassword": (
                _prompt(
                    "Reset existing QTracX DB-user password (yes/no)",
                    "no",
                ).casefold()
                in {"yes", "y"}
            ),
        })
    if engine == "mssql":
        database_payload.update({
            "odbcDriver": _prompt(
                "MSSQL ODBC driver",
                "ODBC Driver 18 for SQL Server",
            ),
            "encrypt": True,
            "trustServerCertificate": (
                _prompt("Trust the SQL Server certificate (yes/no)", "yes")
                .casefold()
                in {"yes", "y"}
            ),
        })
    payload = {
        "database": database_payload,
        "initialAdministrator": {
            "username": _prompt("Initial QTracX administrator", "admin"),
            "displayName": _prompt(
                "Administrator display name",
                "QTracX Administrator",
            ),
            "password": _confirmed_password("QTracX administrator password"),
        },
    }
    return DatabaseBootstrapRequest.from_mapping(payload)


def _stdin_request() -> DatabaseBootstrapRequest:
    try:
        payload = json.load(sys.stdin)
    except (OSError, json.JSONDecodeError) as error:
        raise SetupConfigurationError("stdin must contain valid setup JSON") from error
    if not isinstance(payload, Mapping):
        raise SetupConfigurationError("setup JSON must contain an object")
    return DatabaseBootstrapRequest.from_mapping(payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create the QTracX database")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="prompt for setup values; otherwise read one JSON object from stdin",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="test administrator connectivity without changing the database",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        request = _interactive_request() if arguments.interactive else _stdin_request()
        if arguments.check_only:
            result = check_database_connection(request)
        else:
            result = bootstrap_database(
                request,
                progress=lambda stage, message: print(json.dumps({
                    "status": "working",
                    "stage": stage,
                    "message": message,
                })),
            )
        print(json.dumps(result, indent=2))
        return 0
    except SetupConfigurationError as error:
        print(json.dumps({
            "status": "failed",
            "stage": "validation",
            "message": str(error),
        }))
        return 2
    except DatabaseBootstrapError as error:
        print(json.dumps({
            "status": "failed",
            "stage": error.stage,
            "message": str(error),
        }))
        return 1
    except Exception as error:
        print(json.dumps({
            "status": "failed",
            "stage": "unexpected",
            "message": f"Database setup failed ({error.__class__.__name__})",
        }))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

