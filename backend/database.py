"""Database configuration and session management for QTracX authentication.

Production deployments support PostgreSQL and Microsoft SQL Server. SQLite is
accepted only when explicitly requested by tests; it is not a deployment
database for QTracX.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker


class DatabaseConfigurationError(ValueError):
    """The configured database URL is absent, unsupported, or incomplete."""


def normalize_database_url(
    value: str,
    *,
    allow_test_sqlite: bool = False,
) -> str:
    """Validate a database URL and select the supported permissive driver."""

    raw = value.strip()
    if not raw:
        raise DatabaseConfigurationError("DATABASE_URL is required")
    try:
        url = make_url(raw)
    except Exception as error:
        raise DatabaseConfigurationError("DATABASE_URL is not a valid URL") from error

    drivername = url.drivername.lower()
    backend, _, explicit_driver = drivername.partition("+")
    if backend == "postgres":
        backend = "postgresql"
    if backend == "postgresql":
        if not explicit_driver:
            url = url.set(drivername="postgresql+pg8000")
        elif explicit_driver != "pg8000":
            raise DatabaseConfigurationError(
                "PostgreSQL DATABASE_URL must use the pg8000 driver"
            )
    elif backend == "mssql":
        if not explicit_driver:
            url = url.set(drivername="mssql+pyodbc")
            explicit_driver = "pyodbc"
        if explicit_driver != "pyodbc":
            raise DatabaseConfigurationError(
                "MSSQL DATABASE_URL must use the pyodbc driver"
            )
        if "odbc_connect" not in url.query and "driver" not in url.query:
            raise DatabaseConfigurationError(
                "MSSQL DATABASE_URL must specify an ODBC driver query parameter"
            )
    elif backend == "sqlite" and allow_test_sqlite:
        return raw
    else:
        raise DatabaseConfigurationError(
            "DATABASE_URL must use PostgreSQL or Microsoft SQL Server"
        )

    return url.render_as_string(hide_password=False)


class DatabaseManager:
    """Own one SQLAlchemy engine and short-lived application sessions."""

    def __init__(
        self,
        database_url: str,
        *,
        allow_test_sqlite: bool = False,
    ) -> None:
        self.database_url = normalize_database_url(
            database_url,
            allow_test_sqlite=allow_test_sqlite,
        )
        self.engine: Engine = create_engine(
            self.database_url,
            pool_pre_ping=True,
            future=True,
        )
        self._sessions = sessionmaker(
            bind=self.engine,
            class_=Session,
            autoflush=False,
            expire_on_commit=False,
        )

    def ping(self) -> None:
        """Open a connection and execute a database-neutral health query."""

        with self.engine.connect() as connection:
            connection.execute(text("SELECT 1"))

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Commit a unit of work, rolling it back safely on failure."""

        session = self._sessions()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        self.engine.dispose()


def database_backend(database_url: str) -> str:
    """Return a non-secret backend label for diagnostics and setup output."""

    normalized = normalize_database_url(database_url)
    return make_url(normalized).get_backend_name()
