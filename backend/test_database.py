from __future__ import annotations

import unittest

from database import DatabaseConfigurationError, normalize_database_url


class DatabaseConfigurationTests(unittest.TestCase):
    def test_postgresql_defaults_to_pg8000(self) -> None:
        url = normalize_database_url("postgresql://user:secret@db/qtracx")
        self.assertTrue(url.startswith("postgresql+pg8000://"))
        self.assertIn("secret", url)

    def test_explicit_pg8000_is_preserved(self) -> None:
        url = normalize_database_url("postgresql+pg8000://user:secret@db/qtracx")
        self.assertTrue(url.startswith("postgresql+pg8000://"))

    def test_mssql_requires_an_explicit_odbc_driver(self) -> None:
        with self.assertRaisesRegex(DatabaseConfigurationError, "ODBC driver"):
            normalize_database_url("mssql://user:secret@db/qtracx")
        url = normalize_database_url(
            "mssql+pyodbc://user:secret@db/qtracx"
            "?driver=ODBC+Driver+18+for+SQL+Server"
        )
        self.assertTrue(url.startswith("mssql+pyodbc://"))

    def test_unsupported_production_database_is_rejected(self) -> None:
        for value in ("sqlite:///qtracx.db", "mysql://user:secret@db/qtracx"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    DatabaseConfigurationError,
                    "PostgreSQL or Microsoft SQL Server",
                ):
                    normalize_database_url(value)

    def test_sqlite_is_available_only_for_schema_tests(self) -> None:
        value = "sqlite:///:memory:"
        self.assertEqual(
            normalize_database_url(value, allow_test_sqlite=True),
            value,
        )


if __name__ == "__main__":
    unittest.main()
