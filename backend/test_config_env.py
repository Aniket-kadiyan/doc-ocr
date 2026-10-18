"""Tests for the validated QTracX runtime configuration contract."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from config_env import RuntimeConfigurationError, load_runtime_config


class RuntimeConfigTests(unittest.TestCase):
    def test_defaults_preserve_backend_and_use_ui_port_3002(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = load_runtime_config()

        self.assertEqual(config.environment, "development")
        self.assertEqual(config.ui_port, 3002)
        self.assertEqual(config.backend_host, "127.0.0.1")
        self.assertEqual(config.backend_port, 8000)
        self.assertTrue(config.backend_reload)
        self.assertIsNone(config.cors_origins)
        self.assertFalse(config.https_enabled)

    def test_valid_values_are_normalized(self) -> None:
        values = {
            "QTRACX_ENV": "PRODUCTION",
            "PORT": "3102",
            "OCR_HOST": "0.0.0.0",
            "OCR_PORT": "8100",
            "OCR_RELOAD": "false",
            "OCR_CORS_ORIGINS": "http://qtracx:3102/,https://qtracx.example",
            "CHECKSHEET_DATA_DIR": "deployment-data/checksheets",
            "CHECKSHEET_MAX_DOCUMENT_MB": "512",
            "DATABASE_URL": "postgresql://user:secret@db/qtracx",
            "QTRACX_LOG_DIR": "deployment-data/logs",
            "QTRACX_LOG_LEVEL": "warning",
            "QTRACX_LOG_MAX_MB": "50",
            "QTRACX_LOG_BACKUP_COUNT": "20",
            "QTRACX_HTTPS_ENABLED": "yes",
            "QTRACX_TLS_CERT_FILE": "certs/qtracx.crt",
            "QTRACX_TLS_KEY_FILE": "certs/qtracx.key",
        }
        with patch.dict(os.environ, values, clear=True):
            config = load_runtime_config()

        self.assertEqual(config.environment, "production")
        self.assertEqual(config.ui_port, 3102)
        self.assertEqual(config.backend_host, "0.0.0.0")
        self.assertEqual(config.backend_port, 8100)
        self.assertFalse(config.backend_reload)
        self.assertEqual(
            config.cors_origins,
            ("http://qtracx:3102", "https://qtracx.example"),
        )
        self.assertEqual(config.log_level, "WARNING")
        self.assertEqual(config.log_max_mb, 50)
        self.assertEqual(config.log_backup_count, 20)
        self.assertTrue(config.https_enabled)
        self.assertTrue(config.checksheet_data_dir.is_absolute())
        self.assertTrue(config.log_dir.is_absolute())
        self.assertEqual(config.tls_cert_file, Path(config.tls_cert_file).resolve())

    def test_invalid_port_fails_instead_of_silently_using_default(self) -> None:
        with patch.dict(os.environ, {"OCR_PORT": "not-a-port"}, clear=True):
            with self.assertRaisesRegex(
                RuntimeConfigurationError, "OCR_PORT must be an integer"
            ):
                load_runtime_config()

    def test_invalid_boolean_fails_instead_of_silently_disabling_it(self) -> None:
        with patch.dict(os.environ, {"OCR_RELOAD": "sometimes"}, clear=True):
            with self.assertRaisesRegex(
                RuntimeConfigurationError, "OCR_RELOAD must be one of"
            ):
                load_runtime_config()

    def test_cors_rejects_paths(self) -> None:
        with patch.dict(
            os.environ,
            {"OCR_CORS_ORIGINS": "http://qtracx:3002/application"},
            clear=True,
        ):
            with self.assertRaisesRegex(
                RuntimeConfigurationError, "must not contain paths"
            ):
                load_runtime_config()

    def test_public_summary_does_not_expose_database_credentials(self) -> None:
        secret = "postgresql://user:very-secret@db/qtracx"
        with patch.dict(os.environ, {"DATABASE_URL": secret}, clear=True):
            config = load_runtime_config()
            summary = config.public_summary()

        self.assertTrue(summary["database_configured"])
        self.assertNotIn(secret, repr(config))
        self.assertNotIn(secret, repr(summary))


if __name__ == "__main__":
    unittest.main()
