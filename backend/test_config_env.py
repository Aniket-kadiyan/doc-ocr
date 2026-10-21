from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from config_env import RuntimeConfigurationError, load_runtime_config


class RuntimeConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = load_runtime_config()
        self.assertEqual(config.ui_port, 3002)
        self.assertEqual(config.backend_host, "127.0.0.1")
        self.assertEqual(config.backend_port, 8000)
        self.assertTrue(config.backend_reload)
        self.assertIsNone(config.cors_origins)

    def test_valid_values_are_normalized(self) -> None:
        values = {
            "QTRACX_ENV": "PRODUCTION", "PORT": "3102",
            "OCR_HOST": "0.0.0.0", "OCR_PORT": "8100", "OCR_RELOAD": "false",
            "OCR_CORS_ORIGINS": "http://qtracx:3102/,http://192.168.1.20:3102",
            "DATABASE_URL": "postgresql://user:secret@db/qtracx",
            "QTRACX_LOG_LEVEL": "warning", "QTRACX_LOG_MAX_MB": "50",
            "QTRACX_LOG_BACKUP_COUNT": "20",
        }
        with patch.dict(os.environ, values, clear=True):
            config = load_runtime_config()
        self.assertEqual(config.environment, "production")
        self.assertEqual(config.backend_port, 8100)
        self.assertFalse(config.backend_reload)
        self.assertEqual(config.log_level, "WARNING")
        self.assertEqual(config.cors_origins, (
            "http://qtracx:3102", "http://192.168.1.20:3102"
        ))

    def test_invalid_values_fail(self) -> None:
        with patch.dict(os.environ, {"OCR_PORT": "bad"}, clear=True):
            with self.assertRaisesRegex(RuntimeConfigurationError, "integer"):
                load_runtime_config()
        with patch.dict(os.environ, {"OCR_RELOAD": "sometimes"}, clear=True):
            with self.assertRaisesRegex(RuntimeConfigurationError, "must be one of"):
                load_runtime_config()

    def test_cors_rejects_paths(self) -> None:
        with patch.dict(
            os.environ, {"OCR_CORS_ORIGINS": "http://qtracx:3002/app"}, clear=True
        ):
            with self.assertRaisesRegex(RuntimeConfigurationError, "must not contain paths"):
                load_runtime_config()

    def test_summary_hides_database_credentials(self) -> None:
        secret = "postgresql://user:very-secret@db/qtracx"
        with patch.dict(os.environ, {"DATABASE_URL": secret}, clear=True):
            config = load_runtime_config()
        self.assertTrue(config.public_summary()["database_configured"])
        self.assertNotIn(secret, repr(config))
        self.assertNotIn(secret, repr(config.public_summary()))


if __name__ == "__main__":
    unittest.main()
