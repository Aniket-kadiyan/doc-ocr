from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from config_env import RuntimeConfigurationError, load_runtime_config


class RuntimeConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = load_runtime_config()
        self.assertEqual(config.ui_port, 3002)
        self.assertEqual(config.ui_host, "127.0.0.1")
        self.assertEqual(config.backend_host, "127.0.0.1")
        self.assertEqual(config.backend_port, 8000)
        self.assertEqual(config.cad_max_document_mb, 128)
        self.assertEqual(config.cad_convert_timeout_seconds, 180)
        self.assertIsNone(config.dwg_converter_path)
        self.assertTrue(config.backend_reload)
        self.assertEqual(config.ocr_model_instances, 1)
        self.assertFalse(config.https_enabled)
        self.assertIsNone(config.tls_cert_file)
        self.assertIsNone(config.tls_key_file)
        self.assertEqual(config.auth_session_hours, 8)
        self.assertEqual(config.run_report_keep, 300)
        self.assertEqual(
            config.public_summary()["cors_policy"], "allow_all_origins"
        )

    def test_valid_values_are_normalized(self) -> None:
        values = {
            "QTRACX_ENV": "PRODUCTION", "UI_HOST": "0.0.0.0", "PORT": "3102",
            "OCR_HOST": "0.0.0.0", "OCR_PORT": "8100", "OCR_RELOAD": "false",
            "OCR_MODEL_INSTANCES": "3",
            "CAD_MAX_DOCUMENT_MB": "256",
            "CAD_CONVERT_TIMEOUT_SECONDS": "240",
            "DWG2DXF_ACADSHARP": "tools/dwg2dxf/custom-converter",
            "HTTPS_ENABLED": "true",
            "TLS_CERT_FILE": "certificates/qtracx.crt",
            "TLS_KEY_FILE": "certificates/qtracx.key",
            "DATABASE_URL": "postgresql://user:secret@db/qtracx",
            "QTRACX_LOG_LEVEL": "warning", "QTRACX_LOG_MAX_MB": "50",
            "QTRACX_LOG_BACKUP_COUNT": "20",
            "QTRACX_RUN_REPORT_KEEP": "450",
            "AUTH_SESSION_HOURS": "12",
        }
        with patch.dict(os.environ, values, clear=True):
            config = load_runtime_config()
        self.assertEqual(config.environment, "production")
        self.assertEqual(config.ui_host, "0.0.0.0")
        self.assertEqual(config.backend_port, 8100)
        self.assertFalse(config.backend_reload)
        self.assertEqual(config.ocr_model_instances, 3)
        self.assertEqual(config.public_summary()["ocr_model_instances"], 3)
        self.assertEqual(config.cad_max_document_mb, 256)
        self.assertEqual(config.cad_convert_timeout_seconds, 240)
        self.assertTrue(config.public_summary()["dwg_converter_configured"])
        self.assertTrue(
            config.dwg_converter_path.as_posix().endswith(
                "tools/dwg2dxf/custom-converter"
            )
        )
        self.assertEqual(config.log_level, "WARNING")
        self.assertEqual(config.run_report_keep, 450)
        self.assertEqual(config.auth_session_hours, 12)
        self.assertTrue(config.https_enabled)
        self.assertTrue(
            str(config.tls_cert_file).endswith("certificates/qtracx.crt")
        )
        self.assertTrue(
            str(config.tls_key_file).endswith("certificates/qtracx.key")
        )
        self.assertTrue(config.public_summary()["tls_cert_configured"])

    def test_invalid_values_fail(self) -> None:
        with patch.dict(os.environ, {"OCR_PORT": "bad"}, clear=True):
            with self.assertRaisesRegex(RuntimeConfigurationError, "integer"):
                load_runtime_config()
        with patch.dict(os.environ, {"OCR_RELOAD": "sometimes"}, clear=True):
            with self.assertRaisesRegex(RuntimeConfigurationError, "must be one of"):
                load_runtime_config()
        with patch.dict(os.environ, {"HTTPS_ENABLED": "sometimes"}, clear=True):
            with self.assertRaisesRegex(RuntimeConfigurationError, "must be one of"):
                load_runtime_config()
        with patch.dict(os.environ, {"AUTH_SESSION_HOURS": "0"}, clear=True):
            with self.assertRaisesRegex(RuntimeConfigurationError, "between"):
                load_runtime_config()
        with patch.dict(os.environ, {"OCR_MODEL_INSTANCES": "0"}, clear=True):
            with self.assertRaisesRegex(RuntimeConfigurationError, "between"):
                load_runtime_config()
        with patch.dict(os.environ, {"OCR_MODEL_INSTANCES": "9"}, clear=True):
            with self.assertRaisesRegex(RuntimeConfigurationError, "between"):
                load_runtime_config()
        with patch.dict(os.environ, {"CAD_MAX_DOCUMENT_MB": "0"}, clear=True):
            with self.assertRaisesRegex(RuntimeConfigurationError, "between"):
                load_runtime_config()
        with patch.dict(
            os.environ,
            {"QTRACX_RUN_REPORT_KEEP": "10001"},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeConfigurationError, "between"):
                load_runtime_config()
        with patch.dict(
            os.environ,
            {"CAD_CONVERT_TIMEOUT_SECONDS": "1801"},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeConfigurationError, "between"):
                load_runtime_config()

    def test_absolute_certificate_paths_are_preserved(self) -> None:
        with TemporaryDirectory() as temporary:
            cert = Path(temporary) / "certificate.pem"
            key = Path(temporary) / "private-key.pem"
            with patch.dict(
                os.environ,
                {
                    "TLS_CERT_FILE": str(cert),
                    "TLS_KEY_FILE": str(key),
                },
                clear=True,
            ):
                config = load_runtime_config()
        self.assertEqual(config.tls_cert_file, cert.resolve())
        self.assertEqual(config.tls_key_file, key.resolve())

    def test_summary_hides_database_credentials(self) -> None:
        secret = "postgresql://user:very-secret@db/qtracx"
        with patch.dict(os.environ, {"DATABASE_URL": secret}, clear=True):
            config = load_runtime_config()
        self.assertTrue(config.public_summary()["database_configured"])
        self.assertNotIn(secret, repr(config))
        self.assertNotIn(secret, repr(config.public_summary()))


if __name__ == "__main__":
    unittest.main()
