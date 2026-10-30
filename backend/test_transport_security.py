from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from config_env import load_runtime_config
from transport_security import resolve_transport_security


class TransportSecurityTests(unittest.TestCase):
    def test_http_is_the_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            transport = resolve_transport_security(load_runtime_config())
        self.assertFalse(transport.requested)
        self.assertFalse(transport.active)
        self.assertEqual(transport.scheme, "http")
        self.assertIsNone(transport.fallback_reason)

    def test_missing_certificate_configuration_falls_back_to_http(self) -> None:
        with patch.dict(os.environ, {"HTTPS_ENABLED": "true"}, clear=True):
            transport = resolve_transport_security(load_runtime_config())
        self.assertTrue(transport.requested)
        self.assertFalse(transport.active)
        self.assertIn("both be configured", transport.fallback_reason or "")

    def test_missing_files_fall_back_without_exposing_paths(self) -> None:
        with TemporaryDirectory() as temporary:
            cert = Path(temporary) / "missing-cert.pem"
            key = Path(temporary) / "missing-key.pem"
            with patch.dict(
                os.environ,
                {
                    "HTTPS_ENABLED": "true",
                    "TLS_CERT_FILE": str(cert),
                    "TLS_KEY_FILE": str(key),
                },
                clear=True,
            ):
                transport = resolve_transport_security(load_runtime_config())
        self.assertFalse(transport.active)
        self.assertEqual(
            transport.fallback_reason, "TLS certificate file is unavailable"
        )
        self.assertNotIn(str(cert), repr(transport.public_summary()))

    def test_validated_pair_activates_https(self) -> None:
        with TemporaryDirectory() as temporary:
            cert = Path(temporary) / "cert.pem"
            key = Path(temporary) / "key.pem"
            cert.write_text("test certificate", encoding="utf-8")
            key.write_text("test key", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "HTTPS_ENABLED": "true",
                    "TLS_CERT_FILE": str(cert),
                    "TLS_KEY_FILE": str(key),
                },
                clear=True,
            ), patch(
                "transport_security._certificate_pair_is_valid",
                return_value=True,
            ) as load_pair:
                transport = resolve_transport_security(load_runtime_config())
        self.assertTrue(transport.active)
        self.assertEqual(transport.scheme, "https")
        load_pair.assert_called_once_with(cert.resolve(), key.resolve())

    def test_invalid_pair_falls_back_to_http(self) -> None:
        with TemporaryDirectory() as temporary:
            cert = Path(temporary) / "cert.pem"
            key = Path(temporary) / "key.pem"
            cert.write_text("invalid", encoding="utf-8")
            key.write_text("invalid", encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "HTTPS_ENABLED": "true",
                    "TLS_CERT_FILE": str(cert),
                    "TLS_KEY_FILE": str(key),
                },
                clear=True,
            ):
                transport = resolve_transport_security(load_runtime_config())
        self.assertFalse(transport.active)
        self.assertIn("invalid", transport.fallback_reason or "")


if __name__ == "__main__":
    unittest.main()

