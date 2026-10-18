"""Privacy and schema tests for browser-to-backend diagnostics."""

from __future__ import annotations

import json
import logging
import os
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pydantic import ValidationError

from app_logging import configure_logging, get_logger, log_context, shutdown_logging
from client_diagnostics import ClientDiagnosticEvent, record_client_diagnostic
from config_env import load_runtime_config


class ClientDiagnosticTests(unittest.TestCase):
    def tearDown(self) -> None:
        shutdown_logging()

    def test_free_form_error_content_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ClientDiagnosticEvent.model_validate(
                {
                    "kind": "runtime_error",
                    "operation": "window.error",
                    "error_type": "TypeError",
                    "message": "drawing value 12.5 must stay private",
                }
            )

    def test_invalid_tokens_and_statuses_are_rejected(self) -> None:
        for payload in (
            {
                "kind": "runtime_error",
                "operation": "drawing value 12.5",
                "error_type": "TypeError",
            },
            {
                "kind": "api_error",
                "operation": "scan.create",
                "error_type": "ApiRequestError",
                "http_status": 999,
            },
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(ValidationError):
                    ClientDiagnosticEvent.model_validate(payload)

    def test_safe_event_uses_request_context_without_content(self) -> None:
        with TemporaryDirectory() as temporary:
            try:
                with patch.dict(
                    os.environ,
                    {"QTRACX_LOG_DIR": temporary, "QTRACX_LOG_LEVEL": "INFO"},
                    clear=True,
                ):
                    path = configure_logging(load_runtime_config())
                event = ClientDiagnosticEvent(
                    kind="api_error",
                    operation="scan.create",
                    error_type="ApiRequestError",
                    http_status=503,
                )
                with log_context(request_id="frontend-reference"):
                    record_client_diagnostic(event, get_logger("frontend"))
                for handler in logging.getLogger("qtracx").handlers:
                    handler.flush()

                payload = json.loads(path.read_text(encoding="utf-8").strip())
                self.assertEqual(payload["event"], "frontend_diagnostic")
                self.assertEqual(payload["request_id"], "frontend-reference")
                self.assertEqual(payload["status_code"], 503)
                self.assertEqual(payload["details"]["operation"], "scan.create")
                self.assertNotIn("message", payload["details"])
                self.assertNotIn("stack", payload["details"])
            finally:
                shutdown_logging()


if __name__ == "__main__":
    unittest.main()
