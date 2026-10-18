"""Fast tests for structured logging and safe correlation metadata."""

from __future__ import annotations

import json
import logging
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from unittest.mock import patch

from app_logging import (
    configure_logging,
    get_logger,
    log_context,
    request_id,
    safe_exception_details,
    shutdown_logging,
)
from config_env import load_runtime_config
from scan_jobs import ScanJobManager


class StructuredLoggingTests(unittest.TestCase):
    def tearDown(self) -> None:
        shutdown_logging()

    def test_request_id_keeps_safe_value_and_replaces_unsafe_value(self) -> None:
        self.assertEqual(request_id("client-request_12"), "client-request_12")
        generated = request_id("bad request\nforged-header")
        self.assertNotEqual(generated, "bad request\nforged-header")
        self.assertEqual(len(generated), 36)

    def test_rotating_file_receives_json_context(self) -> None:
        with TemporaryDirectory() as temporary:
            with patch.dict(
                os.environ,
                {
                    "QTRACX_LOG_DIR": temporary,
                    "QTRACX_LOG_LEVEL": "INFO",
                },
                clear=True,
            ):
                path = configure_logging(load_runtime_config())
            with log_context(request_id="request-1", job_id="job-1"):
                get_logger("test").info(
                    "Synthetic event",
                    extra={"event": "synthetic_event", "candidate_count": 3},
                )
            for handler in logging.getLogger("qtracx").handlers:
                handler.flush()

            payload = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(payload["event"], "synthetic_event")
            self.assertEqual(payload["request_id"], "request-1")
            self.assertEqual(payload["job_id"], "job-1")
            self.assertEqual(payload["candidate_count"], 3)
            self.assertEqual(payload["service"], "qtracx-backend")
            self.assertNotIn("password", payload)

    def test_exception_details_do_not_include_exception_message(self) -> None:
        secret = "drawing value 12.5 must not be copied"
        try:
            raise RuntimeError(secret)
        except RuntimeError as error:
            details = safe_exception_details(error)

        self.assertEqual(details["error_type"], "RuntimeError")
        self.assertIn("test_app_logging.py", details["error_location"])
        self.assertNotIn(secret, repr(details))

    def test_database_url_is_absent_from_log_configuration(self) -> None:
        with TemporaryDirectory() as temporary:
            secret = "postgresql://user:secret@database/qtracx"
            with patch.dict(
                os.environ,
                {"QTRACX_LOG_DIR": temporary, "DATABASE_URL": secret},
                clear=True,
            ):
                config = load_runtime_config()
                path = configure_logging(config)
            get_logger("test").info(
                "Startup",
                extra={"event": "startup", "details": config.public_summary()},
            )
            for handler in logging.getLogger("qtracx").handlers:
                handler.flush()

            self.assertTrue(Path(path).exists())
            self.assertNotIn(secret, path.read_text(encoding="utf-8"))

    def test_request_context_reaches_background_scan_job_events(self) -> None:
        with TemporaryDirectory() as temporary:
            with patch.dict(
                os.environ,
                {"QTRACX_LOG_DIR": temporary, "QTRACX_LOG_LEVEL": "INFO"},
                clear=True,
            ):
                path = configure_logging(load_runtime_config())
            manager = ScanJobManager(max_workers=1)
            try:
                with log_context(request_id="scan-request"):
                    initial = manager.submit(
                        lambda _report: {"count": 2, "regions": []},
                        metadata={"scope_kind": "page", "page": 3},
                    )
                deadline = monotonic() + 2.0
                while monotonic() < deadline:
                    snapshot = manager.get(initial["job_id"])
                    if snapshot and snapshot["status"] == "succeeded":
                        break
                    sleep(0.01)
                else:
                    self.fail("background scan did not finish")
            finally:
                manager.shutdown()
            for handler in logging.getLogger("qtracx").handlers:
                handler.flush()

            events = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            completed = next(
                event for event in events if event["event"] == "scan_job_completed"
            )
            self.assertEqual(completed["request_id"], "scan-request")
            self.assertEqual(completed["job_id"], initial["job_id"])
            self.assertEqual(completed["scope_kind"], "page")
            self.assertEqual(completed["page"], 3)
            self.assertEqual(completed["candidate_count"], 2)


if __name__ == "__main__":
    unittest.main()
