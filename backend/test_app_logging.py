from __future__ import annotations

import json
import logging
import os
import unittest
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from unittest.mock import patch

from app_logging import (
    configure_logging, get_logger, log_context, request_id,
    safe_exception_details, shutdown_logging,
)
from config_env import load_runtime_config
from scan_jobs import ScanJobManager


class StructuredLoggingTests(unittest.TestCase):
    def tearDown(self) -> None:
        shutdown_logging()

    def test_request_id_and_exception_privacy(self) -> None:
        self.assertEqual(request_id("client-request_12"), "client-request_12")
        self.assertNotEqual(request_id("bad request\nheader"), "bad request\nheader")
        try:
            raise RuntimeError("private drawing value")
        except RuntimeError as error:
            details = safe_exception_details(error)
        self.assertEqual(details["error_type"], "RuntimeError")
        self.assertNotIn("private drawing value", repr(details))

    def test_json_context_and_database_secret(self) -> None:
        with TemporaryDirectory() as temporary:
            with patch.dict(os.environ, {
                "QTRACX_LOG_DIR": temporary,
                "DATABASE_URL": "postgresql://user:secret@db/qtracx",
            }, clear=True):
                config = load_runtime_config()
                path = configure_logging(config)
            with log_context(request_id="request-1", job_id="job-1"):
                get_logger("test").info(
                    "Synthetic", extra={"event": "synthetic_event", "candidate_count": 3}
                )
            for handler in logging.getLogger("qtracx").handlers:
                handler.flush()
            payload = json.loads(path.read_text(encoding="utf-8").strip())
            self.assertEqual(payload["request_id"], "request-1")
            self.assertEqual(payload["candidate_count"], 3)
            self.assertNotIn("postgresql://", path.read_text(encoding="utf-8"))

    def test_request_context_reaches_background_job(self) -> None:
        with TemporaryDirectory() as temporary:
            with patch.dict(os.environ, {"QTRACX_LOG_DIR": temporary}, clear=True):
                path = configure_logging(load_runtime_config())
            manager = ScanJobManager(max_workers=1)
            try:
                with log_context(request_id="scan-request"):
                    initial = manager.submit(
                        lambda _report: {"count": 2, "regions": []},
                        metadata={"scope_kind": "page", "page": 3},
                    )
                deadline = monotonic() + 2
                while monotonic() < deadline:
                    if manager.get(initial["job_id"])["status"] == "succeeded":
                        break
                    sleep(0.01)
            finally:
                manager.shutdown()
            for handler in logging.getLogger("qtracx").handlers:
                handler.flush()
            events = [json.loads(line) for line in path.read_text().splitlines()]
            completed = next(e for e in events if e["event"] == "scan_job_completed")
            self.assertEqual(completed["request_id"], "scan-request")
            self.assertEqual(completed["candidate_count"], 2)


if __name__ == "__main__":
    unittest.main()
