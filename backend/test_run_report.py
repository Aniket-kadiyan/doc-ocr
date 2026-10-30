from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep

import run_report
from inference_gate import InferenceCapacityGate
from scan_jobs import ScanJobManager


def _wait_for_job(
    manager: ScanJobManager,
    job_id: str,
    timeout: float = 5.0,
) -> dict:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        job = manager.get(job_id)
        if job and job["status"] in {"succeeded", "failed", "cancelled"}:
            return job
        sleep(0.01)
    raise AssertionError("scan job did not finish")


def _wait_for_report(
    directory: Path,
    timeout: float = 5.0,
) -> Path:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        reports = [
            path
            for path in directory.glob("*_scan-job_*.log")
            if path.name != run_report.INDEX_FILE_NAME
        ]
        if reports:
            return reports[0]
        sleep(0.01)
    raise AssertionError("scan report was not written")


class RunReportTests(unittest.TestCase):
    def tearDown(self) -> None:
        run_report.disable_run_reports()

    def test_scan_job_writes_private_readable_report(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = run_report.configure_run_reports(Path(temporary))
            manager = ScanJobManager(heartbeat_interval_seconds=0.01)
            gate = InferenceCapacityGate(1)

            def work(report_progress):
                report_progress(
                    stage="layout",
                    message="Analysing page",
                    percent=10,
                    completed=1,
                    total=1,
                    operation_label="Page layout analysis",
                )
                for index in range(1, 4):
                    report_progress(
                        stage="recognizing",
                        message=f"Recognizing object {index} of 3",
                        percent=50,
                        completed=index - 1,
                        total=3,
                        candidate_count=3,
                        operation_label=f"Object {index}",
                    )
                    with gate.slot("text_recognizer"):
                        sleep(0.01)
                return {"eligible_count": 3, "excluded_count": 1}

            try:
                job = manager.submit(
                    work,
                    metadata={
                        "scope_kind": "page",
                        "page": 2,
                        "gdt": True,
                        "scope_pixel_area": 2_500_000,
                    },
                )
                _wait_for_job(manager, job["job_id"])
                report_path = _wait_for_report(directory)
            finally:
                manager.shutdown()

            text = report_path.read_text(encoding="utf-8")
            self.assertIn("Auto-balloon scan (page scope, page 2)", text)
            self.assertIn("Status              : SUCCEEDED", text)
            self.assertIn("eligible count=3", text)
            self.assertIn("OCR model use: 3 calls", text)
            self.assertIn("recognizing", text)
            self.assertRegex(text, r"Object #\s+3 x")
            self.assertIn("intentionally excludes uploaded files", text)
            index_text = (
                directory / run_report.INDEX_FILE_NAME
            ).read_text(encoding="utf-8")
            self.assertIn(report_path.name, index_text)

    def test_failed_job_does_not_copy_private_error_text(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = run_report.configure_run_reports(Path(temporary))
            manager = ScanJobManager(heartbeat_interval_seconds=0.01)

            def work(report_progress):
                report_progress(
                    stage="detecting",
                    message="Running detector",
                    percent=10,
                )
                raise RuntimeError("private drawing value 12.5")

            try:
                job = manager.submit(
                    work,
                    metadata={"scope_kind": "section", "page": 1},
                )
                _wait_for_job(manager, job["job_id"])
                report_path = _wait_for_report(directory)
            finally:
                manager.shutdown()

            text = report_path.read_text(encoding="utf-8")
            self.assertIn("Status              : FAILED", text)
            self.assertIn("see qtracx-backend.log using this job id", text)
            self.assertNotIn("private drawing value", text)

    def test_prunes_old_reports(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = run_report.configure_run_reports(
                Path(temporary),
                keep=2,
            )
            for index in range(4):
                report = run_report.RunReport(
                    "scan-job",
                    f"run-{index}",
                    "test scan",
                ).start()
                report.finish("succeeded")
                sleep(0.01)
            reports = [
                path
                for path in directory.glob("*.log")
                if path.name != run_report.INDEX_FILE_NAME
            ]
            self.assertEqual(len(reports), 2)


if __name__ == "__main__":
    unittest.main()
