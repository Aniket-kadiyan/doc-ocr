from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep

import run_report
from inference_gate import InferenceCapacityGate
from scan_jobs import ScanJobManager


def _wait_for(path: Path, pattern: str, timeout: float = 5.0) -> list[Path]:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        found = [p for p in path.glob(pattern) if p.name != run_report.INDEX_FILE_NAME]
        if found:
            return found
        sleep(0.02)
    return []


class RunReportTests(unittest.TestCase):
    def test_scan_job_writes_readable_report(self) -> None:
        gate = InferenceCapacityGate(1)
        with TemporaryDirectory() as temporary:
            directory = run_report.configure_run_reports(Path(temporary))
            manager = ScanJobManager(heartbeat_interval_seconds=0.01)

            def work(report_progress):
                report_progress(stage="layout", message="Analysing page", percent=2,
                                operation_label="Page layout analysis")
                sleep(0.05)
                for index in range(1, 4):
                    report_progress(stage="recognizing", message=f"Recognizing object {index} of 3",
                                    percent=50, completed=index - 1, total=3,
                                    operation_label=f"Object {index}")
                    with gate.slot("test-recognition"):
                        sleep(0.02)
                return {"eligible_count": 3, "excluded_count": 1}

            try:
                manager.submit(work, metadata={"scope_kind": "page", "page": 2, "gdt": True})
                reports = _wait_for(directory, "*_scan-job_*.log")
            finally:
                manager.shutdown()
            self.assertEqual(len(reports), 1)
            text = reports[0].read_text(encoding="utf-8")
            self.assertIn("Scan job (page scope, page 2)", text)
            self.assertIn("Status          : SUCCEEDED", text)
            self.assertIn("eligible count=3", text)
            self.assertIn("WHERE THE TIME WENT", text)
            self.assertIn("[3] recognizing", text)
            # 'Object 1..3' collapse into one operation with three runs.
            self.assertRegex(text, r"Object #\s+3 x")
            self.assertIn("OCR models   : 3 calls", text)
            index = (directory / run_report.INDEX_FILE_NAME).read_text(encoding="utf-8")
            self.assertIn(reports[0].name, index)

    def test_failed_job_does_not_copy_error_text(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = run_report.configure_run_reports(Path(temporary))
            manager = ScanJobManager(heartbeat_interval_seconds=0.01)

            def work(report_progress):
                report_progress(stage="detecting", message="Running", percent=10)
                raise RuntimeError("private drawing value 12.5")

            try:
                manager.submit(work, metadata={"scope_kind": "section"})
                reports = _wait_for(directory, "*_scan-job_*.log")
            finally:
                manager.shutdown()
            text = reports[0].read_text(encoding="utf-8")
            self.assertIn("Status          : FAILED", text)
            self.assertIn("failed in step 'detecting'", text)
            self.assertNotIn("private drawing value", text)

    def test_helpers_are_noops_without_a_run(self) -> None:
        with run_report.run_step("anything"):
            run_report.count("nothing")
            run_report.note(image="1 x 1 px")
            run_report.model_call(0.1, 0.2)

    def test_reporting_can_be_disabled(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.assertIsNone(
                run_report.configure_run_reports(directory, enabled=False)
            )
            report = run_report.RunReport("test", "disabled", "disabled").start()
            report.finish("succeeded")
            self.assertEqual(list(directory.iterdir()), [])
            run_report.configure_run_reports(directory)

    def test_prunes_old_reports(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = run_report.configure_run_reports(Path(temporary), keep=2)
            for index in range(4):
                report = run_report.RunReport("request-ocr-x", f"run{index}", "POST /ocr/x").start()
                report.wall_started += index  # distinct file names
                report.finish("succeeded")
            kept = [p for p in directory.glob("*.log") if p.name != run_report.INDEX_FILE_NAME]
            self.assertEqual(len(kept), 2)
            run_report.configure_run_reports(Path(temporary))


if __name__ == "__main__":
    unittest.main()


