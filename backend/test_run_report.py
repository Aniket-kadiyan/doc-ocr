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
            # The in-progress copy is deleted once the final report lands.
            and not path.name.endswith(run_report.RUNNING_SUFFIX)
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

    def test_nested_steps_record_time_runs_and_contribution(self) -> None:
        report = run_report.RunReport("scan-job", "steps", "test scan").start()
        with run_report.activate(report):
            with run_report.step("Classify callouts", "Gate callouts") as outer:
                for _ in range(3):
                    with run_report.step("Dimension-line check") as inner:
                        inner.count(on_line=1)
                    run_report.record_model_call(
                        "paddle", waited_seconds=0.0, active_seconds=0.01
                    )
                outer.count(values=2, excluded=1)
        report.finish("succeeded", rule_counts={"pdf_text_in_table": 4})

        outer_step = report.steps[("Classify callouts",)]
        inner_step = report.steps[("Classify callouts", "Dimension-line check")]
        self.assertEqual(outer_step.calls, 1)
        self.assertEqual(outer_step.model_calls, 3)
        self.assertEqual(outer_step.counts, {"values": 2, "excluded": 1})
        self.assertEqual(inner_step.calls, 3)
        self.assertEqual(inner_step.counts, {"on_line": 3})
        text = report.render()
        self.assertIn("STEP TREE (2 distinct steps, 4 step runs)", text)
        self.assertIn("  Dimension-line check", text)
        self.assertIn("result : values=2, excluded=1", text)
        self.assertIn("pdf_text_in_table", text)

    def test_a_step_repeated_per_parent_is_timed_per_parent(self) -> None:
        report = run_report.RunReport("scan-job", "areas", "test scan").start()
        with run_report.activate(report):
            for index, reads in ((1, 3), (2, 1)):
                with run_report.step(f"Neighbourhood {index:02d}") as area:
                    for _ in range(reads):
                        with run_report.step("Read callout", "OCR one crop"):
                            run_report.record_model_call(
                                "paddle", waited_seconds=0.0, active_seconds=0.01
                            )
                    area.count(values=1)
        report.finish("succeeded")

        first = report.steps[("Neighbourhood 01", "Read callout")]
        second = report.steps[("Neighbourhood 02", "Read callout")]
        self.assertEqual((first.calls, first.model_calls), (3, 3))
        self.assertEqual((second.calls, second.model_calls), (1, 1))
        # The purpose is printed once, not under every neighbourhood.
        self.assertEqual(report.render().count("what   : OCR one crop"), 1)

    def test_steps_without_an_active_report_are_inert(self) -> None:
        with run_report.step("Anything") as handle:
            handle.count(found=1)
        self.assertEqual(handle.counts, {"found": 1})

    def test_a_running_scan_keeps_a_report_until_it_finishes(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = run_report.configure_run_reports(Path(temporary))
            report = run_report.RunReport("scan-job", "live0001", "test scan").start()
            report.progress(stage="detecting", message="Running detector")
            running = list(directory.glob(f"*{run_report.RUNNING_SUFFIX}"))
            self.assertEqual(len(running), 1)
            self.assertIn("detecting", running[0].read_text(encoding="utf-8"))
            report.finish("succeeded")
            self.assertEqual(list(directory.glob(f"*{run_report.RUNNING_SUFFIX}")), [])

    def test_a_scan_killed_mid_run_is_reported_as_interrupted(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = run_report.configure_run_reports(Path(temporary))
            report = run_report.RunReport("scan-job", "dead0001", "test scan").start()
            report.progress(stage="slanted", message="Slanted-callout recovery")
            # The process dies here: finish() never runs. The next backend
            # process configures reports at startup.
            run_report.configure_run_reports(Path(temporary))
            self.assertEqual(list(directory.glob(f"*{run_report.RUNNING_SUFFIX}")), [])
            interrupted = list(directory.glob("*_interrupted.log"))
            self.assertEqual(len(interrupted), 1)
            text = interrupted[0].read_text(encoding="utf-8")
            self.assertTrue(text.startswith("INTERRUPTED"))
            self.assertIn("slanted", text)
            index = (directory / run_report.INDEX_FILE_NAME).read_text(encoding="utf-8")
            self.assertIn("INTERRUPTED", index)


if __name__ == "__main__":
    unittest.main()
