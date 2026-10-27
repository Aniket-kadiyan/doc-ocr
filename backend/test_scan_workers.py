from __future__ import annotations

from concurrent.futures import Future

from PIL import Image, ImageDraw

from ocr_pipeline import _ink_row_count
from scan_workers import PRIORITY_BACKGROUND, PRIORITY_CRITICAL, ScanWork


class _Pipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []

    def double(self, image: Image.Image, value: int) -> int:
        self.calls.append(("double", (value,)))
        return value * 2


class _Pool:
    """A pool that queues jobs and runs nothing until told to."""

    broken = False

    def __init__(self) -> None:
        self.jobs: list[tuple[int, str, tuple, Future]] = []

    def submit(self, priority, path, method, *args, **kwargs) -> Future:
        future: Future = Future()
        self.jobs.append((priority, method, args, future))
        return future


def test_without_a_pool_every_job_runs_in_process() -> None:
    pipeline = _Pipeline()
    work = ScanWork(pipeline, Image.new("RGB", (4, 4)), None)

    futures = [work.submit(PRIORITY_CRITICAL, "double", n) for n in range(3)]

    assert not work.parallel
    assert work.gather(futures) == [0, 2, 4]
    work.close()


def test_gather_takes_queued_jobs_and_keeps_worker_results_in_order() -> None:
    pipeline = _Pipeline()
    pool = _Pool()
    work = ScanWork(pipeline, Image.new("RGB", (4, 4)), pool)
    futures = [work.submit(PRIORITY_CRITICAL, "double", n) for n in range(4)]
    # Two jobs already reached a worker; the other two are still queued.
    for _priority, _method, args, future in pool.jobs[:2]:
        future.set_running_or_notify_cancel()
        future.set_result(args[0] * 100)

    assert work.gather(futures) == [0, 100, 4, 6]
    # The queued ones ran here, from the back.
    assert [args for _name, args in pipeline.calls] == [(3,), (2,)]
    work.close()


def test_a_failed_worker_job_runs_again_in_process() -> None:
    pipeline = _Pipeline()
    pool = _Pool()
    work = ScanWork(pipeline, Image.new("RGB", (4, 4)), pool)
    future = work.submit(PRIORITY_BACKGROUND, "double", 21)
    pool.jobs[0][3].set_running_or_notify_cancel()
    pool.jobs[0][3].set_exception(RuntimeError("worker died"))

    assert work.result(future) == 42
    work.close()


def test_closing_withdraws_queued_jobs_and_refuses_new_ones() -> None:
    pool = _Pool()
    work = ScanWork(_Pipeline(), Image.new("RGB", (4, 4)), pool)
    queued = work.submit(PRIORITY_BACKGROUND, "double", 1)

    work.close()

    assert queued.cancelled()
    assert work.submit(PRIORITY_BACKGROUND, "double", 2).cancelled()


def _text_rows(rows: list[tuple[int, int]], size=(120, 60)) -> Image.Image:
    image = Image.new("L", size, 255)
    draw = ImageDraw.Draw(image)
    for top, bottom in rows:
        for x in range(10, 110, 12):
            draw.rectangle([x, top, x + 7, bottom], fill=0)
    return image.convert("RGB")


def test_ink_rows_tell_a_stacked_pair_from_one_short_value() -> None:
    box = {"x": 0, "y": 0, "width": 120, "height": 60}

    assert _ink_row_count(_text_rows([(10, 50)]), box) == 1
    assert _ink_row_count(_text_rows([(5, 25), (35, 55)]), box) == 2


def test_a_rule_between_stacked_values_does_not_join_them() -> None:
    image = _text_rows([(5, 25), (35, 55)])
    ImageDraw.Draw(image).line([(0, 30), (119, 30)], fill=(0, 0, 0), width=2)

    assert _ink_row_count(image, {"x": 0, "y": 0, "width": 120, "height": 60}) == 2


def test_vertical_text_is_profiled_across_its_reading_direction() -> None:
    upright = _text_rows([(5, 25), (35, 55)])
    turned = upright.rotate(90, expand=True)
    box = {"x": 0, "y": 0, "width": turned.width, "height": turned.height}

    assert _ink_row_count(turned, box, rotation=90) == 2
