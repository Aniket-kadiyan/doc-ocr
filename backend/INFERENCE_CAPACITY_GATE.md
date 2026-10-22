# Shared inference-capacity gate

QTracX currently creates one Paddle model bundle per backend process. This
milestone adds one process-wide gate with one inference slot, shared by every
route and the auto-balloon worker.

## What is gated

The gate is held only while code is executing a Paddle model call:

- the main PaddleOCR pipeline;
- the standalone text detector;
- the standalone text-line orientation classifier; and
- the standalone text recognizer.

Image decoding, preprocessing, page layout, balloon filtering, result parsing,
serialization and database work stay outside the gate and can continue in
parallel.

## Priority and queue behaviour

- Direct/manual OCR work has `interactive` priority by default.
- Auto-balloon jobs run with `background` inference priority.
- If both priorities are waiting when a model slot becomes free, interactive
  work is admitted first.
- An auto-balloon job releases the gate between its individual model calls, so
  an interactive request can run without cancelling or reordering the scan-job
  FIFO queue.

The gate is not a second job queue. Auto-balloon ownership, FIFO ordering,
cancellation and result publication remain the responsibility of
`ScanJobManager`.

## Runtime diagnostics

`GET /health` exposes live values under `ocr_capacity.inference_gate`:

- `configured_slots`;
- `active_calls` and `available_slots`;
- `waiting_interactive` and `waiting_background`;
- `peak_active_calls` and `total_acquisitions`; and
- `active_operations`, containing only fixed component names and no drawing
  content.

The slot count comes from the model-bundle capacity constants. Increasing it
without also creating and validating additional independent model bundles is
not supported.
