# OCR capacity audit

This note records the backend capacity that QTracX can actually schedule at
this milestone. It does not describe the planned multi-user scheduler.

## Current architecture

- Each backend process creates one shared `OcrPipeline` singleton.
- That bundle can contain the main PaddleOCR pipeline, a standalone text
  detector, a text recognizer and a text-line orientation model. These are
  cooperating components inside one bundle; they are not four independent job
  slots.
- Auto-balloon scans run through a `ThreadPoolExecutor` with one worker.
- Therefore the current effective auto-balloon capacity is one scan at a time
  when PaddleOCR loaded successfully, and zero when it did not.
- Paddle/Paddle Inference may use internal compute threads. Those threads can
  accelerate one inference call but do not create additional QTracX job slots.
- Direct OCR endpoints share the same singleton but do not currently pass
  through the auto-balloon scan executor. A later scheduler must coordinate
  all model-using routes rather than only whole-page scans.

## Runtime evidence

`GET /health` now includes `ocr_capacity` with:

- the number of configured, created and inference-ready model bundles;
- the loaded components within the bundle;
- the scan executor's worker count;
- running, queued and active scan counts;
- currently available scan-worker slots; and
- the resulting effective concurrent auto-balloon slots.

The same capacity snapshot is written to the backend log during startup under
the `ocr_capacity_detected` event. This is diagnostic-only: it does not change
model loading, OCR results, job ordering, cancellation, or scan concurrency.

## Scheduler implication

The multi-user scheduler should use model-bundle slots as its limiting
resource. Adding HTTP workers or executor threads alone must not increase OCR
concurrency. A future configurable pool must first create and validate the
requested number of model bundles, then expose that verified count as capacity.
