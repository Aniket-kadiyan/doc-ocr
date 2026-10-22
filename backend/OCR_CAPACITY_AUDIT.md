# OCR capacity audit

This note records the backend capacity that QTracX can actually schedule at
this milestone.

## Current architecture

- Each backend process creates one shared `OcrPipeline` coordinator and a
  configurable pool of independently loaded model bundles.
- Each bundle can contain the main PaddleOCR pipeline, a standalone text
  detector, a text recognizer and a text-line orientation model. These are
  cooperating components inside one bundle; they are not four independent job
  slots.
- `OCR_MODEL_INSTANCES` requests 1-8 complete bundles and defaults to one.
- Additional bundles join the pool only when they load successfully with the
  same component capabilities, API version and device as the first bundle.
- The auto-balloon executor is created with one worker per successfully loaded
  bundle. A failed additional load reduces real scheduler capacity instead of
  advertising an unusable worker.
- Paddle/Paddle Inference may use internal compute threads. Those threads can
  accelerate one inference call but do not create additional QTracX job slots.
- Every Paddle inference call passes through one process-wide inference gate.
  Its slots equal the number of successfully loaded bundles, and every slot
  leases one concrete bundle exclusively for the call.
- Direct region OCR, title-field reads, reference detection and OCR-assisted
  PDF text recovery use interactive priority. Auto-balloon jobs use background
  priority. Image preparation and result parsing remain outside the gate.

## Runtime evidence

`GET /health` now includes `ocr_capacity` with:

- the number of configured, created and inference-ready model bundles;
- the loaded components within the bundle;
- the scan executor's worker count;
- running, queued and active scan counts;
- currently available scan-worker slots; and
- the resulting effective concurrent auto-balloon slots; and
- live inference-gate counts for active calls, available slots and waiting
  interactive/background work.

The same capacity snapshot is written to the backend log during startup under
the `ocr_capacity_detected` event.

## Scheduler implication

The scheduler uses loaded model-bundle slots as its limiting resource. Adding
HTTP workers or executor threads alone does not increase OCR concurrency.
Changing `OCR_MODEL_INSTANCES` requires a backend restart because model loading
and scan-executor sizing happen together during startup.
