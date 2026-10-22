# Configurable OCR model pool

## Configuration

`OCR_MODEL_INSTANCES` requests the number of complete Paddle model bundles to
load in one QTracX backend process. The valid range is 1-8 and the default is 1.

```dotenv
OCR_MODEL_INSTANCES=1
```

One bundle contains the main PaddleOCR pipeline plus the standalone detector,
recognizer and text-line orientation models used by auto-ballooning. Therefore
two instances consume approximately twice the model memory; two worker threads
without two model bundles do not provide two safe inference slots.

Keep the value at `1` on the current 4 GB RTX 2050. On different deployment
hardware, increase it one step at a time only after measuring GPU/CPU memory,
scan throughput and interactive response time. Restart the backend after every
change. The setting is not hot-reloaded.

## Startup validation

The first bundle establishes the required Paddle version, API, effective
device and component availability. Each additional bundle must match it before
joining the pool. Loading stops after the first failed or inconsistent bundle,
which avoids repeatedly attempting allocation after a likely memory failure.

The backend then:

1. creates one inference-gate slot per successfully loaded bundle;
2. creates one auto-balloon worker per successfully loaded bundle;
3. leases one exclusive bundle for every Paddle call; and
4. reports requested, loaded and failed counts through `/health`.

If Paddle itself is unavailable, the pool reports zero inference capacity. A
single scan worker remains only so submitted jobs can return a real failure
instead of waiting forever.

## Runtime verification

```powershell
$health = Invoke-RestMethod http://127.0.0.1:8000/health
$health.ocr_capacity.model_bundles | ConvertTo-Json -Depth 5
$health.ocr_capacity.inference_gate | ConvertTo-Json -Depth 5
$health.ocr_capacity.auto_balloon_scans | ConvertTo-Json -Depth 5
```

Confirm that `created`, `inference_ready`, `configured_slots` and
`worker_threads` equal the number that actually loaded. `configured` records
the requested value and `failed` records any shortfall.

This remains a single-process design. Running multiple Uvicorn worker processes
would create separate pools and queues and is not supported by this milestone.
