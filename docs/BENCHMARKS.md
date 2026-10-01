# Real-drawing benchmark

The benchmark turns the supplied drawings and known failures into repeatable
evidence. It measures recognition, classification, final candidate disposition,
and whether every detected object reaches one explicit final state. It does not
change production OCR, filtering, review UI, persistence, or export behaviour.

## Corpus

| Fixture | Profile | Why it is included |
|---|---|---|
| `47630` | Dimensional | Dual units, limit dimensions, references, and known symbol/type errors |
| `56103-0182B` | Dimensional | Threads, chamfers, angles, repeated values, and characteristic tags |
| `BS1801006.020` | Dimensional | Asymmetric tolerances, diameters, radii, and angle tolerances |
| `129E01-13300-ID` | Dimensional | The supplied failed run: raster source, rotated values, GD&T, notes, BOM content, and existing item callouts |
| `503645_DES001_CF_BALLOON_DRAWING` | Full inspection | Professionally ballooned reference with single-sided tolerances and a gear/material specification table |
| `ballwoon_drawing` | Dimensional | High-resolution raster with vertical/rotated values, GD&T, references, notes, BOM content, and assembly callouts |

Each JSON fixture lives in `backend/benchmarks/documents/`; its source drawing
is in `docs/`. `coverage: curated` means the listed objects are a deliberate
representative set, not a claim that every printable string has been transcribed.

## Profiles and final dispositions

The profile is ground-truth policy recorded by the fixture. It is not yet a
runtime UI switch.

| Profile | Accepted inspection content | Content expected as `other` |
|---|---|---|
| `dimensional` | Dimensions, tolerances, GD&T, and explicitly identified reference dimensions | Notes, title/revision data, BOM/material rows, specification prose, stamps, and existing circled item markers |
| `full_inspection` | Dimensional content plus explicitly curated material and specification-table requirements | Administrative/title content and existing callout numbers |

Every scored candidate uses one of three benchmark dispositions:

- `accepted`: safe to create as an automatic balloon.
- `review`: retained for a user decision because recognition or meaning is
  unresolved.
- `other`: retained as an auditable non-balloon detection. A later milestone
  will persist and show these below reviews; this benchmark does not add that UI.

The target export rule remains accepted balloons only. Review and other
candidates must never enter an export merely because they were detected.

## Accounting contract

For every page result, the benchmark verifies:

```text
detected = accepted + review + other
recognized = detected - unread
```

`unread` is a diagnostic subset, not a fourth final state. Objects protected by
an already-present balloon are reported separately as `skipped_existing`; they
are outside the detected total. The accounting report also checks collection
counts, state names, candidate identity uniqueness, usable geometry, links
between published candidates and outcomes, and reasons for review/exclusion.

Legacy `*.regions.json` files contain accepted regions only. They remain
scorable for text, but candidate accounting is correctly reported as
unavailable. A schema-v2 `*.scan.json` snapshot is required to prove lifecycle
accounting.

## Fixture format

The smallest useful fixture is:

```json
{
  "schema_version": 2,
  "document": "docs/example.pdf",
  "title": "Example drawing",
  "route": "page",
  "profile": "dimensional",
  "coverage": "curated",
  "dpi": 250,
  "requires": {
    "accounting_balanced": true,
    "max_accounting_errors": 0
  },
  "expected": [
    {
      "id": 1,
      "accept": ["R0.3+0.2"],
      "type": "radius",
      "disposition": "accepted",
      "page": 1,
      "bbox": {"x": 0.25, "y": 0.4, "width": 0.05, "height": 0.02}
    }
  ]
}
```

`accept` may contain multiple genuinely equivalent renderings, but should not
whitelist an OCR error. `bbox` is optional normalized page geometry and is used
only as a tie-breaker for repeated text; text remains the primary match. The
available accuracy gates are `matched`, `mean_percent`, `type_correct`,
`semantic_exact`, `disposition_correct`, `accepted_matched`,
`accepted_semantic_exact`, `accepted_disposition_correct`, `review_matched`,
`review_disposition_correct`, `other_matched`, `other_disposition_correct`,
`max_unexpected`, and `max_seconds`. Per-disposition totals keep recognition of
true inspection characteristics separate from auditing non-balloon detections.

## Running the benchmark

Fast, model-free fixture/scoring/accounting tests run in the normal suite:

```bash
npm run test:backend
```

Run one real drawing and save its complete lifecycle snapshot:

```bash
backend/.venv/bin/python backend/scripts/run_benchmark.py \
  129E01-13300-ID --save-snapshot
```

Windows:

```bat
backend\.venv\Scripts\python backend\scripts\run_benchmark.py ^
  129E01-13300-ID --save-snapshot
```

With no fixture name, the runner processes all drawings. This loads OCR models
and may take several minutes per drawing. `route: page` is the production
whole-page Auto-Balloon path and is the default. `legacy_segment` exists only
to reproduce old measurements; do not compare the two routes as if they were
the same pipeline.

Re-score a saved snapshot without loading OCR:

```bash
backend/.venv/bin/python backend/scripts/run_benchmark.py \
  129E01-13300-ID \
  --regions backend/benchmarks/results/129E01-13300-ID.scan.json
```

Useful options:

- `--json` emits a machine-readable report.
- `--dpi 400` or `--dpi 600` measures source-fidelity sensitivity without a
  production architecture change.
- `--save-regions` writes a compatibility accepted/review/other flat list;
  `--save-snapshot` is the authoritative format.
- `--ignore-gates` is for inspecting historical accepted-only results whose
  accounting is necessarily unavailable.

For pytest-based end-to-end execution, set `RUN_DOC_BENCHMARKS=1` and run
`backend/test_benchmark_documents.py` in the configured OCR environment.

## Baselines and calibration

The three original fixtures retain their measured recognition gates. Their old
saved files are accepted-only and therefore cannot satisfy accounting gates;
record new schema-v2 snapshots before treating them as lifecycle baselines.

| Drawing/result | Route | Recorded result | Limitation |
|---|---|---|---|
| `47630.regions.json` | Historical saved file | 1/19 matched, 20.5% mean, 1/19 type-correct, 11 unexpected accepted | Accepted regions only; this file does not reproduce the separate 16/19 baseline recorded in the fixture notes |
| `56103-0182B.regions.json` | Historical | 15/24 matched, 80.8% mean, 18/24 type-correct | Accepted regions only |
| `BS1801006.020.regions.json` | Legacy segment | 12/12 matched | Different route; accepted regions only |
| `129E01-13300-ID.pre-m1.regions.json` | Supplied saved project | 24/34 matched, 8 partial, 2 missed, 87.4% mean; 22 semantic-exact; 28/34 disposition-correct; 5 unexpected accepted | 37 items were all auto-accepted, so review/other accounting is unavailable |

The three newly supplied drawings initially gate only the accounting contract.
After running them on the validated OCR machine, review the overlay and report,
correct ground truth if needed, store the `*.scan.json`, and then add accuracy
thresholds at the observed baseline. This avoids inventing thresholds or
freezing current OCR mistakes into the expected text.

When changing DPI, models, filtering, or recognition, compare the same fixture,
profile, route, and ground truth. Report accuracy, disposition, accounting, and
runtime together; a larger balloon count alone is not an improvement.
