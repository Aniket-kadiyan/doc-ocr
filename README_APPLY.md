# QTracX MVP2 — durable review-candidate lifecycle

Target branch: `mvp2-review-candidate-persistence`

Expected base commit: `19558ab` (`Merge branch 'mvp2-reference-export' into MVP2`)

## Apply

1. Check out `mvp2-review-candidate-persistence` in the QTracX repository.
2. Confirm `git rev-parse --short HEAD` prints `19558ab`.
3. Extract this ZIP into the repository root and allow matching files to be overwritten.
4. Review with `git status --short` and `git diff --check`.
5. Restart both the frontend and backend. The backend health payload should report `ocr_changeset` as `mvp2-review-candidate-persistence`.

## What changes

- Keeps Integration's detector, OCR, filtering, and accepted-balloon pipeline unchanged.
- Converts the backend's existing candidate outcome ledger into durable `Review`, `Other`, and `Ignored` records.
- Shows filtered detections under **Other detections**, below **Review required**.
- Accept creates a normal balloon; Ignore retains the record; Restore returns it to its previous queue.
- Keeps the selected candidate highlighted when its popup is closed. Accepting, ignoring, selecting another object, or clicking blank drawing space clears the highlight.
- Merges geometry-equivalent rescans and records duplicate source identities/counts.
- Persists candidates in IndexedDB, portable `.docbox.json` projects, and checksheet revision audit data.
- Keeps candidates out of inspection rows and normal accepted-value exports.

## Verification completed

- `npm run lint` — passed.
- `npm run typecheck` — passed.
- `npm run test:frontend` — 24 files, 146 tests passed.
- `npm run build` — passed with Next.js 15.5.20.
- `python -m pytest -q test_checksheet_storage.py` — 7 tests passed.
- `python -m compileall -q backend` — passed.

`test_scan_progress.py` has three failures on both untouched base commit `19558ab` and this package. They are existing Integration fixture mismatches, not regressions from candidate persistence:

- `test_segment_reports_real_stages_and_recognition_counters`
- `test_page_scan_uses_adaptive_panels_then_bounded_recovery_batches`
- `test_page_scan_never_publishes_preliminary_garbage_as_review_text`

## Suggested commit message

`feat(review): persist scan candidates and retain filtered detections`
