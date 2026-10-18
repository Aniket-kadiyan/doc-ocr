# Reference workflow validation

This checklist closes milestone E6 for the coordinate-reference workflow. It
covers automatic detection, project persistence, Excel export, checksheet
storage, and checksheet review without changing their formats or UI.

## Automated coverage

| Requirement | Automated evidence |
|---|---|
| Auto Balloon checks each successfully scanned page once | `referenceScanQueue.test.ts` verifies completed-prefix selection, page deduplication, and sequential execution. |
| A reference failure does not discard balloon results or block later pages | `referenceScanQueue.test.ts` verifies API and local-save failure isolation. The reference queue only starts after `runAutoBalloon` has committed its annotations and scan candidates. |
| A sheet without a coordinate table completes silently | `referenceScanQueue.test.ts` covers both no ruled tables and unrelated ruled tables. |
| A rerun replaces one page's reference result without duplicates | `referenceScanQueue.test.ts` verifies page replacement while preserving other pages. |
| A successful rerun with no coordinate table clears only that page's stale result | `referenceScanQueue.test.ts` verifies page-scoped clearing. |
| Project files preserve page-aware reference tables | `project.test.ts` verifies project round-trip persistence. |
| Every point produces exactly X, Y, and Z rows, including blank missing axes | `referenceValues.test.ts` verifies row order, IDs, multiple pages, blank axes, and latest-result normalization. |
| Excel keeps balloons on `Checksheet` and references on `References` | `excelExport.test.ts` verifies worksheet names, row values, continuous numbering, separate images, and captions. |
| Checksheet creation appends reference rows after balloon rows | `checksheetSnapshot.test.ts` verifies numbering, labels, values, and source geometry. |
| Backend storage preserves reference rows, readings, and duplicates | `backend/test_checksheet_storage.py` verifies storage, reading updates, database columns, and checksheet duplication. |
| Review shows missing axes and focuses the table row, name cell, and markers | `checksheetReference.test.ts` verifies display values and combined focus geometry. |
| Backend table/API contracts retain unmatched points and scale geometry | `backend/test_reference_table.py` and `backend/test_reference_points_api.py` verify table recognition, row handling, unmatched symbols, and coordinate scaling. |

## Automated commands

Run from the repository root in PowerShell:

```powershell
npm run typecheck
npm run lint
npm run test:frontend
npm run build
```

Run the reference backend tests from `doc-ocr\backend` in the active CPU or
GPU environment:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest -q `
  test_reference_table.py `
  test_reference_points_api.py `
  test_checksheet_storage.py
```

Use `.\.venv\Scripts\python.exe` instead when validating the CPU-only
environment.

## Manual acceptance test

Use one drawing with a valid coordinate-reference table, one drawing without
such a table, and—if available—a multi-page drawing.

1. Run whole-page Auto Balloon on the drawing with references.
2. Confirm balloon results appear first, followed by the
   `Reading reference points…` status.
3. Confirm there is no separate Reference Points button.
4. Confirm the sidebar lists the detected points and selecting a point
   highlights its table row, name cell, markers, and coordinate tags.
5. Select multiple scan sections on the same page and confirm only one
   reference pass follows the completed section queue.
6. Save the project, reopen it, and confirm the same references return.
7. Rerun Auto Balloon and confirm reference rows are replaced rather than
   duplicated.
8. Export Excel and confirm:
   - `Checksheet` contains only balloon rows and balloon images;
   - `References` contains the point X/Y/Z rows and reference-only images;
   - numbering on `References` continues after the last balloon number;
   - a missing axis remains a blank value.
9. Create a checksheet and confirm reference rows follow balloon rows.
10. Open checksheet review and confirm a blank reference axis reads
    `Not detected` and selecting any reference row focuses its saved geometry.
11. Run Auto Balloon on the drawing without a coordinate table and confirm it
    finishes normally without a reference warning or reference rows.
12. To verify failure isolation, stop the backend after balloon detection has
    completed but before the reference request finishes. Confirm the balloons
    remain and the UI reports that reference detection failed.

## Completion rule

E6 is complete when the automated commands pass and the manual acceptance test
passes on the deployment workstation. Any failure should be corrected on
`exporting_refrences` before it is merged into `MVP1`.
