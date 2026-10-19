QTracX MVP2 — Reference persistence and export milestone
=========================================================

Base
----
Apply this ZIP over the repository root on branch:
  mvp2-reference-export

The change set was prepared from commit:
  7205995 feat(export): add branded Excel checksheet with tolerance normalization

What is included
----------------
1. Reference detection runs automatically after successful whole-page and
   selected-section balloon scans. There is no separate Reference Points button.
2. A reference failure does not roll back or discard saved balloon results.
3. Reference tables persist per project and per page in IndexedDB and portable
   .docbox.json project files. Older project files remain loadable.
4. Every reference point emits exactly three rows in X, Y, Z order. A missing
   coordinate remains present with a blank value.
5. JSON, CSV library output, XML, verification payloads, ballooned PDF tables,
   and saved checksheets place reference values after balloon values.
6. Excel keeps balloon rows/images on Checksheet and adds a separate References
   worksheet containing reference rows and reference-only drawing images.
7. Saved checksheets retain reference source-table and marker geometry and show
   reference rows/markers as references rather than numbered balloons.
8. Existing checksheet SQLite databases are migrated in place with nullable
   reference storage columns. Existing balloon rows default to item_kind=balloon.

Apply
-----
1. Stop the frontend and backend.
2. Extract this ZIP into the doc-ocr repository root and allow it to overwrite
   matching files.
3. Review the changes with:
     git status --short
     git diff --check
4. Reinstall dependencies only if your existing environment requires it.

Verification
------------
Frontend (from doc-ocr):
  npm run lint
  npm run typecheck
  npm run test:frontend
  npm run build

Backend focused test (from doc-ocr/backend):
  .\.venv-gpu\Scripts\python.exe -m pytest -q test_checksheet_storage.py

Manual acceptance
-----------------
1. Open a drawing with a coordinate reference table and run Auto Balloon.
2. Confirm references appear automatically after ballooning completes.
3. Change pages, reload the browser, and reopen/save/load the project; confirm
   each page retains its own reference table.
4. Export JSON/XML/PDF and verify A_X, A_Y and A_Z follow the balloon rows.
5. Export Excel and verify:
   - Checksheet contains only ordinary balloon rows and ballooned images.
   - References contains A_X/A_Y/A_Z rows and reference-only images.
6. Create a saved checksheet and select a reference row. Confirm the preview
   highlights its source row and located marker instead of drawing a balloon.

No commit or push is included in this ZIP.
