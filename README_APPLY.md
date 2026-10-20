# QTracX MVP2 — image-source support

Apply this overlay only after the Review/Other candidate-persistence milestone.
It is intended for the `mvp2-image-source-support` branch.

No commit or push was made while preparing this package.

## Included behavior

- Validates drawing files by their bytes instead of trusting only the extension or MIME type.
- Supports PDF, PNG, JPG/JPEG, TIF/TIFF, and the previously accepted WebP format.
- Decodes single- and multi-page TIFF drawings through the existing backend.
- Makes TIFF pages available to page navigation, OCR/auto-ballooning, reference detection, checksheet previews, and drawing exports.
- Keeps the Integration PDF-first and accepted-balloon detection paths unchanged.
- Preserves the original source file in saved projects and checksheet revisions.

## Apply

1. Stop the frontend and backend.
2. Extract the ZIP into the repository root and allow the listed folders to merge.
3. Review the overlay:

   ```powershell
   git status --short
   git diff --check
   ```

4. Restart the backend. TIFF rendering requires the backend to be running.

## Validate

From the repository root:

```powershell
npm run typecheck
npm run lint
npm run test:frontend
npm run build
```

From `doc-ocr\backend`:

```powershell
.\.venv-gpu\Scripts\python.exe -m unittest -v test_source_images.py
```

Manual checks:

1. Open one PNG and one JPG drawing.
2. Open a multi-page TIFF and navigate between its pages.
3. Auto-balloon a TIFF page and confirm ballooning behaves the same as an image/PDF scan.
4. Create a checksheet from a value on TIFF page 2 and open its drawing preview.
5. Export ballooned pages and confirm the relevant TIFF pages are present.
6. Rename a non-image file to `.tif` and confirm the application rejects it clearly.

## Suggested commit message

```text
feat(source): add validated raster and multipage TIFF support
```
