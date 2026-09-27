# Developer balloon style

Balloon appearance is controlled only by the deployed
`public/balloon-style.json`. There is no user-facing upload, selector, or
per-balloon style field. Saved annotations contain values, numbers, positions,
and metadata, but no artwork.

## Runtime contract

The single JSON file embeds a transparent PNG and the geometry required to
overlay a dynamic number.

| Field | Purpose |
|---|---|
| `schemaVersion` | Compatibility check for the renderer/builder |
| `artwork` | Embedded PNG data URL and declared pixel dimensions |
| `numberArea` | Normalized rectangle for the dynamic number |
| `anchor` | Normalized pointer-tip position placed on the value box |
| `display` | Drawing and sidebar widths in screen pixels |
| `text` | Colour, font, fitting limits, and padding |

The app validates the entire file and then verifies that the decoded image
dimensions match the declarations. Missing JSON, invalid JSON, corrupt artwork,
or incompatible geometry logs an error and leaves the built-in orange-pointer
fallback active.

## Graphical builder

From the repository root on Windows:

```bat
backend\.venv\Scripts\python.exe tools\balloon_builder\balloon_builder.py
```

macOS/Linux:

```bash
backend/.venv/bin/python tools/balloon_builder/balloon_builder.py
```

The Python installation must include Pillow and Tkinter. The validated workflow
is:

1. Open PNG, JPEG, WebP, BMP, or TIFF artwork.
2. Remove only the edge-connected external background.
3. Crop transparent margins.
4. Clear any printed sample number without clearing the enclosed white area.
5. Mark the complete rectangle available for the dynamic number.
6. Click the exact pointer tip.
7. Preview `1`, `35`, and `128` and adjust text/display settings if needed.
8. Export one `balloon-style.json`.
9. Use **Open style** to reopen and validate the exported file.

## Command-line validation and build

Validate an existing file:

```bash
backend/.venv/bin/python tools/balloon_builder/balloon_builder.py \
  --validate public/balloon-style.json
```

On Windows, replace the Python path with
`backend\.venv\Scripts\python.exe` and place the command on one line.

Prepared transparent artwork can also be built without the GUI:

```bash
backend/.venv/bin/python tools/balloon_builder/balloon_builder.py \
  --build prepared-balloon.png \
  --output balloon-style.json \
  --number-area 4,4,32,30 \
  --anchor 20,59
```

The number-area and anchor arguments are source-artwork pixel coordinates.
Optional CLI flags control display widths, text settings, edge-background
removal, and transparent trimming; run with `--help` for the complete list.

## Deployment replacement

1. Validate the candidate JSON.
2. Back up the current `public/balloon-style.json`.
3. Replace only that file with the candidate, retaining the same filename.
4. Refresh the application. The loader requests the style without browser
   caching.
5. Check drawing balloons, sidebar thumbnails, existing annotations, a newly
   created annotation, and a loaded saved project.
6. Confirm exports and annotation data are unchanged.

Restoring the backed-up JSON and refreshing restores the previous appearance.
No backend, database, annotation migration, or project-file conversion is
required.
