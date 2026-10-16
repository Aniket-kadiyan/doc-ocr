# Reference-point detection and review

The reference-point pass reads a ruled coordinate table, rebuilds its named
rows, and locates matching point-name glyphs elsewhere on the drawing. The UI
shows the reconstructed table and places selectable markers on the drawing.

Reference points remain separate from the dimension pipeline:

- they do not become balloons or scan candidates;
- they do not receive tolerances, methods, or tools;
- one result per page is saved in the local project and portable project file;
- Excel exports append X/Y/Z values after balloons and include a separate
  reference-only drawing image; other inspection exports remain unchanged;
- checksheet snapshots append X/Y/Z rows after the balloon rows and retain the
  point name, axis, coordinate row, name cell, and every detected marker;
- a point read from the table but not located in a view is returned in
  `unmatchedSymbols` instead of being discarded.

Each saved point has a shared export-neutral representation with exactly three
rows in X, Y, Z order. Missing axes remain explicit blank values rather than
being discarded. Checksheet preview and auto-balloon integration are handled by
later milestones.

## Endpoint

`POST /ocr/reference-points` accepts multipart form data:

| Field | Meaning |
|---|---|
| `file` | A high-resolution rendered image of one complete drawing page |
| `display_scale` | Submitted image pixels per annotation-space pixel |

The backend divides all returned boxes by `display_scale`, so the next UI
milestone can place them directly in the drawing's existing coordinate space.

The response distinguishes three cases:

- `found: false`, `tableCount: 0`: no ruled table was found;
- `found: false`, `tableCount > 0`: tables were found, but none had a point
  column followed by at least two X/Y/Z coordinate columns;
- `found: true`: the response contains the table geometry, axes, every valid
  point row, its coordinates, zero or more located markers, and an explicit
  list of unmatched point names.

Detection is bounded to six likely table candidates, considered largest-first.
Marker matching uses glyphs cut from the drawing's own point-name cells and
uses coordinate-cell digits as negative evidence. This avoids treating lone
letters in normal notes, headings, dimensions, or other tables as points.
