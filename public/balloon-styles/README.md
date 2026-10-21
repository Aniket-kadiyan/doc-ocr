# Balloon SVG registry

The application loads `manifest.json` and stores only a style id, style version,
and size in projects and checksheets. It never embeds the artwork in saved data.

Each style has one versioned directory and separate `small.svg`, `medium.svg`,
and `large.svg` files. Every SVG must:

- be a complete, self-contained SVG with no scripts or external resources;
- include an empty element carrying `data-balloon-label="true"`;
- match the artwork width and height declared in `manifest.json`.

The application draws the live balloon number over the `numberArea` declared by
the style. `anchor` identifies the point on the SVG that attaches to the top
centre of the detected value box.

To add a style, copy an existing versioned directory, edit its three SVGs, and
add a new entry to `manifest.json`. Increase `version` when an existing style's
geometry changes so saved projects continue to identify the intended design.
Removing a manifest entry is safe: saved projects that reference an unavailable
style fall back to the manifest's `defaultSelection`.
