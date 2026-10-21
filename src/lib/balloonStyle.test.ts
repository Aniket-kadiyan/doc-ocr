import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  fitBalloonFontSize,
  getAnchoredBalloonGeometry,
  normalizeBalloonStyleSelection,
  parseBalloonStyleManifest,
  resolveBalloonStyle,
  validateBalloonSvg,
} from "@/lib/balloonStyle";
import {
  BALLOON_SIZES,
  BALLOON_STYLE_SCHEMA_VERSION,
} from "@/types/balloonStyle";

const manifestPath = fileURLToPath(
  new URL("../../public/balloon-styles/manifest.json", import.meta.url)
);

const loadManifest = () =>
  parseBalloonStyleManifest(JSON.parse(readFileSync(manifestPath, "utf8")));

describe("balloon style registry", () => {
  it("loads four versioned SVG styles with three sizes each", () => {
    const manifest = loadManifest();

    expect(manifest.schemaVersion).toBe(BALLOON_STYLE_SCHEMA_VERSION);
    expect(manifest.styles.map((style) => style.id)).toEqual([
      "circle",
      "square",
      "hexagon",
      "rounded-rectangle",
    ]);
    for (const definition of manifest.styles) {
      for (const size of BALLOON_SIZES) {
        const relative = definition.sizes[size].artwork.path.replace(/^\//, "");
        const path = fileURLToPath(new URL(`../../public/${relative}`, import.meta.url));
        expect(existsSync(path)).toBe(true);
        expect(validateBalloonSvg(readFileSync(path, "utf8"))).toContain(
          "data-balloon-label"
        );
      }
    }
  });

  it("rejects active SVG content and falls back from unavailable selections", () => {
    expect(() =>
      validateBalloonSvg(
        '<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script><text data-balloon-label="true"></text></svg>'
      )
    ).toThrow(/prohibited active content/);

    const manifest = loadManifest();
    expect(
      normalizeBalloonStyleSelection(
        { styleId: "removed-style", styleVersion: 4, size: "large" },
        manifest
      )
    ).toEqual(manifest.defaultSelection);
  });
});

describe("balloon geometry", () => {
  const manifest = loadManifest();
  const svg = readFileSync(
    fileURLToPath(
      new URL("../../public/balloon-styles/circle/v1/medium.svg", import.meta.url)
    ),
    "utf8"
  );
  const style = resolveBalloonStyle(manifest, manifest.defaultSelection, svg);

  it("keeps screen size constant and the pointer anchored across zoom levels", () => {
    const anchorX = 120;
    const anchorY = 80;
    const normal = getAnchoredBalloonGeometry(style, anchorX, anchorY, 1);
    const zoomed = getAnchoredBalloonGeometry(style, anchorX, anchorY, 2);

    expect(normal.width).toBeCloseTo(32);
    expect(zoomed.width * 2).toBeCloseTo(32);
    expect(normal.x + style.anchor.x * normal.width).toBeCloseTo(anchorX);
    expect(normal.y + style.anchor.y * normal.height).toBeCloseTo(anchorY);
    expect(zoomed.x + style.anchor.x * zoomed.width).toBeCloseTo(anchorX);
    expect(zoomed.y + style.anchor.y * zoomed.height).toBeCloseTo(anchorY);
  });

  it("shrinks longer balloon numbers without going below the configured minimum", () => {
    const one = fitBalloonFontSize("1", 20, 27, 32, style);
    const threeDigits = fitBalloonFontSize("128", 20, 27, 32, style);

    expect(threeDigits).toBeLessThan(one);
    expect(threeDigits).toBeGreaterThanOrEqual(style.text.minFontSize);
  });
});
