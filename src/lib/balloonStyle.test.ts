import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  BALLOON_OK_BOUNDARY_COLOR,
  BALLOON_REVIEW_BOUNDARY_COLOR,
  colorBalloonBoundary,
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
        expect(readFileSync(path, "utf8")).toContain(
          "data-balloon-connector"
        );
        expect(readFileSync(path, "utf8")).toContain(
          "data-balloon-boundary"
        );
      }
      expect(definition.sizes.small.display).toEqual({
        drawingWidth: 14,
        sidebarWidth: 10,
      });
      expect(definition.sizes.medium.display).toEqual({
        drawingWidth: 18,
        sidebarWidth: 14,
      });
      expect(definition.sizes.large.display).toEqual({
        drawingWidth: 24,
        sidebarWidth: 18,
      });
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

  it("colours and strengthens the boundary and connector", () => {
    const svg = readFileSync(
      fileURLToPath(
        new URL(
          "../../public/balloon-styles/circle/v1/medium.svg",
          import.meta.url
        )
      ),
      "utf8"
    );
    const accepted = colorBalloonBoundary(svg, BALLOON_OK_BOUNDARY_COLOR);
    const review = colorBalloonBoundary(svg, BALLOON_REVIEW_BOUNDARY_COLOR);
    const acceptedConnector = accepted.match(
      /<[^>]*data-balloon-connector="true"[^>]*>/
    )?.[0];
    const acceptedBoundary = accepted.match(
      /<[^>]*data-balloon-boundary="true"[^>]*>/
    )?.[0];
    const reviewConnector = review.match(
      /<[^>]*data-balloon-connector="true"[^>]*>/
    )?.[0];
    const reviewBoundary = review.match(
      /<[^>]*data-balloon-boundary="true"[^>]*>/
    )?.[0];

    expect(acceptedConnector).toContain(
      `stroke="${BALLOON_OK_BOUNDARY_COLOR}"`
    );
    expect(acceptedConnector).toContain('stroke-width="12.5"');
    expect(acceptedBoundary).toContain(
      `stroke="${BALLOON_OK_BOUNDARY_COLOR}"`
    );
    expect(acceptedBoundary).toContain('stroke-width="17.5"');
    expect(reviewConnector).toContain(
      `stroke="${BALLOON_REVIEW_BOUNDARY_COLOR}"`
    );
    expect(reviewBoundary).toContain(
      `stroke="${BALLOON_REVIEW_BOUNDARY_COLOR}"`
    );
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

    expect(normal.width).toBeCloseTo(18);
    expect(zoomed.width * 2).toBeCloseTo(18);
    expect(normal.x + style.anchor.x * normal.width).toBeCloseTo(anchorX);
    expect(normal.y + style.anchor.y * normal.height).toBeCloseTo(anchorY);
    expect(zoomed.x + style.anchor.x * zoomed.width).toBeCloseTo(anchorX);
    expect(zoomed.y + style.anchor.y * zoomed.height).toBeCloseTo(anchorY);
  });

  it("shrinks longer balloon numbers without going below the configured minimum", () => {
    const one = fitBalloonFontSize("1", 20, 27, 18, style);
    const threeDigits = fitBalloonFontSize("128", 20, 27, 18, style);

    expect(threeDigits).toBeLessThan(one);
    expect(threeDigits).toBeGreaterThanOrEqual(style.text.minFontSize);
  });
});
