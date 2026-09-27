import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  fitBalloonFontSize,
  getAnchoredBalloonGeometry,
  parseBalloonStyle,
} from "@/lib/balloonStyle";
import { BALLOON_STYLE_SCHEMA_VERSION } from "@/types/balloonStyle";

function validStyle() {
  return {
    schemaVersion: BALLOON_STYLE_SCHEMA_VERSION,
    name: "Test balloon",
    artwork: {
      mimeType: "image/png",
      dataUrl: "data:image/png;base64,iVBORw0KGgo=",
      width: 40,
      height: 60,
    },
    numberArea: { x: 0.25, y: 0.05, width: 0.5, height: 0.45 },
    anchor: { x: 0.5, y: 0.98 },
    display: { drawingWidth: 40, sidebarWidth: 22 },
    text: {
      color: "#111111",
      fontFamily: "Arial, sans-serif",
      fontWeight: "bold",
      minFontSize: 7,
      maxFontSize: 13,
      padding: 2,
    },
  };
}

describe("balloon style validation", () => {
  it("accepts the deployed developer-replaceable style file", () => {
    const path = fileURLToPath(
      new URL("../../public/balloon-style.json", import.meta.url)
    );
    const parsed = parseBalloonStyle(JSON.parse(readFileSync(path, "utf8")));

    expect(parsed.schemaVersion).toBe(BALLOON_STYLE_SCHEMA_VERSION);
    expect(parsed.artwork.mimeType).toBe("image/png");
  });

  it("rejects unsafe artwork and out-of-bounds number geometry", () => {
    const badArtwork = validStyle();
    badArtwork.artwork.dataUrl = "not-a-data-url";
    expect(() => parseBalloonStyle(badArtwork)).toThrow(/base64 PNG/);

    const badArea = validStyle();
    badArea.numberArea = { x: 0.8, y: 0.05, width: 0.5, height: 0.45 };
    expect(() => parseBalloonStyle(badArea)).toThrow(/artwork bounds/);
  });
});

describe("balloon geometry", () => {
  it("keeps screen size constant and the pointer anchored across zoom levels", () => {
    const style = parseBalloonStyle(validStyle());
    const anchorX = 120;
    const anchorY = 80;
    const normal = getAnchoredBalloonGeometry(style, anchorX, anchorY, 1);
    const zoomed = getAnchoredBalloonGeometry(style, anchorX, anchorY, 2);

    expect(normal.width).toBeCloseTo(40);
    expect(zoomed.width * 2).toBeCloseTo(40);
    expect(normal.x + style.anchor.x * normal.width).toBeCloseTo(anchorX);
    expect(normal.y + style.anchor.y * normal.height).toBeCloseTo(anchorY);
    expect(zoomed.x + style.anchor.x * zoomed.width).toBeCloseTo(anchorX);
    expect(zoomed.y + style.anchor.y * zoomed.height).toBeCloseTo(anchorY);
  });

  it("shrinks longer balloon numbers without going below the configured minimum", () => {
    const style = parseBalloonStyle(validStyle());
    const one = fitBalloonFontSize("1", 20, 27, 40, style);
    const threeDigits = fitBalloonFontSize("128", 20, 27, 40, style);

    expect(threeDigits).toBeLessThan(one);
    expect(threeDigits).toBeGreaterThanOrEqual(style.text.minFontSize);
  });
});
