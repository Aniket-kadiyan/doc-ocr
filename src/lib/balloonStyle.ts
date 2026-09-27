import {
  BALLOON_STYLE_SCHEMA_VERSION,
  type BalloonGeometry,
  type BalloonStyle,
} from "@/types/balloonStyle";

const FALLBACK_SVG = `
<svg xmlns="http://www.w3.org/2000/svg" width="40" height="60" viewBox="0 0 40 60">
  <path d="M20 59 L7.5 34 A18 18 0 1 1 32.5 34 Z" fill="#f97316" stroke="#111827" stroke-width="2" stroke-linejoin="round"/>
  <circle cx="20" cy="20" r="17.5" fill="#ffffff" stroke="#111827" stroke-width="2"/>
</svg>`;

/** Always-available marker used while the deployed file loads or if it fails. */
export const FALLBACK_BALLOON_STYLE: BalloonStyle = {
  schemaVersion: BALLOON_STYLE_SCHEMA_VERSION,
  name: "Built-in orange pointer fallback",
  artwork: {
    mimeType: "image/svg+xml",
    dataUrl: `data:image/svg+xml;charset=utf-8,${encodeURIComponent(
      FALLBACK_SVG.trim()
    )}`,
    width: 40,
    height: 60,
  },
  numberArea: { x: 0.1, y: 0.04, width: 0.8, height: 0.55 },
  anchor: { x: 0.5, y: 0.985 },
  display: { drawingWidth: 32, sidebarWidth: 22 },
  text: {
    color: "#111827",
    fontFamily: "Arial, sans-serif",
    fontWeight: "bold",
    minFontSize: 7,
    maxFontSize: 13,
    padding: 2,
  },
};

type JsonRecord = Record<string, unknown>;

const isRecord = (value: unknown): value is JsonRecord =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const readRecord = (value: unknown, path: string): JsonRecord => {
  if (!isRecord(value)) throw new Error(`${path} must be an object.`);
  return value;
};

const readString = (value: unknown, path: string): string => {
  if (typeof value !== "string" || !value.trim()) {
    throw new Error(`${path} must be a non-empty string.`);
  }
  return value.trim();
};

const readNumber = (
  value: unknown,
  path: string,
  minimum: number,
  maximum: number
): number => {
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    value < minimum ||
    value > maximum
  ) {
    throw new Error(`${path} must be between ${minimum} and ${maximum}.`);
  }
  return value;
};

const readNormalizedRect = (value: unknown) => {
  const rect = readRecord(value, "numberArea");
  const x = readNumber(rect.x, "numberArea.x", 0, 1);
  const y = readNumber(rect.y, "numberArea.y", 0, 1);
  const width = readNumber(rect.width, "numberArea.width", 0.001, 1);
  const height = readNumber(rect.height, "numberArea.height", 0.001, 1);
  if (x + width > 1.000001 || y + height > 1.000001) {
    throw new Error("numberArea must stay within the artwork bounds.");
  }
  return { x, y, width, height };
};

/** Parse and validate the developer-replaceable public style file. */
export function parseBalloonStyle(input: unknown): BalloonStyle {
  const root = readRecord(input, "balloon style");
  if (root.schemaVersion !== BALLOON_STYLE_SCHEMA_VERSION) {
    throw new Error(
      `schemaVersion must be ${BALLOON_STYLE_SCHEMA_VERSION}.`
    );
  }

  const artwork = readRecord(root.artwork, "artwork");
  if (artwork.mimeType !== "image/png") {
    throw new Error("artwork.mimeType must be image/png.");
  }
  const dataUrl = readString(artwork.dataUrl, "artwork.dataUrl");
  if (!/^data:image\/png;base64,[A-Za-z0-9+/=\r\n]+$/.test(dataUrl)) {
    throw new Error("artwork.dataUrl must contain a base64 PNG data URL.");
  }
  // Keep accidental multi-megabyte source images out of the browser bundle.
  if (dataUrl.length > 3_000_000) {
    throw new Error("artwork.dataUrl is larger than the 3 MB safety limit.");
  }

  const width = readNumber(artwork.width, "artwork.width", 1, 4096);
  const height = readNumber(artwork.height, "artwork.height", 1, 4096);
  if (!Number.isInteger(width) || !Number.isInteger(height)) {
    throw new Error("artwork width and height must be whole pixels.");
  }

  const anchor = readRecord(root.anchor, "anchor");
  const display = readRecord(root.display, "display");
  const text = readRecord(root.text, "text");
  const minFontSize = readNumber(text.minFontSize, "text.minFontSize", 4, 48);
  const maxFontSize = readNumber(text.maxFontSize, "text.maxFontSize", 4, 72);
  if (maxFontSize < minFontSize) {
    throw new Error("text.maxFontSize must be at least text.minFontSize.");
  }

  const color = readString(text.color, "text.color");
  if (!/^#(?:[0-9a-f]{3}|[0-9a-f]{4}|[0-9a-f]{6}|[0-9a-f]{8})$/i.test(color)) {
    throw new Error("text.color must be a hexadecimal CSS colour.");
  }

  const fontWeight = text.fontWeight;
  if (fontWeight !== "normal" && fontWeight !== "bold") {
    throw new Error('text.fontWeight must be "normal" or "bold".');
  }

  return {
    schemaVersion: BALLOON_STYLE_SCHEMA_VERSION,
    name: readString(root.name, "name"),
    artwork: {
      mimeType: "image/png",
      dataUrl,
      width,
      height,
    },
    numberArea: readNormalizedRect(root.numberArea),
    anchor: {
      x: readNumber(anchor.x, "anchor.x", 0, 1),
      y: readNumber(anchor.y, "anchor.y", 0, 1),
    },
    display: {
      drawingWidth: readNumber(
        display.drawingWidth,
        "display.drawingWidth",
        12,
        160
      ),
      sidebarWidth: readNumber(
        display.sidebarWidth,
        "display.sidebarWidth",
        12,
        80
      ),
    },
    text: {
      color,
      fontFamily: readString(text.fontFamily, "text.fontFamily").slice(0, 120),
      fontWeight,
      minFontSize,
      maxFontSize,
      padding: readNumber(text.padding, "text.padding", 0, 24),
    },
  };
}

export function getBalloonScreenSize(
  style: BalloonStyle,
  target: "drawing" | "sidebar"
) {
  const width =
    target === "drawing"
      ? style.display.drawingWidth
      : style.display.sidebarWidth;
  return {
    width,
    height: width * (style.artwork.height / style.artwork.width),
  };
}

/** Position the configured pointer tip exactly on the supplied drawing point. */
export function getAnchoredBalloonGeometry(
  style: BalloonStyle,
  anchorX: number,
  anchorY: number,
  scale: number
): BalloonGeometry {
  const screen = getBalloonScreenSize(style, "drawing");
  const width = screen.width / scale;
  const height = screen.height / scale;
  const x = anchorX - style.anchor.x * width;
  const y = anchorY - style.anchor.y * height;
  return {
    x,
    y,
    width,
    height,
    numberX: x + style.numberArea.x * width,
    numberY: y + style.numberArea.y * height,
    numberWidth: style.numberArea.width * width,
    numberHeight: style.numberArea.height * height,
  };
}

/** Fit typical balloon numbers without storing a font size per annotation. */
export function fitBalloonFontSize(
  value: string,
  areaWidth: number,
  areaHeight: number,
  displayWidth: number,
  style: BalloonStyle
): number {
  const relativeScale = displayWidth / style.display.drawingWidth;
  const minimum = style.text.minFontSize * relativeScale;
  const maximum = style.text.maxFontSize * relativeScale;
  const padding = style.text.padding * relativeScale;
  const availableWidth = Math.max(1, areaWidth - padding * 2);
  const availableHeight = Math.max(1, areaHeight - padding * 2);
  const glyphUnits = [...value].reduce((total, character) => {
    if (character === "1") return total + 0.5;
    if (character === "-" || character === ".") return total + 0.4;
    return total + 0.68;
  }, 0);
  const fitted = Math.min(
    maximum,
    availableHeight * 0.68,
    availableWidth / Math.max(glyphUnits, 0.68)
  );
  return Math.max(minimum, fitted);
}
