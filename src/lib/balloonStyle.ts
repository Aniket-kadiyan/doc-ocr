import {
  BALLOON_SIZES,
  BALLOON_STYLE_SCHEMA_VERSION,
  DEFAULT_BALLOON_STYLE_SELECTION,
  type BalloonGeometry,
  type BalloonSize,
  type BalloonStyle,
  type BalloonStyleDefinition,
  type BalloonStyleManifest,
  type BalloonStyleSelection,
  type NormalizedRect,
} from "@/types/balloonStyle";

const FALLBACK_SVG = `<svg xmlns="http://www.w3.org/2000/svg" width="202" height="310" viewBox="0 0 202 310"><defs><linearGradient id="tail" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ff9a1f"/><stop offset="1" stop-color="#ff7100"/></linearGradient></defs><path d="M101 305 29 164 173 164Z" fill="url(#tail)" stroke="#050505" stroke-width="10" stroke-linejoin="round"/><circle cx="101" cy="99" r="94" fill="#fff" stroke="#050505" stroke-width="10"/><text data-balloon-label="true"></text></svg>`;

const FALLBACK_DEFINITION: BalloonStyleDefinition = {
  id: "circle",
  name: "Circle",
  version: 1,
  numberArea: { x: 0.09901, y: 0.064516, width: 0.80198, height: 0.483871 },
  anchor: { x: 0.5, y: 0.996774 },
  text: {
    color: "#111827",
    fontFamily: "Arial, sans-serif",
    fontWeight: "bold",
    minFontSize: 7,
    maxFontSize: 13,
    padding: 2,
  },
  sizes: {
    small: {
      artwork: { path: "/balloon-styles/circle/v1/small.svg", width: 202, height: 310 },
      display: { drawingWidth: 24, sidebarWidth: 18 },
    },
    medium: {
      artwork: { path: "/balloon-styles/circle/v1/medium.svg", width: 202, height: 310 },
      display: { drawingWidth: 32, sidebarWidth: 22 },
    },
    large: {
      artwork: { path: "/balloon-styles/circle/v1/large.svg", width: 202, height: 310 },
      display: { drawingWidth: 40, sidebarWidth: 28 },
    },
  },
};

const svgDataUrl = (svg: string) =>
  `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg.trim())}`;

/** Always-available marker used while the manifest loads or if it fails. */
export const FALLBACK_BALLOON_STYLE: BalloonStyle = {
  schemaVersion: BALLOON_STYLE_SCHEMA_VERSION,
  id: FALLBACK_DEFINITION.id,
  name: FALLBACK_DEFINITION.name,
  version: FALLBACK_DEFINITION.version,
  size: "medium",
  artwork: {
    mimeType: "image/svg+xml",
    path: FALLBACK_DEFINITION.sizes.medium.artwork.path,
    dataUrl: svgDataUrl(FALLBACK_SVG),
    width: 202,
    height: 310,
  },
  numberArea: FALLBACK_DEFINITION.numberArea,
  anchor: FALLBACK_DEFINITION.anchor,
  display: FALLBACK_DEFINITION.sizes.medium.display,
  text: FALLBACK_DEFINITION.text,
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

const readNormalizedRect = (value: unknown, path: string): NormalizedRect => {
  const rect = readRecord(value, path);
  const x = readNumber(rect.x, `${path}.x`, 0, 1);
  const y = readNumber(rect.y, `${path}.y`, 0, 1);
  const width = readNumber(rect.width, `${path}.width`, 0.001, 1);
  const height = readNumber(rect.height, `${path}.height`, 0.001, 1);
  if (x + width > 1.000001 || y + height > 1.000001) {
    throw new Error(`${path} must stay within the artwork bounds.`);
  }
  return { x, y, width, height };
};

const readSelection = (value: unknown, path: string): BalloonStyleSelection => {
  const record = readRecord(value, path);
  const size = readString(record.size, `${path}.size`);
  if (!BALLOON_SIZES.includes(size as BalloonSize)) {
    throw new Error(`${path}.size must be small, medium, or large.`);
  }
  return {
    styleId: readString(record.styleId, `${path}.styleId`),
    styleVersion: readNumber(record.styleVersion, `${path}.styleVersion`, 1, 999),
    size: size as BalloonSize,
  };
};

const readDefinition = (value: unknown, index: number): BalloonStyleDefinition => {
  const path = `styles[${index}]`;
  const record = readRecord(value, path);
  const anchor = readRecord(record.anchor, `${path}.anchor`);
  const text = readRecord(record.text, `${path}.text`);
  const color = readString(text.color, `${path}.text.color`);
  if (!/^#(?:[0-9a-f]{3}|[0-9a-f]{6}|[0-9a-f]{8})$/i.test(color)) {
    throw new Error(`${path}.text.color must be a hexadecimal CSS colour.`);
  }
  if (text.fontWeight !== "normal" && text.fontWeight !== "bold") {
    throw new Error(`${path}.text.fontWeight must be normal or bold.`);
  }
  const minFontSize = readNumber(text.minFontSize, `${path}.text.minFontSize`, 4, 48);
  const maxFontSize = readNumber(text.maxFontSize, `${path}.text.maxFontSize`, 4, 72);
  if (maxFontSize < minFontSize) {
    throw new Error(`${path}.text.maxFontSize must be at least minFontSize.`);
  }
  const sizesRecord = readRecord(record.sizes, `${path}.sizes`);
  const sizes = Object.fromEntries(
    BALLOON_SIZES.map((size) => {
      const sizeRecord = readRecord(sizesRecord[size], `${path}.sizes.${size}`);
      const artwork = readRecord(sizeRecord.artwork, `${path}.sizes.${size}.artwork`);
      const display = readRecord(sizeRecord.display, `${path}.sizes.${size}.display`);
      const artworkPath = readString(artwork.path, `${path}.sizes.${size}.artwork.path`);
      if (!/^\/balloon-styles\/[a-z0-9/_-]+\.svg$/i.test(artworkPath)) {
        throw new Error(`${path}.sizes.${size}.artwork.path must be a local balloon SVG path.`);
      }
      return [
        size,
        {
          artwork: {
            path: artworkPath,
            width: readNumber(artwork.width, `${path}.sizes.${size}.artwork.width`, 1, 4096),
            height: readNumber(artwork.height, `${path}.sizes.${size}.artwork.height`, 1, 4096),
          },
          display: {
            drawingWidth: readNumber(display.drawingWidth, `${path}.sizes.${size}.display.drawingWidth`, 12, 160),
            sidebarWidth: readNumber(display.sidebarWidth, `${path}.sizes.${size}.display.sidebarWidth`, 12, 80),
          },
        },
      ];
    })
  ) as unknown as BalloonStyleDefinition["sizes"];

  return {
    id: readString(record.id, `${path}.id`),
    name: readString(record.name, `${path}.name`),
    version: readNumber(record.version, `${path}.version`, 1, 999),
    numberArea: readNormalizedRect(record.numberArea, `${path}.numberArea`),
    anchor: {
      x: readNumber(anchor.x, `${path}.anchor.x`, 0, 1),
      y: readNumber(anchor.y, `${path}.anchor.y`, 0, 1),
    },
    text: {
      color,
      fontFamily: readString(text.fontFamily, `${path}.text.fontFamily`).slice(0, 120),
      fontWeight: text.fontWeight,
      minFontSize,
      maxFontSize,
      padding: readNumber(text.padding, `${path}.text.padding`, 0, 24),
    },
    sizes,
  };
};

export function parseBalloonStyleManifest(input: unknown): BalloonStyleManifest {
  const root = readRecord(input, "balloon style manifest");
  if (root.schemaVersion !== BALLOON_STYLE_SCHEMA_VERSION) {
    throw new Error(`schemaVersion must be ${BALLOON_STYLE_SCHEMA_VERSION}.`);
  }
  if (!Array.isArray(root.styles) || root.styles.length === 0) {
    throw new Error("styles must contain at least one balloon style.");
  }
  const styles = root.styles.map(readDefinition);
  if (new Set(styles.map((style) => style.id)).size !== styles.length) {
    throw new Error("style IDs must be unique.");
  }
  const defaultSelection = readSelection(root.defaultSelection, "defaultSelection");
  if (!styles.some((style) => style.id === defaultSelection.styleId && style.version === defaultSelection.styleVersion)) {
    throw new Error("defaultSelection must identify a style in the manifest.");
  }
  return { schemaVersion: BALLOON_STYLE_SCHEMA_VERSION, defaultSelection, styles };
}

export function normalizeBalloonStyleSelection(
  value?: Partial<BalloonStyleSelection> | null,
  manifest?: BalloonStyleManifest | null
): BalloonStyleSelection {
  const fallback = manifest?.defaultSelection ?? DEFAULT_BALLOON_STYLE_SELECTION;
  const size = BALLOON_SIZES.includes(value?.size as BalloonSize)
    ? (value?.size as BalloonSize)
    : fallback.size;
  const styleId = typeof value?.styleId === "string" && value.styleId.trim()
    ? value.styleId.trim()
    : fallback.styleId;
  const styleVersion = Number.isInteger(value?.styleVersion) && Number(value?.styleVersion) > 0
    ? Number(value?.styleVersion)
    : fallback.styleVersion;
  if (!manifest) return { styleId, styleVersion, size };
  const found = manifest.styles.find(
    (style) => style.id === styleId && style.version === styleVersion
  );
  return found ? { styleId, styleVersion, size } : { ...fallback };
}

/** Reject active or externally loaded SVG content before it reaches the image decoder. */
export function validateBalloonSvg(svg: string): string {
  const trimmed = svg.trim();
  if (!/^<svg\b/i.test(trimmed) || !/<\/svg>\s*$/i.test(trimmed)) {
    throw new Error("Balloon artwork must be a complete SVG document.");
  }
  if (trimmed.length > 200_000) throw new Error("Balloon SVG exceeds the 200 KB limit.");
  if (/<(?:script|foreignObject|iframe|object|embed)\b/i.test(trimmed)) {
    throw new Error("Balloon SVG contains prohibited active content.");
  }
  if (/\son[a-z]+\s*=/i.test(trimmed) || /javascript\s*:/i.test(trimmed)) {
    throw new Error("Balloon SVG contains prohibited event or script content.");
  }
  if (/\s(?:href|xlink:href)\s*=\s*["'](?!#)/i.test(trimmed)) {
    throw new Error("Balloon SVG must not load external resources.");
  }
  if (!/data-balloon-label\s*=\s*["']true["']/i.test(trimmed)) {
    throw new Error("Balloon SVG must contain the data-balloon-label marker.");
  }
  return trimmed;
}

export function resolveBalloonStyle(
  manifest: BalloonStyleManifest,
  requested: Partial<BalloonStyleSelection> | null | undefined,
  svg: string
): BalloonStyle {
  const selection = normalizeBalloonStyleSelection(requested, manifest);
  const definition = manifest.styles.find(
    (style) => style.id === selection.styleId && style.version === selection.styleVersion
  ) ?? manifest.styles[0];
  const size = definition.sizes[selection.size];
  return {
    schemaVersion: BALLOON_STYLE_SCHEMA_VERSION,
    id: definition.id,
    name: definition.name,
    version: definition.version,
    size: selection.size,
    artwork: {
      mimeType: "image/svg+xml",
      path: size.artwork.path,
      dataUrl: svgDataUrl(validateBalloonSvg(svg)),
      width: size.artwork.width,
      height: size.artwork.height,
    },
    numberArea: definition.numberArea,
    anchor: definition.anchor,
    display: size.display,
    text: definition.text,
  };
}

export function getBalloonScreenSize(style: BalloonStyle, target: "drawing" | "sidebar") {
  const width = target === "drawing" ? style.display.drawingWidth : style.display.sidebarWidth;
  return { width, height: width * (style.artwork.height / style.artwork.width) };
}

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
  const fitted = Math.min(maximum, availableHeight * 0.68, availableWidth / Math.max(glyphUnits, 0.68));
  return Math.max(minimum, fitted);
}

export function balloonArtworkPngDataUrl(
  style: BalloonStyle,
  image: HTMLImageElement,
  rasterScale = 4
): string {
  const canvas = document.createElement("canvas");
  canvas.width = Math.ceil(style.artwork.width * rasterScale);
  canvas.height = Math.ceil(style.artwork.height * rasterScale);
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Canvas context unavailable for balloon artwork.");
  context.drawImage(image, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/png");
}
