export const BALLOON_STYLE_SCHEMA_VERSION = 2 as const;

export const BALLOON_SIZES = ["small", "medium", "large"] as const;
export type BalloonSize = (typeof BALLOON_SIZES)[number];

export interface BalloonStyleSelection {
  styleId: string;
  styleVersion: number;
  size: BalloonSize;
}

export const DEFAULT_BALLOON_STYLE_SELECTION: BalloonStyleSelection = {
  styleId: "circle",
  styleVersion: 1,
  size: "medium",
};

export interface NormalizedPoint {
  x: number;
  y: number;
}

export interface NormalizedRect extends NormalizedPoint {
  width: number;
  height: number;
}

export interface BalloonStyleSizeDefinition {
  artwork: {
    /** Same-origin path to a trusted, repository-owned SVG file. */
    path: string;
    width: number;
    height: number;
  };
  display: {
    drawingWidth: number;
    sidebarWidth: number;
  };
}

export interface BalloonStyleDefinition {
  id: string;
  name: string;
  version: number;
  numberArea: NormalizedRect;
  anchor: NormalizedPoint;
  text: {
    color: string;
    fontFamily: string;
    fontWeight: "normal" | "bold";
    minFontSize: number;
    maxFontSize: number;
    padding: number;
  };
  sizes: Record<BalloonSize, BalloonStyleSizeDefinition>;
}

export interface BalloonStyleManifest {
  schemaVersion: typeof BALLOON_STYLE_SCHEMA_VERSION;
  defaultSelection: BalloonStyleSelection;
  styles: BalloonStyleDefinition[];
}

/** Fully resolved style used by the viewer and every export renderer. */
export interface BalloonStyle {
  schemaVersion: typeof BALLOON_STYLE_SCHEMA_VERSION;
  id: string;
  name: string;
  version: number;
  size: BalloonSize;
  artwork: {
    mimeType: "image/svg+xml";
    path: string;
    dataUrl: string;
    width: number;
    height: number;
  };
  numberArea: NormalizedRect;
  anchor: NormalizedPoint;
  display: {
    drawingWidth: number;
    sidebarWidth: number;
  };
  text: BalloonStyleDefinition["text"];
}

export interface BalloonGeometry {
  x: number;
  y: number;
  width: number;
  height: number;
  numberX: number;
  numberY: number;
  numberWidth: number;
  numberHeight: number;
}
