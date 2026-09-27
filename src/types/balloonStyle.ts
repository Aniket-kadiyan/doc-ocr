export const BALLOON_STYLE_SCHEMA_VERSION = 1 as const;

export interface NormalizedPoint {
  /** Horizontal position as a fraction of the artwork width. */
  x: number;
  /** Vertical position as a fraction of the artwork height. */
  y: number;
}

export interface NormalizedRect extends NormalizedPoint {
  width: number;
  height: number;
}

export interface BalloonStyle {
  schemaVersion: typeof BALLOON_STYLE_SCHEMA_VERSION;
  name: string;
  artwork: {
    /** Deployed style files use PNG. SVG is reserved for the built-in fallback. */
    mimeType: "image/png" | "image/svg+xml";
    dataUrl: string;
    width: number;
    height: number;
  };
  /** Rectangle in which the application overlays the dynamic balloon number. */
  numberArea: NormalizedRect;
  /** Pointer tip that is placed on the value box's top centre. */
  anchor: NormalizedPoint;
  display: {
    /** Constant on-screen width in pixels in the drawing viewer. */
    drawingWidth: number;
    /** Constant on-screen width in pixels in the sidebar. */
    sidebarWidth: number;
  };
  text: {
    color: string;
    fontFamily: string;
    fontWeight: "normal" | "bold";
    /** Pixel sizes at drawingWidth; thumbnails scale them proportionally. */
    minFontSize: number;
    maxFontSize: number;
    padding: number;
  };
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
