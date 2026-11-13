"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import {
  BALLOON_REVIEW_BOUNDARY_COLOR,
  FALLBACK_BALLOON_STYLE,
  FALLBACK_REVIEW_BALLOON_STYLE,
  normalizeBalloonStyleSelection,
  parseBalloonStyleManifest,
  resolveBalloonStyle,
} from "@/lib/balloonStyle";
import {
  DEFAULT_BALLOON_STYLE_SELECTION,
  type BalloonStyle,
  type BalloonStyleManifest,
  type BalloonStyleSelection,
} from "@/types/balloonStyle";

interface BalloonStyleContextValue {
  style: BalloonStyle;
  artworkImage: HTMLImageElement | null;
  reviewStyle: BalloonStyle;
  reviewArtworkImage: HTMLImageElement | null;
  manifest: BalloonStyleManifest | null;
  selection: BalloonStyleSelection;
  setSelection: (selection: BalloonStyleSelection) => void;
  ready: boolean;
  source: "configured" | "fallback";
}

const defaultValue: BalloonStyleContextValue = {
  style: FALLBACK_BALLOON_STYLE,
  artworkImage: null,
  reviewStyle: FALLBACK_REVIEW_BALLOON_STYLE,
  reviewArtworkImage: null,
  manifest: null,
  selection: DEFAULT_BALLOON_STYLE_SELECTION,
  setSelection: () => undefined,
  ready: false,
  source: "fallback",
};

const BalloonStyleContext = createContext<BalloonStyleContextValue>(defaultValue);

const loadArtwork = (style: BalloonStyle) =>
  new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new window.Image();
    image.decoding = "async";
    image.onload = () => {
      if (
        image.naturalWidth !== style.artwork.width ||
        image.naturalHeight !== style.artwork.height
      ) {
        reject(
          new Error(
            `${style.artwork.path} is ${image.naturalWidth}×${image.naturalHeight}; ` +
              `the manifest declares ${style.artwork.width}×${style.artwork.height}.`
          )
        );
        return;
      }
      resolve(image);
    };
    image.onerror = () => reject(new Error("Balloon SVG artwork is invalid."));
    image.src = style.artwork.dataUrl;
  });

interface BalloonStyleProviderProps {
  children: ReactNode;
  /** A supplied selection makes the provider controlled (used by saved checksheets). */
  selection?: BalloonStyleSelection;
  onSelectionChange?: (selection: BalloonStyleSelection) => void;
}

export function BalloonStyleProvider({
  children,
  selection: controlledSelection,
  onSelectionChange,
}: BalloonStyleProviderProps) {
  const [manifest, setManifest] = useState<BalloonStyleManifest | null>(null);
  const [internalSelection, setInternalSelection] = useState(
    normalizeBalloonStyleSelection(controlledSelection)
  );
  const [style, setStyle] = useState<BalloonStyle>(FALLBACK_BALLOON_STYLE);
  const [artworkImage, setArtworkImage] = useState<HTMLImageElement | null>(null);
  const [reviewStyle, setReviewStyle] = useState<BalloonStyle>(
    FALLBACK_REVIEW_BALLOON_STYLE
  );
  const [reviewArtworkImage, setReviewArtworkImage] =
    useState<HTMLImageElement | null>(null);
  const [ready, setReady] = useState(false);
  const selection = useMemo(
    () =>
      normalizeBalloonStyleSelection(
        controlledSelection ?? internalSelection,
        manifest
      ),
    [controlledSelection, internalSelection, manifest]
  );

  const setSelection = useCallback(
    (next: BalloonStyleSelection) => {
      const normalized = normalizeBalloonStyleSelection(next, manifest);
      if (controlledSelection == null) setInternalSelection(normalized);
      onSelectionChange?.(normalized);
    },
    [controlledSelection, manifest, onSelectionChange]
  );

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const [fallbackImage, fallbackReviewImage] = await Promise.all([
          loadArtwork(FALLBACK_BALLOON_STYLE),
          loadArtwork(FALLBACK_REVIEW_BALLOON_STYLE),
        ]);
        if (!cancelled) {
          setArtworkImage(fallbackImage);
          setReviewArtworkImage(fallbackReviewImage);
        }
      } catch {
        // Balloon.tsx has a vector last-resort rendering path.
      }
      try {
        const response = await fetch("/balloon-styles/manifest.json", {
          cache: "no-store",
        });
        if (!response.ok) {
          throw new Error(`manifest returned HTTP ${response.status}.`);
        }
        const parsed = parseBalloonStyleManifest(await response.json());
        if (!cancelled) setManifest(parsed);
      } catch (error) {
        if (!cancelled) setReady(true);
        console.error("[balloon-style] Could not load the SVG manifest; using the built-in circle.", error);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!manifest) return;
    let cancelled = false;
    setReady(false);
    void (async () => {
      try {
        const normalized = normalizeBalloonStyleSelection(selection, manifest);
        const definition = manifest.styles.find(
          (item) => item.id === normalized.styleId && item.version === normalized.styleVersion
        );
        if (!definition) throw new Error("Selected balloon style is not in the manifest.");
        const asset = definition.sizes[normalized.size].artwork;
        const response = await fetch(asset.path, { cache: "no-store" });
        if (!response.ok) throw new Error(`${asset.path} returned HTTP ${response.status}.`);
        const svg = await response.text();
        const resolved = resolveBalloonStyle(manifest, normalized, svg);
        const resolvedReview = resolveBalloonStyle(
          manifest,
          normalized,
          svg,
          BALLOON_REVIEW_BOUNDARY_COLOR
        );
        const [image, reviewImage] = await Promise.all([
          loadArtwork(resolved),
          loadArtwork(resolvedReview),
        ]);
        if (!cancelled) {
          setStyle(resolved);
          setArtworkImage(image);
          setReviewStyle(resolvedReview);
          setReviewArtworkImage(reviewImage);
          setReady(true);
          if (controlledSelection == null) {
            setInternalSelection((current) =>
              current.styleId === normalized.styleId &&
              current.styleVersion === normalized.styleVersion &&
              current.size === normalized.size
                ? current
                : normalized
            );
          }
        }
      } catch (error) {
        if (!cancelled) setReady(true);
        console.error("[balloon-style] Could not load the selected SVG; retaining the previous style.", error);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [controlledSelection, manifest, selection]);

  const contextValue = useMemo<BalloonStyleContextValue>(
    () => ({
      style,
      artworkImage,
      reviewStyle,
      reviewArtworkImage,
      manifest,
      selection,
      setSelection,
      ready,
      source: manifest && style !== FALLBACK_BALLOON_STYLE ? "configured" : "fallback",
    }),
    [
      artworkImage,
      manifest,
      ready,
      reviewArtworkImage,
      reviewStyle,
      selection,
      setSelection,
      style,
    ]
  );

  return (
    <BalloonStyleContext.Provider value={contextValue}>
      {children}
    </BalloonStyleContext.Provider>
  );
}

export const useBalloonStyle = () => useContext(BalloonStyleContext);
