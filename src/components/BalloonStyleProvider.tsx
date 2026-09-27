"use client";

import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import {
  FALLBACK_BALLOON_STYLE,
  parseBalloonStyle,
} from "@/lib/balloonStyle";
import type { BalloonStyle } from "@/types/balloonStyle";

interface BalloonStyleContextValue {
  style: BalloonStyle;
  artworkImage: HTMLImageElement | null;
  source: "configured" | "fallback";
}

const defaultValue: BalloonStyleContextValue = {
  style: FALLBACK_BALLOON_STYLE,
  artworkImage: null,
  source: "fallback",
};

const BalloonStyleContext =
  createContext<BalloonStyleContextValue>(defaultValue);

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
            `Artwork dimensions are ${image.naturalWidth}×${image.naturalHeight}; ` +
              `the style declares ${style.artwork.width}×${style.artwork.height}.`
          )
        );
        return;
      }
      resolve(image);
    };
    image.onerror = () => reject(new Error("Embedded balloon artwork is invalid."));
    image.src = style.artwork.dataUrl;
  });

export function BalloonStyleProvider({ children }: { children: ReactNode }) {
  const [value, setValue] =
    useState<BalloonStyleContextValue>(defaultValue);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      // Make the fallback visible immediately while the deployed file is read.
      try {
        const fallbackImage = await loadArtwork(FALLBACK_BALLOON_STYLE);
        if (!cancelled) {
          setValue({
            style: FALLBACK_BALLOON_STYLE,
            artworkImage: fallbackImage,
            source: "fallback",
          });
        }
      } catch {
        // Balloon.tsx also contains a vector last-resort rendering path.
      }

      try {
        const response = await fetch(`/balloon-style.json?t=${Date.now()}`, {
          cache: "no-store",
        });
        if (!response.ok) {
          throw new Error(`balloon-style.json returned HTTP ${response.status}.`);
        }
        const style = parseBalloonStyle(await response.json());
        const artworkImage = await loadArtwork(style);
        if (!cancelled) {
          setValue({ style, artworkImage, source: "configured" });
        }
      } catch (error) {
        console.error(
          "[balloon-style] Could not load the deployed style; using the built-in fallback.",
          error
        );
      }
    };

    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  const contextValue = useMemo(
    () => value,
    [value]
  );

  return (
    <BalloonStyleContext.Provider value={contextValue}>
      {children}
    </BalloonStyleContext.Provider>
  );
}

export const useBalloonStyle = () => useContext(BalloonStyleContext);
