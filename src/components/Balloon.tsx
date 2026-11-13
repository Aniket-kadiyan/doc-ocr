"use client";

import type { Annotation } from "@/types/annotation";
import { BalloonMarker } from "@/components/BalloonMarker";

interface BalloonProps {
  annotation: Annotation;
  scale: number;
  selected?: boolean;
  /** Faded because another annotation is the active selection. */
  dimmed?: boolean;
  listening?: boolean;
  onSelect?: (id: string) => void;
  /** Double click / double tap opens the editor for this annotation. */
  onOpen?: (id: string) => void;
}

export function Balloon({
  annotation,
  scale,
  selected,
  dimmed = false,
  listening = true,
  onSelect,
  onOpen,
}: BalloonProps) {
  return (
    <BalloonMarker
      bbox={annotation.orientedBox ?? annotation.bbox}
      label={String(annotation.number)}
      scale={scale}
      status="ok"
      selected={selected}
      dimmed={dimmed}
      listening={listening}
      onSelect={() => onSelect?.(annotation.id)}
      onOpen={() => onOpen?.(annotation.id)}
    />
  );
}
