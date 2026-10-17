"use client";

import { Group, Rect, Text } from "react-konva";
import {
  FRAME_COLOR,
  detectionKey,
  statusColor,
} from "@/lib/gdtSymbols";
import type { GdtFrame, GdtOverlayItem } from "@/types/gdt";

interface GdtOverlayProps {
  frames: GdtFrame[];
  items: GdtOverlayItem[];
  /** Current stage zoom, so strokes and labels stay one size on screen. */
  scale: number;
  /** Detection the panel has selected, by key. */
  selectedKey: string | null;
  onSelect: (key: string | null, item: GdtOverlayItem) => void;
}

/**
 * Detected GD&T symbols and feature control frames, painted over the drawing.
 *
 * Lives in the same page coordinate system as annotations, so stage zoom and
 * pan keep it aligned without any work here. Like `ScanDebugOverlay`, this is
 * presentation only: nothing it draws is exported, saved into a project, or
 * balloon-numbered. A GD&T detection is something to *look at and confirm*,
 * not an inspection value — making one into a balloon silently would put a
 * tolerance on a checksheet that nobody measured.
 *
 * Stroke widths divide by `scale` so a box reads the same at any zoom. At a
 * 10× zoom an unscaled 2px stroke becomes a 20px band that hides the symbol
 * underneath it, which defeats the point of showing the operator the glyph.
 */
export function GdtOverlay({
  frames,
  items,
  scale,
  selectedKey,
  onSelect,
}: GdtOverlayProps) {
  const safeScale = Math.max(scale, 0.01);
  const stroke = 1.5 / safeScale;
  const labelSize = 11 / safeScale;

  return (
    <Group>
      {/* Frames first, so symbol boxes sit on top of their container. */}
      {frames.map((frame) => (
        <Group key={`gdt-frame-${frame.id}`} listening={false}>
          <Rect
            x={frame.bounding_box.x}
            y={frame.bounding_box.y}
            width={frame.bounding_box.width}
            height={frame.bounding_box.height}
            stroke={FRAME_COLOR}
            strokeWidth={stroke}
            dash={[6 / safeScale, 4 / safeScale]}
            fill={FRAME_COLOR}
            opacity={0.06}
          />
          {frame.composite_id && (
            <Text
              x={frame.bounding_box.x}
              y={frame.bounding_box.y - labelSize * 1.2}
              text={`segment ${frame.segment_index + 1}`}
              fontSize={labelSize * 0.85}
              fill={FRAME_COLOR}
            />
          )}
        </Group>
      ))}

      {items.map((item, index) => {
        const key = detectionKey(item.detection, index);
        const selected = key === selectedKey;
        const color = statusColor(item.detection.status);
        return (
          <Group key={`gdt-symbol-${key}`}>
            <Rect
              x={item.bbox.x}
              y={item.bbox.y}
              width={item.bbox.width}
              height={item.bbox.height}
              stroke={color}
              strokeWidth={selected ? stroke * 2.2 : stroke}
              fill={color}
              opacity={selected ? 0.24 : 0.12}
              onClick={() => onSelect(selected ? null : key, item)}
              onTap={() => onSelect(selected ? null : key, item)}
              // A 10px symbol is a 10px click target. Widening the hit area
              // keeps the small modifiers — which are exactly the ones a
              // reviewer most often needs to correct — actually selectable.
              hitStrokeWidth={Math.max(12 / safeScale, stroke * 6)}
            />
            {selected && (
              <Text
                x={item.bbox.x}
                y={item.bbox.y - labelSize * 1.25}
                text={item.detection.name}
                fontSize={labelSize}
                fill={color}
                fontStyle="bold"
                listening={false}
              />
            )}
          </Group>
        );
      })}
    </Group>
  );
}
