"use client";

import { Group, Line, Text } from "react-konva";
import type { DrawingSection } from "@/types/drawingSection";

interface DrawingSectionsOverlayProps {
  sections: DrawingSection[];
  scale: number;
}

/** One distinct hue per section, a golden angle apart, as the backend's
 * section report draws them. */
function sectionColor(index: number): string {
  const hue = Math.round((index * 222.5) % 360);
  return `hsl(${hue}, 85%, ${index % 2 === 0 ? 38 : 30}%)`;
}

/**
 * The page's drawing sections, outlined and labelled. Display only: it lives
 * in page coordinates like the balloons, ignores the pointer, and is never
 * exported.
 */
export function DrawingSectionsOverlay({
  sections,
  scale,
}: DrawingSectionsOverlayProps) {
  const safeScale = Math.max(scale, 0.01);
  return (
    <Group listening={false}>
      {sections.map((section, index) => {
        const color = sectionColor(index);
        const first = section.outlines[0];
        if (!first || first.length < 2) return null;
        // Label at the outline's topmost point, nudged inside.
        let labelX = first[0];
        let labelY = first[1];
        for (let i = 0; i < first.length; i += 2) {
          if (first[i + 1] < labelY) {
            labelX = first[i];
            labelY = first[i + 1];
          }
        }
        return (
          <Group key={section.id}>
            {section.outlines.map((points, outlineIndex) => (
              <Line
                key={outlineIndex}
                points={points}
                closed
                stroke={color}
                strokeWidth={2 / safeScale}
                dash={[8 / safeScale, 5 / safeScale]}
                opacity={0.8}
              />
            ))}
            <Text
              x={labelX + 4 / safeScale}
              y={labelY + 4 / safeScale}
              text={section.id}
              fontSize={14 / safeScale}
              fontStyle="bold"
              fill={color}
            />
          </Group>
        );
      })}
    </Group>
  );
}
