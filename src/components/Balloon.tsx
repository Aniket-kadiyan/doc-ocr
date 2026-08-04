"use client";

import { Circle, Group, Line, Text } from "react-konva";
import type { Annotation } from "@/types/annotation";

interface BalloonProps {
  annotation: Annotation;
  scale: number;
  selected?: boolean;
  /** Faded because another annotation is the active selection. */
  dimmed?: boolean;
  listening?: boolean;
  onSelect?: (id: string) => void;
}

export function Balloon({
  annotation,
  scale,
  selected,
  dimmed = false,
  listening = true,
  onSelect,
}: BalloonProps) {
  const { bbox, number } = annotation;
  const cx = bbox.x + bbox.width / 2;
  const cy = bbox.y - 28 / scale;
  const anchorX = bbox.x + bbox.width / 2;
  const anchorY = bbox.y;
  const radius = 14 / scale;
  const fontSize = 12 / scale;

  const accent = "#dc2626";
  const selectedFill = "#2563eb";
  const selectedStroke = "#1d4ed8";

  return (
    <Group
      listening={listening}
      opacity={dimmed ? 0.15 : 1}
      onClick={() => onSelect?.(annotation.id)}
      onTap={() => onSelect?.(annotation.id)}
    >
      <Line
        points={[cx, cy + radius, anchorX, anchorY]}
        stroke={selected ? selectedFill : accent}
        strokeWidth={1.5 / scale}
      />
      <Circle
        x={cx}
        y={cy}
        radius={radius}
        fill={selected ? selectedFill : "#ffffff"}
        stroke={selected ? selectedStroke : accent}
        strokeWidth={2 / scale}
      />
      <Text
        x={cx}
        y={cy}
        text={String(number)}
        fontSize={fontSize}
        fontStyle="bold"
        fill={selected ? "#ffffff" : accent}
        align="center"
        verticalAlign="middle"
        offsetX={fontSize * 0.35}
        offsetY={fontSize * 0.45}
      />
    </Group>
  );
}
