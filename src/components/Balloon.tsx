"use client";

import { Circle, Group, Image as KonvaImage, Line, Text } from "react-konva";
import type { Annotation } from "@/types/annotation";
import {
  fitBalloonFontSize,
  getAnchoredBalloonGeometry,
} from "@/lib/balloonStyle";
import { useBalloonStyle } from "@/components/BalloonStyleProvider";

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
  const anchorX = bbox.x + bbox.width / 2;
  const anchorY = bbox.y;
  const { style, artworkImage } = useBalloonStyle();
  const geometry = getAnchoredBalloonGeometry(
    style,
    anchorX,
    anchorY,
    scale
  );
  const screenAreaWidth = geometry.numberWidth * scale;
  const screenAreaHeight = geometry.numberHeight * scale;
  const fontSize =
    fitBalloonFontSize(
      String(number),
      screenAreaWidth,
      screenAreaHeight,
      style.display.drawingWidth,
      style
    ) / scale;

  return (
    <Group
      listening={listening}
      opacity={dimmed ? 0.15 : 1}
      onClick={() => onSelect?.(annotation.id)}
      onTap={() => onSelect?.(annotation.id)}
    >
      {artworkImage ? (
        <KonvaImage
          image={artworkImage}
          x={geometry.x}
          y={geometry.y}
          width={geometry.width}
          height={geometry.height}
          shadowColor="#2563eb"
          shadowBlur={selected ? 7 / scale : 0}
          shadowOpacity={selected ? 0.9 : 0}
          shadowEnabled={selected}
        />
      ) : (
        <>
          {/* Last-resort shape if even the built-in artwork cannot decode. */}
          <Line
            points={[
              anchorX,
              anchorY,
              geometry.x + geometry.width * 0.18,
              geometry.y + geometry.height * 0.52,
              geometry.x + geometry.width * 0.82,
              geometry.y + geometry.height * 0.52,
            ]}
            closed
            fill="#f97316"
            stroke="#111827"
            strokeWidth={2 / scale}
          />
          <Circle
            x={geometry.x + geometry.width / 2}
            y={geometry.y + geometry.width / 2}
            radius={geometry.width * 0.45}
            fill="#ffffff"
            stroke={selected ? "#2563eb" : "#111827"}
            strokeWidth={(selected ? 3 : 2) / scale}
          />
        </>
      )}
      <Text
        x={geometry.numberX}
        y={geometry.numberY}
        width={geometry.numberWidth}
        height={geometry.numberHeight}
        text={String(number)}
        fontSize={fontSize}
        fontFamily={style.text.fontFamily}
        fontStyle={style.text.fontWeight}
        fill={style.text.color}
        align="center"
        verticalAlign="middle"
        listening={false}
      />
    </Group>
  );
}
