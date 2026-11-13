"use client";

import { Circle, Group, Image as KonvaImage, Line, Text } from "react-konva";
import type { BBox } from "@/types/annotation";
import {
  BALLOON_OK_BOUNDARY_COLOR,
  BALLOON_REVIEW_BOUNDARY_COLOR,
  fitBalloonFontSize,
  getAnchoredBalloonGeometry,
} from "@/lib/balloonStyle";
import { useBalloonStyle } from "@/components/BalloonStyleProvider";

export type BalloonMarkerStatus = "ok" | "review";

interface BalloonMarkerProps {
  bbox: BBox;
  label: string;
  scale: number;
  status?: BalloonMarkerStatus;
  selected?: boolean;
  dimmed?: boolean;
  listening?: boolean;
  onSelect?: () => void;
  onOpen?: () => void;
}

/** Shared SVG marker used by accepted balloons and unresolved review values. */
export function BalloonMarker({
  bbox,
  label,
  scale,
  status = "ok",
  selected = false,
  dimmed = false,
  listening = true,
  onSelect,
  onOpen,
}: BalloonMarkerProps) {
  const {
    style: okStyle,
    artworkImage: okArtworkImage,
    reviewStyle,
    reviewArtworkImage,
  } = useBalloonStyle();
  const review = status === "review";
  const style = review ? reviewStyle : okStyle;
  const artworkImage = review ? reviewArtworkImage : okArtworkImage;
  const boundaryColor = review
    ? BALLOON_REVIEW_BOUNDARY_COLOR
    : BALLOON_OK_BOUNDARY_COLOR;
  const anchorX = bbox.x + bbox.width / 2;
  const anchorY = bbox.y;
  const geometry = getAnchoredBalloonGeometry(
    style,
    anchorX,
    anchorY,
    scale
  );
  const fontSize =
    fitBalloonFontSize(
      label,
      geometry.numberWidth * scale,
      geometry.numberHeight * scale,
      style.display.drawingWidth,
      style
    ) / scale;

  return (
    <Group
      listening={listening}
      opacity={dimmed ? 0.15 : 1}
      onClick={onSelect}
      onTap={onSelect}
      onDblClick={onOpen}
      onDblTap={onOpen}
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
          <Line
            points={[
              geometry.x + geometry.width / 2,
              geometry.y + geometry.width * 0.95,
              anchorX,
              anchorY,
            ]}
            stroke="#111827"
            strokeWidth={2 / scale}
            lineCap="round"
            listening={false}
          />
          <Circle
            x={geometry.x + geometry.width / 2}
            y={geometry.y + geometry.width / 2}
            radius={geometry.width * 0.45}
            fill="#ffffff"
            stroke={boundaryColor}
            strokeWidth={(selected ? 3 : 2) / scale}
          />
        </>
      )}
      <Text
        x={geometry.numberX}
        y={geometry.numberY}
        width={geometry.numberWidth}
        height={geometry.numberHeight}
        text={label}
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
