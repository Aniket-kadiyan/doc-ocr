"use client";

import { Group, Line, Rect, Text } from "react-konva";
import type {
  ReferenceCoordinate,
  ReferenceMarker as Marker,
} from "@/types/referencePoint";
import type { ReferenceColor } from "@/lib/referenceColors";
import type { TagPlacement } from "@/lib/referenceTagLayout";

const HEADER_HEIGHT = 13;
const AXIS_ROW_HEIGHT = 11;
const PADDING_X = 4;
const AXIS_FONT = 8.5;
const NAME_FONT = 9.5;
const AXIS_COLUMN = 7;
const RING_PAD = 2;
const CORNER = 2.5;
const GLYPH_ADVANCE = 0.62;

/** Screen-pixel size of a stacked point-coordinate tag. */
export function referenceTagSize(coordinates: ReferenceCoordinate[]) {
  const widest = coordinates.reduce(
    (longest, coordinate) => Math.max(longest, coordinate.value.length),
    3
  );
  return {
    width:
      PADDING_X * 2 + AXIS_COLUMN + 2 + widest * AXIS_FONT * GLYPH_ADVANCE,
    height: HEADER_HEIGHT + coordinates.length * AXIS_ROW_HEIGHT,
  };
}

interface ReferenceMarkerProps {
  symbol: string;
  coordinates: ReferenceCoordinate[];
  marker: Marker;
  placement: TagPlacement;
  color: ReferenceColor;
  scale: number;
  selected: boolean;
  /** Faded while another point is selected. */
  dimmed: boolean;
  listening?: boolean;
  onSelect: (symbol: string) => void;
}

/** Ring one printed point name and show its coordinates in a nearby tag. */
export function ReferenceMarker({
  symbol,
  coordinates,
  marker,
  placement,
  color,
  scale,
  selected,
  dimmed,
  listening = true,
  onSelect,
}: ReferenceMarkerProps) {
  const { bbox } = marker;
  const unit = (size: number) => size / scale;
  const screen = referenceTagSize(coordinates);
  const width = unit(screen.width);
  const height = unit(screen.height);
  const ring = {
    x: bbox.x - unit(RING_PAD),
    y: bbox.y - unit(RING_PAD),
    width: bbox.width + unit(RING_PAD * 2),
    height: bbox.height + unit(RING_PAD * 2),
  };
  const { x, y } = placement;
  const anchorX = x + width / 2 < ring.x ? x + width : x;
  const anchorY = y + height / 2 < ring.y ? y + height : y;

  return (
    <Group
      opacity={dimmed ? 0.25 : 1}
      listening={listening}
      onClick={() => onSelect(symbol)}
      onTap={() => onSelect(symbol)}
    >
      <Line
        points={[
          ring.x + ring.width / 2,
          ring.y + ring.height / 2,
          anchorX,
          anchorY,
        ]}
        stroke={color.fill}
        strokeWidth={unit(selected ? 1.75 : 1)}
        listening={false}
      />
      <Rect
        x={ring.x}
        y={ring.y}
        width={ring.width}
        height={ring.height}
        stroke={color.fill}
        strokeWidth={unit(selected ? 2.5 : 1.5)}
        cornerRadius={unit(2)}
        listening={false}
      />

      <Rect
        x={x}
        y={y}
        width={width}
        height={height}
        fill="#ffffff"
        stroke="#1f2937"
        strokeWidth={unit(selected ? 1.5 : 0.75)}
        cornerRadius={unit(CORNER)}
        shadowColor="#000000"
        shadowBlur={selected ? unit(6) : 0}
        shadowOpacity={selected ? 0.35 : 0}
        shadowEnabled={selected}
      />
      <Rect
        x={x}
        y={y}
        width={width}
        height={unit(HEADER_HEIGHT)}
        fill={color.fill}
        cornerRadius={[unit(CORNER), unit(CORNER), 0, 0]}
        listening={false}
      />
      <Text
        x={x}
        y={y}
        width={width}
        height={unit(HEADER_HEIGHT)}
        text={symbol}
        fontSize={unit(NAME_FONT)}
        fontStyle="bold"
        fontFamily="Arial, sans-serif"
        fill={color.ink}
        align="center"
        verticalAlign="middle"
        listening={false}
      />

      {coordinates.map((coordinate, row) => {
        const rowY = y + unit(HEADER_HEIGHT + row * AXIS_ROW_HEIGHT);
        return (
          <Group key={coordinate.axis} listening={false}>
            <Text
              x={x + unit(PADDING_X)}
              y={rowY}
              width={unit(AXIS_COLUMN)}
              height={unit(AXIS_ROW_HEIGHT)}
              text={coordinate.axis}
              fontSize={unit(AXIS_FONT)}
              fontStyle="bold"
              fontFamily="ui-monospace, Menlo, Consolas, monospace"
              fill="#6b7280"
              verticalAlign="middle"
            />
            <Text
              x={x + unit(PADDING_X + AXIS_COLUMN)}
              y={rowY}
              width={width - unit(PADDING_X * 2 + AXIS_COLUMN)}
              height={unit(AXIS_ROW_HEIGHT)}
              text={coordinate.value}
              fontSize={unit(AXIS_FONT)}
              fontFamily="ui-monospace, Menlo, Consolas, monospace"
              fill="#111827"
              align="right"
              verticalAlign="middle"
            />
          </Group>
        );
      })}
    </Group>
  );
}
