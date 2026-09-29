"use client";

import { Group, Rect, Text } from "react-konva";
import type { SegmentRegion } from "@/lib/paddleOcrClient";

interface ScanReviewOverlayProps<T extends SegmentRegion> {
  candidates: T[];
  scale: number;
  disabled?: boolean;
  selectedCandidateId?: string | null;
  onSelect: (candidate: T) => void;
}

/**
 * Post-scan review geometry. These boxes are deliberately separate from
 * annotations, so they are never persisted or exported until the user accepts
 * one through the existing value/tolerance popup.
 */
export function ScanReviewOverlay<T extends SegmentRegion>({
  candidates,
  scale,
  disabled = false,
  selectedCandidateId = null,
  onSelect,
}: ScanReviewOverlayProps<T>) {
  const safeScale = Math.max(scale, 0.01);
  const hasSelection = selectedCandidateId !== null;

  return (
    <Group listening={!disabled}>
      {candidates.map((candidate, index) => {
        const bbox = candidate.valueBox;
        const box = candidate.orientedBox ?? bbox;
        const key =
          candidate.candidateId ??
          `review-${bbox.x}-${bbox.y}-${bbox.width}-${bbox.height}-${index}`;
        const selected = candidate.candidateId === selectedCandidateId;
        return (
          <Group
            key={key}
            opacity={hasSelection && !selected ? 0.15 : 1}
          >
            <Rect
              x={box.x}
              y={box.y}
              width={box.width}
              height={box.height}
              rotation={candidate.orientedBox?.rotation ?? 0}
              fill={selected ? "#2563eb" : "#64748b"}
              opacity={selected ? 0.28 : 0.12}
              stroke={selected ? "#1d4ed8" : "#475569"}
              strokeWidth={(selected ? 4 : 2.5) / safeScale}
              dash={[6 / safeScale, 4 / safeScale]}
              shadowColor="#2563eb"
              shadowBlur={selected ? 10 / safeScale : 0}
              shadowOpacity={selected ? 0.9 : 0}
              shadowEnabled={selected}
              onClick={() => onSelect(candidate)}
              onTap={() => onSelect(candidate)}
              onMouseEnter={(event) => {
                const stage = event.target.getStage();
                if (stage) stage.container().style.cursor = "pointer";
              }}
              onMouseLeave={(event) => {
                const stage = event.target.getStage();
                if (stage) stage.container().style.cursor = "default";
              }}
            />
            <Text
              x={bbox.x + 2 / safeScale}
              y={bbox.y + 1 / safeScale}
              text="?"
              fill={selected ? "#1d4ed8" : "#334155"}
              fontStyle="bold"
              fontSize={(selected ? 16 : 12) / safeScale}
              listening={false}
            />
          </Group>
        );
      })}
    </Group>
  );
}
