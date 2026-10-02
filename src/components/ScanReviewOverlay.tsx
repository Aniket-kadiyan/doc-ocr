"use client";

import { Group, Rect, Text } from "react-konva";
import type { ScanCandidate } from "@/types/scanCandidate";

interface ScanReviewOverlayProps {
  candidates: ScanCandidate[];
  scale: number;
  disabled?: boolean;
  selectedCandidateId?: string | null;
  onSelect: (candidate: ScanCandidate) => void;
}

/**
 * Persistent post-scan candidate geometry. Review candidates use amber and
 * filtered "Other" detections use a quieter slate treatment. Ignored records
 * are retained in storage/sidebar but are intentionally absent from this layer.
 */
export function ScanReviewOverlay({
  candidates,
  scale,
  disabled = false,
  selectedCandidateId = null,
  onSelect,
}: ScanReviewOverlayProps) {
  const safeScale = Math.max(scale, 0.01);
  const hasSelection = selectedCandidateId !== null;

  return (
    <Group listening={!disabled}>
      {candidates.map((candidate, index) => {
        const bbox = candidate.valueBox;
        const box = candidate.orientedBox ?? bbox;
        const key =
          candidate.id ||
          `candidate-${bbox.x}-${bbox.y}-${bbox.width}-${bbox.height}-${index}`;
        const selected = candidate.id === selectedCandidateId;
        const review = candidate.state === "review";
        const idleFill = review ? "#d97706" : "#64748b";
        const idleStroke = review ? "#b45309" : "#475569";
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
              fill={selected ? "#2563eb" : idleFill}
              opacity={selected ? 0.28 : review ? 0.14 : 0.08}
              stroke={selected ? "#1d4ed8" : idleStroke}
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
              text={review ? "?" : "·"}
              fill={selected ? "#1d4ed8" : idleStroke}
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
