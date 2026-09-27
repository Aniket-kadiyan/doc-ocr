"use client";

import { Group, Rect, Text } from "react-konva";
import type {
  ScanDebugOverlay as ScanDebugOverlayModel,
  ScanOverlayCandidateState,
  ScanOverlayPanelState,
} from "@/types/scanJob";

interface ScanDebugOverlayProps {
  overlay: ScanDebugOverlayModel;
  scale: number;
}

const panelColor: Record<ScanOverlayPanelState, string> = {
  pending: "#2563eb",
  active: "#16a34a",
  completed: "#64748b",
};

const candidateColor: Record<ScanOverlayCandidateState, string> = {
  detected: "#475569",
  recovering: "#7c3aed",
  eligible: "#16a34a",
  excluded: "#f97316",
  unread: "#94a3b8",
  review: "#64748b",
};

/**
 * Temporary, non-exportable processing geometry.  It lives in the same base
 * page coordinate system as annotations, so Stage zoom/pan keeps it aligned.
 */
export function ScanDebugOverlay({ overlay, scale }: ScanDebugOverlayProps) {
  const safeScale = Math.max(scale, 0.01);
  return (
    <Group listening={false}>
      {overlay.overlaps.map((bbox, index) => (
        <Rect
          key={`scan-overlap-${index}`}
          {...bbox}
          fill="#f59e0b"
          opacity={0.14}
        />
      ))}

      {overlay.tableMasks.map((bbox, index) => (
        <Rect
          key={`scan-table-${index}`}
          {...bbox}
          fill="#dc2626"
          stroke="#b91c1c"
          strokeWidth={1.5 / safeScale}
          opacity={0.2}
        />
      ))}

      {overlay.panels.map((panel) => {
        const color = panelColor[panel.state];
        const active = panel.state === "active";
        return (
          <Group key={`scan-panel-${panel.id}`}>
            <Rect
              x={panel.x}
              y={panel.y}
              width={panel.width}
              height={panel.height}
              stroke={color}
              strokeWidth={(active ? 4 : 2) / safeScale}
              dash={active ? undefined : [10 / safeScale, 6 / safeScale]}
              opacity={active ? 1 : 0.78}
            />
            <Text
              x={panel.x + 5 / safeScale}
              y={panel.y + 5 / safeScale}
              text={panel.label}
              fontSize={14 / safeScale}
              fontStyle="bold"
              fill={color}
              padding={2 / safeScale}
            />
          </Group>
        );
      })}

      {overlay.candidates.map((candidate, index) => {
        const color = candidateColor[candidate.state];
        return (
          <Rect
            key={candidate.id ?? `scan-candidate-${index}`}
            x={candidate.bbox.x}
            y={candidate.bbox.y}
            width={candidate.bbox.width}
            height={candidate.bbox.height}
            stroke={color}
            strokeWidth={2 / safeScale}
            dash={
              candidate.state === "detected"
                ? [4 / safeScale, 3 / safeScale]
                : undefined
            }
            opacity={0.95}
          />
        );
      })}
    </Group>
  );
}
