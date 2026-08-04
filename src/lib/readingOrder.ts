import type { OCRWordBox } from "@/types/annotation";
import { fixEngineeringSymbols } from "@/lib/engineeringSymbols";

const LINE_THRESHOLD = 15;

export type TextOrientation = "horizontal" | "vertical" | "rotated";

export function detectOrientationFromBBox(
  width: number,
  height: number
): TextOrientation {
  const ratio = height / Math.max(width, 1);
  if (ratio > 2.5) return "vertical";
  if (width / Math.max(height, 1) > 2.5) return "horizontal";
  return "horizontal";
}

export function detectOrientationFromAngle(angle: number): TextOrientation {
  const normalized = ((angle % 180) + 180) % 180;
  if (normalized < 15 || normalized > 165) return "horizontal";
  if (normalized > 75 && normalized < 105) return "vertical";
  return "rotated";
}

export function sortReadingOrder(
  boxes: OCRWordBox[],
  orientation: TextOrientation
): OCRWordBox[] {
  const sorted = [...boxes];

  if (orientation === "vertical") {
    return sorted.sort((a, b) => a.y - b.y);
  }

  if (orientation === "rotated") {
    return sorted.sort((a, b) => {
      const distA = a.x + a.y;
      const distB = b.x + b.y;
      return distA - distB;
    });
  }

  return sorted.sort((a, b) => {
    const sameLine = Math.abs(a.y - b.y) < LINE_THRESHOLD;
    if (sameLine) return a.x - b.x;
    return a.y - b.y;
  });
}

export function assembleText(
  boxes: OCRWordBox[],
  orientation: TextOrientation
): string {
  const sorted = sortReadingOrder(boxes, orientation);
  if (sorted.length === 0) return "";

  if (orientation === "vertical") {
    return sorted.map((b) => b.text.trim()).join("");
  }

  const lines: string[] = [];
  let currentLine: OCRWordBox[] = [sorted[0]];

  for (let i = 1; i < sorted.length; i++) {
    const prev = currentLine[currentLine.length - 1];
    const curr = sorted[i];
    if (Math.abs(curr.y - prev.y) < LINE_THRESHOLD) {
      currentLine.push(curr);
    } else {
      lines.push(
        sortReadingOrder(currentLine, "horizontal")
          .map((b) => b.text)
          .join(" ")
          .replace(/\s+/g, " ")
      );
      currentLine = [curr];
    }
  }

  lines.push(
    sortReadingOrder(currentLine, "horizontal")
      .map((b) => b.text)
      .join(" ")
      .replace(/\s+/g, " ")
  );

  return lines.join(" ").replace(/\s+/g, " ").trim();
}

export function normalizeEngineeringText(text: string): string {
  return fixEngineeringSymbols(text);
}
