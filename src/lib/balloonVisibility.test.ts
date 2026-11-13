import { describe, expect, it } from "vitest";
import {
  areAllValueBalloonsHidden,
  areAllValueBalloonsVisible,
  areAnyValueBalloonsVisible,
} from "@/lib/balloonVisibility";

describe("global balloon visibility", () => {
  it("does not report a global state when there are no value balloons", () => {
    expect(areAllValueBalloonsVisible([])).toBe(false);
    expect(areAllValueBalloonsHidden([])).toBe(false);
    expect(areAnyValueBalloonsVisible([])).toBe(false);
  });

  it("reports the fully visible state", () => {
    const annotations = [{ kind: "dimension" as const, hidden: false }, {}];

    expect(areAllValueBalloonsVisible(annotations)).toBe(true);
    expect(areAllValueBalloonsHidden(annotations)).toBe(false);
    expect(areAnyValueBalloonsVisible(annotations)).toBe(true);
  });

  it("reports the fully hidden state used for whiteout", () => {
    const annotations = [
      { kind: "dimension" as const, hidden: true },
      { kind: "dimension" as const, hidden: true },
    ];

    expect(areAllValueBalloonsVisible(annotations)).toBe(false);
    expect(areAllValueBalloonsHidden(annotations)).toBe(true);
    expect(areAnyValueBalloonsVisible(annotations)).toBe(false);
  });

  it("does not white out values while visibility is mixed", () => {
    const annotations = [
      { kind: "dimension" as const, hidden: true },
      { kind: "dimension" as const, hidden: false },
    ];

    expect(areAllValueBalloonsVisible(annotations)).toBe(false);
    expect(areAllValueBalloonsHidden(annotations)).toBe(false);
    expect(areAnyValueBalloonsVisible(annotations)).toBe(true);
  });

  it("ignores legacy label records", () => {
    const annotations = [
      { kind: "dimension" as const, hidden: true },
      { kind: "label" as const, hidden: false },
    ];

    expect(areAllValueBalloonsHidden(annotations)).toBe(true);
  });
});
