import { describe, expect, it } from "vitest";
import {
  nextClockwiseRotation,
  normalizePageRotation,
  normalizePageRotations,
  pageRotationFor,
} from "@/lib/pageRotation";

describe("page rotation", () => {
  it("advances through clockwise quarter turns", () => {
    expect(nextClockwiseRotation(0)).toBe(90);
    expect(nextClockwiseRotation(90)).toBe(180);
    expect(nextClockwiseRotation(180)).toBe(270);
    expect(nextClockwiseRotation(270)).toBe(0);
  });

  it("normalizes persisted page maps and drops invalid entries", () => {
    expect(
      normalizePageRotations({
        1: 90,
        2: 0,
        3: 270,
        0: 180,
        bad: 90,
        4: 45,
      })
    ).toEqual({ 1: 90, 3: 270 });
  });

  it("defaults missing and malformed rotations to zero", () => {
    expect(normalizePageRotation("90")).toBe(90);
    expect(normalizePageRotation(45)).toBe(0);
    expect(pageRotationFor({ 2: 180 }, 1)).toBe(0);
    expect(pageRotationFor({ 2: 180 }, 2)).toBe(180);
  });
});
