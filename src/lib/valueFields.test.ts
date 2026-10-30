import { describe, expect, it } from "vitest";
import {
  deriveRange,
  singleDeviationRange,
  stackedDeviationRange,
} from "@/lib/valueFields";

describe("deriveRange", () => {
  it("defaults a plain numeric value to zero tolerance", () => {
    expect(deriveRange("0.02")).toBe("0");
    expect(deriveRange("R25")).toBe("0");
  });

  it("normalizes decimal and asymmetric tolerances", () => {
    expect(deriveRange("58.21±0.05")).toBe("+0.05, -0.05");
    expect(deriveRange("25 +0.1 -0.2")).toBe("+0.1, -0.2");
  });

  it("converts complete angular tolerances to decimal degrees", () => {
    expect(deriveRange("90°±0°30′")).toBe("+0.5, -0.5");
    expect(deriveRange("90°±0°30′15″")).toBe(
      "+0.504167, -0.504167"
    );
  });

  it("reads minutes whose mark the text layer dropped", () => {
    expect(deriveRange("60°±0°30")).toBe("+0.5, -0.5");
  });

  it("ignores counts, notes, dual units and thread classes", () => {
    expect(deriveRange("M54×1.5-6H")).toBe("0");
    expect(deriveRange("4X M5-6H")).toBe("0");
    expect(deriveRange("2-Ø6")).toBe("0");
    expect(deriveRange("10±0.1 (2X)")).toBe("+0.1, -0.1");
    expect(deriveRange("3.2 ±0.1 2PLCS")).toBe("+0.1, -0.1");
    expect(deriveRange("14.99±0.1[.590]")).toBe("+0.1, -0.1");
    expect(deriveRange("Ø10 H7 (+0.015/0)")).toBe("+0.015, -0");
  });

  it("leaves incomplete or malformed tolerance intent empty", () => {
    expect(deriveRange("25 ±")).toBe("");
    expect(deriveRange("25 +0.1 +0.2")).toBe("");
  });
});

describe("stackedDeviationRange", () => {
  it("reads an upper/lower pair after the nominal", () => {
    expect(stackedDeviationRange("Ø8+0.2/0")).toBe("+0.2, -0");
    expect(stackedDeviationRange("21+1/-0.5")).toBe("+1, -0.5");
    expect(stackedDeviationRange("Ø8+0.2/0.2")).toBe("+0.2, -0.2");
  });

  it("does not reinterpret fractions or unsupported pairs", () => {
    expect(stackedDeviationRange("1/1")).toBe("");
    expect(stackedDeviationRange("1 3/8")).toBe("");
    expect(stackedDeviationRange("Ø8-0.1/-0.2")).toBe("");
    expect(stackedDeviationRange("Ø8+0.2/+0.1")).toBe("");
  });
});

describe("singleDeviationRange", () => {
  it("takes the nominal as the other limit", () => {
    expect(deriveRange("Ø37.65+0.01")).toBe("+0.01, -0");
    expect(deriveRange("*Ø33.21-0.01")).toBe("+0, -0.01");
    expect(deriveRange("R0.3+0.05(34x)")).toBe("+0.05, -0");
  });

  it("rejects signs that are not a single deviation", () => {
    expect(singleDeviationRange("+0.5")).toBe("");
    expect(singleDeviationRange("R-5")).toBe("");
    expect(deriveRange("R-5")).toBe("");
    expect(singleDeviationRange("25 +0.1 +0.2")).toBe("");
  });
});
