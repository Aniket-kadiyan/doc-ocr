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

  it("leaves incomplete or malformed tolerance intent empty", () => {
    expect(deriveRange("90°±0°30")).toBe("");
    expect(deriveRange("25 ±")).toBe("");
    expect(deriveRange("25 +0.1 +0.2")).toBe("");
  });
});

describe("stackedDeviationRange", () => {
  it("reads an upper/lower pair after the nominal", () => {
    expect(stackedDeviationRange("Ø8+0.2/0")).toBe("+0.2, -0");
    expect(stackedDeviationRange("Ø32.3+0.2/0")).toBe("+0.2, -0");
    expect(stackedDeviationRange("21+1/-0.5")).toBe("+1, -0.5");
    expect(stackedDeviationRange("21°+1°/0")).toBe("+1, -0");
    // The drawn minus is lost from the text layer.
    expect(stackedDeviationRange("Ø8+0.2/0.2")).toBe("+0.2, -0.2");
    expect(deriveRange("Ø8+0.2/0")).toBe("+0.2, -0");
  });

  it("leaves what is not a deviation pair, or does not fit the field", () => {
    expect(stackedDeviationRange("1/1")).toBe("");
    expect(stackedDeviationRange("1 3/8")).toBe("");
    expect(stackedDeviationRange("Ø8-0.1/-0.2")).toBe("");
    expect(stackedDeviationRange("Ø8+0.2/+0.1")).toBe("");
  });
});

describe("singleDeviationRange", () => {
  it("takes the nominal as the other limit of a one-sided deviation", () => {
    expect(deriveRange("Ø37.65+0.01")).toBe("+0.01, -0");
    expect(deriveRange("*Ø37.65+0.01")).toBe("+0.01, -0");
    expect(deriveRange("*Ø33.21-0.01")).toBe("+0, -0.01");
    expect(deriveRange("R0.6 -0.1")).toBe("+0, -0.1");
    // The count printed after it is not part of the tolerance.
    expect(deriveRange("R0.3+0.05(34x)")).toBe("+0.05, -0");
  });

  it("leaves what is not one deviation after a nominal", () => {
    expect(singleDeviationRange("+0.5")).toBe("");
    expect(singleDeviationRange("R-5")).toBe("");
    expect(singleDeviationRange("25 +0.1 +0.2")).toBe("");
    expect(deriveRange("25 +0.1 +0.2")).toBe("");
    expect(deriveRange("●Ø37.50")).toBe("0");
  });
});
