import { describe, expect, it } from "vitest";
import { deriveRange } from "@/lib/valueFields";

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
