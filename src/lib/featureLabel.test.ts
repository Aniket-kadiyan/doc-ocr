import { describe, expect, it } from "vitest";
import { balloonLabel, labelForDimensionType } from "@/lib/featureLabel";
import { makeAnnotation } from "@/test/annotationFixture";

describe("balloon labels", () => {
  it("matches the backend rule engine for each category", () => {
    // The names here are the ones feature_classifier._label_for returns, so a
    // rescan overwrites a derived label with an identical one.
    expect(labelForDimensionType("Linear")).toBe("Linear Dimension");
    expect(labelForDimensionType("Diameter")).toBe("Diameter");
    expect(labelForDimensionType("Reference")).toBe("Reference Dimension");
    expect(labelForDimensionType("Basic")).toBe("Basic Dimension");
    expect(labelForDimensionType("Tolerance")).toBe("Toleranced Dimension");
    expect(labelForDimensionType("GD&T")).toBe("Feature Control Frame");
    expect(labelForDimensionType("General Note")).toBe("General Note");
    expect(labelForDimensionType("Unknown")).toBe("Feature");
  });

  it("keeps a label the scan or a person supplied", () => {
    expect(
      balloonLabel(makeAnnotation({ label: "4X Through Hole", type: "Hole" }))
    ).toBe("4X Through Hole");
  });

  it("falls back to the category for a balloon that carries none", () => {
    // Every balloon scanned before the page route published the rule label,
    // and every balloon drawn by hand, arrives with an empty label.
    expect(balloonLabel(makeAnnotation({ label: "", type: "Diameter" }))).toBe(
      "Diameter"
    );
    expect(
      balloonLabel(makeAnnotation({ label: "   ", type: "Linear" }))
    ).toBe("Linear Dimension");
  });
});
