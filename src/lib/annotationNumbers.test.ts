import { describe, expect, it } from "vitest";
import { renumberValueAnnotations } from "@/lib/annotationNumbers";
import { makeAnnotation } from "@/test/annotationFixture";

describe("renumberValueAnnotations", () => {
  it("makes value numbers contiguous while preserving array order and ids", () => {
    const input = [
      makeAnnotation({ id: "third", number: 3 }),
      makeAnnotation({ id: "first", number: 1 }),
      makeAnnotation({ id: "fifth", number: 5 }),
    ];

    const result = renumberValueAnnotations(input);

    expect(result.map(({ id }) => id)).toEqual(["third", "first", "fifth"]);
    expect(result.map(({ number }) => number)).toEqual([2, 1, 3]);
  });

  it("ignores legacy label records when allocating visible numbers", () => {
    const result = renumberValueAnnotations([
      makeAnnotation({ id: "value-a", number: 2 }),
      makeAnnotation({ id: "label", number: 1, kind: "label" }),
      makeAnnotation({ id: "value-b", number: 4 }),
    ]);

    expect(result.find(({ id }) => id === "value-a")?.number).toBe(1);
    expect(result.find(({ id }) => id === "value-b")?.number).toBe(2);
    expect(result.find(({ id }) => id === "label")?.number).toBe(1);
  });

  it("returns the original array when no changes are needed", () => {
    const input = [
      makeAnnotation({ id: "a", number: 1 }),
      makeAnnotation({ id: "b", number: 2 }),
    ];

    expect(renumberValueAnnotations(input)).toBe(input);
  });
});
