import { describe, expect, it } from "vitest";
import {
  moveValueAnnotationToNumber,
  renumberValueAnnotations,
} from "@/lib/annotationNumbers";
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

describe("moveValueAnnotationToNumber", () => {
  const annotations = [
    makeAnnotation({ id: "a", number: 1, value: "10" }),
    makeAnnotation({ id: "b", number: 2, value: "20" }),
    makeAnnotation({ id: "c", number: 3, value: "30" }),
    makeAnnotation({ id: "d", number: 4, value: "40" }),
    makeAnnotation({ id: "e", number: 5, value: "50" }),
  ];

  it("moves a later balloon upward and shifts the intervening range", () => {
    const result = moveValueAnnotationToNumber(annotations, "e", 2);

    expect(result.map(({ id }) => id)).toEqual(["a", "b", "c", "d", "e"]);
    expect(result.map(({ id, number }) => ({ id, number }))).toEqual([
      { id: "a", number: 1 },
      { id: "b", number: 3 },
      { id: "c", number: 4 },
      { id: "d", number: 5 },
      { id: "e", number: 2 },
    ]);
    expect(result.find(({ id }) => id === "e")?.value).toBe("50");
  });

  it("moves an earlier balloon downward and shifts the intervening range", () => {
    const result = moveValueAnnotationToNumber(annotations, "b", 5);

    expect(result.map(({ id, number }) => ({ id, number }))).toEqual([
      { id: "a", number: 1 },
      { id: "b", number: 5 },
      { id: "c", number: 2 },
      { id: "d", number: 3 },
      { id: "e", number: 4 },
    ]);
  });

  it("does not change annotations for an invalid target or unknown id", () => {
    expect(moveValueAnnotationToNumber(annotations, "e", 0)).toBe(annotations);
    expect(moveValueAnnotationToNumber(annotations, "missing", 2)).toBe(
      annotations
    );
  });
});
