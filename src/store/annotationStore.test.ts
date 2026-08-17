import { beforeEach, describe, expect, it } from "vitest";
import { useAnnotationStore } from "@/store/annotationStore";
import { makeAnnotation } from "@/test/annotationFixture";

describe("annotationStore numbering", () => {
  beforeEach(() => {
    useAnnotationStore.setState({
      annotations: [],
      pending: null,
      editingValueId: null,
    });
  });

  it("preserves the user-drawn box when adding a value", () => {
    const bbox = { x: 11, y: 22, width: 83, height: 19 };
    useAnnotationStore
      .getState()
      .addAnnotation(makeAnnotation({ id: "drawn", bbox }));

    expect(useAnnotationStore.getState().annotations[0].bbox).toEqual(bbox);
  });

  it("renumbers remaining values after deletion and reuses the next number", () => {
    useAnnotationStore.getState().setAnnotations([
      makeAnnotation({ id: "a", number: 1 }),
      makeAnnotation({ id: "b", number: 2 }),
      makeAnnotation({ id: "c", number: 3 }),
      makeAnnotation({ id: "d", number: 4 }),
    ]);
    useAnnotationStore.setState({ editingValueId: "b" });

    useAnnotationStore.getState().removeAnnotation("b");

    const state = useAnnotationStore.getState();
    expect(state.annotations.map(({ id, number }) => ({ id, number }))).toEqual([
      { id: "a", number: 1 },
      { id: "c", number: 2 },
      { id: "d", number: 3 },
    ]);
    expect(state.editingValueId).toBeNull();
    expect(state.getNextNumber()).toBe(4);
  });

  it("normalizes gaps when loading annotations", () => {
    useAnnotationStore.getState().setAnnotations([
      makeAnnotation({ id: "one", number: 1 }),
      makeAnnotation({ id: "three", number: 3 }),
      makeAnnotation({ id: "nine", number: 9 }),
    ]);

    expect(
      useAnnotationStore
        .getState()
        .annotations.map(({ number }) => number)
    ).toEqual([1, 2, 3]);
  });

  it("adds an automatic scan batch in one state update with contiguous numbers", () => {
    useAnnotationStore
      .getState()
      .setAnnotations([makeAnnotation({ id: "manual", number: 1 })]);
    let updates = 0;
    const unsubscribe = useAnnotationStore.subscribe(() => {
      updates += 1;
    });

    useAnnotationStore.getState().addAnnotations([
      makeAnnotation({ id: "auto-a", number: 0 }),
      makeAnnotation({ id: "auto-b", number: 0 }),
    ]);
    unsubscribe();

    expect(updates).toBe(1);
    expect(
      useAnnotationStore
        .getState()
        .annotations.map(({ id, number }) => ({ id, number }))
    ).toEqual([
      { id: "manual", number: 1 },
      { id: "auto-a", number: 2 },
      { id: "auto-b", number: 3 },
    ]);
  });

  it("moves a balloon to a new number in one state update", () => {
    useAnnotationStore.getState().setAnnotations([
      makeAnnotation({ id: "a", number: 1 }),
      makeAnnotation({ id: "b", number: 2 }),
      makeAnnotation({ id: "c", number: 3 }),
      makeAnnotation({ id: "d", number: 4 }),
    ]);
    let updates = 0;
    const unsubscribe = useAnnotationStore.subscribe(() => {
      updates += 1;
    });

    useAnnotationStore.getState().moveAnnotationToNumber("d", 2);
    unsubscribe();

    expect(updates).toBe(1);
    expect(
      useAnnotationStore
        .getState()
        .annotations.map(({ id, number }) => ({ id, number }))
    ).toEqual([
      { id: "a", number: 1 },
      { id: "b", number: 3 },
      { id: "c", number: 4 },
      { id: "d", number: 2 },
    ]);
  });

  it("uses the same hidden field for master and individual visibility", () => {
    useAnnotationStore.getState().setAnnotations([
      makeAnnotation({ id: "visible" }),
      makeAnnotation({ id: "hidden", hidden: true }),
    ]);

    useAnnotationStore.getState().setAllBalloonVisibility(true);
    expect(
      useAnnotationStore.getState().annotations.map((item) => item.hidden)
    ).toEqual([false, false]);

    useAnnotationStore.getState().updateAnnotation("visible", { hidden: true });
    expect(useAnnotationStore.getState().annotations[0].hidden).toBe(true);

    useAnnotationStore.getState().setAllBalloonVisibility(false);
    expect(
      useAnnotationStore.getState().annotations.map((item) => item.hidden)
    ).toEqual([true, true]);
  });
});
