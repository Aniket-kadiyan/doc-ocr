import { describe, expect, it } from "vitest";
import {
  emptySectionScanSummary,
  runSectionScanQueue,
  type QueuedScanSection,
} from "@/lib/sectionScanQueue";

const sections: QueuedScanSection[] = [
  {
    id: "one",
    page: 1,
    bbox: { x: 10, y: 10, width: 50, height: 30 },
  },
  {
    id: "two",
    page: 1,
    bbox: { x: 70, y: 10, width: 50, height: 30 },
  },
  {
    id: "three",
    page: 1,
    bbox: { x: 130, y: 10, width: 50, height: 30 },
  },
];

describe("ordered section scan queue", () => {
  it("finishes and publishes each section before starting the next", async () => {
    const events: string[] = [];
    const result = await runSectionScanQueue({
      sections,
      runSection: async (section, position) => {
        events.push("run:" + section.id);
        return {
          status: "succeeded" as const,
          summary: {
            ...emptySectionScanSummary(),
            added: position.current,
            detected: position.current + 1,
            reviewRequired: position.current,
          },
        };
      },
      afterSection: async (section, position) => {
        events.push("paint:" + section.id + ":" + position.current);
      },
    });

    expect(events).toEqual([
      "run:one",
      "paint:one:1",
      "run:two",
      "paint:two:2",
      "run:three",
      "paint:three:3",
    ]);
    expect(result).toMatchObject({
      status: "succeeded",
      completedSections: 3,
      summary: { added: 6, detected: 9, reviewRequired: 3 },
    });
  });

  it("keeps completed sections and does not start queued work after cancellation", async () => {
    const started: string[] = [];
    const published: string[] = [];
    const result = await runSectionScanQueue({
      sections,
      runSection: async (section) => {
        started.push(section.id);
        if (section.id === "two") return { status: "cancelled" as const };
        return {
          status: "succeeded" as const,
          summary: { ...emptySectionScanSummary(), added: 2 },
        };
      },
      afterSection: async (section) => {
        published.push(section.id);
      },
    });

    expect(started).toEqual(["one", "two"]);
    expect(published).toEqual(["one"]);
    expect(result).toMatchObject({
      status: "cancelled",
      completedSections: 1,
      summary: { added: 2 },
    });
  });
});
