import { afterEach, describe, expect, it, vi } from "vitest";
import { readPdfTextLayer, regionsWithin } from "@/lib/pdfTextLayer";
import type { SegmentRegion } from "@/lib/paddleOcrClient";

afterEach(() => {
  vi.unstubAllGlobals();
});

const PAGE = { x: 0, y: 0, width: 892, height: 1263 };
const pdf = () => new Blob(["%PDF-1.4"], { type: "application/pdf" });

const reply = (body: unknown, ok = true) =>
  vi.fn(async () => ({ ok, status: ok ? 200 : 500, json: async () => body }));

describe("reading a PDF text layer", () => {
  it("returns the callouts a vector drawing carries", async () => {
    const fetchMock = reply({
      usable: true,
      fonts: ["AIGDT", "ArialMT"],
      regions: [
        {
          bbox: { x: 500, y: 100, width: 61, height: 21 },
          text: "Ø33-0.05-0.1",
          confidence: 1,
          type: "diameter",
          category: "Diameter",
          label: "Diameter",
          recognized: true,
          needs_review: false,
        },
      ],
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await readPdfTextLayer(pdf(), 1, PAGE);

    expect(result.usable).toBe(true);
    expect(result.fonts).toContain("AIGDT");
    expect(result.regions).toHaveLength(1);
    // Boxes arrive in the viewer's own pixels, so they are used as they are.
    expect(result.regions[0].valueBox).toEqual({
      x: 500,
      y: 100,
      width: 61,
      height: 21,
    });
    expect(result.regions[0].label).toBe("Diameter");
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringMatching(/\/ocr\/text-layer$/),
      expect.objectContaining({ method: "POST" })
    );
  });

  it("reports a scanned drawing as unusable so the caller scans it", async () => {
    vi.stubGlobal("fetch", reply({ usable: false, regions: [], fonts: [] }));

    expect(await readPdfTextLayer(pdf(), 1, PAGE)).toEqual({
      usable: false,
      regions: [],
      fonts: [],
    });
  });

  it("treats a page with no callouts as unusable", async () => {
    vi.stubGlobal("fetch", reply({ usable: true, regions: [], fonts: ["Arial"] }));

    expect((await readPdfTextLayer(pdf(), 1, PAGE)).usable).toBe(false);
  });

  it("never throws: an old backend or a network failure just means scanning", async () => {
    vi.stubGlobal("fetch", reply({ detail: "Not Found" }, false));
    expect((await readPdfTextLayer(pdf(), 1, PAGE)).usable).toBe(false);

    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new Error("offline");
      })
    );
    expect((await readPdfTextLayer(pdf(), 1, PAGE)).usable).toBe(false);
  });
});

describe("limiting callouts to a drawn selection", () => {
  const region = (x: number, y: number): SegmentRegion =>
    ({
      text: "25",
      confidence: 1,
      orientation: "horizontal",
      rotation: 0,
      needsReview: false,
      recognized: true,
      valueBox: { x, y, width: 40, height: 12 },
    }) as SegmentRegion;

  it("keeps the ones whose box sits inside it", () => {
    const scope = { x: 100, y: 100, width: 200, height: 200 };
    const kept = regionsWithin(
      [region(120, 120), region(500, 500), region(280, 280)],
      scope
    );

    expect(kept.map((r) => r.valueBox.x)).toEqual([120, 280]);
  });
});
