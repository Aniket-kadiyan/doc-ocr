import { afterEach, describe, expect, it, vi } from "vitest";
import {
  planTextLayerRefinement,
  readPdfTextLayer,
  regionsWithin,
} from "@/lib/pdfTextLayer";
import type { SegmentRegion } from "@/lib/paddleOcrClient";
import type { Annotation } from "@/types/annotation";

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
      titleFields: [],
    });
  });

  it("asks for the fast read with the title keywords, the refined one without", async () => {
    const fetchMock = reply({ usable: false, regions: [], fonts: [] });
    vi.stubGlobal("fetch", fetchMock);

    await readPdfTextLayer(pdf(), 1, PAGE, { refine: false });
    await readPdfTextLayer(pdf(), 1, PAGE);

    const forms = fetchMock.mock.calls.map(
      (call) => (call as unknown as [string, { body: FormData }])[1].body
    );
    expect(forms[0].get("refine")).toBe("false");
    expect(forms[0].get("keywords")).toBeTruthy();
    expect(forms[1].get("refine")).toBe("true");
    expect(forms[1].get("keywords")).toBeNull();
  });

  it("asks for no feature control frames when GD&T reading is off", async () => {
    const fetchMock = reply({ usable: false, regions: [], fonts: [] });
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("window", new EventTarget());
    vi.stubGlobal("CustomEvent", class extends Event {});
    const items = new Map<string, string>();
    vi.stubGlobal("localStorage", {
      getItem: (key: string) => items.get(key) ?? null,
      setItem: (key: string, value: string) => void items.set(key, value),
    });
    const { setGdtScanEnabled } = await import("@/lib/scanSettings");

    await readPdfTextLayer(pdf(), 1, PAGE);
    setGdtScanEnabled(false);
    await readPdfTextLayer(pdf(), 1, PAGE);
    setGdtScanEnabled(true);

    const forms = fetchMock.mock.calls.map(
      (call) => (call as unknown as [string, { body: FormData }])[1].body
    );
    expect(forms[0].get("gdt")).toBe("true");
    expect(forms[1].get("gdt")).toBe("false");
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

describe("folding the refined read into the fast one", () => {
  const box = (x: number) => ({ x, y: 10, width: 40, height: 12 });
  const region = (candidateId: string, text: string, x: number): SegmentRegion =>
    ({
      candidateId,
      text,
      confidence: 1,
      orientation: "horizontal",
      rotation: 0,
      needsReview: false,
      recognized: true,
      valueBox: box(x),
    }) as SegmentRegion;
  const balloon = (id: string, value: string, x: number): Annotation =>
    ({ id, value, bbox: box(x), number: 1, page: 1, kind: "dimension" }) as Annotation;

  const placed = new Map([
    ["tl-0", { annotationId: "a", value: "0.2", bbox: box(0) }],
    ["tl-1", { annotationId: "b", value: "45", bbox: box(100) }],
    ["tl-2", { annotationId: "c", value: "25", bbox: box(200) }],
  ]);
  const fastIds = new Set(["tl-0", "tl-1", "tl-2"]);
  const refined = [
    region("tl-1", "1×45°", 80),
    region("tl-2", "25", 200),
    region("fcf-0", "⌓ 0.2 Ⓤ", 0),
  ];

  it("drops what a frame absorbed, rereads changed values and adds the frame", () => {
    const plan = planTextLayerRefinement(placed, fastIds, refined, [
      balloon("a", "0.2", 0),
      balloon("b", "45", 100),
      balloon("c", "25", 200),
    ]);

    expect(plan.remove).toEqual(["a"]);
    expect(plan.update.map((item) => [item.id, item.region.text])).toEqual([
      ["b", "1×45°"],
    ]);
    expect(plan.add.map((item) => item.text)).toEqual(["⌓ 0.2 Ⓤ"]);
  });

  it("leaves a balloon the inspector edited, moved or deleted alone", () => {
    const plan = planTextLayerRefinement(placed, fastIds, refined, [
      balloon("a", "0.25", 0),
      { ...balloon("b", "45", 100), bbox: box(110) },
    ]);

    expect(plan.remove).toEqual([]);
    expect(plan.update).toEqual([]);
  });
});
