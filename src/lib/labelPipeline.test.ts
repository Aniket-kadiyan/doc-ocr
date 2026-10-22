import { describe, expect, it, vi } from "vitest";
import { mapPageSegmentRegions, type ApiSegmentRegion } from "@/lib/paddleOcrClient";
import { annotationsFromRegions } from "@/lib/scanAnnotations";
import { buildInspectionSheet } from "@/lib/project";
import { exportInspectionCSV, exportInspectionJSON, exportXML } from "@/lib/export";
import { buildInspectionWorkbook } from "@/lib/excelExport";
import { findLabelGaps } from "@/lib/labelRules";
import { renumberValueAnnotations } from "@/lib/annotationNumbers";
import scan from "@/test/fixtures/ts2-page-scan.json";

vi.mock("@/lib/titleKeywords", () => ({ getTitleKeywords: () => [] }));

/**
 * One real whole-page scan, end to end: the API response for TS2 (a scanned
 * drawing, read by the route that used to publish its values unlabelled),
 * through the client mapping, the annotations, and every export.
 *
 * Each stage is compared with the one before it, so a label lost anywhere is
 * caught at the stage that lost it.
 */
describe("labels on a real scanned drawing", () => {
  const api = scan as { regions: ApiSegmentRegion[]; page_width: number; page_height: number };
  const page = { x: 0, y: 0, width: api.page_width, height: api.page_height };

  it("reach every stage and every export", async () => {
    // Stage 1: the API. Every value carries the rules' label.
    const apiPairs = api.regions
      .filter((r) => r.text.trim())
      .map((r) => r.label ?? "");
    expect(apiPairs.length).toBeGreaterThan(20);
    expect(apiPairs.filter((label) => !label.trim())).toEqual([]);

    // Stage 2: the client mapping keeps them.
    const mapped = mapPageSegmentRegions(api.regions, page);
    expect(mapped.map((r) => r.label)).toEqual(api.regions.map((r) => r.label));

    // Stage 3: the annotations the drawing view lists.
    const annotations = renumberValueAnnotations(
      annotationsFromRegions(mapped, 1, 1).map((a, i) => ({ ...a, number: i + 1 }))
    );
    expect(findLabelGaps(annotations, [])).toEqual([]);
    const expected = annotations
      .slice()
      .sort((a, b) => a.number - b.number)
      .map((a) => `${a.label} = ${a.value.replace(/\s*\n\s*/g, " ")}`);

    // Stage 4: every export, row for row, in order.
    const pairsOf = (rows: string[][]) =>
      rows.filter((row) => /^\d+$/.test(row[0] ?? "")).map((row) => `${row[1]} = ${row[2]}`);
    expect(pairsOf(buildInspectionSheet(annotations).rows)).toEqual(expected);

    const json = JSON.parse(exportInspectionJSON(annotations)) as { data: Record<string, string>[] };
    expect(pairsOf(json.data.map((row) => [row["S.no"], row.Label, row.Value]))).toEqual(expected);

    const csv = exportInspectionCSV(annotations)
      .split("\n")
      .slice(6)
      .map((line) => [...line.matchAll(/"((?:[^"]|"")*)"/g)].map((m) => m[1].replace(/""/g, '"')));
    expect(pairsOf(csv)).toEqual(expected);

    const xml = [...exportXML(annotations).matchAll(/<row>([\s\S]*?)<\/row>/g)].map((match) =>
      [...match[1].matchAll(/<cell name="[^"]*">([\s\S]*?)<\/cell>/g)].map((cell) =>
        cell[1].replace(/&quot;/g, '"').replace(/&apos;/g, "'").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&")
      )
    );
    expect(pairsOf(xml)).toEqual(expected);

    const workbook = await buildInspectionWorkbook({
      annotations,
      pages: [],
      options: { fileName: "TS2", partCount: 1, metadata: { partName: "", documentNumber: "", revisionNumber: "" } },
    });
    const excel: string[][] = [];
    workbook.getWorksheet("Checksheet")!.eachRow((row) => {
      const values = row.values as unknown[];
      excel.push([1, 2, 3].map((i) => String(values[i] ?? "")));
    });
    expect(pairsOf(excel)).toEqual(expected);
  });
});

describe("a feature control frame", () => {
  it("keeps its symbols, modifier and adjacent text from the API to every export", async () => {
    // The profile frame of 3015-100-2232F as the text-layer route returns it.
    const [mapped] = mapPageSegmentRegions(
      [
        {
          bbox: { x: 514.1, y: 263.1, width: 65.7, height: 17.7 },
          text: "⌓ 0.2 Ⓤ",
          confidence: 0.64,
          type: "GD&T",
          category: "GD&T",
          subtype: "Profile of a Surface",
          label: "GD&T Profile of a Surface · OUTWARDS",
        },
      ],
      { x: 0, y: 0, width: 1785, height: 1262 }
    );
    // The symbol fixer that tidies dimension text must not touch a frame.
    expect(mapped.text).toBe("⌓ 0.2 Ⓤ");
    const annotations = annotationsFromRegions([mapped], 1, 1).map((a) => ({ ...a, number: 1 }));
    expect(annotations[0]).toMatchObject({ value: "⌓ 0.2 Ⓤ", label: "GD&T Profile of a Surface · OUTWARDS" });

    const row = (rows: string[][]) => rows.find((r) => r[0] === "1")?.slice(1, 3);
    const expected = ["GD&T Profile of a Surface · OUTWARDS", "⌓ 0.2 Ⓤ"];
    expect(row(buildInspectionSheet(annotations).rows)).toEqual(expected);
    const json = JSON.parse(exportInspectionJSON(annotations)) as { data: Record<string, string>[] };
    expect([json.data[0].Label, json.data[0].Value]).toEqual(expected);
    expect(exportInspectionCSV(annotations)).toContain('"GD&T Profile of a Surface · OUTWARDS","⌓ 0.2 Ⓤ"');
    expect(exportXML(annotations)).toContain('<cell name="Value">⌓ 0.2 Ⓤ</cell>');
    const workbook = await buildInspectionWorkbook({
      annotations,
      pages: [],
      options: { fileName: "Frame", partCount: 1, metadata: { partName: "", documentNumber: "", revisionNumber: "" } },
    });
    const values: string[] = [];
    workbook.getWorksheet("Checksheet")!.eachRow((r) => values.push(String((r.values as unknown[])[3] ?? "")));
    expect(values).toContain("⌓ 0.2 Ⓤ");
  });
});
