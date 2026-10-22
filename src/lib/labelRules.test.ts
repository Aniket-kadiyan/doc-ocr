import { inflateSync } from "node:zlib";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  describeLabelGaps,
  findLabelGaps,
  relabelForEdit,
  ruleForValue,
  titleFieldAnnotations,
  titleFieldFor,
} from "@/lib/labelRules";
import { buildInspectionSheet } from "@/lib/project";
import { exportInspectionCSV, exportInspectionJSON, exportXML } from "@/lib/export";
import { buildInspectionWorkbook } from "@/lib/excelExport";
import { buildBalloonedPdf } from "@/lib/balloonedPdf";
import { makeAnnotation } from "@/test/annotationFixture";
import type { Annotation } from "@/types/annotation";
import type { TitleField } from "@/lib/paddleOcrClient";

// The configured "Keywords to look for". "DWG NO." is the case the old
// export matching lost: the field is read as "DWG".
vi.mock("@/lib/titleKeywords", () => ({ getTitleKeywords: () => ["DWG NO.", "REV"] }));

// The backend rule engine, as /ocr/classify answers for these texts.
const RULES: Record<string, { category: string; label: string }> = {
  "Ø12": { category: "Diameter", label: "Diameter" },
  "R5": { category: "Radius", label: "Radius" },
  "4X Ø6.6 THRU": { category: "Hole", label: "Through Hole" },
  "0": { category: "Linear", label: "Linear Dimension" },
};

function stubRules() {
  const fetchMock = vi.fn(async (_url: string, init?: RequestInit) => {
    const { texts } = JSON.parse(String(init?.body)) as { texts: string[] };
    return {
      ok: true,
      status: 200,
      json: async () => ({
        results: texts.map((text) => ({ text, ...(RULES[text] ?? { category: "Unknown", label: "" }) })),
      }),
    };
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

// ── the labelling rules applied to typed and accepted values ─────────────

describe("labelling a value that did not come from a scan", () => {
  it("reuses the verdict a read already carries while the text is unchanged", async () => {
    const fetchMock = stubRules();
    const verdict = await ruleForValue("Ø12", { category: "Diameter", label: "Diameter" }, "Ø12");
    expect(verdict).toEqual({ category: "Diameter", label: "Diameter" });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("asks the rules again once the value was corrected", async () => {
    stubRules();
    // The read said Ø12; the inspector corrected it to R5.
    const verdict = await ruleForValue("R5", { category: "Diameter", label: "Diameter" }, "Ø12");
    expect(verdict).toEqual({ category: "Radius", label: "Radius" });
  });

  it("labels a value of zero: falsy is not empty", async () => {
    stubRules();
    expect(await ruleForValue("0")).toEqual({ category: "Linear", label: "Linear Dimension" });
  });

  it("returns nothing, rather than a guess, when the rules cannot be reached", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 503, json: async () => ({}) })));
    expect(await ruleForValue("Ø12")).toBeNull();
  });

  it("gives an empty value no label", async () => {
    const fetchMock = stubRules();
    expect(await ruleForValue("   ")).toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("editing a value", () => {
  beforeEach(() => {
    stubRules();
  });

  it("leaves an unchanged value's label and type alone", async () => {
    const result = await relabelForEdit({ value: "Ø12", label: "Diameter" }, "Ø12", "Diameter");
    expect(result).toEqual({ label: "Diameter" });
  });

  it("follows the rules when the label was still the rules' own", async () => {
    const result = await relabelForEdit({ value: "Ø12", label: "Diameter" }, "R5", "Diameter");
    expect(result).toEqual({ label: "Radius", type: "Radius" });
  });

  it("fills in a missing label when the value is edited", async () => {
    const result = await relabelForEdit({ value: "Ø12", label: "" }, "R5", "");
    expect(result.label).toBe("Radius");
  });

  it("never replaces a label the inspector wrote", async () => {
    const typed = await relabelForEdit({ value: "Ø12", label: "Bore" }, "R5", "Bore");
    expect(typed.label).toBe("Bore");
    const editedNow = await relabelForEdit({ value: "Ø12", label: "Diameter" }, "R5", "Corner");
    expect(editedNow.label).toBe("Corner");
  });
});

// ── title-block fields ────────────────────────────────────────────────────

const field = (keyword: string, label: string, value: string): TitleField => ({
  keyword,
  label,
  value,
  confidence: 0.9,
  bbox: value ? { x: 900, y: 700, width: 80, height: 20 } : null,
});

let ids = 0;
const nextId = () => `tf-${++ids}`;

describe("title-block fields", () => {
  it("are linked to the keyword they were read for", () => {
    const made = titleFieldAnnotations([field("DWG NO.", "DWG", "47630")], [], 1, 1, nextId);
    expect(made).toHaveLength(1);
    expect(made[0]).toMatchObject({ titleKeyword: "DWG NO.", label: "DWG", value: "47630", type: "Title Block" });
    // The export finds it by keyword, though the read label is "DWG" — the
    // old prefix match on the label lost exactly this field.
    expect(titleFieldFor(made, "DWG NO.")?.value).toBe("47630");
  });

  it("are not duplicated by a second scan, and a missed keyword makes no annotation", () => {
    const first = titleFieldAnnotations([field("REV", "REV", "2"), field("DATE", "DATE", "")], [], 1, 1, nextId);
    expect(first.map((a) => a.titleKeyword)).toEqual(["REV"]);
    const again = titleFieldAnnotations([field("REV", "REV", "2")], first, 1, 2, nextId);
    expect(again).toEqual([]);
  });

  it("still match older projects that recorded no keyword", () => {
    const legacy = [makeAnnotation({ id: "old", type: "Title Block", label: "DWG NO.", value: "GEP5A34" })];
    expect(titleFieldFor(legacy, "DWG")?.value).toBe("GEP5A34");
    expect(titleFieldFor(legacy, "DWG NO.")?.value).toBe("GEP5A34");
  });
});

// ── completeness ──────────────────────────────────────────────────────────

describe("label completeness", () => {
  it("reports values without a label and keywords without a value, by reference only", () => {
    const annotations = [
      makeAnnotation({ id: "a", number: 1, value: "Ø12", label: "Diameter" }),
      makeAnnotation({ id: "b", number: 2, value: "0", label: "" }),
      makeAnnotation({ id: "c", number: 3, value: "", label: "" }),
      makeAnnotation({ id: "t", number: 4, value: "2", label: "REV", type: "Title Block", titleKeyword: "REV" }),
    ];
    const gaps = findLabelGaps(annotations, ["REV", "DWG"]);
    // "0" is a value and needs its label; the empty one legitimately has none.
    expect(gaps).toEqual([
      { kind: "unlabelled_value", ref: "2" },
      { kind: "title_keyword_unread", ref: "DWG" },
    ]);
    expect(describeLabelGaps(gaps)).toBe("1 value has no label (balloon 2); title field not read: DWG.");
    expect(describeLabelGaps(findLabelGaps(annotations.slice(0, 1), []))).toBe("");
  });
});

// ── every export carries the same labels and values ──────────────────────

/** A drawing that exercises the cases that went missing. */
function drawing(): Annotation[] {
  return [
    makeAnnotation({ id: "v1", number: 1, value: "Ø12", label: "Diameter", type: "Diameter" }),
    makeAnnotation({ id: "v2", number: 2, value: "0", label: "Linear Dimension", range: "0" }),
    makeAnnotation({ id: "v3", number: 3, value: "4X Ø6.6 THRU", label: 'Through Hole, "c/bore" & <spec>', type: "Hole" }),
    makeAnnotation({ id: "v4", number: 4, value: "1. FREE FROM BURRS\n2. BREAK EDGES", label: "General Notes", type: "General Note", range: undefined }),
    makeAnnotation({ id: "v5", number: 5, value: "R5", label: "Radius", type: "Radius" }),
    makeAnnotation({ id: "t1", number: 6, value: "47630", label: "DWG", type: "Title Block", titleKeyword: "DWG NO.", range: undefined }),
  ];
}

/** The (label, value) pairs of a sheet's balloon rows, in row order. */
function pairs(rows: string[][]): string[] {
  return rows.filter((row) => /^\d+$/.test(row[0] ?? "")).map((row) => `${row[1]} = ${row[2]}`);
}

describe("exports", () => {
  it("carry every label and value, in order, in CSV, JSON, XML, Excel and PDF", async () => {
    const annotations = drawing();
    const sheet = buildInspectionSheet(annotations);
    const expected = pairs(sheet.rows);
    // The sheet itself: one row per value, nothing dropped or merged, the
    // falsy "0" kept, the multiline note on one line.
    expect(expected).toEqual([
      "Diameter = Ø12",
      "Linear Dimension = 0",
      'Through Hole, "c/bore" & <spec> = 4X Ø6.6 THRU',
      "General Notes = 1. FREE FROM BURRS 2. BREAK EDGES",
      "Radius = R5",
      "DWG = 47630",
    ]);
    // The title-field section: the read field by its keyword, and the
    // keyword that was not read as a blank row rather than no row.
    const titleRows = sheet.rows.filter((row) => row[0] === "" && row[1]).slice(0, 2);
    expect(titleRows.map((row) => [row[1], row[2]])).toEqual([["DWG", "47630"], ["REV", ""]]);

    const json = JSON.parse(exportInspectionJSON(annotations)) as { data: Record<string, string>[] };
    expect(pairs(json.data.map((row) => [row["S.no"], row.Label, row.Value]))).toEqual(expected);

    const csvRows = exportInspectionCSV(annotations)
      .split("\n")
      .slice(6)
      .map((line) => [...line.matchAll(/"((?:[^"]|"")*)"/g)].map((m) => m[1].replace(/""/g, '"')));
    expect(pairs(csvRows)).toEqual(expected);

    const xmlRows = [...exportXML(annotations).matchAll(/<row>([\s\S]*?)<\/row>/g)].map((match) =>
      [...match[1].matchAll(/<cell name="[^"]*">([\s\S]*?)<\/cell>/g)].map((cell) =>
        cell[1].replace(/&quot;/g, '"').replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&")
      )
    );
    expect(pairs(xmlRows)).toEqual(expected);

    const workbook = await buildInspectionWorkbook({
      annotations,
      pages: [],
      options: {
        fileName: "Labels",
        partCount: 1,
        metadata: { partName: "", documentNumber: "", revisionNumber: "" },
      },
    });
    const worksheet = workbook.getWorksheet("Checksheet")!;
    const excelRows: string[][] = [];
    worksheet.eachRow((row) => {
      const values = row.values as unknown[];
      excelRows.push([1, 2, 3].map((i) => String(values[i] ?? "")));
    });
    expect(pairs(excelRows)).toEqual(expected);

    const pdf = await buildBalloonedPdf({
      projectName: "Labels",
      metadata: { partName: "", documentNumber: "", revisionNumber: "" },
      pages: [],
      annotations,
    });
    const text = pdfText(Buffer.from(await pdf.arrayBuffer()));
    for (const pair of expected) {
      const [label, value] = pair.split(" = ");
      expect(text).toContain(label);
      expect(text).toContain(value);
    }
  });
});

/** Every text string drawn in a PDF, from its inflated content streams. */
function pdfText(bytes: Buffer): string {
  const raw = bytes.toString("latin1");
  const strings: string[] = [];
  for (const match of raw.matchAll(/stream\r?\n([\s\S]*?)\r?\nendstream/g)) {
    let content: string;
    try {
      content = inflateSync(Buffer.from(match[1], "latin1")).toString("latin1");
    } catch {
      continue;
    }
    for (const literal of content.matchAll(/\(((?:\\.|[^\\)])*)\)\s*Tj/g)) {
      strings.push(literal[1].replace(/\\([()\\])/g, "$1"));
    }
    for (const hex of content.matchAll(/<([0-9A-Fa-f]+)>\s*Tj/g)) {
      const h = hex[1];
      let decoded = "";
      for (let i = 0; i + 3 < h.length; i += 4) decoded += String.fromCharCode(parseInt(h.slice(i, i + 4), 16));
      strings.push(decoded);
    }
  }
  return strings.join("\n");
}
