export type SourceFileKind =
  | "pdf"
  | "png"
  | "jpeg"
  | "tiff"
  | "webp"
  | "dwg"
  | "dxf";

const EXTENSION_KIND: Record<string, SourceFileKind> = {
  pdf: "pdf",
  png: "png",
  jpg: "jpeg",
  jpeg: "jpeg",
  tif: "tiff",
  tiff: "tiff",
  webp: "webp",
  dwg: "dwg",
  dxf: "dxf",
};

const MIME_KIND: Record<string, SourceFileKind> = {
  "application/pdf": "pdf",
  "image/png": "png",
  "image/jpeg": "jpeg",
  "image/jpg": "jpeg",
  "image/tiff": "tiff",
  "image/x-tiff": "tiff",
  "image/webp": "webp",
  "image/vnd.dwg": "dwg",
  "application/acad": "dwg",
  "image/vnd.dxf": "dxf",
  "application/dxf": "dxf",
};

function bytesEqual(bytes: Uint8Array, offset: number, expected: number[]) {
  return expected.every((value, index) => bytes[offset + index] === value);
}

function signatureKind(bytes: Uint8Array): SourceFileKind | null {
  const ascii = new TextDecoder("latin1").decode(bytes);
  const pdfHeader = ascii.indexOf("%PDF-");
  if (pdfHeader >= 0 && pdfHeader <= 1024) return "pdf";
  if (bytesEqual(bytes, 0, [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])) {
    return "png";
  }
  if (bytesEqual(bytes, 0, [0xff, 0xd8, 0xff])) return "jpeg";
  if (
    bytesEqual(bytes, 0, [0x49, 0x49, 0x2a, 0x00]) ||
    bytesEqual(bytes, 0, [0x4d, 0x4d, 0x00, 0x2a]) ||
    bytesEqual(bytes, 0, [0x49, 0x49, 0x2b, 0x00]) ||
    bytesEqual(bytes, 0, [0x4d, 0x4d, 0x00, 0x2b])
  ) {
    return "tiff";
  }
  if (ascii.startsWith("RIFF") && ascii.slice(8, 12) === "WEBP") {
    return "webp";
  }
  // Every DWG opens with its version tag, e.g. "AC1015" for AutoCAD 2000.
  if (/^AC(10\d\d|1\.40|2\.)/.test(ascii)) return "dwg";
  if (ascii.startsWith("AutoCAD Binary DXF")) return "dxf";
  // ASCII DXF is group-code/value pairs and begins with "0 / SECTION".
  if (/^\s*0\s*\r?\n\s*SECTION\s*\r?\n/.test(ascii)) return "dxf";
  return null;
}

function declaredKind(file: File): SourceFileKind | null {
  const extension = file.name.split(".").pop()?.toLowerCase() ?? "";
  const byExtension = EXTENSION_KIND[extension];
  const mime = file.type.split(";", 1)[0].trim().toLowerCase();
  return byExtension ?? MIME_KIND[mime] ?? null;
}

/** Validate the file bytes first; Windows may provide no useful MIME type. */
export async function detectSourceFileKind(file: File): Promise<SourceFileKind> {
  if (file.size === 0) throw new Error("The selected drawing file is empty.");
  const head = new Uint8Array(await file.slice(0, 2048).arrayBuffer());
  const actual = signatureKind(head);
  if (!actual) {
    throw new Error(
      "Unsupported drawing format. Choose a PDF, DWG, DXF, PNG, JPG/JPEG, TIF/TIFF, or WebP file."
    );
  }
  const declared = declaredKind(file);
  if (declared && declared !== actual) {
    throw new Error(
      `The selected file contains ${actual.toUpperCase()} data but is named or labelled as ${declared.toUpperCase()}.`
    );
  }
  return actual;
}

export const isPdfKind = (kind: SourceFileKind) => kind === "pdf";
export const isTiffKind = (kind: SourceFileKind) => kind === "tiff";
export const isCadKind = (kind: SourceFileKind) =>
  kind === "dwg" || kind === "dxf";
