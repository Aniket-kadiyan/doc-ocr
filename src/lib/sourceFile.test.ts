import { describe, expect, it } from "vitest";
import { detectSourceFileKind } from "@/lib/sourceFile";

function file(bytes: number[] | string, name: string, type = ""): File {
  const blob = new Blob(
    [typeof bytes === "string" ? new TextEncoder().encode(bytes) : new Uint8Array(bytes)],
    { type }
  );
  Object.defineProperty(blob, "name", { value: name });
  return blob as File;
}

describe("source drawing format detection", () => {
  it("uses signatures when Windows supplies no MIME type", async () => {
    await expect(
      detectSourceFileKind(file([0x89, 0x50, 0x4e, 0x47, 13, 10, 26, 10], "a.png"))
    ).resolves.toBe("png");
    await expect(
      detectSourceFileKind(file([0xff, 0xd8, 0xff, 0xe0], "a.jpg"))
    ).resolves.toBe("jpeg");
    await expect(
      detectSourceFileKind(file([0x49, 0x49, 0x2a, 0x00], "a.tif"))
    ).resolves.toBe("tiff");
  });

  it("recognizes PDFs even when the header follows a short preamble", async () => {
    await expect(
      detectSourceFileKind(file("\n%PDF-1.7\n", "drawing.pdf", "application/pdf"))
    ).resolves.toBe("pdf");
  });

  it("rejects misleading extensions and unknown data", async () => {
    await expect(
      detectSourceFileKind(
        file([0x89, 0x50, 0x4e, 0x47, 13, 10, 26, 10], "wrong.jpg", "image/jpeg")
      )
    ).rejects.toThrow(/contains PNG data/i);
    await expect(
      detectSourceFileKind(file("not an image", "drawing.tif"))
    ).rejects.toThrow(/Unsupported drawing format/i);
  });

  it("keeps the existing WebP and BMP inputs supported", async () => {
    await expect(
      detectSourceFileKind(file("RIFF0000WEBP", "drawing.webp", "image/webp"))
    ).resolves.toBe("webp");
    await expect(
      detectSourceFileKind(file("BM0000", "drawing.bmp", "image/bmp"))
    ).resolves.toBe("bmp");
  });
});
