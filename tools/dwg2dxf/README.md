# dwg2dxf-acadsharp

Converts a DWG drawing to DXF so `backend/cad_documents.py` can draw it to PDF.
It is built on [ACadSharp](https://github.com/DomCR/ACadSharp) (MIT) and reads
DWG from R14 to AutoCAD 2018+ (AC1014–AC1032), keeping model space, every
paper-space layout and their viewports.

Copyleft or proprietary converters (LibreDWG, the ODA File Converter) are
deliberately not used; every dependency here is MIT licensed.

## Build

Needs the .NET 10 SDK only on the build machine (`brew install dotnet`, or see
https://dotnet.microsoft.com/download). The published binary is
self-contained, so servers need no .NET runtime.

```sh
cd tools/dwg2dxf
dotnet publish -c Release -r linux-x64 --self-contained -p:PublishSingleFile=true -o dist/linux-x64
dotnet publish -c Release -r osx-arm64 --self-contained -p:PublishSingleFile=true -o dist/osx-arm64
dotnet publish -c Release -r win-x64   --self-contained -p:PublishSingleFile=true -o dist/win-x64
```

The backend looks for the converter in this order:

1. `DWG2DXF_ACADSHARP`, an explicit path to the binary;
2. `dwg2dxf-acadsharp` on `PATH`;
3. `tools/dwg2dxf/dist/<runtime id>/dwg2dxf-acadsharp`, the publish output above.

Without it, DXF and PDF uploads still work and DWG uploads are rejected with
a message saying the converter is missing.

## Usage

```sh
dwg2dxf-acadsharp input.dwg output.dxf
```

Exit code 0 means converted (objects ACadSharp does not know are skipped and
counted on stderr), 1 means the drawing could not be read or written, and 2
means wrong arguments.
