# QTracX PDFium Windows test-fixture fix

Apply this after `QTracX_MVP2_Chitra_Port_1` on branch
`mvp2-chitra-new-feature-port`.

This changes only `backend/test_pdf_text_layer.py`. The runtime detection code
is unchanged.

The synthetic PDF test previously embedded raw, uncompressed pixel bytes.
Some Windows PDFium builds did not expose that hand-built image as a page
object. The fixture now uses the standard `FlateDecode` image stream, matching
normal PDF image encoding and making the test portable across platforms.

Copy the included `backend` folder over the repository's `backend` folder,
then run:

```powershell
cd backend
.\.venv-gpu\Scripts\python.exe -m pytest -q `
  test_dimension_digits.py test_pdf_text_layer.py `
  test_text_layer_scan.py test_page_scan.py `
  test_gdt_text.py test_gdt_context.py
```

Expected result: `187 passed, 9 skipped` when the optional drawing fixtures
are not installed.

Suggested commit message:

`test(pdf): make embedded-image fixture portable on Windows PDFium`
