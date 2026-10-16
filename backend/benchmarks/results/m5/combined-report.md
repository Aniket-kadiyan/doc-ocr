# QTracX combined OCR benchmark

**Suite status:** FAIL

## Combined results

| Measure | Result |
|---|---:|
| Documents passing | 5 / 6 |
| Expected callouts matched | 142 / 203 |
| Partial / missed | 40 / 21 |
| Mean recognition score | 90.3% |
| Semantically exact | 137 / 203 |
| Correct type | 136 / 203 |
| Correct final state | 111 / 203 |
| Unexpected accepted detections | 30 |
| Accounting pages passing | 5 / 6 |
| Accounting issues | 1 |
| Accounting unavailable documents | 0 |
| Total runtime | 549.1 s |

## Per drawing

| Drawing | Status | Matched | Exact | State | Unexpected accepted | Runtime |
|---|---|---:|---:|---:|---:|---:|
| PIPE, EXH PRESSURE, drawing 129E01-13300-ID | PASS | 26 / 34 | 24 / 34 | 21 / 34 | 11 | 182.9 s |
| ASSY HITCH PIN 5/8 x 5, drawing 47630 | FAIL | 18 / 19 | 17 / 19 | 17 / 19 | 1 | 39.6 s |
| ICE Connection Shaft Center, drawing 503645_DES001 R2 | PASS | 32 / 47 | 32 / 47 | 32 / 47 | 7 | 23.4 s |
| INLINE LOCKNUT, drawing 56103-0182 rev B | PASS | 16 / 24 | 15 / 24 | 9 / 24 | 2 | 56.5 s |
| PIPE ASM, EXH PRESS SENSOR, drawing 7475870300 | PASS | 38 / 67 | 37 / 67 | 20 / 67 | 8 | 243.9 s |
| PUNCH, drawing BS1801006.020 | PASS | 12 / 12 | 12 / 12 | 12 / 12 | 1 | 2.7 s |

## Recognition sources

- **Page profiles:** raster: 4, vector: 2
- **Text-layer strategies:** native_text_first: 2, raster_ocr: 4
- **Recognized candidates:** native_pdf: 236, ocr: 782

## Gate failures

- **ASSY HITCH PIN 5/8 x 5, drawing 47630**
  - candidate accounting issues 1 exceed maximum 0
