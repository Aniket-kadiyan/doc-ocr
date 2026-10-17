# GD&T symbol recognition

Finds feature control frames and GD&T symbols on a drawing, names them, and
reads a frame's compartments into structured fields instead of one flat
string.

This is a separate pass that runs *alongside* OCR. Nothing in the existing
ballooning, scanning or checksheet workflow changes, and nothing here writes
into a project: a GD&T detection carries no balloon number, no tolerance of
its own and no measured result, and it is neither saved into a `.docbox.json`
nor exported. It is shown for an operator to read and correct.

## Two recognition paths

**Graphical** (`gdt_detect`, `fcf_parse`) finds frames and glyphs in pixels.
**Textual** (`gdt_text`, spec section 16) reads controls a drawing spells out
in words — `FLATNESS 0.05`, `POSITION Ø0.1 A B` — which have no glyph for a
template to match and are invisible to the graphical pass by construction.

Section 16 was switched off for most of this project's life, with the reason
recorded only as "marked out of scope in the spec". Establishing it by
measurement rather than assumption: the old rule matched a GD&T *word*
anywhere in a string, and over 170 OCR strings from four real drawings that
gave **no true positives and one false positive** — a 400-character accuracy
note reading `... ROUNDNESS(2PT./180)50µm MAX. ... SQUARENESS WITH FACE 50µm
MAX. ...` would have been published as a perpendicularity control.

The vocabulary was never the problem; the guards were. `gdt_text` requires the
shape of a callout before accepting a term:

- text no longer than 64 characters / 10 words
- not prose, a note block, or a measurement specification (`MAX.`, `MIN.`,
  `µm`, `Ra`, `(2PT.)`)
- an unambiguous glyph, a pipe-delimited frame, **or** a control name with a
  tolerance value beside it
- ambiguous marks never decide: `○`, `//`, `∥`, `⌒`, `∠`, `↗` all mean other
  things on a drawing

Measured: **precision 1.00, recall 1.00** on 18 genuine spelled-out callouts,
and **zero claims** on the same 170 real strings. That is the evidence the
section was enabled on.

Where both paths describe one callout, `link_text_annotations` attaches the
textual read to the frame — `duplicate_of_frame: true` — instead of reporting
it twice, and records whether the two agree.

## How a symbol is recognized

No single method is good enough on its own, and each is used for what it is
good at.

**OCR alone loses.** PaddleOCR is trained on text. A `⌖` comes back as `+`, a
`⌭` as `101`, a `⏥` as `/7` — confidently, so its score cannot be used to
reject them.

**Template matching alone over-fires.** A drawing is full of circles and right
angles, and `matchTemplate` will score a bolt hole as circularity.

**Contour topology alone cannot read a letter.**

So naming is done by **shape comparison at an aligned scale**: the glyph is
cropped to its own ink, padded to a square so its aspect survives, resized to
a fixed box, and compared against templates normalised identically. The whole
outline and the glyph's *interior* are scored separately and blended.

That alignment is the thing that makes it work, and it was measured. Over the
41 templates:

| Method | Named correctly |
|---|---|
| Sliding-window multi-scale matching | 23 / 41 |
| Same templates, aligned scale | 40 / 41 |

Sliding-window matching sent every circled-letter modifier to "all over": with
freedom to choose a scale and an offset, the correlation settles on a smaller
template that fits the glyph's ring and lets the rest fall on blank paper.
Adding interior scoring at weight 0.30 lifted top-1 from 118 to 120 of 123
across clean, blurred and noisy renders and widened the mean margin to the
runner-up from 0.264 to 0.289.

The interior term has one exception, measured on a scanned drawing. The
position template's interior is 82% ink — a cross through a ring leaves
almost no paper in the middle — and a scanned ⌖ fills its middle with a soft
blob. Correlating two near-solid patches measures their noise: every
position frame's outline scored 0.73–0.82 and its interior 0.03–0.30, and the
blend landed at 0.52–0.64, under the accept bar. Where the *template's*
interior is mostly ink, the comparison is how much ink the two carry, as a
ratio; the same nine frames then score 0.75–0.87. Keyed on the template, not
the glyph: the circled modifiers at 18px are rings full of blurred letter,
and judged by ink alone every one of them read as the same modifier.

Topology — ring, slash, nested ring, stroke figure — is measured and reported
as evidence, but it does **not** move a score. It was built as a prior first
and measuring it showed why that was wrong: hole-counting misreads
concentricity's inner ring and all-over's at 40px, so a confident wrong family
penalised correct template matches out of first place.

### Templates are drawn, not shipped

Each glyph is a handful of lines and circles the standard specifies, so
`gdt_templates.py` draws them procedurally — at whatever size and stroke
weight the page needs, with no binary assets, no licence question about the
artwork, and geometry a reviewer can check against the standard by reading the
code. It also keeps the promise not to add a machine-learning dependency
without demonstrating need: nothing here requires training data.

Templates are **ink white on black**, matching `cad_ink_to_gray`. Getting this
backwards is not subtle — a correctly shaped template scores about 0.08
against its own glyph inverted, and 1.0 the right way up.

## The ± matcher in `symbol_vision`

Separate from this pass but on the same preprocessing path, and fixed in the
same change. `symbol_vision._match_pm` had two faults that compounded:

1. `_make_pm_template` drew **black ink on white** while `_to_gray` (via
   `cad_ink_to_gray`) produces **white ink on black**. A correlation between
   inverted images anti-correlates.
2. The score was taken over the whole padded zone. `pad_image` adds a 36px
   white border, which becomes a large black field after the ink conversion,
   so the correlation was dominated by background agreement rather than by the
   glyph.

Neither could be fixed alone. Measured over 86 real regions from four drawings
(23 with a ±, 63 without):

| Configuration | threshold | precision | recall |
|---|---|---|---|
| as written | 0.34 | 1.00 | **0.04** |
| polarity fixed only | any | ≤0.26 | — |
| polarity + ink-scoped | 0.50 | 0.76 | 0.70 |
| polarity + ink-scoped | **0.60** | **0.83** | **0.22** |
| polarity + ink-scoped | 0.65 | 1.00 | 0.09 |

Fixing the polarity *alone* makes things worse: with a black-background
template the padding matches everywhere, and negatives outscore positives.

`PM_THRESHOLD` moved from 0.34 to **0.60**, chosen for precision rather than
F1. The reason is what the flag is for: `detect_symbols`' result is OR-ed with
`detect_prefix_from_ocr_text`, and the OCR text path reads ± on **all 23** real
regions — it survives 5px blur, σ=30 noise and half resolution at 23/23. On
deliberately degraded crops where OCR finally dropped the glyph, the matcher
rescued **1 of 16**. So its marginal recall is near zero, and 0.60 buys a
fivefold recall improvement on a dead signal for one false positive across 63
real negatives, leaving merged precision at 0.96.

The template's *geometry* was also suspect — its vertical runs past the lower
bar, so it renders closer to a `∓` than a `±`. A faithful glyph was built and
measured, and it was **worse** (median gap +0.105 against +0.145, best F1 0.52
against 0.73), because the template is never matched against an isolated glyph
but against a whole ink-cropped value line such as `3,41±0.30`. It was left
alone, with that recorded in the code.

## Three answers, not two

Every detection is `confirmed`, `ambiguous`, or `unknown`.

`ambiguous` exists because some pairs **are the same mark**. Counterbore and
spotface are drawn identically; profile-of-a-line and arc length share one
glyph; `Ø` and `SØ` differ only by a letter. Shape cannot separate those, and
reporting the more common one would be a guess dressed as an answer. They come
back with both candidates attached and a warning saying why.

A detection also goes ambiguous when a rival is within `GDT_RIVAL_MARGIN` of
the best score, even above the accept bar. "It is one of these two" is the
honest answer when two readings score within a hair of each other.

## Context resolves, it never overrules

`gdt_context.py` settles what position can settle:

- `⌒` is profile-of-a-line inside a frame, arc length above a dimension.
- `⌴` is a counterbore unless an `SF` annotation makes it a spotface.
- A circle is circularity in a frame's first compartment, a diameter before a
  value, all-around on a leader kink, and a drilled hole everywhere else.
- `R` is a radius before a number, and part of a word in `RA`/`Rc`/`REF`.
- A capital letter is a datum reference inside a frame, and a datum feature
  only when a triangle is attached.

**Context only touches a detection the detector left ambiguous.** A confirmed
reading is never swapped. That guard is load-bearing: without it, the rule "a
circle in compartment 0 is circularity" fired whenever circularity appeared
anywhere in the alternatives, and rewrote a confident profile-of-a-line
reading (shape 0.87, runner-up 0.59) to circularity on the strength of
circularity sitting fourth in its list.

Geometric characteristics found *outside* any frame are demoted to ambiguous
and told why. This is the biggest single source of false positives: a drawing
is full of right angles and parallel lines, and every one correlates with
perpendicularity or parallelism.

## Feature control frames

A frame is the one place on a drawing where GD&T is unambiguous, so it is not
just another thing to detect — it is the context that makes the rest safe.

Frames are found from their **rules**, not from text and not from a contour.
Two horizontal rules one frame-height apart with the same extent, closed by
a column of ink at each end and divided by at least one more, are a frame —
whatever else touches them. The detected rectangle is split into composite
segments at its full-width internal rules (the segments of a composite frame
share side rules, so they always detect as one rectangle), then into
compartments at the vertical rules.

That last clause is the point, and it was measured on a scanned drawing with
13 frames (nine position controls, two flatness, a two-segment and a
three-segment composite):

| Finder | Frames found at 2x |
|---|---|
| Outer contour of the ruled grid (as shipped) | 2 of 13 |
| Enclosed compartments found as holes | 7 of 13 |
| Rule pairs | **13 of 13** at 1x, 2x, 3x and 4x |

The contour finder lost frames two ways. Nearly every frame on a drawing is
joined to the part by a leader, and the leader's horizontal run is a rule as
far as morphology is concerned, so the outer contour of a frame was the outer
contour of the whole part outline. And the page binarizer's despeckle drops
components smaller than 2% of the *largest* one, which on a glyph crop is
scanner grain and on a whole page is every frame border and tolerance digit
on the sheet — the largest component is the part outline. The finder reads
the page without despeckling; the rule morphology ignores specks on its own.

Finding compartments as holes of the ruling was tried next and lost every
frame whose tolerance digits rested on the bottom rule, which on a scan they
do. Rule pairs do not care what is drawn between the rules.

Two things the rule pairs needed that a clean render never shows:

- **A rule is a dense row, not a component.** Text resting on a rule joins
  its connected component, and on a page read enlarged the digits between
  two rules join them into one component the height of the frame. Each run
  of rows at least half as dense as the densest is taken as its own rule.
- **A divider leaves room beside it.** The stem of a ⟂ runs nearly the full
  height of a scanned frame and was taken for the divider after the symbol,
  which cut the symbol in half. No compartment is narrower than 70% of the
  frame's height, so a column of ink that would leave one is a stroke. A
  divider drawn one weight lighter than the borders is kept when it runs
  from rule to rule without a break or has clear paper either side; the
  upright of a digit does neither.

The rule length and the smallest frame are fractions of the **page**, not of
the tile being read: judged from a 1400px tile of a 2000px page they came
out small enough to let a 21px box through as a frame.

The finder over-produces by design — every boxed basic dimension and ruled
table row is a candidate — and the parser turns a box away after reading its
*first* compartment, so a candidate that opens with no characteristic costs
one OCR call. A box nested inside a parsed frame (the legs and crossbar of a
datum "A" pair with the rule above them) gives way to the frame. What is
left is a few boxes from the part outline whose first cell holds a stroke
that passes for straightness or perpendicularity; those are published only
when the symbol is confirmed or a tolerance under 5 was read beside it, and
flagged for review otherwise.

The frame's own lines are **erased** before any compartment is read, rather
than cropped around. Cropping cannot be made to work: an inset wide enough to
clear a border at one frame size clips a modifier drawn flush to a divider at
another, and both failures are silent. Erasure uses a length threshold of 80%
of the whole frame for horizontal rules, so the straightness glyph — a
horizontal stroke filling its own compartment — is never mistaken for one;
the vertical rules erased are exactly the dividers the compartment split
found, so a ⟂ stem is not. The erased band is grown with the rule's own
thickness, because a rule's antialiased halo is below one threshold and
above the next, and what one binarization left the glyph crop's own
binarization picked up as ink at the corner of the cell.

OCR of the symbol compartment is used only when it returns the glyph itself.
PaddleOCR reads a ⌖ as "0", "4", "2" or "中", and a bare "0" corroborates the
ring-shaped symbols: two position frames came back as circularity,
*confirmed*, on the strength of that one character.

### Missing means three different things

| State | Meaning |
|---|---|
| `present` | Read, with a value. |
| `absent` | The frame genuinely has no such field. A flatness callout has no datum, and that is correct. |
| `unreadable` | Something is there and it could not be read. |
| `not_applicable` | The field cannot apply to this control. |

This distinction is not decoration. For a material condition, `absent` means
RFS — a specific callout with a specific acceptance limit — while `unreadable`
means nobody knows yet. Collapsing them silently deletes a bonus tolerance an
inspector is entitled to.

### Units are never invented

`tolerance_unit` is always `drawing_unit`. Nothing in OCR establishes whether
a sheet is in millimetres or inches; that comes from the drawing's title
block, a units note, or a dual-unit bracket. A unit that was never read is an
invention someone might act on.

## Resolution is the main accuracy lever

Measured on synthetic frames with OCR enabled, per-frame field accuracy
(symbol, tolerance, material condition, datum sequence):

| Frame height | Fields correct |
|---|---|
| 44 px | 92% |
| 66 px | 100% |
| 88 px | 100% |

At 44px the failures are a datum letter misread by OCR and a modifier below
the resolution floor — a circled `Ⓜ` at that frame height is about 20px with a
10px letter inside it. The recognizer correctly refuses to name it and reports
the field `unreadable` rather than `absent`.

This is why the frontend re-renders the page at `GDT_RENDER_SCALE` before
reading it, the same way the reference-point pass does, instead of using the
displayed canvas.

A raster upload — a photo, a screenshot — arrives at whatever size it was
saved at, and the page scan reads it as is. `analyze_page` enlarges a page
whose shorter edge is under `MIN_WORK_SHORT_EDGE_PX` (2000) by up to 3x
before reading its frames: on a 1291px-wide sheet a frame is 14px tall and
its symbol 9px, below the floor at which confusable glyphs can be told apart.
The loose-symbol pass stays at the page's own size, where its glyph-size
limits were tuned; run over the enlarged page it kept 3 of 28 glyphs on the
confusables benchmark. The adaptive binarization threshold moved from 120px
to 400px for the same reason: a frame crop at 3x is 140px, and an adaptive
block an eighth of that is narrower than a stroke and hollows it out.

## Skew

Compartment splitting and glyph crops are axis-aligned, so the page is
deskewed first and every box is mapped back, leaving all published
coordinates in the caller's original space. The contour-based finder this
replaced found nothing at all on a page skewed by 2.5°; the rule-pair finder
tolerates that much on its own, and deskewing stays for what comes after it.

The skew estimate uses the **long lines** — border, frames, dimension lines —
which is the opposite of `image_preprocess.estimate_skew_angle`, which
suppresses long lines to find a text baseline inside a small crop.

## In the page scan

Frames reach the ballooning flow through `gdt_pipeline.scan_regions`, which
`segment_page` runs after its own detection. The text detector the scan is
built on finds *text*: handed `⌖ | Ø0.3` it returns a box around `0.3` and
publishes a linear dimension with no tolerance, and the rest of the frame
comes back as a review box no OCR could read. Each frame found here replaces
everything the scan produced inside it and is published as one region of
type `GD&T` whose text is the callout — `⌖ Ø0.3 A B`, the registry glyph
then the fields — with the structured read under a `gdt` key. The segments
of a composite frame are one region, because they are one requirement.

A region needs review when the symbol is not confirmed, the tolerance could
not be read, or a modifier is present and unreadable; every reason is
listed, not the first. The ring-shaped "unreadable modifier" is only looked
for in the trailing 40% of the tolerance compartment, where a modifier
stands — the digit 0 is a ring too, and judged anywhere in the cell every
tolerance under 1 sent its frame to review.

## API

All routes are new; no existing response shape changed.

| Route | Purpose |
|---|---|
| `GET /gdt/registry` | The whole symbol registry |
| `GET /gdt/registry/{id}` | One symbol, with its confusions resolved |
| `GET /gdt/config` | The thresholds in force |
| `POST /ocr/gdt-symbols` | Recognize one rendered page |
| `POST /gdt/corrections` | Record an operator correction |
| `GET /gdt/corrections` | Read corrections back |
| `GET /gdt/corrections/summary` | Counts, and which pairs are actually confused |

`/ocr/recognize` gains one **optional** query flag, `gdt=1`, off by default.
When set, the response carries an extra `gdt` key alongside everything it
already returned. A failure there degrades that one key to an error record and
leaves the OCR result intact.

## Configuration

Read from the project's own `.env.local` / `.env` through `config_env`, so the
backend and frontend draw from one file. All default sensibly; set none of
them and the pass works.

```dotenv
GDT_ENABLED=true                  # run the pass at all
GDT_ACCEPT_CONFIDENCE=0.62        # publish a named symbol at or above this
GDT_AMBIGUOUS_CONFIDENCE=0.42     # report nothing at all below this
GDT_RIVAL_MARGIN=0.06             # a rival this close makes it ambiguous
GDT_TEMPLATE_MIN_SCORE=0.52
GDT_MIN_GLYPH_PX=9                # below this, confusables cannot be told apart
GDT_MAX_GLYPH_FRAC=0.12           # a "symbol" larger than this is geometry
GDT_TILE_PX=1400                  # page tiling for large drawings
GDT_TILE_OVERLAP_PX=180           # must exceed the largest symbol
GDT_MERGE_IOU=0.55                # overlap at which two detections are one
GDT_CORRECTIONS_DIR=backend/data/gdt_corrections
```

The two confidence bars are deliberately *two*. Above `accept`, a symbol is
named. Below `ambiguous`, nothing is reported. Between them it is published
with `status="ambiguous"` and its alternatives attached — an uncertain glyph
is never promoted to a definitive engineering symbol, and never silently
dropped either, because an operator reviewing the sheet wants to see that
something is there.

## Standards

Recognition and interpretation are kept apart.

- Retired controls are **recorded, not deleted**. Concentricity and symmetry
  were removed in ASME Y14.5-2018 and are still current in ISO and on any
  drawing released to an earlier revision. They are recognized, and flagged
  with the revision that retired them.
- ASME-only (`independency`, `controlled_radius`) and ISO-only
  (`envelope_requirement`) symbols say so.
- A symbol found on a drawing declared to a different standard is **reported,
  not rewritten**.
- Nothing declares a drawing compliant or non-compliant. That needs the whole
  drawing, its revision and its notes; OCR establishes none of them.

Derived concepts — virtual condition, bonus tolerance, RFS, the datum
reference frame, Rule #1 — are in the registry as `kind: concept` and are
**not detectable**. They are real and important, and none of them is drawn.
Reporting a bonus tolerance as a found glyph would be a fabrication.

## Corrections

An operator who sees a wrong detection is the best labelling source this
system will ever have. Corrections append to JSON Lines under
`GDT_CORRECTIONS_DIR` and record only the symbol, box, page and an optional
document hash — **no image data and no file contents**, so a correction ties
back to a drawing without this store holding one.

`GET /gdt/corrections/summary` reports which symbol pairs are actually
confused in service, which is a better guide to what to improve than any
synthetic benchmark.

## Evaluation

See [GDT_DATASET.md](GDT_DATASET.md) for what is measured, the honest limits
of the synthetic suite, and the annotation workflow for real drawings.

```bash
cd backend
PYTHONPATH=. .venv/bin/python scripts/run_gdt_eval.py --ocr --save
```

Macro F1 by condition, before and after the rule-pair finder, the page-scale
binarization fixes and the position interior rule (2026-10-16, no OCR):

| Condition | Before | After | Frames found |
|---|---|---|---|
| clean | 0.80 | 0.83 | 13/13 |
| blurred | 0.64 | 0.80 | 12/13 |
| noisy | 0.77 | 0.84 | 13/13 |
| low_res | 0.47 | 0.81 | 13/13 |
| skewed | 0.55 | 0.68 | 13/13 |
| scan | 0.61 | 0.79 | 13/13 |
| poor_scan | 0.30 | 0.56 | 7/13 (was 3/13) |

Seven symbol compartments cropped from a scanned drawing (position,
flatness, perpendicularity, straightness, parallelism) sit beside the
synthetic suite in `benchmarks/fixtures/glyphs/` and must classify confirmed.

## Source map

| File | Purpose |
|---|---|
| `backend/gdt_registry.py` | The 59-entry symbol registry |
| `backend/gdt_templates.py` | Procedural glyph artwork |
| `backend/gdt_detect.py` | Normalisation, matching, confidence, merging |
| `backend/fcf_parse.py` | Frame detection, segmentation, field parsing |
| `backend/gdt_context.py` | Context disambiguation and standards checks |
| `backend/gdt_pipeline.py` | Tiling, deskew, orchestration, error isolation |
| `backend/gdt_config.py` | Thresholds, via `config_env` |
| `backend/gdt_text.py` | Guarded textual GD&T (spec section 16) |
| `backend/gdt_corrections.py` | Operator-correction store |
| `backend/gdt_routes.py` | FastAPI routes |
| `backend/benchmarks/gdt_synthetic.py` | Scene generator with exact truth |
| `backend/benchmarks/gdt_eval.py` | Per-class scoring |
| `src/lib/gdtSymbols.ts` | Presentation helpers |
| `src/lib/gdtClient.ts` | API client and coordinate mapping |
| `src/components/GdtOverlay.tsx` | Konva overlay |
| `src/components/GdtPanel.tsx` | Review and correction panel |
