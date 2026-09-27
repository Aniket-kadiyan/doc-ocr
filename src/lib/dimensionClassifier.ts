import type { DimensionType } from "@/types/annotation";

/**
 * Text-only mirror of the backend GD&T rule engine
 * (backend/feature_classifier.py). Used for manually-typed values and as a
 * fallback when the backend didn't return a `category`. The backend remains
 * authoritative because it also sees detected symbols + geometry.
 *
 * Priority order matches the spec's decision engine.
 */

const DIAMETER_GLYPHS = /[Øø⌀φΦ∅]/;
const RE_THREAD =
  /\b(M\d+(?:\s*[×xX]\s*\d+(?:\.\d+)?)?|\d+\s*-\s*\d+\s*(?:UNC|UNF|UNEF)|UNC|UNF|UNEF|NPT|NPTF|BSP|BSPT|BSPP|G\s*\d+(?:\/\d+)?|PT\s*\d+)\b/i;
const RE_CHAMFER = /\d+(?:\.\d+)?\s*[×xX]\s*(?:45|30|60)\s*°?/;
const RE_SURFACE = /\b(?:Ra|Rz|Rmax|Rq)\s*\d|\bN\s?(?:1[0-2]|[1-9])\b/i;
const RE_ANGLE = /\d+\s*°|\d+\s*DEG\b/i;
const RE_TOLERANCE = /±|\+\/-/;
const RE_REFERENCE = /^\(.*\)$|(?<![A-Z])REF(?:\.|ERENCE)?\b/i;

const HOLE_MODIFIERS =
  /\b(THRU|THROUGH|CBORE|COUNTERBORE|CSK|COUNTERSINK|SPOTFACE|DEEP|DEPTH|DRILL|REAM|TAP|TAPPED)\b/i;
const NOTE_KEYWORDS =
  /\b(NOTE|REMOVE|BURR|BURRS|EDGE|UNLESS|SPECIFIED|OTHERWISE|DO NOT SCALE|BREAK|SHARP|TYP)\b/i;
const MATERIAL_KEYWORDS =
  /\b(MATERIAL|MAT'?L|ASTM|AISI|SAE|STEEL|ALUMINI?UM|BRASS|BRONZE|CAST IRON|SUJ2|FORGED|EN\d+|AL\d{4}|S45C|SS30[46])\b/i;
const HEAT_KEYWORDS =
  /\b(HEAT TREAT|H\/T|HARDNESS|HARDEN|TEMPER|NITRID|ANNEAL|CARBURI[SZ]E|CASE DEPTH|QUENCH|HRC|HB|HV)\b/i;
const COATING_KEYWORDS =
  /\b(ZINC|BLACK OXIDE|ANODI[SZ]E|CHROME|NICKEL|GALVANI[SZ]ED|PAINT|POWDER COAT|PHOSPHATE|PASSIVATE|PLATE|PLATING|COATING)\b/i;
const GDT_KEYWORDS =
  /\b(STRAIGHTNESS|FLATNESS|CIRCULARITY|ROUNDNESS|CYLINDRICITY|PROFILE|PARALLELISM|PERPENDICULARITY|SQUARENESS|ANGULARITY|POSITION|CONCENTRICITY|SYMMETRY|RUNOUT|ECCENTRICITY)\b/i;
const WELD_KEYWORDS = /\b(WELD|FILLET WELD|ALL AROUND|AWS|ISO ?2553)\b/i;

export function classifyDimension(text: string): DimensionType {
  const t = text.trim();
  if (!t) return "Unknown";
  const u = t.toUpperCase();

  // 1. GD&T / Feature Control Frame
  if (GDT_KEYWORDS.test(u) || (t.split("|").length - 1 >= 2 && /\d/.test(t)))
    return "GD&T";
  // 3. Weld
  if (WELD_KEYWORDS.test(u)) return "Weld";
  // 4. Surface finish
  if (RE_SURFACE.test(t)) return "Surface Finish";
  // 5. Thread (Rc hardness guarded by heat keywords)
  if (RE_THREAD.test(u) && !(/\bRC\b/i.test(u) && /HARDNESS|HRC/i.test(u)))
    return "Thread";
  // 6. Hole feature
  if (HOLE_MODIFIERS.test(u)) return "Hole";
  // 7. Geometric dimensions — chamfer before angle
  if (RE_CHAMFER.test(t) || /\bCHAMFER\b/i.test(u)) return "Chamfer";
  if (/\bTAPER\b/i.test(u)) return "Taper";
  if (DIAMETER_GLYPHS.test(t)) return "Diameter";
  if (/^\s*R\s*\d/.test(u) || /\bFILLET\b/i.test(u)) return "Radius";
  if (RE_ANGLE.test(u)) return "Angle";
  // 8. Manufacturing notes
  if (HEAT_KEYWORDS.test(u)) return "Heat Treatment";
  if (COATING_KEYWORDS.test(u)) return "Coating";
  if (MATERIAL_KEYWORDS.test(u)) return "Material";
  // 9. Reference / basic
  if (RE_REFERENCE.test(t)) return "Reference";
  // 10. General notes
  if (NOTE_KEYWORDS.test(u)) return "General Note";
  // 11. Remaining numeric
  if (/^\s*[ØøRr]?\s*[+-]?\d/.test(t) || /\d/.test(t))
    return RE_TOLERANCE.test(t) ? "Tolerance" : "Linear";

  return "Note";
}
