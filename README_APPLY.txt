QTracX UI redesign - Batch 2
Review numbering, review balloons, and status colours

REQUIRES
Batch 1 must already be applied and committed on the ui-redesign branch.

APPLY
1. Make sure you are on the ui-redesign branch containing Batch 1.
2. Extract this ZIP.
3. Copy the extracted src folder into the root of doc-ocr.
4. Allow Windows to replace the existing files and add BalloonMarker.tsx.

BEHAVIOUR
- Review candidates receive stable R-1, R-2, ... identifiers.
- Old saved candidates receive deterministic R-n identifiers when loaded.
- Review candidates use the selected SVG balloon style with a red boundary.
- Accepted values use green value-box and balloon boundaries.
- Selected values keep a blue shadow while retaining their status colour.
- Other detections remain grey and do not receive R balloons.
- Ignored review candidates retain their R number when restored.
- Accepting a review candidate removes it from the R sequence and creates the
  normal numbered value through the existing acceptance workflow.

TEST FROM THE doc-ocr ROOT
npx vitest run src/lib/scanCandidateLifecycle.test.ts src/lib/balloonStyle.test.ts
npm run typecheck
npm run test:frontend
npm run build

VERIFIED BEFORE PACKAGING
- Focused tests: 10 passed
- Full frontend tests: 197 passed
- TypeScript typecheck: passed
- Production build: passed

The production build reported one pre-existing warning in Toolbar.tsx for an
unused ReactNode import. Batch 2 does not modify Toolbar.tsx.

SUGGESTED COMMIT MESSAGE
feat(ui): add numbered review balloons and status colours
