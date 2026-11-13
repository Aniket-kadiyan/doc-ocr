QTracX UI redesign - Batch 2.1 correction
Visible status outlines and matching connector colours

REQUIRES
Batch 2 must already be applied and committed on the ui-redesign branch.

APPLY
1. Extract this ZIP.
2. Copy the extracted src folder into the root of doc-ocr.
3. Allow Windows to replace the three existing files.

FIXED BEHAVIOUR
- Accepted balloon boundary and connector are both green.
- Review balloon boundary and connector are both red.
- SVG boundary width is strengthened by 1.75x before rendering.
- SVG connector width is strengthened by 1.25x before rendering.
- The emergency canvas fallback uses the same status colours and stronger
  strokes.
- The blue selection effect remains a shadow and does not replace the status
  colour.

TEST FROM THE doc-ocr ROOT
npx vitest run src/lib/balloonStyle.test.ts
npm run typecheck
npm run test:frontend
npm run build

VERIFIED BEFORE PACKAGING
- Balloon style tests: 5 passed
- Full frontend tests: 197 passed
- TypeScript typecheck: passed
- Production build: passed

The production build reported one pre-existing unused ReactNode warning in
Toolbar.tsx. This correction does not modify Toolbar.tsx.

SUGGESTED COMMIT MESSAGE
fix(ui): restore balloon outlines and colour status connectors
