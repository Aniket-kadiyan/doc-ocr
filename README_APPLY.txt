QTracX UI redesign - Batch 1
Balloon size reduction and connector-line replacement

SOURCE BASE
ui-redesign branch, using the complete files currently pushed to GitHub.

APPLY
1. Make sure you are on the ui-redesign branch.
2. Extract this ZIP.
3. Copy the extracted public and src folders into the root of doc-ocr.
4. Allow Windows to replace the existing files.

This package contains complete replacement files only. It does not contain
node_modules, Python environments, build output, or unrelated source files.

BALLOON SIZES
Small:  drawing 14 px, sidebar 10 px
Medium: drawing 18 px, sidebar 14 px
Large:  drawing 24 px, sidebar 18 px

The orange triangular pointer is replaced with a neutral straight connector
line in every SVG style and in the emergency fallback renderer.

TEST FROM THE doc-ocr ROOT
npx vitest run src/lib/balloonStyle.test.ts
npm run typecheck
npm run test:frontend
npm run build

SUGGESTED COMMIT MESSAGE
feat(ui): reduce balloon sizes and replace pointer with connector line
