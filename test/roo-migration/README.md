# Roo Migration Test Harness

A hostless Mocha/esbuild harness to run converted tests against the real source in this repo without modifying production code.

## What it does
- Bundles TypeScript on-the-fly via esbuild and loads real `src/` modules.
- Stubs host-only deps (`vscode`, `tcp-port-used`) under `test/roo-migration/node_modules/` for deterministic runs.
- Discovers and runs all `*.test.js` under `test/roo-migration/suite/`.

## Run
```bash
npm run test:roo-migration
```
or directly:
```bash
node test/roo-migration/runTest.js
```

## Add tests
- Drop converted tests into `test/roo-migration/suite/` as `*-converted.test.js`.
- Use `helpers/thea-loader.js` to import TS entrypoints from this repo when needed.

## Notes
- This harness is non-invasive; it doesn’t require building `dist/`.
- The `vscode` and `tcp-port-used` stubs cover common cases used by the suites we migrated; extend if future tests need more APIs.
