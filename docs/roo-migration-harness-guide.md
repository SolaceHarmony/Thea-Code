# Roo Migration Test Harness – Tips, Tricks, and Style Guide

This guide covers best practices for writing and converting tests to the in-repo hostless harness under `test/roo-migration/`.

## Quick recap
- Runner: `node test/roo-migration/runTest.js` or `npm run test:roo-migration`
- Discovers: all `*.test.js` under `test/roo-migration/suite/`
- Loader: `helpers/thea-loader.js` bundles TS from this repo, externalizing host-only deps
- Stubs: local `vscode/` and `tcp-port-used/` under `test/roo-migration/node_modules/`

## Writing and converting tests
- Naming: Use `*-converted.test.js` to mark migrated tests from upstream sources.
- Asserts: Prefer Node's built-in `assert` module for zero-dependency assertions.
- Imports:
  - For plain JS utilities, `require` production modules normally (the harness runs in Node CJS).
  - For TS entrypoints (or when pathing is tricky), use the loader:
    ```js
    const { loadTheaModule } = require('../helpers/thea-loader')
    const { someFn } = loadTheaModule('src/utils/some-file.ts')
    ```
- Isolation: Keep tests deterministic, hostless, and side-effect free.
  - No real network calls or VS Code host APIs.
  - No real `git` calls; mock `child_process.exec` inline and restore.
  - No real timers that sleep long; use small, deterministic waits.

## Stubbing and mocking
- VS Code API:
  - Minimal stubs live at `test/roo-migration/node_modules/vscode/index.js`.
  - If a suite needs more APIs (e.g., `workspace`, `window`), extend the stub conservatively and add methods your tests require.
- Ports and networking:
  - `tcp-port-used` is stubbed to simulate port checks and retries.
  - Avoid adding real socket/listener logic in tests; simulate with the stub surface.
- Child processes:
  - Use inline stubs for `child_process.exec` or `util.promisify(exec)` and restore after each test.
  - Ensure mocks do not leak between tests.

## Determinism and cross-platform behavior
- Paths:
  - Normalize paths you assert against. Prefer comparing relative or POSIX-ified strings when possible.
  - Avoid hard-coding Windows or POSIX separators in expectations; derive using the same helper under test when feasible.
- Randomness and jitter:
  - If code uses jitter/backoff, assert on observable outcomes (e.g., "eventually succeeds after retries"), not exact delays.
- Logging:
  - The harness prints some info-level logs from utilities (e.g., port utils); allow benign logs but avoid test expectations that depend on log text.

## File I/O guidelines
- Temp files:
  - If a test needs real files, write them into `os.tmpdir()` or a unique subfolder under `test/roo-migration/.cache/` and clean up.
- Fixtures:
  - Keep fixtures small and inline when possible to ease maintenance.

## Mocha usage tips
- Timeouts:
  - The default per-test timeout is 20s (set in `suite/index.js`). For slow tests, adjust per-spec:
    ```js
    it('does X', function () {
      this.timeout(5000)
      // ...
    })
    ```
- Running a subset:
  - Current runner loads all tests. A lightweight pattern is to temporarily `only` a spec or suite:
    ```js
    describe.only('utils/path', () => { /* ... */ })
    it.only('specific case', () => { /* ... */ })
    ```
  - If desired, the runner can be trivially extended to read `process.env.GREP` and call `mocha.grep()`.

## Converting from upstream tests
- Avoid external frameworks (Jest, ts-node) — this harness is plain Mocha in Node CJS with esbuild for TS bundling.
- Replace Jest matchers with Node `assert` equivalents.
- Replace `import` syntax with `require` or use the loader for TS sources.
- Replace TS-only test files by converting to JS and importing TS targets through the loader.
- Replace environment- or host-dependent behavior with local stubs/mocks.

## Extending the harness
- Adding new externals:
  - If a production module requires a host or binary your tests should not load, add its package name to `external` in `helpers/thea-loader.js` and provide a local stub under `test/roo-migration/node_modules/<name>/`.
- Enhancing the VS Code stub:
  - Add the smallest surface area needed, and include comments describing behavior assumptions.

## Common pitfalls
- Accidental network/FS coupling: keep interactions local and deterministic.
- Brittle string assertions (e.g., full log lines): assert on structured data or normalized substrings.
- Path separator mismatches: normalize using helpers before asserting.
- Leaky mocks: always restore monkey patches in `afterEach`.

## FAQ
- Why not use the built extension build?
  - This harness tests real source with minimal build steps to speed iteration and avoid coupling to VS Code.
- Can I use ESM in tests?
  - Tests run in CJS; prefer `require`. For TS sources, use the loader.
- How do I simulate VS Code workspace?
  - Extend the `vscode` stub's `workspace` as needed (e.g., `workspaceFolders`, `getConfiguration`). Keep defaults minimal.
