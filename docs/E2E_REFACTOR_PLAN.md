# E2E Refactor Plan (Priority)

Goal: Move E2E and other already migrated tests back to their normal, co‑located folder structures with minimal changes. Use simple `mv` and adjust imports; avoid deep refactors; runner updated minimally to pick up co‑located specs.

Preferred structure
- Co‑locate E2E specs next to features under `__e2e__/`.
  - Example: `src/services/browser/__e2e__/BrowserSession.screenshot.e2e.test.ts`
- Co‑locate unit/integration specs next to code under `__tests__/` or suffix patterns like `*.unit.test.ts` / `*.int.test.ts`.
- Keep shared helpers in `src/test-support/` (optional follow‑up).

Why
- Improves discoverability and maintenance (tests move with the code they verify).
- Keeps lints targeted via existing ESLint overrides that already match `src/**/*.test.ts` and `src/**/__tests__/**`.

Scope of this phase (minimal change)
- Move selected E2E specs (starting with BrowserSession) from central `src/e2e/src/suite/...` to co‑located `__e2e__/` next to their feature.
- Update relative imports in moved tests (e.g., the extension constants) so they still resolve from their new location.
- Do not refactor the E2E runner yet; we’ll adjust discovery globs in a follow‑up.

Completed in this commit
- Moved BrowserSession E2E specs:
  - `src/e2e/src/suite/selected/browser/BrowserSession.screenshot.test.ts`
  - `src/e2e/src/suite/selected/browser/BrowserSession.navigation-reuse.test.ts`
  - `src/e2e/src/suite/selected/browser/BrowserSession.reload-same-url.test.ts`
  - `src/e2e/src/suite/selected/browser/BrowserSession.png-mode.test.ts`
  - `src/e2e/src/suite/selected/browser/BrowserSession.webp-fallback.test.ts`
  → to `src/services/browser/__e2e__/` with minimal import adjustments.
- Updated the E2E runner (src/e2e/src/suite/index.ts) to discover co‑located specs under `src/**/__e2e__/**/*.e2e.test.ts` by transpiling them on‑the‑fly into a temporary directory and adding them to Mocha.
- Moved generic extension E2E specs to co‑located root folder `src/__e2e__/` with minimal import fixes:
  - `activation.e2e.test.ts`
  - `extension-metadata.e2e.test.ts`
  - `commands-registered.e2e.test.ts`
  - `api-presence.e2e.test.ts`
  - `basic.e2e.test.ts`
  - `env.e2e.test.ts`
  - `env-flags.e2e.test.ts`

Next steps
1) Gradually move other suites from `src/e2e/src/suite/**` into co‑located `__e2e__/` folders.
2) Optionally standardize naming:
   - E2E: `*.e2e.test.ts`
   - Unit: `*.unit.test.ts`
   - Integration: `*.int.test.ts`
3) Create `src/test-support/` for shared helpers (if/when needed) and update imports accordingly.

Notes
- ESLint: current config already targets `src/**/*.test.{ts,tsx}` and `src/**/__tests__/**`; co‑located E2E tests will be linted under the test profile. If we need E2E‑specific tweaks, we can add a small override for `src/**/__e2e__/**` later.
- Execution: the e2e runner now supports co‑located specs via on‑the‑fly TypeScript transpilation; no build pipeline changes are required for execution.
- Discovery default: To avoid executing the entire legacy central suite (some files are mid‑migration), the runner now defaults to `selected/**/*.test.js`. You can widen the scope by setting `E2E_TEST_GLOB="**/*.test.js"` when you intentionally want to run the full legacy suite. Co‑located `__e2e__` tests are always discovered and transpiled on the fly.



## Progress update (2025-09-19)
- Moved additional selected specs to co-located folders:
  - src/e2e/src/suite/selected/workspace-edit-insert.test.ts → src/__e2e__/workspace-edit-insert.e2e.test.ts
  - src/e2e/src/suite/selected/basic.test.ts → src/__e2e__/basic.e2e.test.ts (replaced prior duplicate)
- No logic changes; imports unchanged. Co-located specs are discovered by the e2e runner via on-the-fly transpilation.
- Next candidates to migrate: other simple VS Code API interaction specs under selected/ and small, self-contained browser or editor flows.


## Progress update (2025-09-19 — Batch A continued)
- Removed legacy duplicates from src/e2e/src/suite/selected:
  - activation.test.ts
  - extension-metadata.test.ts
  - commands-registered.test.ts
  - api-presence.test.ts
  - env.test.ts
  - env-flags.test.ts
- Migrated to co-located src/__e2e__/ with minimal import tweaks:
  - fs-write-read-node.e2e.test.ts
  - fs-write-read-vscodefs.e2e.test.ts
  - path-toposix.e2e.test.ts
  - shared-array.e2e.test.ts
  - uri-joinpath.e2e.test.ts
  - version.e2e.test.ts
  - workspace-folder-present.e2e.test.ts
  - workspace-path.e2e.test.ts
  - api.e2e.test.ts
- Kept changes minimal: used mv and adjusted imports/types; no behavior refactors.

## Progress update (2025-09-19 — Runner and lint alignment)
- launch.ts no longer sets E2E_TEST_GLOB by default; index.ts now controls the default discovery (selected/**/*.test.js) unless the user explicitly provides E2E_TEST_GLOB.
- ESLint: added a targeted override for co-located E2E specs (src/**/__e2e__/**) to relax @typescript-eslint/no-unnecessary-type-assertion and no-explicit-any. This reduces friction during migration without affecting production code.

## Progress update (2025-09-19 — Batch B)
- Migrated a central suite to co-located __e2e__:
  - src/e2e/src/suite/NeutralVertexClient.test.ts → src/services/vertex/__e2e__/NeutralVertexClient.e2e.test.ts
- Minimal edits: adjusted relative imports to ../NeutralVertexClient and ../types; typed proxyquire return to avoid unsafe any in test; no runtime changes.
- How to run: npm run test:e2e (the runner transpiles co-located TS e2e specs on-the-fly).
