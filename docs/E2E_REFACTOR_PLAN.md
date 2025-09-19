# E2E Refactor Plan (Priority)

Goal: Move E2E and other already migrated tests back to their normal, co‑located folder structures with minimal changes. Use simple `mv` and adjust imports; avoid deep refactors or runner rewrites in this step.

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

Next steps
1) Update the E2E runner discovery to also pick up `src/**/__e2e__/**/*.e2e.test.ts` (without changing how the central e2e package works).
2) Gradually move other suites from `src/e2e/src/suite/**` into co‑located `__e2e__/` folders.
3) Optionally standardize naming:
   - E2E: `*.e2e.test.ts`
   - Unit: `*.unit.test.ts`
   - Integration: `*.int.test.ts`
4) Create `src/test-support/` for shared helpers (if/when needed) and update imports accordingly.

Notes
- ESLint: current config already targets `src/**/*.test.{ts,tsx}` and `src/**/__tests__/**`; co‑located E2E tests will be linted under the test profile. If we need E2E‑specific tweaks, we can add a small override for `src/**/__e2e__/**` later.
- TypeScript: editor tooling will work out of the box; execution wiring comes in the follow‑up that adjusts E2E discovery globs.
