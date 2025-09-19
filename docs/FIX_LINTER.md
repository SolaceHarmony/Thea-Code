# Lint Hardening & Suppression Remediation Plan

_Last updated: 2025-09-19_

## 1. Executive Summary
The codebase currently relies on broad, global ESLint rule disablements and large directory ignore patterns rather than targeted, contextual suppressions. This eliminates meaningful static feedback for core safety, correctness, and hygiene concerns (e.g. unsafe argument/return usage, unhandled promises, unused variables). There are **no inline suppressions** (good), but that is only because rules are disabled globally. This document defines a structured, low‑risk path to progressively restore lint signal while avoiding PR noise or team disruption.

## 2. Current State Snapshot
| Aspect | Status | Notes |
|--------|--------|-------|
| Inline `eslint-disable` / `@ts-ignore` / `@ts-nocheck` | None found | Clean of ad‑hoc silencing. |
| Global rule disables | Extensive | Many core TypeScript safety rules set to `off`. |
| Directory ignores | Very broad | Tests, scripts, e2e, benchmarks, locales, assets ignored. |
| Partial re‑enabling attempt | Present (comments about re-adding `webview-ui` / `e2e`) | But still re‑suppressed by other patterns. |
| Type safety posture | Degraded | `no-unsafe-*`, `no-explicit-any`, `no-misused-promises` all off. |
| Technical risk | Elevated | Silent logic / async issues may slip through review. |

## 3. Key Global Suppressions (From `eslint.config.mjs`)
Disabled (all set to `off`):
- `@typescript-eslint/no-unsafe-argument`
- `@typescript-eslint/no-unsafe-return`
- `@typescript-eslint/restrict-plus-operands`
- `@typescript-eslint/restrict-template-expressions`
- `@typescript-eslint/unbound-method`
- `@typescript-eslint/no-misused-promises`
- `@typescript-eslint/require-await`
- `@typescript-eslint/no-unused-vars`
- `@typescript-eslint/no-explicit-any`
- `@typescript-eslint/no-require-imports`
- `no-undef`
- `no-import-assign`

## 4. Major Ignore Patterns
Ignored directories/files (selected examples):
- `scripts/`, `benchmark/`, `test/`, `src/e2e/src/**`
- All tests via glob: `src/**/__tests__/**`, `src/**/*.test.ts(x)`
- Non-code formats: `**/*.md`, `**/*.json`, `**/*.yaml` (could host config logic)
- Assets, locales, mocks — understandable, but a few TS utility files sometimes creep into these.

### Impact
1. **Tests not linted** → flaky / dead helpers accumulate.
2. **Automation (scripts) unreviewed** → deployment / release logic risk.
3. **Benchmarks & e2e** → drift, outdated API usage undetected.

## 5. Risks of Maintaining Status Quo
| Risk | Description | Potential Consequence |
|------|-------------|-----------------------|
| Silent unsafe usage | `no-unsafe-return/argument` disabled | Runtime type mismatch / subtle logic errors. |
| Async footguns | `no-misused-promises`, `require-await` off | Lost promise rejections, race conditions, swallowed failures. |
| Hidden dead code | `no-unused-vars` off | Cognitive load + missed refactor opportunities. |
| Weak typing discipline | `no-explicit-any` off w/ other disables | Erodes reliability of TS as a design tool. |
| Config / test drift | Entire categories un-linted | Slower onboarding, brittle CI. |

## 6. Guiding Principles for Remediation
1. **Progressive Activation** – Introduce rules in `warn` mode first; escalate once warning counts decline.
2. **Differential Enforcement** – Apply stricter rules to *new or recently touched* files (git diff basis).
3. **Stable Baseline** – Record current warning counts; CI blocks only if they *increase*.
4. **No Surprise PR Noise** – Avoid mass auto-fix commits until after consensus; batch strategically.
5. **Visibility First** – Prefer “warn” over silence; devs learn patterns before enforcement.
6. **Document Justification** – Any permanent disable must include rationale + expiry review date.

## 7. Phased Rollout Plan
| Phase | Duration | Focus | Exit Criteria |
|-------|----------|-------|---------------|
| 0 – Baseline | Day 0 | Add reporting only | Baseline JSON snapshot stored. |
| 1 – Visibility | Week 1–2 | Turn critical safety rules to `warn` | < 5% increase in build time; devs acknowledge. |
| 2 – Containment | Week 3–4 | Enforce (error) on changed files only | Pre-commit hook passes for 95% of touches. |
| 3 – Expansion | Week 5–6 | Include tests & scripts (warn) | Test dirs yield < X warnings per KLoC. |
| 4 – Hardening | Week 7+ | Escalate key rules to `error` | Warning trend negative for 3 successive weeks. |
| 5 – Tighten Typing | Continuous | Address `any`, unsafe ops | `any` usage reduced by 30% vs baseline. |

## 8. Initial Rule Reintroduction (Phase 1: warn)
Safety / correctness (highest leverage):
- `@typescript-eslint/no-misused-promises`
- `@typescript-eslint/no-unsafe-return`
- `@typescript-eslint/no-unsafe-argument`
- `@typescript-eslint/restrict-plus-operands`
- `@typescript-eslint/restrict-template-expressions` (allow number)
- `@typescript-eslint/unbound-method`
- `@typescript-eslint/no-unused-vars` (configure ignore patterns)
- `@typescript-eslint/require-await`
- `@typescript-eslint/no-explicit-any` (warn only)
- Add (new): `@typescript-eslint/no-floating-promises` (warn)

## 9. Example ESLint Config Patch (Conceptual)
```diff
 // In eslint.config.mjs (add BEFORE broad override)
+{
+  files: ["src/**/*.{ts,tsx}"],
+  rules: {
+    "@typescript-eslint/no-misused-promises": ["warn", { checksVoidReturn: { attributes: false }}],
+    "@typescript-eslint/no-unsafe-return": "warn",
+    "@typescript-eslint/no-unsafe-argument": "warn",
+    "@typescript-eslint/restrict-plus-operands": "warn",
+    "@typescript-eslint/restrict-template-expressions": ["warn", { allowNumber: true }],
+    "@typescript-eslint/unbound-method": "warn",
+    "@typescript-eslint/no-floating-promises": "warn",
+    "@typescript-eslint/no-unused-vars": ["warn", { argsIgnorePattern: "^_", varsIgnorePattern: "^_" }],
+    "@typescript-eslint/require-await": "warn",
+    "@typescript-eslint/no-explicit-any": "warn"
+  }
+},
+{
+  files: ["scripts/**/*.{ts,js}"],
+  rules: {
+    "@typescript-eslint/no-require-imports": "off"
+  }
+},
+{
+  files: ["src/**/__tests__/**", "src/**/*.test.{ts,tsx}"],
+  env: { node: true, mocha: true },
+  rules: {
+    "@typescript-eslint/no-unused-expressions": "off"
+  }
+}
```
_This is staged; do **not** remove global disables until warning counts are measured._

## 10. Directory Inclusion Strategy
1. **Phase 1**: Add `scripts/` & unit tests (warn only).
2. **Phase 2**: Add `benchmark/` & `src/e2e/src/**` (subset: only syntactic + unused vars).
3. **Phase 3**: Consider linting Markdown for link rot / fenced language tags (optional).
4. **Phase 4**: Add JSON/YAML schema lint (tooling: `eslint-plugin-jsonc`, `eslint-plugin-yml`).

## 11. Metrics & CI Gating
Artifacts to generate (store under `lint-output/`):
- `eslint-baseline.json` – full JSON output (Phase 0).
- `eslint-trend.json` – appended daily/CI run metrics: `{ date, rule, warnings }`.
- `any-usage.json` – list of files + count of `any` occurrences (simple grep or TS AST). 

Introduce script (pseudo):
```bash
npm run lint:json -- --output-file lint-output/current.json
node scripts/compare-lint-baseline.mjs
```
Comparison logic blocks CI **only if** total warnings > baseline + threshold (e.g. +2%).

## 12. Rule Escalation Matrix
| Rule | Phase Introduced | Escalate To Error | Criteria |
|------|------------------|-------------------|----------|
| no-misused-promises | 1 (warn) | Phase 3 | < 25 active warnings |
| no-unsafe-return | 1 (warn) | Phase 4 | Manual audit complete |
| no-unsafe-argument | 1 (warn) | Phase 4 | Same as above |
| restrict-plus-operands | 1 (warn) | 3 | < 10 warnings |
| restrict-template-expressions | 1 (warn) | 4 | All string concat hotspots refactored |
| unbound-method | 1 (warn) | 3 | Event handler patterns validated |
| no-floating-promises | 1 (warn) | 2 | Critical paths wrapped |
| no-unused-vars | 1 (warn) | 2 | < 40 warnings (then selective error) |
| no-explicit-any | 1 (warn) | Maybe never global error | Track trending only |
| require-await | 1 (warn) | 3 | False positives addressed |

## 13. Developer Workflow Changes
| Practice | New Expectation |
|----------|-----------------|
| New feature modules | Must be warning-free (treat warns as errors locally). |
| Refactors | Remove adjacent `any` where trivial. |
| Reviews | Ask “Can this unsafe return be typed?” before approving. |
| Suppressions | If needed, add inline `// eslint-disable-next-line <rule> -- reason (date)` – require justification. |

## 14. Automation Enhancements
Tasks to add under `scripts/`:
1. `generate-lint-baseline.mjs` – produce & store baseline.
2. `lint-diff.mjs` – lint only changed files (git diff) with `--max-warnings=0`.
3. `analyze-any-usage.mjs` – count & diff `any` occurrences vs baseline.
4. `enforce-new-code.mjs` – fail if new/changed files introduce *new* categories of warnings.

CI steps (GitHub Actions concept):
```yaml
- run: npm ci
- run: npm run lint:json
- run: node scripts/lint-diff.mjs
- run: node scripts/compare-lint-baseline.mjs
```

## 15. Success Criteria / KPIs
| Metric | Baseline | Target (Quarter) |
|--------|----------|------------------|
| Warnings / KLoC (core src) | TBD (capture) | -40% |
| `any` usages | Baseline count | -30% |
| Unsafe returns | Baseline | < 5 remaining |
| Unused vars | Baseline | 0 (outside tests) |
| Floating promises | Baseline | 0 in `/src/services` |

## 16. Handling Legacy / Hard Files
If a file is **high-churn + high-warning**, fast-track refactor.
If **low-churn + high-warning**, isolate w/ file-level TODO header:
```ts
/* LINT_DEBT: Tracked in issue #123 – reduce unsafe returns (5) & floating promises (3). */
```

## 17. Long-Term Enhancements (Optional)
- Semantic lint (custom rule) for model provider capability metadata completeness.
- Ban raw `fetch` in favor of a wrapped HTTP client with typed response guards.
- Introduce `eslint-plugin-security` (light mode) once noise is reduced.

## 18. Immediate Action Checklist (Phase 0 → 1)
- [ ] Commit this document.
- [ ] Add baseline lint run script & capture JSON.
- [ ] Insert Phase 1 rule block (warn) before existing global disables.
- [ ] Add test + scripts override blocks.
- [ ] Run full lint → record counts.
- [ ] Open tracking issue: “Incremental ESLint Hardening”.
- [ ] Announce in team channel with migration timeline.

## 19. Decommissioning Original Global Disables
Only after:
1. All planned rules have a warning baseline.
2. Trend line shows decreasing warnings over ≥ 3 measurement intervals.
3. High-risk rules (async / unsafe) have < agreed threshold.
Then remove specific `off` entries *one cluster at a time* → escalate to `warn` / `error`.

## 20. Appendix: Quick Commands
Baseline capture:
```bash
npm run lint -- --format json --output-file lint-output/baseline.json
```
Changed-files lint (example):
```bash
git diff --name-only origin/main...HEAD -- 'src/**/*.{ts,tsx}' \
  | xargs npx eslint --max-warnings=0 --color
```
Any usage (rough first pass):
```bash
grep -R --line-number "\bany\b" src | wc -l
```

---
**Questions / adjustments?** Open an issue titled `lint-hardening: <topic>` to refine scope.

**Owner (initial):** Platform / Tooling Maintainers
**Revision cadence:** Review progress every 2 weeks; update KPIs quarterly.



## 21. Progress Log

- 2025-09-19 12:19 — Re-enabled strict ESLint (type-aware) across src, tests, and webview-ui; escalated critical safety rules (no-unsafe-*, no-floating-promises, no-misused-promises, require-await, restrict-*) to error. Brought tests and scripts back under linting with targeted overrides (kept only no-unused-expressions off for BDD). Converted high-noise Jest/Chai tests to node:assert/strict + sinon to eliminate unsafe assertion chains. Added unit tests for custom-instructions and checkExistApiConfig. Completed conversion for neutral-ollama-format tests. Implemented real-browser E2E screenshot test via hidden command (thea-code.test.browserCapture) to exercise BrowserSession without stubs. Standardized error handling using shared helpers (getErrorMessage, toError) and refactored BrowserSession.screenshot flow to remove throw-caught-locally patterns.

- 2025-09-18 — Phase 1 rules introduced and lint scripts added (lint, lint:fix, lint:json, lint:baseline, lint:changed). Initial cleanup of warnings in prompts, UI components, and build scripts.

### Next checkpoints
- Convert remaining legacy Jest tests under src/api/providers/__tests__ to assert/strict + sinon (no unsafe shims).
- Prune temporary global ignores as files are migrated; lint scripts/JS under JS-specific override or migrate to TS.
- Capture and commit baseline JSON (npm run lint:baseline) and wire compare script in CI.
- Add small unit tests for modes helpers and webview path utilities.

## 22. Testing Conventions (enforced)

- Use node:assert/strict for assertions; use sinon for stubs/spies. Do not use chai expect chains to avoid error-typed chains under TS.
- Always restore stubs in afterEach via sinon.restore().
- Do not disable no-unsafe-* rules around tests; fix by adding types/guards.
- Favor real integrations (e.g., real browser via BrowserSession) for critical paths; avoid mocks unless necessary.

## 23. Error Handling Patterns

- Prefer getErrorMessage(err) and getErrorCode(err) from src/shared/errors.ts when logging.
- When rethrowing unknown, wrap with toError(err) to preserve stack/message safely.
- Avoid accessing err.message/err.code directly on unknown; add narrow guards or use helpers.
