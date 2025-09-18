import * as path from "path"
import Mocha from "mocha"
import * as fs from "fs"

import type { TheaCodeAPI } from "../../../exports/thea-code"

const repoRoot = path.resolve(__dirname, "../../../..")
const testRoot = path.resolve(repoRoot, ".vscode-test")
const debugFile = path.join(testRoot, "e2e-index.log")
function writeDebug(msg: string) {
	try { fs.appendFileSync(debugFile, msg + "\n") } catch {}
}

console.log("[e2e] index module loaded")
writeDebug("[index] loaded")

declare global {
	var api: TheaCodeAPI
}

export async function run() {
	console.log("Starting e2e test suite...")
	writeDebug("[index] run() entered")

	try {
		// Increase timeout to 10 minutes for extension host initialization
		const mocha = new Mocha({ ui: "tdd", timeout: 600_000, reporter: "spec" })

		writeDebug("[index] mocha created")
		// Ensure TDD globals (suite/test) are registered for subsequently loaded test files
		try {
			// Register TDD globals for subsequent requires
			const suiteRef: Mocha.Suite = (mocha as unknown as { suite: Mocha.Suite }).suite
			suiteRef.emit("pre-require", global, "global", mocha)
			writeDebug("[index] pre-require emitted")
		} catch (e) {
			const msg = e instanceof Error ? e.stack ?? String(e) : String(e)
			writeDebug(`[index] pre-require failed: ${msg}`)
			throw e
		}

		writeDebug("[index] mapping globals start")
		// Support both BDD (describe/it) and TDD (suite/test) style tests
		// Minimal typed shape of the Mocha globals we rely on to avoid `any`.
		// Mocha-style test function: supports optional done callback or returning a Promise.
		type TestFn = (this: unknown, done?: (err?: unknown) => void) => void | Promise<void>
		type FnWithModifiers = ((title: string, fn: TestFn) => void) & {
			only?: (title: string, fn: TestFn) => void
			skip?: (title: string, fn: TestFn) => void
		}
		interface TestGlobals {
			suite: FnWithModifiers
			test: FnWithModifiers
			suiteSetup: (fn: TestFn) => void
			suiteTeardown: (fn: TestFn) => void
			setup: (fn: TestFn) => void
			teardown: (fn: TestFn) => void
			describe?: FnWithModifiers
			it?: FnWithModifiers
			before?: (fn: TestFn) => void
			after?: (fn: TestFn) => void
			beforeEach?: (fn: TestFn) => void
			afterEach?: (fn: TestFn) => void
			context?: FnWithModifiers
			specify?: FnWithModifiers
			xdescribe?: FnWithModifiers
			xit?: FnWithModifiers
		}
		const g = globalThis as unknown as TestGlobals
		// Core mappings
		if (!g.describe) g.describe = (title: string, fn: TestFn) => g.suite(title, fn)
		if (!g.it) g.it = (title: string, fn: TestFn) => g.test(title, fn)
		if (!g.before) g.before = (fn: TestFn) => g.suiteSetup(fn)
		if (!g.after) g.after = (fn: TestFn) => g.suiteTeardown(fn)
		if (!g.beforeEach) g.beforeEach = (fn: TestFn) => g.setup(fn)
		if (!g.afterEach) g.afterEach = (fn: TestFn) => g.teardown(fn)
		// Aliases & modifiers commonly used in BDD tests
		if (!g.context) g.context = g.describe
		if (!g.specify) g.specify = g.it
		if (!g.xdescribe) g.xdescribe = (title: string, fn: TestFn) => {
			// Prefer describe.skip if available, otherwise fall back to suite.skip
			if (g.describe?.skip) return g.describe.skip(title, fn)
			if (g.suite?.skip) return g.suite.skip(title, fn)
		}
		if (!g.xit) g.xit = (title: string, fn: TestFn) => {
			// Prefer it.skip if available, otherwise fall back to test.skip
			if (g.it?.skip) return g.it.skip(title, fn)
			if (g.test?.skip) return g.test.skip(title, fn)
		}
		// Ensure modifier properties exist before assigning (use non-null assertions
		// because we've already ensured `describe`/`it` above).
		if (g.describe && !g.describe.only) g.describe.only = (title: string, fn: TestFn) => {
			// Prefer suite.only if available, otherwise call suite normally
			if (g.suite?.only) return g.suite.only(title, fn)
			return g.suite(title, fn)
		}
		if (g.describe && !g.describe.skip) g.describe.skip = (title: string, fn: TestFn) => {
			// Prefer suite.skip if available, otherwise call suite normally
			if (g.suite?.skip) return g.suite.skip(title, fn)
			return g.suite(title, fn)
		}
		if (g.it && !g.it.only) g.it.only = (title: string, fn: TestFn) => {
			// Prefer test.only if available, otherwise call test normally
			if (g.test?.only) return g.test.only(title, fn)
			return g.test(title, fn)
		}
		if (g.it && !g.it.skip) g.it.skip = (title: string, fn: TestFn) => {
			// Prefer test.skip if available, otherwise call test normally
			if (g.test?.skip) return g.test.skip(title, fn)
			return g.test(title, fn)
		}
		writeDebug("[index] mapping globals done")

			// Optional: smoke-only run to validate harness without Mocha
			if (process.env.E2E_SMOKE_ONLY === "1") {
				console.log("[e2e] SMOKE mode: skip Mocha, exit early OK")
				writeDebug("[index] smoke mode enabled; early exit")
				return
			} else {
			// Add setup first to activate extension and expose global.api
				// Resolve compiled suite output directory robustly
				const defaultSuiteOutDir = __dirname
				const candidates = [
					defaultSuiteOutDir,
					path.resolve(repoRoot, "src", "e2e", "out", "suite", "suite"),
					path.resolve(repoRoot, "e2e", "out", "suite", "suite"),
				]
				const suiteOutDir = candidates.find((dir) => {
					try { return fs.existsSync(path.resolve(dir, "setup.test.js")) } catch { return false }
				}) ?? defaultSuiteOutDir
				writeDebug(`[index] suiteOutDir=${suiteOutDir}`)
				const setupFile = path.resolve(suiteOutDir, "setup.test.js")
			if (process.env.E2E_SKIP_SETUP !== "1") {
				mocha.addFile(setupFile)
				console.log("[e2e] setup added")
				writeDebug("[index] setup added")
			} else {
				console.log("[e2e] setup skipped via E2E_SKIP_SETUP=1")
				writeDebug("[index] setup skipped")
			}

			// Direct test mode: add a single known test without glob to avoid ESM issues/hangs
			if (process.env.E2E_DIRECT_TEST === "1") {
				const activation = path.resolve(suiteOutDir, "activation.test.js")
				console.log(`[e2e] Direct test mode: adding ${activation}`)
				mocha.addFile(activation)
			} else {
				// Then add all test files from the compiled output
				const cwd = suiteOutDir // out/suite/suite
				// Discover E2E tests. Default to all *.test.js to include migrated tests; can narrow via E2E_TEST_GLOB
				const pattern = process.env.E2E_TEST_GLOB ?? "**/*.test.js"
				console.log(`[e2e] Test discovery cwd=${cwd} pattern=${pattern}`)
				writeDebug(`[index] discover start cwd=${cwd} pattern=${pattern}`)
				// Dynamic import to support latest glob (ESM) from a CJS-compiled test bundle
				const { glob } = await import("glob")
				const files = await glob(pattern, {
					cwd,
					ignore: [
						"**/node_modules/**",
						"**/.vscode-test/**",
						"**/setup.js",
						"**/*.converted.test.js",
						"**/*.node.test.js",
						"**/*.unit.test.js",
						"**/__tests__/**",
						"**/selected/**",
					],
				})

				console.log(`[e2e] Discovered ${files.length} test file(s)`)
				writeDebug(`[index] discovered ${files.length} file(s)`) 

				if (files.length === 0) {
					writeDebug("[index] no tests found; adding trivial baseline to avoid hang")
					suite("NO TESTS FOUND", () => {
						test("baseline", () => {})
					})
				}
				files.forEach((f) => {
					const resolved = path.resolve(cwd, f)
					console.log(`[e2e] addFile ${resolved}`)
					writeDebug(`[index] addFile ${resolved}`)
					mocha.addFile(resolved)
				})
			}
		}

		console.log("[e2e] Starting Mocha run...")
		writeDebug("[index] mocha.run start")
		return await new Promise<void>((resolve, reject) =>
			mocha.run((failures) => {
				console.log(`[e2e] Mocha finished with ${failures} failure(s)`)
				writeDebug(`[index] mocha.run done failures=${failures}`)
				if (failures === 0) {
					resolve()
				} else {
					reject(new Error(`${failures} tests failed.`))
				}
			}),
		)
	} catch (err) {
		console.error("[e2e] Test runner failed:", err)
		const msg = err instanceof Error ? err.stack ?? String(err) : String(err)
		writeDebug(`[index] run() error ${msg}`)
		throw err
	}
}
