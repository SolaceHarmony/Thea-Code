import * as path from "path"
import Mocha from "mocha"
import { glob } from "glob"

import type { TheaCodeAPI } from "../../../src/exports/thea-code"

declare global {
	var api: TheaCodeAPI
}

export async function run() {
	console.log("Starting e2e test suite...")

	try {
		const mocha = new Mocha({ ui: "tdd", timeout: 300_000, reporter: "spec" })

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

		// Add setup first to activate extension and expose global.api
		const setupFile = path.resolve(__dirname, "./setup.js")
		mocha.addFile(setupFile)

		// Then add all test files from the compiled output
		const cwd = path.resolve(__dirname) // out/suite
		// Only pick files that are explicitly marked as E2E tests
		const pattern = "**/*.e2e.test.js"
		const files = await glob(pattern, {
			cwd,
			ignore: [
				"**/node_modules/**",
				"**/.vscode-test/**",
				"**/setup.js",
				// Exclude legacy/converted/unit test patterns that may import vscode in plain Node
				"**/*.converted.test.js",
				"**/*.node.test.js",
				"**/*.unit.test.js",
				"**/__tests__/**"
			],
		})

		files.forEach((f) => mocha.addFile(path.resolve(cwd, f)))

		console.log("[e2e] Starting Mocha run...")
		return await new Promise<void>((resolve, reject) =>
			mocha.run((failures) => (failures === 0 ? resolve() : reject(new Error(`${failures} tests failed.`)))),
		)
	} catch (err) {
		console.error("[e2e] Test runner failed:", err)
		throw err
	}
}
