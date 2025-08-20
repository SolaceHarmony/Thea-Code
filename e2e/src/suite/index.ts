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
		const g: any = globalThis as any
		// Core mappings
		if (!g.describe) g.describe = (title: string, fn: Function) => g.suite(title, fn)
		if (!g.it) g.it = (title: string, fn: Function) => g.test(title, fn)
		if (!g.before) g.before = (fn: Function) => g.suiteSetup(fn)
		if (!g.after) g.after = (fn: Function) => g.suiteTeardown(fn)
		if (!g.beforeEach) g.beforeEach = (fn: Function) => g.setup(fn)
		if (!g.afterEach) g.afterEach = (fn: Function) => g.teardown(fn)
		// Aliases & modifiers commonly used in BDD tests
		if (!g.context) g.context = g.describe
		if (!g.specify) g.specify = g.it
		if (!g.xdescribe) g.xdescribe = (title: string, fn: Function) => g.describe.skip(title, fn)
		if (!g.xit) g.xit = (title: string, fn: Function) => g.it.skip(title, fn)
		if (!g.describe.only) g.describe.only = (title: string, fn: Function) => g.suite.only(title, fn)
		if (!g.describe.skip) g.describe.skip = (title: string, fn: Function) => g.suite.skip(title, fn)
		if (!g.it.only) g.it.only = (title: string, fn: Function) => g.test.only(title, fn)
		if (!g.it.skip) g.it.skip = (title: string, fn: Function) => g.test.skip(title, fn)

		// Add setup first to activate extension and expose global.api
		const setupFile = path.resolve(__dirname, "./setup.js")
		mocha.addFile(setupFile)

		// Then add all test files from the compiled output
		const cwd = path.resolve(__dirname) // out/suite
		const pattern = "**/*.test.js"
		const files = await glob(pattern, {
			cwd,
			ignore: ["**/node_modules/**", "**/.vscode-test/**", "**/setup.js"],
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
