import * as path from "path"
import Mocha from "mocha"
import { glob } from "glob"

import { TheaCodeAPI } from "../../../src/exports/thea-code"

declare global {
	var api: TheaCodeAPI
}

export async function run() {
	console.log("Starting e2e test suite...")
	console.log("Skipping extension activation in test runner - tests will handle it individually")

	// Add all the tests to the runner.
	const mocha = new Mocha({ ui: "tdd", timeout: 300_000 })
	const cwd = path.resolve(__dirname, "..")
	;(await glob("**/**.test.js", { cwd })).forEach((testFile) => mocha.addFile(path.resolve(cwd, testFile)))

	// Let's go!
	return new Promise<void>((resolve, reject) =>
		mocha.run((failures) => (failures === 0 ? resolve() : reject(new Error(`${failures} tests failed.`)))),
	)
}
