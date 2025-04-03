import * as path from "path"
import Mocha from "mocha"
import { EXTENSION_ID } from "../../../dist/thea-config"; // Import branded constant
import { glob } from "glob"
import * as vscode from "vscode"

import { TheaCodeAPI } from "../../../src/exports/thea-code"

import { waitUntilReady } from "./utils"

declare global {
	var api: TheaCodeAPI
}

export async function run() {
	const extension = vscode.extensions.getExtension<TheaCodeAPI>(EXTENSION_ID)

	if (!extension) {
		throw new Error("Extension not found")
	}

	// Activate the extension if it's not already active.
	const api = extension.isActive ? extension.exports : await extension.activate()

	// TODO: We might want to support a "free" model out of the box so
	// contributors can run the tests locally without having to pay.
	await api.setConfiguration({
		apiProvider: "openrouter",
		openRouterApiKey: process.env.OPENROUTER_API_KEY!,
		openRouterModelId: "anthropic/claude-3.5-sonnet",
	})

	await waitUntilReady({ api })

	// Expose the API to the tests.
	globalThis.api = api

	// Add all the tests to the runner.
	const mocha = new Mocha({ ui: "tdd", timeout: 300_000 })
	const cwd = path.resolve(__dirname, "..")
	;(await glob("**/**.test.js", { cwd })).forEach((testFile) => mocha.addFile(path.resolve(cwd, testFile)))

	// Let's go!
	return new Promise<void>((resolve, reject) =>
		mocha.run((failures) => (failures === 0 ? resolve() : reject(new Error(`${failures} tests failed.`)))),
	)
}
