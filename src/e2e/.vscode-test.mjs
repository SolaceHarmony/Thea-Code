/**
 * See: https://code.visualstudio.com/api/working-with-extensions/testing-extension
 */

import { defineConfig } from "@vscode/test-cli"
import fs from "fs"
import path from "path"
const __dirname = path.dirname(new URL(import.meta.url).pathname)
const pkg = JSON.parse(fs.readFileSync(path.join(__dirname, "../package.json"), "utf-8"))
const EXTENSION_ID = `${pkg.publisher}.${pkg.name}`

export default defineConfig({
	label: "integrationTest",
	files: "out/suite/**/*.test.js",
	workspaceFolder: ".",
	mocha: {
		ui: "tdd",
		timeout: 60000,
	},
	launchArgs: [`--enable-proposed-api=${EXTENSION_ID}`, "--disable-extensions"],
})