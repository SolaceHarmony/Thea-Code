/**
 * See: https://code.visualstudio.com/api/working-with-extensions/testing-extension
 */

import { defineConfig } from "@vscode/test-cli"
import path from "path"
const __dirname = path.dirname(new URL(import.meta.url).pathname)
// const pkg = JSON.parse(fs.readFileSync(path.join(__dirname, "../package.json"), "utf-8"))
// const EXTENSION_ID = `${pkg.publisher}.${pkg.name}`

export default defineConfig({
	label: "integrationTest",
	files: "out/suite/selected/**/*.test.js",
	workspaceFolder: ".",
	mocha: {
		ui: "tdd",
		timeout: 60000,
	},
	// Don't reference workspace TS constants in this ESM config; pass minimal args only.
	launchArgs: ["--disable-extensions"],
})