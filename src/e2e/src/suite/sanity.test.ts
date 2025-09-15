import { strict as assert } from "node:assert"
import * as vscode from "vscode"
import { EXTENSION_ID } from "../thea-constants"

describe("Sanity: VS Code + API availability", () => {
	it("vscode module is resolvable", () => {
		assert.ok(typeof vscode.workspace !== "undefined", "vscode.workspace should be defined")
	})

	it("extension can activate and export API", async function () {
		this.timeout(20000)
		const ext = vscode.extensions.getExtension(EXTENSION_ID)
		assert.ok(ext, `Extension ${EXTENSION_ID} not found`)
		if (!ext.isActive) {
			await ext.activate()
		}

		assert.ok(ext.exports, "Extension exports should be defined")
	})
})
