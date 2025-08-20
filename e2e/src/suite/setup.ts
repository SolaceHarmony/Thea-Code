import * as assert from "assert"
import * as vscode from "vscode"
import { EXTENSION_ID } from "../thea-constants"
import type { TheaCodeAPI } from "../../../src/exports/thea-code"

declare global {
	var api: TheaCodeAPI
}

suite("Global E2E Setup", () => {
	suiteSetup(async function () {
		this.timeout(60_000)

		const extension = vscode.extensions.getExtension(EXTENSION_ID)
		assert.ok(extension, `Extension ${EXTENSION_ID} not found`)

		if (!extension!.isActive) {
			await extension!.activate()
		}

		const exportsObj = extension!.exports as
			| TheaCodeAPI
			| { getAPI?: () => TheaCodeAPI; api?: TheaCodeAPI }

		const resolvedApi =
			exportsObj && typeof (exportsObj as any).getAPI === "function"
				? (exportsObj as any).getAPI()
				: ((exportsObj as any).api ?? (exportsObj as TheaCodeAPI))

		assert.ok(resolvedApi, "TheaCodeAPI not available from extension exports")
		global.api = resolvedApi
	})
})
