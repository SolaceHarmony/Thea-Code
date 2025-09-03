import * as assert from "assert"
import * as vscode from "vscode"
import { EXTENSION_ID } from "../thea-constants"
import type { TheaCodeAPI } from "../../../exports/thea-code"

declare global {
	var api: TheaCodeAPI
}

suite("Global E2E Setup", () => {
	suiteSetup(async function () {
		this.timeout(60_000)

		const extension = vscode.extensions.getExtension(EXTENSION_ID)
		assert.ok(extension, `Extension ${EXTENSION_ID} not found`)

		if (!extension.isActive) {
			await extension.activate()
		}

		const exp: unknown = extension.exports
		let resolvedApi: TheaCodeAPI | undefined

		function hasGetAPI(x: unknown): x is { getAPI: () => TheaCodeAPI } {
			return typeof (x as { getAPI?: unknown })?.getAPI === "function"
		}
		function hasApiOrTest(x: unknown): x is { api?: TheaCodeAPI; isTestMode?: boolean } {
			const o = x as { api?: unknown; isTestMode?: unknown }
			return typeof o === "object" && o !== null && ("api" in o || "isTestMode" in o)
		}

		if (hasApiOrTest(exp) && (typeof exp.isTestMode === "boolean" || typeof exp.api !== "undefined")) {
			resolvedApi = (exp.api as TheaCodeAPI | undefined) ?? (exp as TheaCodeAPI)
		} else if (hasGetAPI(exp)) {
			resolvedApi = exp.getAPI()
		} else {
			resolvedApi = exp as TheaCodeAPI
		}

		assert.ok(resolvedApi, "TheaCodeAPI not available from extension exports")
		global.api = resolvedApi
	})
})
