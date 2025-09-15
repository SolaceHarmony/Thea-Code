
import { GLOBAL_STATE_KEYS } from "../../../../schemas"
import { strict as assert } from "node:assert"

describe("GLOBAL_STATE_KEYS", () => {
	it("should contain provider settings keys", () => {
		assert.ok(GLOBAL_STATE_KEYS.includes("autoApprovalEnabled"))
	})

	it("should contain provider settings keys", () => {
		assert.ok(GLOBAL_STATE_KEYS.includes("anthropicBaseUrl"))
	})

	it("should not contain secret state keys", () => {
		assert.ok(!GLOBAL_STATE_KEYS.includes("openRouterApiKey"))
	})
})
