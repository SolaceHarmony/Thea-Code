
import { GLOBAL_STATE_KEYS } from "../index"

import * as assert from 'assert'
suite("GLOBAL_STATE_KEYS", () => {
	test("should contain provider settings keys", () => {
		assert.ok(GLOBAL_STATE_KEYS.includes("autoApprovalEnabled"))

	test("should contain provider settings keys", () => {
		assert.ok(GLOBAL_STATE_KEYS.includes("anthropicBaseUrl"))

	test("should not contain secret state keys", () => {
		assert.ok(!GLOBAL_STATE_KEYS.includes("openRouterApiKey"))
