// npx jest src/schemas/__tests__/index.test.ts

import { GLOBAL_STATE_KEYS } from "../index"

import * as assert from 'assert'
suite("GLOBAL_STATE_KEYS", () => {
	test("should contain provider settings keys", () => {
		assert.ok(GLOBAL_STATE_KEYS.includes("autoApprovalEnabled"))

	test("should contain provider settings keys", () => {
		assert.ok(GLOBAL_STATE_KEYS.includes("anthropicBaseUrl"))

	test("should not contain secret state keys", () => {
		expect(GLOBAL_STATE_KEYS).not.toContain("openRouterApiKey")
