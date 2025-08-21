import * as assert from "assert"

suite("Minimal E2E Test Suite", () => {
	test("Basic test should pass", () => {
		assert.strictEqual(1 + 1, 2)

	test("Should be able to run without extension", () => {
		const result = "test"
		assert.ok(result === "test", "String comparison should work")

})})})
