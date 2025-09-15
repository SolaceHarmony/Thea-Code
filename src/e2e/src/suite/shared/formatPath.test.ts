import { formatPath } from "../../../../shared/formatPath"
import { strict as assert } from "node:assert"

describe("formatPath", () => {
	it("adds leading backslash on Windows", () => {
		const result = formatPath("folder/file", "win32")
		assert.strictEqual(result, "\\folder/file")
	})

	it("preserves existing leading separator", () => {
		const result = formatPath("/already", "darwin")
		assert.strictEqual(result, "/already")
	})

	it("escapes spaces according to platform", () => {
		assert.equal(formatPath("my file", "win32"), "\\my/ file")
		assert.equal(formatPath("my file", "linux"), "/my\\ file")
	})

	it("can skip space escaping", () => {
		const result = formatPath("my file", "win32", false)
		assert.strictEqual(result, "\\my file")
	})
})
