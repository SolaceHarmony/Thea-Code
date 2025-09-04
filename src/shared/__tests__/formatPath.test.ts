// npx jest src/shared/__tests__/formatPath.test.ts
import { strict as assert } from "node:assert"
import { formatPath } from "../formatPath"

describe("formatPath", () => {
	it("adds leading backslash on Windows", () => {
		const result = formatPath("folder/file", "win32")
		assert.equal(result, "\\folder/file")
	})

	it("preserves existing leading separator", () => {
		const result = formatPath("/already", "darwin")
		assert.equal(result, "/already")
	})

	it("escapes spaces according to platform", () => {
		assert.equal(formatPath("my file", "win32"), "\\my/ file")
		assert.equal(formatPath("my file", "linux"), "/my\\ file")
	})

	it("can skip space escaping", () => {
		const result = formatPath("my file", "win32", false)
		assert.equal(result, "\\my file")
	})
})
