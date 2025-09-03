import { formatPath } from "../formatPath"

import * as assert from 'assert'
suite("formatPath", () => {
	test("adds leading backslash on Windows", () => {
		const result = formatPath("folder/file", "win32")
		assert.strictEqual(result, "\\folder/file")

	test("preserves existing leading separator", () => {
		const result = formatPath("/already", "darwin")
		assert.strictEqual(result, "/already")

	test("escapes spaces according to platform", () => {
		expect(formatPath("my file", "win32")).toBe("\\my/ file")
		expect(formatPath("my file", "linux")).toBe("/my\\ file")

	test("can skip space escaping", () => {
		const result = formatPath("my file", "win32", false)
		assert.strictEqual(result, "\\my file")
