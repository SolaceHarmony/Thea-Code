// npx jest src/shared/__tests__/doesFileMatchRegex.test.ts
import { doesFileMatchRegex } from "../modes"

import * as assert from 'assert'
import * as sinon from 'sinon'
suite("doesFileMatchRegex", () => {
	test("returns true when pattern matches", () => {
		expect(doesFileMatchRegex("src/file.ts", "\\.ts$")).toBe(true)

	test("returns false when pattern does not match", () => {
		expect(doesFileMatchRegex("src/file.ts", "\\.js$")).toBe(false)

	test("handles invalid regex gracefully", () => {
		const errSpy = sinon.spy(console, "error").callsFake(() => {})
		expect(doesFileMatchRegex("src/file.ts", "[")).toBe(false)
		assert.ok(errSpy.called)
		errSpy.restore()
