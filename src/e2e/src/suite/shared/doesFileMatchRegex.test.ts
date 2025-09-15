import { doesFileMatchRegex } from "../../../../shared/modes"
import { strict as assert } from "node:assert"
import * as sinon from "sinon"

describe("doesFileMatchRegex", () => {
	it("returns true when pattern matches", () => {
		assert.equal(doesFileMatchRegex("src/file.ts", "\\.ts$"), true)
	})

	it("returns false when pattern does not match", () => {
		assert.equal(doesFileMatchRegex("src/file.ts", "\\.js$"), false)
	})

	it("handles invalid regex gracefully", () => {
		const errSpy = sinon.stub(console, "error")
		assert.equal(doesFileMatchRegex("src/file.ts", "["), false)
		assert.ok(errSpy.called)
		errSpy.restore()
	})
})
