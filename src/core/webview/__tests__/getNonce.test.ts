import { strict as assert } from "node:assert"
import { getNonce } from "../getNonce"

describe("getNonce", () => {
	it("generates a 32-character alphanumeric string", () => {
		const nonce = getNonce()
		assert.match(nonce, /^[A-Za-z0-9]{32}$/)
	})

	it("returns a new value for each call", () => {
		const first = getNonce()
		const second = getNonce()
		assert.notEqual(first, second)
		assert.match(first, /^[A-Za-z0-9]{32}$/)
		assert.match(second, /^[A-Za-z0-9]{32}$/)
	})
})
