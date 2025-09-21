import * as assert from 'assert'
import { expect } from 'chai'
import { getNonce } from "../getNonce"
import * as sinon from 'sinon'

suite("getNonce", () => {
	test("generates a 32-character alphanumeric string", () => {
		const nonce = getNonce()
		expect(nonce).toMatch(/^[A-Za-z0-9]{32}$/)
	})

	test("returns a new value for each call", () => {
		const first = getNonce()
		const second = getNonce()
		assert.notStrictEqual(first, second)
		expect(first).toMatch(/^[A-Za-z0-9]{32}$/)
		expect(second).toMatch(/^[A-Za-z0-9]{32}$/)
	})
// Mock cleanup
