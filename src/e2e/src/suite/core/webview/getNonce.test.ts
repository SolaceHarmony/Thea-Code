import { getNonce } from "../getNonce"

import * as assert from 'assert'
suite("getNonce", () => {
	test("generates a 32-character alphanumeric string", () => {
		const nonce = getNonce()
		assert.ok(nonce.match(/^[A-Za-z0-9]{32}$/))

	test("returns a new value for each call", () => {
		const first = getNonce()
		const second = getNonce()
		assert.notStrictEqual(first, second)
		assert.ok(first.match(/^[A-Za-z0-9]{32}$/))
		assert.ok(second.match(/^[A-Za-z0-9]{32}$/))
