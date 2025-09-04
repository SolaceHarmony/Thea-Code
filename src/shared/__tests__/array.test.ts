// npx jest src/shared/__tests__/array.test.ts
import { strict as assert } from "node:assert"
import { findLastIndex, findLast } from "../array"

describe("array utilities", () => {
	describe("findLastIndex", () => {
		it("returns last index matching predicate", () => {
			const arr = [1, 2, 3, 2]
			const idx = findLastIndex(arr, (x) => x === 2)
			assert.equal(idx, 3)
		})

		it("returns -1 when no match", () => {
			const arr = [1, 2, 3]
			const idx = findLastIndex(arr, (x) => x === 4)
			assert.equal(idx, -1)
		})
	})

	describe("findLast", () => {
		it("returns last element matching predicate", () => {
			const arr = ["a", "b", "c", "b"]
			const val = findLast(arr, (x) => x === "b")
			assert.equal(val, "b")
		})

		it("returns undefined when no match", () => {
			const arr: number[] = []
			const val = findLast(arr, (x) => x > 0)
			assert.strictEqual(val, undefined)
	})
	})
})
