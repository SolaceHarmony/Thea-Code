import * as assert from 'assert'
import { findLastIndex, findLast } from "../../../../shared/array"
suite("array utilities", () => {
	suite("findLastIndex", () => {
		test("returns last index matching predicate", () => {
			const arr = [1, 2, 3, 2]
			const idx = findLastIndex(arr, (x) => x === 2)
			assert.strictEqual(idx, 3)

		test("returns -1 when no match", () => {
			const arr = [1, 2, 3]
			const idx = findLastIndex(arr, (x) => x === 4)
			assert.strictEqual(idx, -1)

	suite("findLast", () => {
		test("returns last element matching predicate", () => {
			const arr = ["a", "b", "c", "b"]
			const val = findLast(arr, (x) => x === "b")
			assert.strictEqual(val, "b")

		test("returns undefined when no match", () => {
			const arr: number[] = []
			const val = findLast(arr, (x) => x > 0)
			assert.strictEqual(val, undefined)
