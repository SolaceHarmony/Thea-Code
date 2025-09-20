import * as assert from "assert"
import * as path from "path"

// Load from the built extension output to avoid TS rootDir restrictions
const repoRoot = path.resolve(__dirname, "../../../../../..")
// eslint-disable-next-line @typescript-eslint/no-require-imports
const arrayMod = require(path.join(repoRoot, "out", "shared", "array.js")) as {
  findLastIndex: <T>(arr: T[], pred: (x: T) => boolean) => number
  findLast: <T>(arr: T[], pred: (x: T) => boolean) => T | undefined
}
const { findLastIndex, findLast } = arrayMod

suite("shared/array", () => {
  suite("findLastIndex", () => {
    test("returns last index matching predicate", () => {
      const arr = [1, 2, 3, 2]
      const idx = findLastIndex(arr, (x: number) => x === 2)
      assert.strictEqual(idx, 3)
    })

    test("returns -1 when no match", () => {
      const arr = [1, 2, 3]
      const idx = findLastIndex(arr, (x: number) => x === 4)
      assert.strictEqual(idx, -1)
    })
  })

  suite("findLast", () => {
    test("returns last element matching predicate", () => {
      const arr = ["a", "b", "c", "b"]
      const val = findLast(arr, (x: string) => x === "b")
      assert.strictEqual(val, "b")
    })

    test("returns undefined when no match", () => {
      const arr: number[] = []
      const val = findLast(arr, (x: number) => x > 0)
      assert.strictEqual(val, undefined)
    })
  })
})
