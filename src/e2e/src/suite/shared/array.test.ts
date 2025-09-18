import * as assert from "assert"
import * as path from "path"
import * as fs from "fs"

function findRepoRoot(startDir: string): string {
  let dir = startDir
  for (let i = 0; i < 10; i++) {
    try {
      const pkg = JSON.parse(fs.readFileSync(path.join(dir, "package.json"), "utf8"))
      if (pkg && pkg.name === "thea-code") return dir
    } catch {}
    const parent = path.dirname(dir)
    if (parent === dir) break
    dir = parent
  }
  return path.resolve(startDir, "../../../../..")
}

const repoRoot = findRepoRoot(__dirname)
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { findLastIndex, findLast } = require(path.join(repoRoot, "out", "shared", "array.js"))

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
