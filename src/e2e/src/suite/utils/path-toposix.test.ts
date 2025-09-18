import * as assert from "assert"
import * as path from "path"
import * as fs from "fs"

declare global {
  interface String { toPosix(): string }
}

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

// Load side-effect module from built output to install toPosix()
const repoRoot = findRepoRoot(__dirname)
// eslint-disable-next-line @typescript-eslint/no-var-requires
require(path.join(repoRoot, "out", "utils", "path.js"))

suite("String.prototype.toPosix", () => {
  test("converts backslashes to slashes", () => {
    const input = "C\\Users\\test\\file.txt"
    assert.strictEqual(input.toPosix(), "C/Users/test/file.txt")
  })

  test("leaves extended-length Windows paths intact", () => {
    const input = "\\\\?\\C:\\Very\\Long\\Path"
    assert.strictEqual(input.toPosix(), "\\\\?\\C:\\Very\\Long\\Path")
  })

  test("idempotent on POSIX style", () => {
    const input = "/home/user/file.txt"
    assert.strictEqual(input.toPosix(), "/home/user/file.txt")
  })
})
