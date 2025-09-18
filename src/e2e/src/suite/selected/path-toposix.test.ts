import * as assert from "assert"
import * as path from "path"

declare global {
  interface String { toPosix(): string }
}

// Load side-effect module from built output to install toPosix()
const repoRoot = path.resolve(__dirname, "../../../../../..")
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
