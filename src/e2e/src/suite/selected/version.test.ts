import * as assert from "assert"
import * as fs from "fs"
import * as path from "path"

suite("Repository version", () => {
  test("package.json has a semver version", () => {
    const repoRoot = path.resolve(__dirname, "../../../../../..")
    const pkg = JSON.parse(fs.readFileSync(path.join(repoRoot, "package.json"), "utf8"))
    assert.match(pkg.version, /^\d+\.\d+\.\d+(?:-.+)?$/)
  })
})
