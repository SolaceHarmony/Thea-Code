import * as assert from "assert"
import * as fs from "fs"
import * as path from "path"

suite("Repository version", () => {
  test("package.json has a semver version", () => {
    // Find repo root by walking up to a directory that contains package.json with name "thea-code"
    function findRepoRoot(startDir: string): string {
      let dir = startDir
      for (let i = 0; i < 10; i++) {
        try {
          const pkgPath = path.join(dir, "package.json")
          const pkg = JSON.parse(fs.readFileSync(pkgPath, "utf8"))
          if (pkg && pkg.name === "thea-code") return dir
        } catch {}
        const parent = path.dirname(dir)
        if (parent === dir) break
        dir = parent
      }
      return path.resolve(startDir, "../../../../..")
    }

    const repoRoot = findRepoRoot(__dirname)
    const pkg = JSON.parse(fs.readFileSync(path.join(repoRoot, "package.json"), "utf8"))
    assert.match(pkg.version, /^\d+\.\d+\.\d+(?:-.+)?$/)
  })
})
