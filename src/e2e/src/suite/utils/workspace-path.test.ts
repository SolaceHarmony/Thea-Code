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
const { getWorkspacePath } = require(path.join(repoRoot, "out", "utils", "path-vscode.js"))

suite("getWorkspacePath (VS Code)", () => {
  test("returns an existing directory (workspace or homedir)", () => {
    const p = getWorkspacePath()
    assert.strictEqual(typeof p, "string")
    assert.ok(p.length > 0)
    assert.ok(fs.existsSync(p), `Path does not exist: ${p}`)
  })
})
