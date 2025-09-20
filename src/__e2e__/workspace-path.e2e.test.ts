import * as assert from "assert"
import * as fs from "fs"
import * as path from "path"

// Load from built extension output to avoid TS rootDir restrictions
const repoRoot = path.resolve(__dirname, "../../../../../..")
// eslint-disable-next-line @typescript-eslint/no-require-imports
const pathVscode = require(path.join(repoRoot, "out", "utils", "path-vscode.js")) as { getWorkspacePath: () => string }
const { getWorkspacePath } = pathVscode

suite("getWorkspacePath (VS Code)", () => {
  test("returns an existing directory (workspace or homedir)", () => {
    const p = getWorkspacePath()
    assert.strictEqual(typeof p, "string")
    assert.ok(p.length > 0)
    assert.ok(fs.existsSync(p), `Path does not exist: ${p}`)
  })
})
