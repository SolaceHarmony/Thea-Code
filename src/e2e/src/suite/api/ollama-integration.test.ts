import * as assert from "assert"
import * as path from "path"
import * as fs from "fs"
import type { TheaCodeAPI } from "../../../exports/thea-code"

declare global {
  // Provided by setup.test.ts
  var api: TheaCodeAPI
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

const repoRoot = findRepoRoot(__dirname)
// Use the real xml-matcher implementation checked in under src/utils
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { XmlMatcher } = require(path.join(repoRoot, "src", "utils", "xml-matcher.js"))

suite("Ollama Integration (recovered)", () => {
  test("API is ready", () => {
    assert.ok(globalThis.api && typeof globalThis.api.isReady === "function", "Thea API present")
    assert.strictEqual(globalThis.api.isReady(), true, "Thea API should be ready after setup")
  })

  test("Configuration is accessible", () => {
    const cfg = globalThis.api.getConfiguration()
    assert.ok(cfg, "Configuration object should be returned")
  })

  test("XmlMatcher reasoning parses", () => {
    const matcher = new XmlMatcher("think")
    const chunks = [...matcher.update("<think>hello</think>"), ...matcher.final()]
    assert.strictEqual(chunks.length, 1)
    const c = chunks[0]
    assert.strictEqual(Boolean(c.matched), true)
    if (typeof c.data === "string") {
      assert.strictEqual(c.data, "hello")
    }
  })
})
