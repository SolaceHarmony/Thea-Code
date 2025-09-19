import assert from "node:assert/strict"
import sinon from "sinon"
import * as fsp from "fs/promises"
import path from "path"

import { loadRuleFiles, addCustomInstructions } from "../custom-instructions"

function errWith(code?: string, message?: string): NodeJS.ErrnoException {
  const e = new Error(message ?? code) as NodeJS.ErrnoException
  if (code) e.code = code as NodeJS.ErrnoException["code"]
  return e
}

describe("custom-instructions", () => {
  let readStub: sinon.SinonStub

  beforeEach(() => {
    readStub = sinon.stub(fsp, "readFile")
  })

  afterEach(() => {
    sinon.restore()
  })

  it("loadRuleFiles: returns empty string for ENOENT/EISDIR and preserves order", async () => {
    const cwd = "/proj"
    const files = [".Thearules", ".cursorrules", ".windsurfrules"]

    // Simulate: first exists, second ENOENT (by message), third EISDIR (by code)
    readStub.callsFake((p: string | Buffer | URL) => {
      const fp = String(p)
      if (fp === path.join(cwd, files[0])) return Promise.resolve("first")
      if (fp === path.join(cwd, files[1])) return Promise.reject(errWith(undefined, "ENOENT: not found"))
      if (fp === path.join(cwd, files[2])) return Promise.reject(errWith("EISDIR"))
      return Promise.reject(new Error("unexpected path " + fp))
    })

    const rules = await loadRuleFiles(cwd)
    assert.ok(rules.includes("# Rules from .Thearules:"))
    assert.ok(rules.includes("first"))
    // Non-existing or directory should be omitted silently
    assert.ok(!rules.includes(".cursorrules"))
    assert.ok(!rules.includes(".windsurfrules"))

    // Ensure ordering (first-only)
    const firstIndex = rules.indexOf("# Rules from .Thearules:")
    assert.ok(firstIndex > -1)
  })

  it("addCustomInstructions: composes language, global, mode, and rules with correct sections and priority", async () => {
    const cwd = "/proj"
    const mode = "architect"

    // Stub mode-specific file and generic rules
    readStub.callsFake((p: string | Buffer | URL) => {
      const fp = String(p)
      if (fp.endsWith(`.Thearules-${mode}`)) return Promise.resolve("mode-rule-1\nmode-rule-2")
      if (fp.endsWith(".Thearules")) return Promise.resolve("generic-A")
      if (fp.endsWith(".cursorrules")) return Promise.resolve("generic-B")
      if (fp.endsWith(".windsurfrules")) return Promise.reject(errWith("ENOENT"))
      return Promise.reject(errWith("ENOENT")) // any other path
    })

    const result = await addCustomInstructions(
      "Use architecture best practices.",
      "Always keep responses concise.",
      cwd,
      mode,
      { language: "en", theaIgnoreInstructions: "# .thea_ignore: ignore node_modules" }
    )

    // Ensure presence of top scaffold
    assert.ok(result.includes("USER'S CUSTOM INSTRUCTIONS"))

    // Language section
    assert.ok(result.includes("Language Preference:"))
    assert.ok(result.includes("\"English\" (en)"))

    // Global then mode-specific ordering
    const gi = result.indexOf("Global Instructions:")
    const mi = result.indexOf("Mode-specific Instructions:")
    assert.ok(gi < mi)

    // Rules section includes mode-specific rules, then theaIgnore, then generic rules
    const ri = result.indexOf("Rules:")
    assert.ok(ri > -1)
    const mri = result.indexOf(`# Rules from .Thearules-${mode}:`)
    const ti = result.indexOf("# .thea_ignore: ignore node_modules")
    const gri = result.indexOf("# Rules from .Thearules:")
    assert.ok(mri > ri)
    assert.ok(ti > mri)
    assert.ok(gri > ti)

    // Content checks
    assert.ok(result.includes("mode-rule-1"))
    assert.ok(result.includes("generic-A"))
    assert.ok(result.includes("generic-B"))
  })

  it("addCustomInstructions: handles empty cwd by falling back to '.' and warning (non-throwing)", async () => {
    // nothing must throw if cwd is empty; our implementation logs a warning and uses '.'
    readStub.rejects(errWith("ENOENT"))
    const result = await addCustomInstructions("", "", "", "", { language: "en" })
    assert.strictEqual(result, "")
  })
})
