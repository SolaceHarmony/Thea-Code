import { expect } from "chai"
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
    readStub.callsFake(async (p: string | Buffer | URL) => {
      const fp = String(p)
      if (fp === path.join(cwd, files[0])) return "first"
      if (fp === path.join(cwd, files[1])) throw errWith(undefined, "ENOENT: not found")
      if (fp === path.join(cwd, files[2])) throw errWith("EISDIR")
      throw new Error("unexpected path " + fp)
    })

    const rules = await loadRuleFiles(cwd)
    expect(rules).to.contain("# Rules from .Thearules:")
    expect(rules).to.contain("first")
    // Non-existing or directory should be omitted silently
    expect(rules).to.not.contain(".cursorrules")
    expect(rules).to.not.contain(".windsurfrules")

    // Ensure ordering (first-only)
    const firstIndex = rules.indexOf("# Rules from .Thearules:")
    expect(firstIndex).to.be.greaterThan(-1)
  })

  it("addCustomInstructions: composes language, global, mode, and rules with correct sections and priority", async () => {
    const cwd = "/proj"
    const mode = "architect"

    // Stub mode-specific file and generic rules
    readStub.callsFake(async (p: string | Buffer | URL) => {
      const fp = String(p)
      if (fp.endsWith(`.Thearules-${mode}`)) return "mode-rule-1\nmode-rule-2"
      if (fp.endsWith(".Thearules")) return "generic-A"
      if (fp.endsWith(".cursorrules")) return "generic-B"
      if (fp.endsWith(".windsurfrules")) throw errWith("ENOENT")
      throw errWith("ENOENT") // any other path
    })

    const result = await addCustomInstructions(
      "Use architecture best practices.",
      "Always keep responses concise.",
      cwd,
      mode,
      { language: "en", theaIgnoreInstructions: "# .thea_ignore: ignore node_modules" }
    )

    // Ensure presence of top scaffold
    expect(result).to.contain("USER'S CUSTOM INSTRUCTIONS")

    // Language section
    expect(result).to.contain('Language Preference:')
    expect(result).to.contain('"English" (en)')

    // Global then mode-specific ordering
    const gi = result.indexOf("Global Instructions:")
    const mi = result.indexOf("Mode-specific Instructions:")
    expect(gi).to.be.lessThan(mi)

    // Rules section includes mode-specific rules, then theaIgnore, then generic rules
    const ri = result.indexOf("Rules:")
    expect(ri).to.be.greaterThan(-1)
    const mri = result.indexOf(`# Rules from .Thearules-${mode}:`)
    const ti = result.indexOf("# .thea_ignore: ignore node_modules")
    const gri = result.indexOf("# Rules from .Thearules:")
    expect(mri).to.be.greaterThan(ri)
    expect(ti).to.be.greaterThan(mri)
    expect(gri).to.be.greaterThan(ti)

    // Content checks
    expect(result).to.contain("mode-rule-1")
    expect(result).to.contain("generic-A")
    expect(result).to.contain("generic-B")
  })

  it("addCustomInstructions: handles empty cwd by falling back to '.' and warning (non-throwing)", async () => {
    // nothing must throw if cwd is empty; our implementation logs a warning and uses '.'
    readStub.rejects(errWith("ENOENT"))
    const result = await addCustomInstructions("", "", "", "", { language: "en" })
    expect(result).to.equal("")
  })
})
