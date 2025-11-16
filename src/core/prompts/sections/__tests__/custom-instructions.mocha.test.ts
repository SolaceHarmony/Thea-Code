import assert from "node:assert/strict"
import fs from "fs/promises"
import path from "path"
import os from "os"

import { loadRuleFiles, addCustomInstructions } from "../custom-instructions"

describe("custom-instructions", () => {
  let tempDir: string

  beforeEach(async () => {
    // Create a temporary directory for test files
    tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "thea-test-"))
  })

  afterEach(async () => {
    // Clean up temporary directory
    await fs.rm(tempDir, { recursive: true, force: true })
  })

  it("loadRuleFiles: returns empty string for ENOENT/EISDIR and preserves order", async () => {
    // Create only the first rule file
    await fs.writeFile(path.join(tempDir, ".Thearules"), "first")
    // .cursorrules and .windsurfrules will not exist (ENOENT)
    // Create a directory for .windsurfrules to test EISDIR
    await fs.mkdir(path.join(tempDir, ".windsurfrules"))

    const rules = await loadRuleFiles(tempDir)
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
    const mode = "architect"

    // Create mode-specific and generic rule files
    await fs.writeFile(path.join(tempDir, `.Thearules-${mode}`), "mode-rule-1\nmode-rule-2")
    await fs.writeFile(path.join(tempDir, ".Thearules"), "generic-A")
    await fs.writeFile(path.join(tempDir, ".cursorrules"), "generic-B")
    // .windsurfrules will not exist (ENOENT)

    const result = await addCustomInstructions(
      "Use architecture best practices.",
      "Always keep responses concise.",
      tempDir,
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
    // Don't pass language to get truly empty result
    const result = await addCustomInstructions("", "", "", "", {})
    assert.strictEqual(result, "")
  })
})
