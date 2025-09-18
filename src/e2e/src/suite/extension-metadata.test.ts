import * as assert from "assert"
import * as vscode from "vscode"
import { EXTENSION_ID } from "../thea-constants"

suite("Extension metadata", () => {
  test("extension is present and active", async function () {
    this.timeout(60000)
    const ext = vscode.extensions.getExtension(EXTENSION_ID)
    assert.ok(ext, `Extension ${EXTENSION_ID} not found`)
    if (!ext!.isActive) {
      await ext!.activate()
    }
    assert.ok(ext!.isActive, "extension did not activate")
  })

  test("package.json basic fields", () => {
    const ext = vscode.extensions.getExtension(EXTENSION_ID)!
    const pkg = ext.packageJSON as any
    assert.strictEqual(pkg.publisher, "SolaceHarmony")
    assert.strictEqual(pkg.name, "thea-code")
    assert.ok(Array.isArray(pkg.activationEvents))
  })
})
