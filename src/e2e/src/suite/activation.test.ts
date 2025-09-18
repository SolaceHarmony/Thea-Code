import * as assert from "assert"
import * as vscode from "vscode"
import { EXTENSION_ID } from "../thea-constants"

suite("Activation", () => {
  test("extension activates and exposes API", async function () {
    this.timeout(120000)
    const ext = vscode.extensions.getExtension(EXTENSION_ID)
    assert.ok(ext, `Extension ${EXTENSION_ID} not found`)
    if (!ext!.isActive) {
      await ext!.activate()
    }
    assert.ok(ext!.isActive, "extension did not activate")
    assert.ok(ext!.exports, "no exports after activation")
    console.log(`[e2e] Activated ${EXTENSION_ID}`)
  })
})
