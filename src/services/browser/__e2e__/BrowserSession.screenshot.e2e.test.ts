import * as assert from "assert"
import * as vscode from "vscode"
import { EXTENSION_ID } from "../../../e2e/src/thea-constants"

suite("BrowserSession real browser screenshot", function () {
  // Real browser launch + navigation can take time in CI
  this.timeout(180_000)

  test("captures screenshot and logs for example.com", async () => {
    const ext = vscode.extensions.getExtension(EXTENSION_ID)
    assert.ok(ext, `Extension ${EXTENSION_ID} not found`)
    if (!ext.isActive) {
      await ext.activate()
    }

    // Execute hidden E2E command that drives a real BrowserSession instance
    const result = await vscode.commands.executeCommand(
      "thea-code.test.browserCapture",
      { url: "https://example.com" }
    )

    assert.ok(result, "No result returned from browserCapture command")
    assert.strictEqual(typeof result, "object")

    const r = result as { screenshot?: string; logs?: string; currentUrl?: string }

    assert.ok(typeof r.screenshot === "string" && r.screenshot.startsWith("data:image/"), "Expected image data URL")
    assert.ok(typeof r.logs === "string", "Expected logs string")
    assert.ok(typeof r.currentUrl === "string" && r.currentUrl.includes("example.com"), "URL should include example.com")
  })
})
