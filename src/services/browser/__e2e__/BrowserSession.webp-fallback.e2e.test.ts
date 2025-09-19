import * as assert from "assert"
import * as vscode from "vscode"
import { EXTENSION_ID } from "../../../e2e/src/thea-constants"

suite("BrowserSession WEBP screenshot (no fallback)", function () {
  // Real browser launch + navigation can take time in local runs
  this.timeout(180_000)

  test("captures WEBP screenshot successfully", async () => {
    const ext = vscode.extensions.getExtension(EXTENSION_ID)
    assert.ok(ext, `Extension ${EXTENSION_ID} not found`)
    if (!ext.isActive) {
      await ext.activate()
    }

    const result = await vscode.commands.executeCommand(
      "thea-code.test.browserCapture",
      {
        url: "https://example.com",
        format: "webp",
        fullPage: false,
        clipping: false,
        viewport: "900x600",
      }
    )

    assert.ok(result, "No result returned from browserCapture command")
    assert.strictEqual(typeof result, "object")

    const r = result as { screenshot?: string; logs?: string; currentUrl?: string }

    assert.ok(
      typeof r.screenshot === "string" && r.screenshot.startsWith("data:image/webp;base64,"),
      "Expected WEBP image data URL without fallback"
    )
    assert.ok(typeof r.logs === "string", "Expected logs string field to be present")
    assert.ok(
      typeof r.currentUrl === "string" && r.currentUrl.includes("example.com"),
      "URL should include example.com"
    )
  })
})
