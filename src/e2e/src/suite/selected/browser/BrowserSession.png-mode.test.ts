import * as assert from "assert"
import * as vscode from "vscode"
import { EXTENSION_ID } from "../../thea-constants"

suite("BrowserSession PNG screenshot (primary path)", function () {
  // Real browser launch + navigation can take time locally
  this.timeout(180_000)

  test("captures PNG screenshot successfully", async () => {
    const ext = vscode.extensions.getExtension(EXTENSION_ID)
    assert.ok(ext, `Extension ${EXTENSION_ID} not found`)
    if (!ext!.isActive) {
      await ext!.activate()
    }

    const result = (await vscode.commands.executeCommand(
      "thea-code.test.browserCapture",
      {
        url: "https://example.com",
        format: "png" as const,
        fullPage: false,
        clipping: false,
        viewport: "900x600",
      }
    )) as unknown

    assert.ok(result, "No result returned from browserCapture command")
    assert.strictEqual(typeof result, "object")

    const r = result as { screenshot?: string; logs?: string; currentUrl?: string }

    assert.ok(
      typeof r.screenshot === "string" && r.screenshot.startsWith("data:image/png;base64,"),
      "Expected PNG image data URL"
    )
    assert.ok(typeof r.logs === "string", "Expected logs string field to be present")
    assert.ok(
      typeof r.currentUrl === "string" && r.currentUrl.includes("example.com"),
      "URL should include example.com"
    )
  })
})
