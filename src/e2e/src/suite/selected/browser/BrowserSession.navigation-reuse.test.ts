import * as assert from "assert"
import * as vscode from "vscode"
import { EXTENSION_ID } from "../../thea-constants"

suite("BrowserSession navigation reuse (same root domain)", function () {
  // Real browser launch + navigation can take time locally
  this.timeout(180_000)

  test("reuses the same tab when navigating within example.com", async () => {
    const ext = vscode.extensions.getExtension(EXTENSION_ID)
    assert.ok(ext, `Extension ${EXTENSION_ID} not found`)
    if (!ext!.isActive) {
      await ext!.activate()
    }

    const result = (await vscode.commands.executeCommand(
      "thea-code.test.browserCapture",
      {
        urls: [
          "https://example.com",
          "https://example.com/foo",
        ],
        viewport: "900x600",
        fullPage: false,
        clipping: false,
      }
    )) as unknown

    assert.ok(result, "No result returned from browserCapture command")
    assert.strictEqual(typeof result, "object")

    const r = result as { steps?: Array<Record<string, any>> }
    assert.ok(Array.isArray(r.steps), "Expected steps array in result")
    assert.strictEqual(r.steps!.length, 2, "Expected two navigation steps")

    const first = r.steps![0]!
    const second = r.steps![1]!

    // Basic payload checks
    assert.ok(typeof first.screenshot === "string" && first.screenshot.startsWith("data:image/"), "Step 1 should include screenshot")
    assert.ok(typeof second.screenshot === "string" && second.screenshot.startsWith("data:image/"), "Step 2 should include screenshot")

    // Reuse assertions
    assert.ok(first.createdNewTab, "First navigation should create a new tab")
    assert.ok(second.reusedTab, "Second navigation within same domain should reuse the tab")

    // URL assertions
    assert.ok(typeof second.currentUrl === "string" && second.currentUrl.includes("example.com"), "URL should include example.com")
    assert.ok(second.currentUrl.includes("/foo"), "Second URL should reflect the /foo path")

    // Page count should remain the same between steps when reusing the same tab
    if (typeof first.pageCount === "number" && typeof second.pageCount === "number") {
      assert.strictEqual(second.pageCount, first.pageCount, "Page count should not increase when reusing the same tab")
    }
  })
})
