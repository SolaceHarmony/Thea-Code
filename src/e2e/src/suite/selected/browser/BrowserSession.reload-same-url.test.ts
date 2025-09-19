import * as assert from "assert"
import * as vscode from "vscode"
import { EXTENSION_ID } from "../../thea-constants"

suite("BrowserSession reload on same URL", function () {
  // Real browser launch + navigation can take time locally
  this.timeout(180_000)

  test("reloads when navigating to the same URL again", async () => {
    const ext = vscode.extensions.getExtension(EXTENSION_ID)
    assert.ok(ext, `Extension ${EXTENSION_ID} not found`)
    if (!ext!.isActive) {
      await ext!.activate()
    }

    const result = (await vscode.commands.executeCommand(
      "thea-code.test.browserCapture",
      {
        urls: [
          "https://example.com/",
          "https://example.com/",
        ],
        viewport: "900x600",
        fullPage: false,
        clipping: false,
        format: "webp",
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

    // Semantics: first creates new tab, second should be reload + reuse
    assert.ok(first.createdNewTab, "First navigation should create a new tab")
    assert.ok(second.reusedTab, "Second navigation to same URL should reuse the tab")
    assert.ok(second.reloaded, "Second navigation to same URL should trigger a reload")

    // URL assertions (normalized equality)
    assert.ok(typeof first.currentUrl === "string" && first.currentUrl.includes("example.com"))
    assert.ok(typeof second.currentUrl === "string" && second.currentUrl.includes("example.com"))

    // Page count should remain stable across reload
    if (typeof first.pageCount === "number" && typeof second.pageCount === "number") {
      assert.strictEqual(second.pageCount, first.pageCount, "Page count should remain the same across reload")
    }

    // Ensure a later capture timestamp (sanity check that a second action occurred)
    if (typeof first.captureTimestamp === "number" && typeof second.captureTimestamp === "number") {
      assert.ok(second.captureTimestamp >= first.captureTimestamp, "Second capture should be same time or later than first")
    }
  })
})
