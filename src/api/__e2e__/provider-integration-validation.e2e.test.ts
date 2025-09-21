import * as assert from "assert"
import type { TheaCodeAPI } from "../../../exports/thea-code"

declare global { var api: TheaCodeAPI }

suite("Provider Integration Validation (recovered)", () => {
  test("API is ready", () => {
    assert.ok(globalThis.api && typeof globalThis.api.isReady === "function")
    assert.strictEqual(globalThis.api.isReady(), true)
  })

  test("Configuration can be read and updated minimally", async () => {
    const cfg = globalThis.api.getConfiguration()
    assert.ok(cfg)
    // Round-trip a no-op setConfiguration to validate call path
    await globalThis.api.setConfiguration(cfg)
  })
})
