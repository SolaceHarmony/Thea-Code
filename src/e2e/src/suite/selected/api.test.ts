import * as assert from "assert"
import type { TheaCodeAPI } from "../../exports/thea-code"

suite("API exposure", () => {
  test("global.api is set by setup", () => {
    const api = (globalThis as unknown as { api?: TheaCodeAPI }).api
    assert.ok(api, "global.api not set")
    assert.strictEqual(typeof api, "object")
  })
})
