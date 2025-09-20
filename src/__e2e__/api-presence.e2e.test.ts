import * as assert from "assert"
import type { TheaCodeAPI } from "../exports/thea-code"

suite("API Presence", () => {
  test("global api is available after setup", () => {
    // setup.test.ts activates the extension and sets global.api
    const g = globalThis as unknown as { api?: TheaCodeAPI }
    assert.ok(g.api, "global.api should be defined")
  })
})
