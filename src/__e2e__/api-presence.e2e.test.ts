import * as assert from "assert"

suite("API Presence", () => {
  test("global api is available after setup", () => {
    // setup.test.ts activates the extension and sets global.api
    assert.ok((globalThis as any).api, "global.api should be defined")
  })
})
