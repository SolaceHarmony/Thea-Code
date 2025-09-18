import * as assert from "assert"

suite("Env Flags", () => {
  test("required env flags are set", () => {
    assert.strictEqual(process.env.THEA_E2E, "1", "THEA_E2E should be 1 during e2e")
    assert.strictEqual(process.env.NODE_ENV, "test", "NODE_ENV should default to test during e2e")
  })
})
