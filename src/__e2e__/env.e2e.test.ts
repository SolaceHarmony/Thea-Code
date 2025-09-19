import * as assert from "assert"

suite("E2E env", () => {
  test("E2E flags are set", () => {
    // These are set by our e2e launchers
    assert.strictEqual(process.env.THEA_E2E, "1")
    assert.strictEqual(process.env.NODE_ENV ?? "test", "test")
  })
})
