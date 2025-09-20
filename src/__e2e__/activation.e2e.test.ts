import * as assert from "assert"
import { EXTENSION_ID } from "../e2e/src/thea-constants"
import { ensureActivated } from "../test-support/e2e"

suite("Activation", () => {
  test("extension activates and exposes API", async function () {
    this.timeout(120000)
    const ext = await ensureActivated(EXTENSION_ID)
    assert.ok(ext.exports, "no exports after activation")
    console.log(`[e2e] Activated ${EXTENSION_ID}`)
  })
})
