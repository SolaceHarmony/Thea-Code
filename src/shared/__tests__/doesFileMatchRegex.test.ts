import { assert } from "chai"
import * as sinon from "sinon"
import { doesFileMatchRegex } from "../modes"

describe("doesFileMatchRegex", () => {
  it("returns true when pattern matches", () => {
    assert.strictEqual(doesFileMatchRegex("src/file.ts", "\\.ts$"), true)
  })

  it("returns false when pattern does not match", () => {
    assert.strictEqual(doesFileMatchRegex("src/file.ts", "\\.js$"), false)
  })

  it("handles invalid regex gracefully", () => {
    const errSpy = sinon.spy(console, "error")
    try {
      assert.strictEqual(doesFileMatchRegex("src/file.ts", "["), false)
      assert.isTrue(errSpy.called)
    } finally {
      errSpy.restore()
    }
  })
})
