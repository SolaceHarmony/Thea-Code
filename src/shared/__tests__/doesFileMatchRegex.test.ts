import assert from "node:assert/strict"
import sinon from "sinon"
import { doesFileMatchRegex } from "../modes"

describe("doesFileMatchRegex", () => {
  it("returns true when pattern matches", () => {
    assert.strictEqual(doesFileMatchRegex("src/file.ts", "\\.ts$"), true)
  })

  it("returns false when pattern does not match", () => {
    assert.strictEqual(doesFileMatchRegex("src/file.ts", "\\.js$"), false)
  })

  it("handles invalid regex gracefully and logs", () => {
    const errSpy = sinon.spy(console, "error")
    try {
      assert.strictEqual(doesFileMatchRegex("src/file.ts", "["), false)
      assert.ok(errSpy.called, "Expected console.error to be called for invalid regex")
    } finally {
      errSpy.restore()
    }
  })
})
