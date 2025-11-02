import * as assert from "assert"
import type { TheaCodeAPI } from "../../../exports/thea-code"
import { XmlMatcher } from "../../utils/xml-matcher"

declare global {
  // Provided by setup.test.ts
  var api: TheaCodeAPI
}

suite("Ollama Integration (recovered)", () => {
  test("API is ready", () => {
    assert.ok(globalThis.api && typeof globalThis.api.isReady === "function", "Thea API present")
    assert.strictEqual(globalThis.api.isReady(), true, "Thea API should be ready after setup")
  })

  test("Configuration is accessible", () => {
    const cfg = globalThis.api.getConfiguration()
    assert.ok(cfg, "Configuration object should be returned")
  })

  test("XmlMatcher reasoning parses", () => {
    const matcher = new XmlMatcher("think")
    const chunks = [...matcher.update("<think>hello</think>"), ...matcher.final()]
    assert.strictEqual(chunks.length, 1)
    const c = chunks[0]
    assert.strictEqual(Boolean(c.matched), true)
    if (typeof c.data === "string") {
      assert.strictEqual(c.data, "hello")
    }
  })
})
