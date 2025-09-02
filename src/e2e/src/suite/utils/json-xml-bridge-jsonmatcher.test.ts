import { JsonMatcher } from "../json-xml-bridge"

import * as assert from 'assert'
suite("JsonMatcher buffer cap", () => {
  test("spills early text when exceeding max buffer length to avoid unbounded growth", () => {
    const matcher = new JsonMatcher("tool_use")
    // Create a long non-JSON string exceeding 256KB
    const chunk = "x".repeat(300 * 1024)
    const results = matcher.update(chunk)
    // Expect at least one non-matched spill result
    assert.ok(results.length > 0)
    // All results should be non-matched text since there is no JSON
    for (const r of results) {
      assert.ok(!r.matched)
      assert.ok(typeof r.data === "string")
