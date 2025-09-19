import assert from "node:assert/strict"
import { getToolsForMode } from "../modes"
import { TOOL_GROUPS, ALWAYS_AVAILABLE_TOOLS } from "../tool-groups"

describe("getToolsForMode", () => {
  it("accumulates tools from multiple groups and includes ALWAYS_AVAILABLE_TOOLS without duplicates", () => {
    const groups = ["read", "edit"] as const

    const result = getToolsForMode(groups)

    // Should be an array of strings
    assert.ok(Array.isArray(result), "Expected result array")
    result.forEach((t) => assert.equal(typeof t, "string"))

    // Build expected union from config plus ALWAYS_AVAILABLE_TOOLS
    const expectedSet = new Set<string>()
    groups.forEach((g) => {
      TOOL_GROUPS[g].tools.forEach((t) => expectedSet.add(t))
    })
    ALWAYS_AVAILABLE_TOOLS.forEach((t) => expectedSet.add(t))

    // All expected tools should be present
    for (const tool of expectedSet) {
      assert.ok(result.includes(tool), `Expected tool ${tool} to be included`)
    }

    // No duplicates
    assert.equal(new Set(result).size, result.length, "Tools array should not contain duplicates")
  })

  it("works with a single group and still includes ALWAYS_AVAILABLE_TOOLS", () => {
    const groups = ["browser"] as const
    const result = getToolsForMode(groups)

    TOOL_GROUPS.browser.tools.forEach((t) => {
      assert.ok(result.includes(t), `Expected tool ${t}`)
    })

    ALWAYS_AVAILABLE_TOOLS.forEach((t) => {
      assert.ok(result.includes(t), `Expected always-available tool ${t}`)
    })
  })
})
