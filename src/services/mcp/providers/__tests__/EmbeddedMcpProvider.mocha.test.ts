import * as assert from "assert"
import { EmbeddedMcpProvider } from "../EmbeddedMcpProvider"

declare const global: typeof globalThis & { __MCP_PROVIDER__?: EmbeddedMcpProvider }

describe("EmbeddedMcpProvider (Mocha)", () => {
  let provider: EmbeddedMcpProvider

  it("starts on a dynamic port and exposes URL", async () => {
    provider = global.__MCP_PROVIDER__ as EmbeddedMcpProvider
    const url = provider.getServerUrl()
    assert.ok(url, "expected server URL after start")
    assert.ok(Number.parseInt(url.port, 10) > 0)
  })

  it("registers and executes a tool", async () => {
    provider = global.__MCP_PROVIDER__ as EmbeddedMcpProvider

    provider.registerTool("echo_tool", "Echo tool", {}, async (args: Record<string, unknown>) => {
      return { content: [{ type: "text", text: JSON.stringify(args) }], isError: false }
    })

    const result = await provider.executeTool("echo_tool", { foo: "bar" })
    assert.strictEqual(result.isError, false)
    assert.strictEqual(result.content[0].type, "text")
    assert.ok((result.content[0] as { type: string; text: string }).text.includes("\"foo\":\"bar\""))
  })
})
