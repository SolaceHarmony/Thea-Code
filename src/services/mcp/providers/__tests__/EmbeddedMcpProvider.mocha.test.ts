import * as assert from "assert"
import { EmbeddedMcpProvider } from "../EmbeddedMcpProvider"

describe("EmbeddedMcpProvider (Mocha)", () => {
  let provider: EmbeddedMcpProvider | undefined

  afterEach(async () => {
    if (provider) {
      try { await provider.stop() } catch {}
      provider = undefined
    }
  })

  it("starts on a dynamic port and exposes URL", async () => {
    provider = await EmbeddedMcpProvider.create({ port: 0, host: "127.0.0.1" })
    await provider.start()
    const url = provider.getServerUrl()
    assert.ok(url, "expected server URL after start")
    assert.ok(Number.parseInt(url!.port, 10) > 0)
  })

  it("registers and executes a tool", async () => {
    provider = await EmbeddedMcpProvider.create({ port: 0, host: "127.0.0.1" })
    await provider.start()

    provider.registerTool("echo_tool", "Echo tool", async (args: Record<string, unknown>) => {
      return { content: [{ type: "text", text: JSON.stringify(args) }], isError: false }
    })

    const result = await provider.executeTool("echo_tool", { foo: "bar" })
    assert.strictEqual(result.isError, false)
    assert.strictEqual(result.content[0].type, "text")
    assert.ok((result.content[0] as { type: string; text: string }).text.includes("\"foo\":\"bar\""))
  })
})

