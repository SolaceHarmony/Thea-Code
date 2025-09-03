import * as assert from 'assert'
import * as sinon from 'sinon'
import { McpClient } from "../McpClient"
import { SseClientFactory } from "../SseClientFactory"

suite("SseClientFactory", () => {
  const testUrl = new URL("http://localhost:3000")

  setup(() => {
    sinon.restore()
  })

  suite("createClient", () => {
    test("creates SDK-backed client when SDK is available (lazy)", async () => {
      const client = await SseClientFactory.createClient(testUrl, { lazy: true })
      assert.ok(client instanceof McpClient)
      const info = client.getInfo()
      assert.strictEqual(info.name, "TheaCodeMcpClient")
    })

    test("falls back to mock client when SDK disabled via env (lazy)", async () => {
      const prev = process.env.THEA_DISABLE_MCP_SDK
      process.env.THEA_DISABLE_MCP_SDK = "1"
      try {
        const client = await SseClientFactory.createClient(testUrl, { lazy: true })
        assert.ok(client instanceof McpClient)
        const list = await client.listTools()
        assert.deepStrictEqual(list, { tools: [] })
        const call = await client.callTool({ name: "test" })
        assert.deepStrictEqual(call, { content: [] })
      } finally {
        if (prev === undefined) delete process.env.THEA_DISABLE_MCP_SDK
        else process.env.THEA_DISABLE_MCP_SDK = prev
      }
    })
  })
})
