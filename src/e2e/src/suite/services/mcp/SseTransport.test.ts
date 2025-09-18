
import { SseClientFactory } from "../client/SseClientFactory"
import { EmbeddedMcpProvider } from "../providers/EmbeddedMcpProvider"
import * as assert from "assert"

suite("SSE Transport (E2E)", () => {
  let server: EmbeddedMcpProvider

  setup(async () => {
    server = await EmbeddedMcpProvider.create({ port: 0, hostname: "localhost" })
  })

  teardown(async () => {
    await server.stop()
  })

  test("should start server and get URL", async () => {
    await server.start()
    const url = server.getServerUrl()
    assert.notStrictEqual(url, undefined)
    assert.strictEqual(url?.protocol, "http:")
    assert.strictEqual(url?.hostname, "localhost")
    assert.ok(url?.port)
  })

  test("should connect client to server and call a tool", async () => {
    await server.start()
    const url = server.getServerUrl()
    assert.notStrictEqual(url, undefined)

    // Register a simple test tool
    server.registerTool(
      "test_tool",
      "A test tool",
      { message: { type: "string" } },
      async (args: any) => {
        return {
          content: [{ type: "text", text: `Received: ${String(args?.message)}` }],
        }
      },
    )

    const client = await SseClientFactory.createClient(url!)
    try {
      const toolsResult = (await client.listTools()) as {
        tools: Array<{ name: string; description: string; inputSchema: unknown }>
      }
      assert.strictEqual(toolsResult.tools.length, 1)
      assert.strictEqual(toolsResult.tools[0].name, "test_tool")

      const result = (await client.callTool({
        name: "test_tool",
        arguments: { message: "Hello, world!" },
      })) as { content: Array<{ type: string; text: string }> }

      assert.strictEqual(result.content.length, 1)
      assert.strictEqual(result.content[0].type, "text")
      assert.strictEqual(result.content[0].text, "Received: Hello, world!")
    } finally {
      await client.close()
    }
  })

  test("should handle multiple clients", async () => {
    await server.start()
    const url = server.getServerUrl()
    assert.notStrictEqual(url, undefined)

    server.registerTool(
      "test_tool",
      "A test tool",
      { message: { type: "string" } },
      async (args: any) => ({ content: [{ type: "text", text: `Received: ${String(args?.message)}` }] }),
    )

    const client1 = await SseClientFactory.createClient(url!)
    const client2 = await SseClientFactory.createClient(url!)
    const client3 = await SseClientFactory.createClient(url!)

    try {
      const [r1, r2, r3] = await Promise.all([
        client1.callTool({ name: "test_tool", arguments: { message: "Client 1" } }) as Promise<{
          content: Array<{ type: string; text: string }>
        }>,
        client2.callTool({ name: "test_tool", arguments: { message: "Client 2" } }) as Promise<{
          content: Array<{ type: string; text: string }>
        }>,
        client3.callTool({ name: "test_tool", arguments: { message: "Client 3" } }) as Promise<{
          content: Array<{ type: string; text: string }>
        }>,
      ])

      assert.strictEqual(r1.content[0].text, "Received: Client 1")
      assert.strictEqual(r2.content[0].text, "Received: Client 2")
      assert.strictEqual(r3.content[0].text, "Received: Client 3")
    } finally {
      await Promise.all([client1.close(), client2.close(), client3.close()])
    }
  })
})
