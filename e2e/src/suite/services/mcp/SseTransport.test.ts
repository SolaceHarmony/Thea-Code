import { EmbeddedMcpProvider } from "../providers/EmbeddedMcpProvider"
import { SseClientFactory } from "../client/SseClientFactory"
import { SseTransportConfig } from "../transport/config/SseTransportConfig"
import * as assert from 'assert'
import * as sinon from 'sinon'

suite("SSE Transport", () => {
	let server: EmbeddedMcpProvider

	setup(async () => {
		// Create a new server with a random port for each test
		server = await EmbeddedMcpProvider.create({
			port: 0,
			hostname: "localhost",
		})
	})

	teardown(async () => {
		// Clean up after each test
		await server.stop()
	})

	test("should start server and get URL", async () => {
		// Start the server
		await server.start()

		// Get the server URL
		const url = server.getServerUrl()

		// Verify that the URL is defined
		assert.notStrictEqual(url, undefined)
		assert.strictEqual(url?.protocol, "http:")
		assert.strictEqual(url?.hostname, "localhost")
		assert.ok(url?.port) // Should have a port assigned
	})

	test("should connect client to server", async () => {
		// Start the server
		await server.start()

		// Get the server URL
		const url = server.getServerUrl()
		assert.notStrictEqual(url, undefined)

		// Register a test tool
		server.registerTool(
			"test_tool",
			"A test tool",
			{
				message: { type: "string" },
			},
			async (args) => {
				await Promise.resolve()
				return {
					content: [{ type: "text", text: `Received: ${String(args.message)}` }],
				}
			},
		)

		// Create a client and connect to the server
		const client = await SseClientFactory.createClient(url!)

		try {
			// List available tools
			const toolsResult = (await client.listTools()) as {
				tools: Array<{ name: string; description: string; inputSchema: unknown }>
			}
			assert.strictEqual(toolsResult.tools.length, 1)
			assert.strictEqual(toolsResult.tools[0].name, "test_tool")

			// Call the tool
			const result = (await client.callTool({
				name: "test_tool",
				arguments: { message: "Hello, world!" },
			})) as {
				content: Array<{ type: string; text: string }>
			}

			// Verify the result
			assert.strictEqual(result.content.length, 1)
			assert.strictEqual(result.content[0].type, "text")
			assert.strictEqual(result.content[0].text, "Received: Hello, world!")
		} finally {
			// Close the client
			await client.close()
		}
	})

	test("should handle multiple clients", async () => {
		// Start the server
		await server.start()

		// Get the server URL
		const url = server.getServerUrl()
		assert.notStrictEqual(url, undefined)

		// Register a test tool
		server.registerTool(
			"test_tool",
			"A test tool",
			{
				message: { type: "string" },
			},
			async (args) => {
				await Promise.resolve()
				return {
					content: [{ type: "text", text: `Received: ${String(args.message)}` }],
				}
			},
		)

		// Create multiple clients
		const client1 = await SseClientFactory.createClient(url!)
		const client2 = await SseClientFactory.createClient(url!)
		const client3 = await SseClientFactory.createClient(url!)

		try {
			// Call the tool from each client
			const promiseAll = Promise.all([
				client1.callTool({
					name: "test_tool",
					arguments: { message: "Client 1" },
				}) as Promise<{ content: Array<{ type: string; text: string }> }>,
				client2.callTool({
					name: "test_tool",
					arguments: { message: "Client 2" },
				}) as Promise<{ content: Array<{ type: string; text: string }> }>,
				client3.callTool({
					name: "test_tool",
					arguments: { message: "Client 3" },
				}) as Promise<{ content: Array<{ type: string; text: string }> }>,
			]) as Promise<
				[
					{ content: Array<{ type: string; text: string }> },
					{ content: Array<{ type: string; text: string }> },
					{ content: Array<{ type: string; text: string }> },
				]
			>
			const [result1, result2, result3] = await promiseAll

			// Verify the results
			assert.strictEqual(result1.content[0].text, "Received: Client 1")
			assert.strictEqual(result2.content[0].text, "Received: Client 2")
			assert.strictEqual(result3.content[0].text, "Received: Client 3")
		} finally {
			// Close all clients
			await Promise.all([client1.close(), client2.close(), client3.close()])
		}
	})

	test("should use custom configuration", async () => {
		// Create a server with custom configuration
		const customConfig: SseTransportConfig = {
			port: 8080,
			hostname: "127.0.0.1",
			allowExternalConnections: true,
			eventsPath: "/custom/events",
			apiPath: "/custom/api",
		}

		const customServer = await EmbeddedMcpProvider.create(customConfig)

		try {
			// Start the server
			await customServer.start()

			// Get the server URL
			const url = customServer.getServerUrl()

			// Verify that the URL uses the custom configuration
			assert.notStrictEqual(url, undefined)
			assert.strictEqual(url?.hostname, "127.0.0.1")

			// Note: We can't verify the port is exactly 8080 because it might be changed
			// if the port is already in use, but we can verify it's a valid port
			assert.ok(url?.port)
		} finally {
			// Clean up
			await customServer.stop()
		}
	})

	test("should handle server restart", async () => {
		// Start the server
		await server.start()

		// Get the server URL
		const url1 = server.getServerUrl()
		assert.notStrictEqual(url1, undefined)

		// Stop the server
		await server.stop()

		// Verify that the URL is no longer available
		expect(server.getServerUrl()).toBeUndefined()

		// Restart the server
		await server.start()

		// Get the new server URL
		const url2 = server.getServerUrl()
		assert.notStrictEqual(url2, undefined)

		// The new URL should be different from the old one (different port)
		expect(url2?.toString()).not.toBe(url1?.toString())
	})
})
