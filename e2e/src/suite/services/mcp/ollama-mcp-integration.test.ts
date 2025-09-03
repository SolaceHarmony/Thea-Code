import OpenAI from "openai"
import { McpClient } from "../client/McpClient"
import { SseClientFactory } from "../client/SseClientFactory"
import { McpIntegration } from "../integration/McpIntegration"
import * as assert from 'assert'
import { McpConverters } from "../core/McpConverters"
import * as sinon from 'sinon'

// Mock the OpenAI client
// Mock needs manual implementation
		chat: {
			completions: {
				create: sinon.stub
					.fn()
					.callsFake(async ({ functions: _functions, function_call: _functionCall }) => {
						void _functions
						void _functionCall
						await Promise.resolve()
						const stream = {
							[Symbol.asyncIterator]: () => {
								let count = 0
								const messages = [
									{ choices: [{ delta: { content: "I will use a tool to help you." } }] },
									{
										choices: [
											{
												delta: {
													function_call: {
														name: "test_tool",
														arguments: '{"param":"test value"}',
													},
												},
											},
										],
									},
									{ choices: [{ delta: { content: "Here is the result from the tool." } }] },
								]
// Mock removed - needs manual implementation
// Mock removed - needs manual implementation,
// 								}
							},
						}
						return stream
					}),
			},
		},
	})
// Mock removed - needs manual implementation
// Mock cleanup
suite("Ollama MCP Integration with SSE Transport", () => {
	let mcpIntegration: McpIntegration
	let client: McpClient

	setup(async () => {
		// Initialize MCP integration with SSE transport
		mcpIntegration = McpIntegration.getInstance({
			port: 0, // Use random port for tests
			hostname: "localhost",
		})

		// Initialize the MCP integration
		await mcpIntegration.initialize()

		// Register a test tool
		mcpIntegration.registerTool({
			name: "test_tool",
			description: "A test tool for demonstration",
			paramSchema: {
				type: "object",
				properties: {
					param: {
						type: "string",
						description: "A test parameter",
					},
				},
				required: ["param"],
			},
			handler: async (args) => {
				await Promise.resolve()
// Mock removed - needs manual implementation,
// 					],
// 				}
			},
		})

		// Get the server URL
		const serverUrl = mcpIntegration.getServerUrl()
		assert.notStrictEqual(serverUrl, undefined)

		// Create a client and connect to the server
		client = await SseClientFactory.createClient(serverUrl!)
	})

	teardown(async () => {
		// Close the client
		if (client) {
			await client.close()
		}

		// Shutdown the MCP integration
		await mcpIntegration.shutdown()
	})

	test("should list available tools", async () => {
		// List available tools
		const toolsResult = (await client.listTools()) as {
			tools: Array<{ name: string; description: string; inputSchema: unknown }>
		}

		// Verify that the test tool is available
		assert.strictEqual(toolsResult.tools.length, 1)
		assert.strictEqual(toolsResult.tools[0].name, "test_tool")
		assert.strictEqual(toolsResult.tools[0].description, "A test tool for demonstration")
		assert.deepStrictEqual(toolsResult.tools[0].inputSchema, {
			type: "object",
			properties: {
				param: {
					type: "string",
					description: "A test parameter",
				},
			},
			required: ["param"],
		})
	})

	test("should call a tool and get result", async () => {
		// Call the tool
		const result = (await client.callTool({
			name: "test_tool",
			arguments: { param: "test value" },
		})) as {
			content: Array<{ type: string; text: string }>
			isError: boolean
		}

		// Verify the result
		assert.strictEqual(result.content.length, 1)
		assert.strictEqual(result.content[0].type, "text")
		assert.strictEqual(result.content[0].text, "Tool executed with param: test value")
		assert.ok(!result.isError)
	})

	test("should convert tool definitions to OpenAI functions", () => {
		// Get all tools from the MCP integration
		const toolRegistry = mcpIntegration["mcpToolSystem"]["toolRegistry"]
		const availableTools = toolRegistry.getAllTools()

		// Convert tool definitions to OpenAI function definitions
		const functions = McpConverters.toolDefinitionsToOpenAiFunctions(availableTools)

		// Verify the conversion
		assert.strictEqual(functions.length, 1)
		assert.deepStrictEqual(functions[0], {
			name: "test_tool",
			description: "A test tool for demonstration",
			parameters: {
				type: "object",
				properties: {
					param: {
						type: "string",
						description: "A test parameter",
					},
				},
				required: ["param"],
			},
		})
	})

	test("should handle multiple concurrent connections", async () => {
		// Create additional clients
		const serverUrl = mcpIntegration.getServerUrl()
		const client2 = await SseClientFactory.createClient(serverUrl!)
		const client3 = await SseClientFactory.createClient(serverUrl!)

		try {
			// Call the tool from each client
			const promiseAll = Promise.all([
				client.callTool({
					name: "test_tool",
					arguments: { param: "client 1" },
				}) as Promise<{ content: Array<{ type: string; text: string }> }>,
				client2.callTool({
					name: "test_tool",
					arguments: { param: "client 2" },
				}) as Promise<{ content: Array<{ type: string; text: string }> }>,
				client3.callTool({
					name: "test_tool",
					arguments: { param: "client 3" },
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
			assert.strictEqual(result1.content[0].text, "Tool executed with param: client 1")
			assert.strictEqual(result2.content[0].text, "Tool executed with param: client 2")
			assert.strictEqual(result3.content[0].text, "Tool executed with param: client 3")
		} finally {
			// Close the additional clients
			await Promise.all([client2.close(), client3.close()])
		}
	})

	test("should integrate with Ollama handler", async () => {
		// Mock the Ollama handler
		const ollamaHandler = {
			mcpIntegration: mcpIntegration,
			client: new OpenAI(),
			getModel: () => ({ id: "llama2" }),
			options: { modelTemperature: 0.7 },
		}

		// Get all available tools from the MCP registry
		const toolRegistry = mcpIntegration["mcpToolSystem"]["toolRegistry"]
		const availableTools = toolRegistry.getAllTools()

		// Convert tool definitions to OpenAI function definitions
		const functions = McpConverters.toolDefinitionsToOpenAiFunctions(availableTools)

		// Create a mock conversation history
		const openAiMessages = [
			{ role: "system", content: "You are a helpful assistant." },
			{ role: "user", content: "Can you help me test a tool?" },
		]

		// Create stream with functions included
		const stream = await ollamaHandler.client.chat.completions.create({
			model: ollamaHandler.getModel().id,
			messages: openAiMessages,
			temperature: ollamaHandler.options.modelTemperature ?? 0,
			stream: true,
			functions: functions,
			function_call: "auto",
		})

		// Collect stream chunks
		const chunks = []
		for await (const chunk of stream) {
			chunks.push(chunk)
		}

		// Verify that the function call was included
		const functionCallChunk = chunks.find((chunk) => chunk.choices[0]?.delta?.function_call?.name === "test_tool")

		assert.notStrictEqual(functionCallChunk, undefined)
		assert.strictEqual(functionCallChunk.choices[0].delta.function_call.name, "test_tool")
		expect(JSON.parse(functionCallChunk.choices[0].delta.function_call.arguments)).toEqual({
			param: "test value",
		})
	})
// Mock cleanup
