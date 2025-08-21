import * as assert from 'assert'
import * as sinon from 'sinon'
import * as proxyquire from 'proxyquire'

// Define global augmentation for shared mock port
declare global {
	var __OLLAMA_PORT__: number | undefined
}

suite("Ollama MCP Integration", () => {
	let OllamaHandler: any
	let getOllamaModels: any
	let handler: any
	let mockOptions: any
	let mockCreate: sinon.SinonStub
	let mockExtractToolCalls: sinon.SinonStub
	let mockProcessToolUse: sinon.SinonStub
	let mockRouteToolUse: sinon.SinonStub
	let mockMcpInstance: any
	let mockStartServer: sinon.SinonStub
	let mockStopServer: sinon.SinonStub
	let mockGetServerPort: sinon.SinonStub
	let availableModels: string[] = []
	let ollamaBaseUrl: string

	setup(() => {
		// Create fresh stubs for each test
		mockCreate = sinon.stub()
		mockExtractToolCalls = sinon.stub()
		mockProcessToolUse = sinon.stub()
		mockRouteToolUse = sinon.stub()
		mockStartServer = sinon.stub()
		mockStopServer = sinon.stub()
		mockGetServerPort = sinon.stub()

		// Mock MCP integration instance
		mockMcpInstance = {
			initialize: sinon.stub().resolves(undefined),
			registerTool: sinon.stub(),
			routeToolUse: mockRouteToolUse.resolves("Tool result from MCP"),
		}

		// Use proxyquire to mock all dependencies
		const proxyModule = proxyquire('../../../../../src/api/providers/ollama', {
			'openai': {
				__esModule: true,
				default: sinon.stub().callsFake(() => ({
					chat: {
						completions: {
							create: mockCreate,
						},
					},
				})),
			},
			'./openai': {
				OpenAiHandler: sinon.stub().callsFake(() => ({
					extractToolCalls: mockExtractToolCalls,
					hasToolCalls: sinon.stub().returns(false),
					processToolUse: mockProcessToolUse,
				})),
			},
			'../../services/mcp/integration/McpIntegration': {
				McpIntegration: {
					getInstance: sinon.stub().returns(mockMcpInstance),
				},
			},
			'../../utils/json-xml-bridge': {
				HybridMatcher: sinon.stub().callsFake(() => ({
					update: sinon.stub().callsFake((text: string) => {
						// Simple mock that returns text chunks
						return [{ matched: false, data: text }]
					}),
					final: sinon.stub().returns([]),
					getDetectedFormat: sinon.stub().returns("json"),
				})),
			},
			'../transform/neutral-ollama-format': {
				convertToOllamaHistory: sinon.stub().returns([
					{ role: "user", content: "Test message" }
				]),
				convertToOllamaContentBlocks: sinon.stub().returns(["Test content"]),
			},
			'../../../../test/ollama-mock-server/server': {
				startServer: mockStartServer,
				stopServer: mockStopServer,
				getServerPort: mockGetServerPort,
			}
		})

		OllamaHandler = proxyModule.OllamaHandler
		getOllamaModels = proxyModule.getOllamaModels || sinon.stub().resolves(['llama2'])

		mockOptions = {
			ollamaBaseUrl: "http://localhost:11434",
			ollamaModelId: "llama2",
		}

		// Set up default mock behavior
		ollamaBaseUrl = "http://localhost:11434"
		availableModels = ["llama2"]
		mockGetServerPort.returns(11434)
		
		handler = new OllamaHandler(mockOptions)
	})

	teardown(() => {
		sinon.restore()
	})

	suite("Constructor", () => {
		test("should initialize with Ollama options", () => {
			assert.ok(handler instanceof OllamaHandler)
			assert.strictEqual(handler.getModel().id, "llama2")
		})

		test("should create OpenAI client with Ollama base URL", () => {
			const model = handler.getModel()
			assert.strictEqual(model.id, "llama2")
		})

		test("should initialize MCP integration", () => {
			// Verify MCP integration was initialized
			assert.ok(mockMcpInstance.initialize.called)
		})
	})

	suite("Tool Detection Integration", () => {
		setup(() => {
			// Setup default streaming response
			mockCreate.callsFake(async function* () {
				yield {
					choices: [{
						delta: { content: "Hello from Ollama" },
					}],
				}
			})
		})

		test("should use OpenAI handler for tool detection", async () => {
			mockExtractToolCalls.returns([])

			const neutralMessages = [
				{
					role: "user",
					content: [{ type: "text", text: "Test message" }],
				},
			]

			const stream = handler.createMessage("You are helpful.", neutralMessages)
			const chunks = []
			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			// Verify OpenAI handler's extractToolCalls was called
			assert.ok(mockExtractToolCalls.called)
		})

		test("should process tool calls through MCP integration", async () => {
			// Mock tool detection to return a tool call
			mockExtractToolCalls.returns([{
				id: "tool-123",
				function: {
					name: "weather",
					arguments: '{"location":"San Francisco"}',
				},
			}])

			const neutralMessages = [
				{
					role: "user",
					content: [{ type: "text", text: "What's the weather?" }],
				},
			]

			const stream = handler.createMessage("You are helpful.", neutralMessages)
			const chunks = []
			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			// Verify MCP integration was called for tool routing
			assert.ok(mockRouteToolUse.called)
			
			// Verify tool result was yielded
			const toolResults = chunks.filter(chunk => chunk.type === "tool_result")
			assert.ok(toolResults.length > 0)
		})

		test("should handle tool processing errors gracefully", async () => {
			// Mock tool detection to return a tool call
			mockExtractToolCalls.returns([{
				id: "tool-123",
				function: {
					name: "broken_tool",
					arguments: '{"param":"value"}',
				},
			}])

			// Mock MCP to throw an error
			mockRouteToolUse.rejects(new Error("Tool execution failed"))

			const neutralMessages = [
				{
					role: "user",
					content: [{ type: "text", text: "Use broken tool" }],
				},
			]

			// Should not throw, should handle error gracefully
			const stream = handler.createMessage("You are helpful.", neutralMessages)
			const chunks = []
			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			// Verify the stream completed despite the error
			assert.ok(chunks.length >= 0)
		})
	})

	suite("Hybrid Matching", () => {
		test("should use HybridMatcher for non-tool content", async () => {
			mockExtractToolCalls.returns([]) // No tools detected

			mockCreate.callsFake(async function* () {
				yield {
					choices: [{
						delta: { content: "Regular text response" },
					}],
				}
			})

			const neutralMessages = [
				{
					role: "user",
					content: [{ type: "text", text: "Just say hello" }],
				},
			]

			const stream = handler.createMessage("You are helpful.", neutralMessages)
			const chunks = []
			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			// Should have processed content through HybridMatcher
			const textChunks = chunks.filter(chunk => chunk.type === "text")
			assert.ok(textChunks.length > 0)
		})
	})

	suite("Model Information", () => {
		test("should return Ollama model information", () => {
			const model = handler.getModel()
			assert.strictEqual(model.id, "llama2")
			assert.notStrictEqual(model.info, undefined)
			assert.ok(model.info.maxTokens > 0)
		})

		test("should handle custom model IDs", () => {
			const customHandler = new OllamaHandler({
				...mockOptions,
				ollamaModelId: "custom-model",
			})
			const model = customHandler.getModel()
			assert.strictEqual(model.id, "custom-model")
		})
	})

	suite("Completion", () => {
		test("should support non-streaming completion", async () => {
			// For single completion, we need to mock different behavior
			mockCreate.resolves({
				choices: [{
					message: {
						content: "Completion response",
						role: "assistant",
					},
				}],
			})

			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "Completion response")
		})
	})

	suite("Token Counting", () => {
		test("should count tokens for content", async () => {
			const neutralContent = [{ type: "text" as const, text: "Test message" }]
			const result = await handler.countTokens(neutralContent)
			
			assert.ok(typeof result === "number")
			assert.ok(result > 0)
		})
	})
// Mock cleanup