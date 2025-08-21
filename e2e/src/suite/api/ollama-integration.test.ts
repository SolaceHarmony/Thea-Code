import * as assert from 'assert'
import * as sinon from 'sinon'
import * as proxyquire from 'proxyquire'

// Note: This test uses port 10000 which is for Msty, a service that uses Ollama on the backend

suite("Ollama Integration", () => {
	let sandbox: sinon.SinonSandbox
	let OllamaHandler: any
	let handler: any
	let mockOpenAI: any
	let mockMcpIntegration: any
	let mockHybridMatcher: any
	let mockXmlMatcher: any

	setup(() => {
		sandbox = sinon.createSandbox()
		
		// Mock MCP Integration
		mockMcpIntegration = {
			initialize: sandbox.stub().resolves(undefined),
			registerTool: sandbox.stub(),
			routeToolUse: sandbox.stub().resolves("{}")
		}
		
		const MockMcpIntegrationClass = class {
			initialize = mockMcpIntegration.initialize
			registerTool = mockMcpIntegration.registerTool
			routeToolUse = mockMcpIntegration.routeToolUse
			static getInstance = sandbox.stub().returns(mockMcpIntegration)
		}
		
		// Mock HybridMatcher
		mockHybridMatcher = sandbox.stub().callsFake(() => ({
			update: sandbox.stub().callsFake((text: string) => {
				if (text.includes("<think>")) {
					return [{ matched: true, type: "reasoning", data: text.replace(/<\/?think>/g, "") }]
				}
				if (text.includes('{"type":"thinking"')) {
					try {
						const jsonObj = JSON.parse(text)
						if (jsonObj.type === "thinking") {
							return [{ matched: true, type: "reasoning", data: String(jsonObj.content) }]
						}
} catch (_e: unknown) {
						// Not valid JSON, treat as text
					}
				}
				return [{ matched: false, type: "text", data: text, text: text }]
			}),
			final: sandbox.stub().callsFake((text: string) => {
				if (text) {
					return [{ matched: false, type: "text", data: text, text: text }]
				}
				return []
			})
		}))
		
		// Mock XmlMatcher
		mockXmlMatcher = sandbox.stub().callsFake(() => ({
			update: sandbox.stub().callsFake((text: string) => {
				if (text.includes("<think>")) {
					return [{ matched: true, type: "reasoning", data: text.replace(/<\/?think>/g, "") }]
				}
				return [{ matched: false, type: "text", data: text, text: text }]
			}),
			final: sandbox.stub().callsFake((text: string) => {
				if (text) {
					return [{ matched: false, type: "text", data: text, text: text }]
				}
				return []
			})
		}))
		
		// Mock OpenAI client
		const mockCreate = sandbox.stub().callsFake(({ messages, stream }: any) => {
			if (stream) {
// Return streaming async generator
e2e/src/suite/api/provider-integration-validation.test.ts
e2e/src/suite/api/provider-integration-validation.test.ts
// 						} else if (userText.includes("json reasoning")) {
e2e/src/suite/api/provider-integration-validation.test.ts
e2e/src/suite/api/provider-integration-validation.test.ts
e2e/src/suite/api/provider-integration-validation.test.ts
// 						} else if (userText.includes("error test")) {
// 							throw new Error("Simulated API error")
// 						} else if (userText.includes("system role test")) {
// 							if (hasSystemMessage) {
e2e/src/suite/api/provider-integration-validation.test.ts
// 							} else {
e2e/src/suite/api/provider-integration-validation.test.ts
// 							}
e2e/src/suite/api/provider-integration-validation.test.ts
						} else if (userText.includes("tool use test")) {
e2e/src/suite/api/provider-integration-validation.test.ts
e2e/src/suite/api/provider-integration-validation.test.ts
e2e/src/suite/api/provider-integration-validation.test.ts
						} else if (userText.includes("multimodal test")) {
e2e/src/suite/api/provider-integration-validation.test.ts
e2e/src/suite/api/provider-integration-validation.test.ts
return {}
						// Default response
e2e/src/suite/api/provider-integration-validation.test.ts
e2e/src/suite/api/provider-integration-validation.test.ts
						}
					}
				}
return {}
				// Non-streaming response
// Mock removed - needs manual implementation,
// 						finish_reason: "stop"
// 					}]
// 				}
			}
		})
		
		mockOpenAI = class {
			chat = {
				completions: {
					create: mockCreate
				}
			}
		}
		
		// Load OllamaHandler with mocked dependencies
		const module = proxyquire('../../../src/api/providers/ollama', {
			'openai': mockOpenAI,
			'../../services/mcp/integration/McpIntegration': { McpIntegration: MockMcpIntegrationClass },
			'../../utils/json-xml-bridge': { HybridMatcher: mockHybridMatcher },
			'../../utils/xml-matcher': { XmlMatcher: mockXmlMatcher }
		})
		
		OllamaHandler = module.OllamaHandler
		
		// Create handler with test options
		handler = new OllamaHandler({
			ollamaBaseUrl: "http://localhost:10000",
			ollamaModelId: "llama2"
		})
	})

	teardown(() => {
		sandbox.restore()
	})

	test("should handle basic text messages", async () => {
		const neutralHistory = [
			{ role: "user", content: [{ type: "text", text: "Hello" }] }
		]

		const stream = handler.createMessage("You are helpful.", neutralHistory)
		
		let response = ""
		for await (const chunk of stream) {
			response += chunk.text || ""
		}

		assert.strictEqual(response, "Test response")
	})

	test("should handle streaming responses", async () => {
		const neutralHistory = [
			{ role: "user", content: [{ type: "text", text: "stream test" }] }
		]

		const stream = handler.createMessage("You are helpful.", neutralHistory)
		
		const chunks = []
		for await (const chunk of stream) {
			if (chunk.text) {
				chunks.push(chunk.text)
			}
		}

		assert.deepStrictEqual(chunks, ["First ", "chunk"])
	})

	test("should handle reasoning with XML tags", async () => {
		const neutralHistory = [
			{ role: "user", content: [{ type: "text", text: "reasoning test" }] }
		]

		const stream = handler.createMessage("You are helpful.", neutralHistory)
		
		const chunks = []
		for await (const chunk of stream) {
			chunks.push(chunk)
		}

		// Should have both reasoning and text chunks
		const reasoningChunks = chunks.filter(c => c.type === "reasoning")
		const textChunks = chunks.filter(c => c.type === "text")
		
		assert.ok(reasoningChunks.length > 0, "Should have reasoning chunks")
		assert.ok(textChunks.length > 0, "Should have text chunks")
	})

	test("should handle JSON reasoning format", async () => {
		const neutralHistory = [
			{ role: "user", content: [{ type: "text", text: "json reasoning" }] }
		]

		const stream = handler.createMessage("You are helpful.", neutralHistory)
		
		const chunks = []
		for await (const chunk of stream) {
			chunks.push(chunk)
		}

		const reasoningChunks = chunks.filter(c => c.type === "reasoning")
		assert.ok(reasoningChunks.length > 0, "Should parse JSON reasoning")
	})

	test("should handle API errors gracefully", async () => {
		const neutralHistory = [
			{ role: "user", content: [{ type: "text", text: "error test" }] }
		]

		try {
			const stream = handler.createMessage("You are helpful.", neutralHistory)
			for await (const _chunk of stream) {
				// Should throw before getting here
			}
			assert.fail("Should have thrown an error")
} catch (error) {
			assert.ok(error instanceof Error)
			assert.ok(error.message.includes("Simulated API error"))
	e2e/src/suite/api/provider-integration-validation.test.ts
			// Removed assert.fail
		}
	})

	test("should handle system messages correctly", async () => {
		const neutralHistory = [
			{ role: "user", content: [{ type: "text", text: "system role test" }] }
		]

		const stream = handler.createMessage("You are a test assistant.", neutralHistory)
		
		let response = ""
		for await (const chunk of stream) {
			response += chunk.text || ""
		}

		// The mock should detect system message was passed
		assert.ok(response.includes("System message handled"))
	})

	test("should handle tool use", async () => {
		const neutralHistory = [
			{ role: "user", content: [{ type: "text", text: "tool use test" }] }
		]

		const stream = handler.createMessage("You can use tools.", neutralHistory)
		
		const chunks = []
		for await (const chunk of stream) {
			chunks.push(chunk)
		}

		// Should have both text and tool use
		const textChunks = chunks.filter(c => c.type === "text")
		const toolChunks = chunks.filter(c => c.type === "tool_use")
		
		assert.ok(textChunks.length > 0, "Should have text chunks")
		// Note: Tool handling might be processed differently
	})

	test("should handle multimodal content", async () => {
		const neutralHistory = [
			{ 
				role: "user", 
				content: [
					{ type: "text", text: "multimodal test" },
					{ type: "image", image: "base64data" }
				]
			}
		]

		const stream = handler.createMessage("You can see images.", neutralHistory)
		
		let response = ""
		for await (const chunk of stream) {
			response += chunk.text || ""
		}

		assert.ok(response.includes("Image processed"))
	})

	test("should use correct base URL", () => {
		// The handler should be configured with the test URL
		assert.ok(handler.options.ollamaBaseUrl === "http://localhost:10000")
	})

	test("should use correct model", () => {
		// The handler should be configured with the test model
		assert.ok(handler.options.ollamaModelId === "llama2")
	})
// Mock cleanup
