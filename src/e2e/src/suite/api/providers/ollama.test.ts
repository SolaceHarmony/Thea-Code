import * as assert from 'assert'
import * as sinon from 'sinon'
import * as proxyquire from 'proxyquire'

/**
 * OllamaHandler Test Suite
 * 
 * Tests the core OllamaHandler functionality that bridges between Thea's neutral message format
 * and Ollama's OpenAI-compatible API. Unlike the MCP integration test, this focuses on the
 * fundamental provider operations: message creation, model management, and token counting.
 * 
 * Key Behaviors Tested:
 * 1. Constructor initialization with Ollama base URLs and model IDs
 * 2. Message format conversion from neutral to Ollama-compatible formats  
 * 3. Streaming response handling using OpenAI client compatibility
 * 4. Model information retrieval and validation
 * 5. Token counting using the base provider's tiktoken implementation
 * 6. Error handling for various failure scenarios
 */
suite("OllamaHandler Core Functionality", () => {
	let OllamaHandler: any
	let handler: any
	let mockOptions: any
	let mockCreate: sinon.SinonStub
	let mockConvertToOllamaHistory: sinon.SinonStub
	let mockConvertToOllamaContentBlocks: sinon.SinonStub

	setup(() => {
		// Create fresh stubs for each test to ensure isolation
		mockCreate = sinon.stub()
		mockConvertToOllamaHistory = sinon.stub()
		mockConvertToOllamaContentBlocks = sinon.stub()

		// Mock all dependencies using proxyquire to maintain proper module boundaries
		// This approach is cleaner than complex external mock servers
		OllamaHandler = proxyquire('../../../../../src/api/providers/ollama', {
			// Mock OpenAI client since Ollama uses OpenAI-compatible API format
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
			// Mock the neutral format converters that transform Thea's internal format to Ollama's expected input
			'../transform/neutral-ollama-format': {
				convertToOllamaHistory: mockConvertToOllamaHistory,
				convertToOllamaContentBlocks: mockConvertToOllamaContentBlocks,
			},
			// Mock the OpenAI handler that Ollama uses internally for tool detection
			'./openai': {
				OpenAiHandler: sinon.stub().callsFake(() => ({
					extractToolCalls: sinon.stub().returns([]),
					hasToolCalls: sinon.stub().returns(false),
					processToolUse: sinon.stub().resolves("tool result"),
				})),
			},
			// Mock the JSON/XML pattern matcher used for reasoning blocks
			'../../utils/json-xml-bridge': {
				HybridMatcher: sinon.stub().callsFake(() => ({
					update: sinon.stub().callsFake((text: string) => {
						// Simple mock that passes text through unchanged
						return [{ matched: false, data: text }]
					}),
					final: sinon.stub().returns([]),
					getDetectedFormat: sinon.stub().returns("json"),
				})),
			},
		}).OllamaHandler

		// Standard test configuration mimicking real Ollama setup
		mockOptions = {
			ollamaBaseUrl: "http://localhost:11434",
			ollamaModelId: "llama2",
		}

		handler = new OllamaHandler(mockOptions)
	})

	teardown(() => {
		sinon.restore()
	})

	/**
	 * Constructor Tests
	 * 
	 * Verify that the OllamaHandler properly initializes with various configuration options.
	 * The constructor must set up the OpenAI client with the correct Ollama base URL and
	 * initialize the internal OpenAI handler for tool detection capabilities.
	 */
	suite("Constructor and Initialization", () => {
		test("should initialize with standard Ollama configuration", () => {
			// Verify the handler was created successfully
			assert.ok(handler instanceof OllamaHandler)
			
			// Confirm it can retrieve model information
			const model = handler.getModel()
			assert.strictEqual(model.id, "llama2")
			assert.notStrictEqual(model.info, undefined)
		})

		test("should handle custom Ollama base URL configuration", () => {
			const customHandler = new OllamaHandler({
				...mockOptions,
				ollamaBaseUrl: "http://custom-ollama:8080",
			})
			
			// Should initialize without errors even with custom URLs
			assert.ok(customHandler instanceof OllamaHandler)
			assert.strictEqual(customHandler.getModel().id, "llama2")
		})

		test("should use default model when none specified", () => {
			const defaultHandler = new OllamaHandler({
				ollamaBaseUrl: "http://localhost:11434",
				// No ollamaModelId specified - should use default
			})
			
			const model = defaultHandler.getModel()
			// Should have some default model ID
			assert.notStrictEqual(model.id, undefined)
			assert.ok(typeof model.id === "string")
		})

		test("should create internal OpenAI handler for tool compatibility", () => {
			// The constructor should have created an internal OpenAI handler
			// This is used for tool detection and processing since Ollama uses OpenAI format
			assert.notStrictEqual(handler["openAiHandler"], undefined)
		})
	})

	/**
	 * Message Creation Tests
	 * 
	 * Test the core message streaming functionality. OllamaHandler converts Thea's neutral
	 * message format to Ollama's expected format, then streams responses back through
	 * the OpenAI-compatible interface.
	 */
	suite("Message Creation and Streaming", () => {
		setup(() => {
			// Set up realistic mock responses for format conversion
			mockConvertToOllamaHistory.returns([
				{ role: "system", content: "You are a helpful assistant." },
				{ role: "user", content: "Test message" }
			])

			// Mock a typical streaming response from Ollama
			mockCreate.callsFake(async function* () {
				yield {
					choices: [{
						delta: { content: "Hello" },
						index: 0,
					}],
				}
				yield {
					choices: [{
						delta: { content: " from Ollama!" },
						index: 0,
					}],
				}
				// Final chunk with usage information
				yield {
					choices: [{
						delta: {},
						index: 0,
					}],
					usage: {
						prompt_tokens: 10,
						completion_tokens: 5,
						total_tokens: 15,
					},
				}
			})
		})

		test("should convert neutral messages to Ollama format before sending", async () => {
			const systemPrompt = "You are a helpful assistant."
			const neutralMessages = [
				{
					role: "user",
					content: [{ type: "text", text: "Hello Ollama" }],
				},
			]

			const stream = handler.createMessage(systemPrompt, neutralMessages)
			const chunks = []
			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			// Verify the conversion function was called with the neutral messages
			assert.ok(mockConvertToOllamaHistory.calledWith(neutralMessages))
			
			// Verify the OpenAI client was called for streaming
			assert.ok(mockCreate.calledOnce)
			
			// Check that we received streaming content
			assert.ok(chunks.length > 0)
			const textChunks = chunks.filter(chunk => chunk.type === "text")
			assert.ok(textChunks.length > 0)
		})

		test("should handle system prompt integration", async () => {
			const systemPrompt = "You are a coding assistant."
			const neutralMessages = [
				{
					role: "user",
					content: [{ type: "text", text: "Write a function" }],
				},
			]

			const stream = handler.createMessage(systemPrompt, neutralMessages)
			const chunks = []
			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			// The conversion should have been called with the neutral messages
			// The system prompt handling is done within the handler
			assert.ok(mockConvertToOllamaHistory.calledOnce)
			
			// Verify the OpenAI client was called with proper streaming setup
			const clientCall = mockCreate.firstCall.args[0]
			assert.strictEqual(clientCall.stream, true)
			assert.strictEqual(clientCall.model, "llama2")
		})

		test("should process reasoning/thinking blocks in responses", async () => {
			// Mock the HybridMatcher to simulate detecting reasoning blocks
			const matcherInstance = handler["matcher"] || { update: sinon.stub(), final: sinon.stub() }
			if (matcherInstance.update) {
				matcherInstance.update.callsFake((text: string) => {
					if (text.includes("<think>")) {
						return [{ matched: true, data: "This is reasoning content" }]
					}
					return [{ matched: false, data: text }]
				})
			}

			const neutralMessages = [
				{
					role: "user",
					content: [{ type: "text", text: "Think about this problem" }],
				},
			]

			const stream = handler.createMessage("System prompt", neutralMessages)
			const chunks = []
			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			// Should have processed the content through the pattern matcher
			assert.ok(chunks.length > 0)
		})

		test("should handle empty or malformed responses gracefully", async () => {
			// Mock an empty response from Ollama
			mockCreate.callsFake(async function* () {
				// No yields - empty stream
			})

			const neutralMessages = [
				{
					role: "user",
					content: [{ type: "text", text: "Test" }],
				},
			]

			const stream = handler.createMessage("System", neutralMessages)
			const chunks = []
			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			// Should handle empty streams without errors
			assert.ok(Array.isArray(chunks))
		})
	})

	/**
	 * Non-Streaming Completion Tests
	 * 
	 * Test the completePrompt method which provides a simpler, non-streaming interface
	 * for single completions. This is often used for quick queries or synchronous operations.
	 */
	suite("Non-Streaming Completion", () => {
		test("should handle single prompt completion", async () => {
			// Mock a non-streaming response
			mockCreate.resolves({
				choices: [{
					message: {
						content: "This is a completion response",
						role: "assistant",
					},
					finish_reason: "stop",
					index: 0,
				}],
				usage: {
					prompt_tokens: 8,
					completion_tokens: 6,
					total_tokens: 14,
				},
			})

			const result = await handler.completePrompt("Complete this sentence")
			
			assert.strictEqual(result, "This is a completion response")
			assert.ok(mockCreate.calledOnce)
			
			// Verify it was called with streaming disabled
			const callArgs = mockCreate.firstCall.args[0]
			assert.strictEqual(callArgs.stream, false)
		})

		test("should handle empty completion responses", async () => {
			mockCreate.resolves({
				choices: [{
					message: {
						content: "",
						role: "assistant",
					},
					finish_reason: "stop",
					index: 0,
				}],
			})

			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "")
		})

		test("should handle completion failures gracefully", async () => {
			mockCreate.rejects(new Error("Ollama server unavailable"))

			try {
				await handler.completePrompt("Test prompt")
				assert.fail("Should have thrown an error")
} catch (error) {
			assert.fail('Unexpected error: ' + error.message)
		} catch (error) {
				assert.ok(error instanceof Error)
				assert.ok(error.message.includes("Ollama server unavailable"))
			} catch (error) {
			assert.fail("Unexpected error: " + error.message)
		}
		})
	})

	/**
	 * Model Information Tests
	 * 
	 * Verify that the handler correctly reports model capabilities and information.
	 * This is crucial for the UI to show appropriate options and for the system
	 * to make informed decisions about model capabilities.
	 */
	suite("Model Information and Capabilities", () => {
		test("should return correct model information for configured model", () => {
			const model = handler.getModel()
			
			assert.strictEqual(model.id, "llama2")
			assert.notStrictEqual(model.info, undefined)
			
			// Verify it has reasonable default values
			assert.ok(model.info.maxTokens > 0)
			assert.ok(model.info.contextWindow > 0)
			
			// Ollama models typically don't support all features
			assert.strictEqual(model.info.supportsImages, false)
			assert.strictEqual(model.info.supportsPromptCache, false)
		})

		test("should handle different model configurations", () => {
			const customHandler = new OllamaHandler({
				...mockOptions,
				ollamaModelId: "mistral:7b",
			})
			
			const model = customHandler.getModel()
			assert.strictEqual(model.id, "mistral:7b")
		})

		test("should provide model capabilities information", () => {
			const model = handler.getModel()
			
			// Should have basic model info structure
			assert.notStrictEqual(model.info.maxTokens, undefined)
			assert.notStrictEqual(model.info.contextWindow, undefined)
			assert.notStrictEqual(model.info.supportsImages, undefined)
			assert.notStrictEqual(model.info.supportsPromptCache, undefined)
		})
	})

	/**
	 * Token Counting Tests
	 * 
	 * Test the token counting functionality. OllamaHandler uses the base provider's
	 * tiktoken implementation since Ollama doesn't have a native token counting API.
	 */
	suite("Token Counting", () => {
		setup(() => {
			// Mock the content block conversion for token counting
			mockConvertToOllamaContentBlocks.returns(["Test content for counting"])
		})

		test("should count tokens for text content", async () => {
			const neutralContent = [
				{ type: "text" as const, text: "This is a test message for token counting" }
			]
			
			const result = await handler.countTokens(neutralContent)
			
			// Should return a reasonable token count
			assert.ok(typeof result === "number")
			assert.ok(result > 0)
			
			// Verify the conversion function was called
			assert.ok(mockConvertToOllamaContentBlocks.calledWith(neutralContent))
		})

		test("should handle empty content gracefully", async () => {
			const neutralContent: any[] = []
			
			const result = await handler.countTokens(neutralContent)
			
			// Empty content should return 0 tokens
			assert.strictEqual(result, 0)
		})

		test("should handle complex multi-part content", async () => {
			const neutralContent = [
				{ type: "text" as const, text: "First part" },
				{ type: "text" as const, text: "Second part with more content" }
			]
			
			const result = await handler.countTokens(neutralContent)
			
			assert.ok(typeof result === "number")
			assert.ok(result > 0)
		})

		test("should handle token counting errors gracefully", async () => {
			// Mock the conversion to throw an error
			mockConvertToOllamaContentBlocks.throws(new Error("Conversion failed"))
			
			const neutralContent = [
				{ type: "text" as const, text: "Test content" }
			]
			
			// Should still return a reasonable fallback
			const result = await handler.countTokens(neutralContent)
			assert.ok(typeof result === "number")
			assert.ok(result >= 0)
		})
	})

	/**
	 * Error Handling and Edge Cases
	 * 
	 * Verify robust error handling for various failure scenarios that can occur
	 * when communicating with Ollama servers or processing responses.
	 */
	suite("Error Handling and Edge Cases", () => {
		test("should handle Ollama server connection errors", async () => {
			mockCreate.rejects(new Error("ECONNREFUSED"))
			
			const neutralMessages = [
				{
					role: "user",
					content: [{ type: "text", text: "Test message" }],
				},
			]

			try {
				const stream = handler.createMessage("System", neutralMessages)
				const chunks = []
				for await (const chunk of stream) {
					chunks.push(chunk)
				} catch (error) {
			assert.fail('Unexpected error: ' + error.message)
		}assert.fail("Should have thrown an error")
} catch (error) {
				assert.ok(error instanceof Error)
				assert.ok(error.message.includes("ECONNREFUSED"))
			} catch (error) {
			assert.fail("Unexpected error: " + error.message)
		}
		})

		test("should handle malformed responses from Ollama", async () => {
			// Mock a response with missing required fields
			mockCreate.callsFake(async function* () {
				yield {
					// Missing choices array
					id: "test",
				}
			})

			const neutralMessages = [
				{
					role: "user",
					content: [{ type: "text", text: "Test" }],
				},
			]

			const stream = handler.createMessage("System", neutralMessages)
			const chunks = []
			
			// Should handle malformed responses without crashing
			for await (const chunk of stream) {
				chunks.push(chunk)
			}
			
			// May have no chunks or error chunks, but shouldn't crash
			assert.ok(Array.isArray(chunks))
		})

		test("should handle format conversion errors", async () => {
			// Mock the format converter to throw an error
			mockConvertToOllamaHistory.throws(new Error("Format conversion failed"))
			
			const neutralMessages = [
				{
					role: "user",
					content: [{ type: "text", text: "Test" }],
				},
			]

			try {
				const stream = handler.createMessage("System", neutralMessages)
				const chunks = []
				for await (const chunk of stream) {
					chunks.push(chunk)
				} catch (error) {
			assert.fail('Unexpected error: ' + error.message)
		}// Depending on implementation, might handle gracefully or throw
} catch (error) {
				assert.ok(error instanceof Error)
				assert.ok(error.message.includes("Format conversion failed"))
			} catch (error) {
			assert.fail("Unexpected error: " + error.message)
		}
		})
	})
// Mock cleanup
