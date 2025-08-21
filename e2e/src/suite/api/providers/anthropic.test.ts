import * as assert from 'assert'
import * as sinon from 'sinon'
import * as proxyquire from 'proxyquire'

suite("AnthropicHandler", () => {
	let AnthropicHandler: any
	let handler: any
	let mockOptions: any
	let mockCreateMessage: sinon.SinonStub
	let mockCountTokens: sinon.SinonStub

	setup(() => {
		// Create fresh stubs for each test
		mockCreateMessage = sinon.stub()
		mockCountTokens = sinon.stub()

		// Use proxyquire to mock the NeutralAnthropicClient
		AnthropicHandler = proxyquire('../../../../../src/api/providers/anthropic', {
			'../../services/anthropic/NeutralAnthropicClient': {
				NeutralAnthropicClient: sinon.stub().callsFake(() => ({
					createMessage: mockCreateMessage,
					countTokens: mockCountTokens,
				})),
			}
		}).AnthropicHandler

		mockOptions = {
			apiKey: "test-api-key",
			apiModelId: "claude-3-5-sonnet-20241022",
		}
		handler = new AnthropicHandler(mockOptions)
	})

	teardown(() => {
		sinon.restore()
	})

	suite("constructor", () => {
		test("should initialize with provided options", () => {
			assert.ok(handler instanceof AnthropicHandler)
			assert.strictEqual(handler.getModel().id, mockOptions.apiModelId)
		})

		test("should initialize with undefined API key", () => {
			const handlerWithoutKey = new AnthropicHandler({
				...mockOptions,
				apiKey: undefined,
			})
			assert.ok(handlerWithoutKey instanceof AnthropicHandler)
		})

		test("should use custom base URL if provided", () => {
			const customBaseUrl = "https://custom.anthropic.com"
			const handlerWithCustomUrl = new AnthropicHandler({
				...mockOptions,
				anthropicBaseUrl: customBaseUrl,
			})
			assert.ok(handlerWithCustomUrl instanceof AnthropicHandler)
		})
	})

	suite("createMessage", () => {
		const systemPrompt = "You are a helpful assistant."

		setup(() => {
			// Setup a default mock for createMessage that returns expected chunks
			mockCreateMessage.callsFake(async function* () {
				yield {
					type: "usage",
					inputTokens: 100,
					outputTokens: 50,
					cacheWriteTokens: 20,
					cacheReadTokens: 10,
				}
				await Promise.resolve()
				yield { type: "text", text: "Hello" }
				yield { type: "text", text: " world" }
			})
		})

		test("should handle prompt caching for supported models", async () => {
			const neutralMessages = [
				{
					role: "user",
					content: [{ type: "text", text: "First message" }],
				},
				{
					role: "assistant",
					content: [{ type: "text", text: "Response" }],
				},
				{
					role: "user",
					content: [{ type: "text", text: "Second message" }],
				},
			]

			const stream = handler.createMessage(systemPrompt, neutralMessages)

			const chunks: Array<{
				type: string
				inputTokens?: number
				outputTokens?: number
				cacheWriteTokens?: number
				cacheReadTokens?: number
				text?: string
			}> = []
			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			// Verify usage information
			const usageChunk = chunks.find((chunk) => chunk.type === "usage")
			assert.notStrictEqual(usageChunk, undefined)
			assert.strictEqual(usageChunk?.inputTokens, 100)
			assert.strictEqual(usageChunk?.outputTokens, 50)
			assert.strictEqual(usageChunk?.cacheWriteTokens, 20)
			assert.strictEqual(usageChunk?.cacheReadTokens, 10)

			// Verify text content
			const textChunks = chunks.filter((chunk) => chunk.type === "text")
			assert.strictEqual(textChunks.length, 2)
			assert.strictEqual(textChunks[0].text, "Hello")
			assert.strictEqual(textChunks[1].text, " world")

			// Verify the neutral client was called
			assert.ok(mockCreateMessage.calledOnce)
		})
	})

	suite("completePrompt", () => {
		test("should complete prompt successfully", async () => {
			mockCreateMessage.callsFake(async function* () {
				yield { type: "text", text: "Test response" }
				await Promise.resolve()
			})

			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "Test response")
			assert.ok(mockCreateMessage.calledOnce)
		})

		test("should handle multiple text chunks", async () => {
			mockCreateMessage.callsFake(async function* () {
				yield { type: "text", text: "Hello" }
				yield { type: "text", text: " world" }
				await Promise.resolve()
			})

			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "Hello world")
		})

		test("should handle empty response", async () => {
			mockCreateMessage.callsFake(async function* () {
				// No yields, empty stream
			})

			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "")
		})
	})

	suite("getModel", () => {
		test("should return default model if no model ID is provided", () => {
			const handlerWithoutModel = new AnthropicHandler({
				...mockOptions,
				apiModelId: undefined,
			})
			const model = handlerWithoutModel.getModel()
			assert.notStrictEqual(model.id, undefined)
			assert.notStrictEqual(model.info, undefined)
		})

		test("should return specified model if valid model ID is provided", () => {
			const model = handler.getModel()
			assert.strictEqual(model.id, mockOptions.apiModelId)
			assert.notStrictEqual(model.info, undefined)
			assert.strictEqual(model.info.maxTokens, 8192)
			assert.strictEqual(model.info.contextWindow, 200_000)
			assert.strictEqual(model.info.supportsImages, true)
			assert.strictEqual(model.info.supportsPromptCache, true)
		})

		test("honors custom maxTokens for thinking models", () => {
			const thinkingHandler = new AnthropicHandler({
				apiKey: "test-api-key",
				apiModelId: "claude-3-7-sonnet-20250219:thinking",
				modelMaxTokens: 32_768,
				modelMaxThinkingTokens: 16_384,
			})

			const result = thinkingHandler.getModel()
			assert.strictEqual(result.maxTokens, 32_768)
			assert.deepStrictEqual(result.thinking, { type: "enabled", budget_tokens: 16_384 })
			assert.strictEqual(result.temperature, 1.0)
		})

		test("does not honor custom maxTokens for non-thinking models", () => {
			const nonThinkingHandler = new AnthropicHandler({
				apiKey: "test-api-key",
				apiModelId: "claude-3-7-sonnet-20250219",
				modelMaxTokens: 32_768,
				modelMaxThinkingTokens: 16_384,
			})

			const result = nonThinkingHandler.getModel()
			assert.strictEqual(result.maxTokens, 8192)
			assert.strictEqual(result.thinking, undefined)
			assert.strictEqual(result.temperature, 0)
		})
	})

	suite("countTokens", () => {
		test("should count tokens using NeutralAnthropicClient", async () => {
			mockCountTokens.resolves(42)

			const neutralContent = [{ type: "text" as const, text: "Test message" }]
			const result = await handler.countTokens(neutralContent)

			assert.strictEqual(result, 42)
			assert.ok(mockCountTokens.calledWith("claude-3-5-sonnet-20241022", neutralContent))
		})

		test("should fall back to base provider implementation on error", async () => {
			mockCountTokens.rejects(new Error("Failed to count tokens: API Error"))

			const mockBaseCountTokens = sinon.stub()
				.resolves(24)
			
			// Mock the prototype method
			const originalCountTokens = Object.getPrototypeOf(Object.getPrototypeOf(handler)).countTokens
			Object.getPrototypeOf(Object.getPrototypeOf(handler)).countTokens = mockBaseCountTokens

			const neutralContent = [{ type: "text" as const, text: "Test message" }]
			const result = await handler.countTokens(neutralContent)

			assert.strictEqual(result, 24)
			assert.ok(mockBaseCountTokens.calledWith(neutralContent))

			// Restore original method
			Object.getPrototypeOf(Object.getPrototypeOf(handler)).countTokens = originalCountTokens
		})
	})
// Mock cleanup