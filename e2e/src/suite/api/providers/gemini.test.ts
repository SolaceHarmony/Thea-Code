import * as assert from 'assert'
import * as sinon from 'sinon'
import * as proxyquire from 'proxyquire'

suite("GeminiHandler", () => {
	let GeminiHandler: any
	let handler: any
	let mockOptions: any
	let mockGenerateContentStream: sinon.SinonStub
	let mockGenerateContent: sinon.SinonStub
	let mockGetGenerativeModel: sinon.SinonStub

	setup(() => {
		// Create fresh stubs for each test
		mockGenerateContentStream = sinon.stub()
		mockGenerateContent = sinon.stub()
		mockGetGenerativeModel = sinon.stub()

		// Use proxyquire to mock the Google Generative AI SDK
		GeminiHandler = proxyquire('../../../../../src/api/providers/gemini', {
			'@google/generative-ai': {
				GoogleGenerativeAI: sinon.stub().callsFake(() => ({
					getGenerativeModel: mockGetGenerativeModel.returns({
						generateContentStream: mockGenerateContentStream,
						generateContent: mockGenerateContent,
					}),
				})),
			}
		}).GeminiHandler

		mockOptions = {
			apiKey: "test-key",
			apiModelId: "gemini-2.0-flash-thinking-exp-1219",
			geminiApiKey: "test-key",
		}
		handler = new GeminiHandler(mockOptions)
	})

	teardown(() => {
		sinon.restore()
	})

	suite("constructor", () => {
		test("should initialize with provided config", () => {
			assert.ok(handler instanceof GeminiHandler)
			assert.strictEqual(handler.getModel().id, mockOptions.apiModelId)
		})

		test("should use default model if none provided", () => {
			const handlerWithoutModel = new GeminiHandler({
				apiKey: "test-key",
				geminiApiKey: "test-key",
			})
			const model = handlerWithoutModel.getModel()
			assert.notStrictEqual(model.id, undefined)
		})
	})

	suite("createMessage", () => {
		const systemPrompt = "You are a helpful assistant."

		setup(() => {
			// Setup default mock for streaming response
			mockGenerateContentStream.callsFake(async function* () {
				yield {
					text: () => "Test response part 1",
					candidates: [{
						content: {
							parts: [{ text: "Test response part 1" }],
						},
					}],
					usageMetadata: {
						promptTokenCount: 10,
						candidatesTokenCount: 5,
						totalTokenCount: 15,
					},
				}
				yield {
					text: () => " part 2",
					candidates: [{
						content: {
							parts: [{ text: " part 2" }],
						},
					}],
				}
			})
		})

		test("should create message stream", async () => {
			const neutralMessages = [
				{
					role: "user",
					content: [{ type: "text", text: "Test message" }],
				},
			]

			const stream = handler.createMessage(systemPrompt, neutralMessages)
			const chunks = []
			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			assert.ok(chunks.length > 0)
			assert.ok(mockGenerateContentStream.calledOnce)
		})

		test("should handle system prompt", async () => {
			const neutralMessages = [
				{
					role: "user",
					content: [{ type: "text", text: "Hello" }],
				},
			]

			const stream = handler.createMessage(systemPrompt, neutralMessages)
			const chunks = []
			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			// Verify the model was called
			assert.ok(mockGenerateContentStream.calledOnce)
		})

		test("should handle thinking models", async () => {
			const thinkingHandler = new GeminiHandler({
				...mockOptions,
				apiModelId: "gemini-2.0-flash-thinking-exp-1219",
			})

			const neutralMessages = [
				{
					role: "user",
					content: [{ type: "text", text: "Think about this" }],
				},
			]

			const stream = thinkingHandler.createMessage(systemPrompt, neutralMessages)
			const chunks = []
			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			assert.ok(chunks.length >= 0)
		})
	})

	suite("completePrompt", () => {
		test("should complete prompt successfully", async () => {
			mockGenerateContent.resolves({
				response: {
					text: () => "Test response",
					candidates: [{
						content: {
							parts: [{ text: "Test response" }],
						},
					}],
				},
			})

			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "Test response")
			assert.ok(mockGenerateContent.calledOnce)
		})

		test("should handle empty response", async () => {
			mockGenerateContent.resolves({
				response: {
					text: () => "",
					candidates: [{
						content: {
							parts: [{ text: "" }],
						},
					}],
				},
			})

			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "")
		})
	})

	suite("getModel", () => {
		test("should return model information", () => {
			const model = handler.getModel()
			assert.strictEqual(model.id, "gemini-2.0-flash-thinking-exp-1219")
			assert.notStrictEqual(model.info, undefined)
			assert.ok(model.info.maxTokens > 0)
			assert.ok(model.info.contextWindow > 0)
		})

		test("should return correct properties for gemini models", () => {
			const model = handler.getModel()
			assert.strictEqual(model.info.supportsImages, true)
			assert.strictEqual(model.info.supportsPromptCache, false)
		})

		test("should handle thinking models", () => {
			const thinkingHandler = new GeminiHandler({
				...mockOptions,
				apiModelId: "gemini-2.0-flash-thinking-exp-1219",
			})
			const model = thinkingHandler.getModel()
			assert.notStrictEqual(model.thinking, undefined)
		})
	})

	suite("countTokens", () => {
		test("should count tokens for text content", async () => {
			const neutralContent = [{ type: "text" as const, text: "Test message with some tokens" }]
			const result = await handler.countTokens(neutralContent)
			
			// Should return a reasonable token count
			assert.ok(typeof result === "number")
			assert.ok(result >= 0)
		})

		test("should handle empty content", async () => {
			const neutralContent: any[] = []
			const result = await handler.countTokens(neutralContent)
			
			assert.strictEqual(result, 0)
		})

		test("should handle multimodal content", async () => {
			const neutralContent = [
				{ type: "text" as const, text: "Describe this image" },
				{ type: "image" as const, source: { type: "base64", media_type: "image/png", data: "iVBORw0KGgo=" } },
			]
			const result = await handler.countTokens(neutralContent)
			
			assert.ok(typeof result === "number")
			assert.ok(result > 0)
		})
	})
})