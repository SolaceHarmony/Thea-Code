import * as assert from 'assert'
import * as sinon from 'sinon'
import * as proxyquire from 'proxyquire'

suite("LmStudioHandler", () => {
	let LmStudioHandler: any
	let handler: any
	let mockOptions: any
	let mockCreate: sinon.SinonStub

	setup(() => {
		// Create fresh stubs for each test
		mockCreate = sinon.stub()

		// Use proxyquire to mock OpenAI
		LmStudioHandler = proxyquire('../lmstudio', {
			'openai': {
				__esModule: true,
				default: sinon.stub().callsFake(() => ({
					chat: {
						completions: {
							create: mockCreate,
						},
					},
				})),
			}
		}).LmStudioHandler

		mockOptions = {
			apiKey: "test-api-key",
			apiModelId: "lmstudio-model",
			lmStudioBaseUrl: "http://localhost:1234",
		}
		handler = new LmStudioHandler(mockOptions)
	})

	teardown(() => {
		sinon.restore()
	})

	suite("constructor", () => {
		test("should initialize with provided options", () => {
			assert.ok(handler instanceof LmStudioHandler)
			assert.strictEqual(handler.getModel().id, mockOptions.apiModelId)
		})

		test("should use default model if none provided", () => {
			const handlerWithoutModel = new LmStudioHandler({
				apiKey: "test-api-key",
				lmStudioBaseUrl: "http://localhost:1234",
			})
			const model = handlerWithoutModel.getModel()
			assert.notStrictEqual(model.id, undefined)
		})

		test("should use custom base URL", () => {
			const customHandler = new LmStudioHandler({
				...mockOptions,
				lmStudioBaseUrl: "http://custom:8080",
			})
			assert.ok(customHandler instanceof LmStudioHandler)
		})
	})

	suite("createMessage", () => {
		const systemPrompt = "You are a helpful assistant."

		setup(() => {
			// Setup default mock for streaming response
			mockCreate.callsFake(async function* () {
				yield {
					choices: [{
						delta: { content: "Test response" },
						index: 0,
					}],
					usage: null,
				}
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
			assert.ok(mockCreate.calledOnce)

			// Verify the OpenAI client was called with correct parameters
			const callArgs = mockCreate.firstCall.args[0]
			assert.strictEqual(callArgs.model, "lmstudio-model")
			assert.strictEqual(callArgs.stream, true)
			assert.ok(Array.isArray(callArgs.messages))
		})

		test("should handle system prompt in messages", async () => {
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

			const callArgs = mockCreate.firstCall.args[0]
			assert.ok(callArgs.messages.some((msg: any) => msg.role === "system"))
		})
	})

	suite("completePrompt", () => {
		test("should complete prompt successfully", async () => {
			mockCreate.resolves({
				id: "test-completion",
				choices: [{
					message: { role: "assistant", content: "Test response" },
					finish_reason: "stop",
					index: 0,
				}],
				usage: {
					prompt_tokens: 10,
					completion_tokens: 5,
					total_tokens: 15,
				},
			})

			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "Test response")
			assert.ok(mockCreate.calledOnce)

			const callArgs = mockCreate.firstCall.args[0]
			assert.strictEqual(callArgs.stream, false)
		})

		test("should handle empty response", async () => {
			mockCreate.resolves({
				id: "test-completion",
				choices: [{
					message: { role: "assistant", content: "" },
					finish_reason: "stop",
					index: 0,
				}],
			})

			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "")
		})
	})

	suite("getModel", () => {
		test("should return model information", () => {
			const model = handler.getModel()
			assert.strictEqual(model.id, "lmstudio-model")
			assert.notStrictEqual(model.info, undefined)
			assert.ok(model.info.maxTokens > 0)
			assert.ok(model.info.contextWindow > 0)
		})

		test("should return correct properties for lmstudio models", () => {
			const model = handler.getModel()
			assert.strictEqual(model.info.supportsImages, false)
			assert.strictEqual(model.info.supportsPromptCache, false)
		})
	})

	suite("countTokens", () => {
		test("should count tokens for text content", async () => {
			const neutralContent = [{ type: "text" as const, text: "Test message with some tokens" }]
			const result = await handler.countTokens(neutralContent)
			
			// Should return a reasonable token count
			assert.ok(typeof result === "number")
			assert.ok(result > 0)
		})

		test("should handle empty content", async () => {
			const neutralContent: any[] = []
			const result = await handler.countTokens(neutralContent)
			
			assert.strictEqual(result, 0)
		})
	})
// Mock cleanup
