import * as assert from 'assert'
import * as sinon from 'sinon'
import * as proxyquire from 'proxyquire'

suite("GlamaHandler", () => {
	let GlamaHandler: any
	let handler: any
	let mockOptions: any
	let mockCreate: sinon.SinonStub
	let mockWithResponse: sinon.SinonStub

	setup(() => {
		// Create fresh stubs for each test
		mockCreate = sinon.stub()
		mockWithResponse = sinon.stub()

		// Use proxyquire to mock OpenAI
		GlamaHandler = proxyquire('../glama', {
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
		}).GlamaHandler

		mockOptions = {
			apiKey: "test-api-key",
			apiModelId: "llama-70b-chat",
		}
		handler = new GlamaHandler(mockOptions)
	})

	teardown(() => {
		sinon.restore()
	})

	suite("constructor", () => {
		test("should initialize with provided options", () => {
			assert.ok(handler instanceof GlamaHandler)
			assert.strictEqual(handler.getModel().id, mockOptions.apiModelId)
		})

		test("should use default model if none provided", () => {
			const handlerWithoutModel = new GlamaHandler({
				apiKey: "test-api-key",
			})
			const model = handlerWithoutModel.getModel()
			assert.notStrictEqual(model.id, undefined)
		})

		test("should initialize with custom base URL", () => {
			const customHandler = new GlamaHandler({
				...mockOptions,
				openAiBaseUrl: "https://custom.glama.ai",
			})
			assert.ok(customHandler instanceof GlamaHandler)
		})
	})

	suite("createMessage", () => {
		const systemPrompt = "You are a helpful assistant."

		setup(() => {
			// Setup default mock for streaming response
			const streamData = {
				[Symbol.asyncIterator]: async function* () {
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
				},
			}

			mockCreate.returns({
				withResponse: mockWithResponse.resolves({
					data: streamData,
					response: {
						headers: {
							get: (name: string) => name === "x-completion-request-id" ? "test-request-id" : null,
						},
					},
				}),
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
			assert.strictEqual(callArgs.model, "llama-70b-chat")
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
					message: { role: "assistant", content: "Test response", refusal: null },
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
					message: { role: "assistant", content: "", refusal: null },
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
			assert.strictEqual(model.id, "llama-70b-chat")
			assert.notStrictEqual(model.info, undefined)
			assert.ok(model.info.maxTokens > 0)
			assert.ok(model.info.contextWindow > 0)
		})

		test("should return correct properties for glama models", () => {
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
