import * as assert from 'assert'
import * as sinon from 'sinon'
import * as proxyquire from 'proxyquire'

suite("MistralHandler", () => {
	let MistralHandler: any
	let handler: any
	let mockOptions: any
	let mockStream: sinon.SinonStub
	let mockComplete: sinon.SinonStub
	let mockConvertToMistralMessages: sinon.SinonStub
	let mockConvertToMistralContent: sinon.SinonStub

	setup(() => {
		// Create fresh stubs for each test
		mockStream = sinon.stub()
		mockComplete = sinon.stub()
		mockConvertToMistralMessages = sinon.stub()
		mockConvertToMistralContent = sinon.stub()

		// Use proxyquire to mock dependencies
		MistralHandler = proxyquire('../../../../../src/api/providers/mistral', {
			'@mistralai/mistralai': {
				Mistral: sinon.stub().callsFake(() => ({
					chat: {
						stream: mockStream,
						complete: mockComplete,
					},
				})),
			},
			'../../transform/neutral-mistral-format': {
				convertToMistralMessages: mockConvertToMistralMessages,
				convertToMistralContent: mockConvertToMistralContent,
			}
		}).MistralHandler

		mockOptions = {
			mistralApiKey: "test-key",
			apiModelId: "mistral-medium",
		}
		handler = new MistralHandler(mockOptions)
	})

	teardown(() => {
		sinon.restore()
	})

	suite("constructor", () => {
		test("should initialize with provided options", () => {
			assert.ok(handler instanceof MistralHandler)
			assert.strictEqual(handler.getModel().id, mockOptions.apiModelId)
		})

		test("should use default model if none provided", () => {
			const handlerWithoutModel = new MistralHandler({
				mistralApiKey: "test-key",
			})
			const model = handlerWithoutModel.getModel()
			assert.notStrictEqual(model.id, undefined)
		})
	})

	suite("createMessage", () => {
		const systemPrompt = "You are a helpful assistant."

		setup(() => {
			// Setup default mocks
			mockConvertToMistralMessages.returns([{ role: "user", content: "Test message" }])
			
			// Setup streaming response
			mockStream.returns({
				[Symbol.asyncIterator]: async function* () {
					yield {
						data: {
							choices: [{
								delta: {
									content: "Test response",
								},
							}],
							usage: {
								promptTokens: 10,
								completionTokens: 5,
							},
						},
					}
				},
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
			assert.ok(mockStream.calledOnce)
			assert.ok(mockConvertToMistralMessages.calledWith(systemPrompt, neutralMessages))
		})

		test("should handle system prompt conversion", async () => {
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

			// Verify conversion functions were called
			assert.ok(mockConvertToMistralMessages.calledOnce)
		})
	})

	suite("completePrompt", () => {
		test("should complete prompt successfully", async () => {
			mockComplete.resolves({
				choices: [{
					message: {
						content: "Test completion",
					},
				}],
			})

			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "Test completion")
			assert.ok(mockComplete.calledOnce)
		})

		test("should handle empty response", async () => {
			mockComplete.resolves({
				choices: [{
					message: {
						content: "",
					},
				}],
			})

			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "")
		})

		test("should handle missing choices", async () => {
			mockComplete.resolves({
				choices: [],
			})

			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "")
		})
	})

	suite("getModel", () => {
		test("should return model information", () => {
			const model = handler.getModel()
			assert.strictEqual(model.id, "mistral-medium")
			assert.notStrictEqual(model.info, undefined)
			assert.ok(model.info.maxTokens > 0)
			assert.ok(model.info.contextWindow > 0)
		})

		test("should return correct properties for mistral models", () => {
			const model = handler.getModel()
			assert.strictEqual(model.info.supportsImages, false)
			assert.strictEqual(model.info.supportsPromptCache, false)
		})
	})

	suite("countTokens", () => {
		setup(() => {
			mockConvertToMistralContent.returns("Test content")
		})

		test("should count tokens for text content", async () => {
			const neutralContent = [{ type: "text" as const, text: "Test message with some tokens" }]
			const result = await handler.countTokens(neutralContent)
			
			// Should return a reasonable token count
			assert.ok(typeof result === "number")
			assert.ok(result > 0)
			assert.ok(mockConvertToMistralContent.calledWith(neutralContent))
		})

		test("should handle empty content", async () => {
			mockConvertToMistralContent.returns("")
			const neutralContent: any[] = []
			const result = await handler.countTokens(neutralContent)
			
			assert.strictEqual(result, 0)
		})
	})
})