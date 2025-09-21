import * as assert from 'assert'
import * as sinon from 'sinon'
import * as proxyquire from 'proxyquire'

import type { ApiHandlerOptions } from "../../../shared/api"
import type { NeutralConversationHistory, NeutralMessageContent } from "../../../shared/neutral-history"
import type { ApiStreamChunk } from "../../transform/stream"

type DeepSeekHandlerConstructor = typeof import("../deepseek").DeepSeekHandler
type DeepSeekHandlerInstance = InstanceType<DeepSeekHandlerConstructor>

interface MockOpenAiUsageDetails {
	cache_miss_tokens?: number
	cached_tokens?: number
}

interface MockOpenAiUsage {
	prompt_tokens?: number
	completion_tokens?: number
	total_tokens?: number
	prompt_tokens_details?: MockOpenAiUsageDetails | null
}

interface MockOpenAiStreamChoice {
	delta: Record<string, unknown>
	index: number
}

interface MockOpenAiStreamChunk {
	choices: MockOpenAiStreamChoice[]
	usage: MockOpenAiUsage | null
}

interface MockOpenAiMessage {
	role: string
	content: unknown
}

interface MockOpenAiCreateParams {
	model: string
	stream: boolean
	messages: MockOpenAiMessage[]
}

interface MockOpenAiCompletionChoice {
	message: { role: "assistant"; content: string; refusal: null }
	finish_reason: string
	index: number
}

interface MockOpenAiCompletion {
	id: string
	choices: MockOpenAiCompletionChoice[]
	usage?: MockOpenAiUsage
}

interface MockOpenAiClient {
	chat: {
		completions: {
			create: sinon.SinonStub<
				[MockOpenAiCreateParams],
				Promise<MockOpenAiCompletion> | AsyncGenerator<MockOpenAiStreamChunk, void, unknown>
			>
		}
	}
}

interface OpenAiConstructorArgs {
	apiKey: string
	baseURL: string
	defaultHeaders: Record<string, string>
}

suite("DeepSeekHandler", () => {
	let DeepSeekHandlerClass: DeepSeekHandlerConstructor
	let handler: DeepSeekHandlerInstance
	let mockOptions: ApiHandlerOptions
	let mockCreate: MockOpenAiClient["chat"]["completions"]["create"]
	let openAiConstructorStub: sinon.SinonStub<[OpenAiConstructorArgs], MockOpenAiClient>

	setup(() => {
		// Create fresh stubs for each test
		mockCreate = sinon.stub<
			[MockOpenAiCreateParams],
			Promise<MockOpenAiCompletion> | AsyncGenerator<MockOpenAiStreamChunk, void, unknown>
		>()

		const mockClient: MockOpenAiClient = {
			chat: {
				completions: {
					create: mockCreate,
				},
			},
		}

		openAiConstructorStub = sinon.stub<[OpenAiConstructorArgs], MockOpenAiClient>().returns(mockClient)

		// Use proxyquire to mock OpenAI so we can inject the stubbed client
		const deepSeekModule = proxyquire("../deepseek", {
			openai: {
				__esModule: true,
				default: openAiConstructorStub,
			},
		}) as { DeepSeekHandler: DeepSeekHandlerConstructor }
		DeepSeekHandlerClass = deepSeekModule.DeepSeekHandler

		mockOptions = {
			deepSeekApiKey: "test-api-key",
			apiModelId: "deepseek-chat",
		}
		handler = new DeepSeekHandlerClass(mockOptions)
	})

	teardown(() => {
		sinon.restore()
	})

	suite("constructor", () => {
		test("should initialize with provided options", () => {
			assert.ok(handler instanceof DeepSeekHandlerClass)
			assert.strictEqual(handler.getModel().id, mockOptions.apiModelId)
		})

		test("should use default model if none provided", () => {
			const handlerWithoutModel = new DeepSeekHandlerClass({
				deepSeekApiKey: "test-api-key",
			})
			const model = handlerWithoutModel.getModel()
			assert.notStrictEqual(model.id, undefined)
		})

		test("should initialize with custom base URL", () => {
			const customHandler = new DeepSeekHandlerClass({
				...mockOptions,
				deepSeekBaseUrl: "https://custom.deepseek.com",
			})
			assert.ok(customHandler instanceof DeepSeekHandlerClass)
		})
	})

	suite("createMessage", () => {
		const systemPrompt = "You are a helpful assistant."

		setup(() => {
			mockCreate.resetBehavior()
			mockCreate.resetHistory()
			// Setup default mock for streaming response
			mockCreate.callsFake(async function* (): AsyncGenerator<MockOpenAiStreamChunk, void, unknown> {
				await Promise.resolve()
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
						prompt_tokens_details: {
							cache_miss_tokens: 8,
							cached_tokens: 2,
						},
					},
				}
			})
		})

		test("should create message stream", async () => {
			const neutralMessages: NeutralConversationHistory = [
				{
					role: "user",
					content: [{ type: "text", text: "Test message" }],
				},
			]

			const stream = handler.createMessage(systemPrompt, neutralMessages)
			const chunks: ApiStreamChunk[] = []
			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			assert.ok(chunks.length > 0)
			assert.ok(mockCreate.calledOnce)

			// Verify the OpenAI client was called with correct parameters
			const callArgs = mockCreate.firstCall.args[0]
			assert.strictEqual(callArgs.model, "deepseek-chat")
			assert.strictEqual(callArgs.stream, true)
			assert.ok(Array.isArray(callArgs.messages))
		})

		test("should handle system prompt in messages", async () => {
			const neutralMessages: NeutralConversationHistory = [
				{
					role: "user",
					content: [{ type: "text", text: "Hello" }],
				},
			]

			const stream = handler.createMessage(systemPrompt, neutralMessages)
			const chunks: ApiStreamChunk[] = []
			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			const callArgs = mockCreate.firstCall.args[0]
			assert.ok(callArgs.messages.some((msg) => msg.role === "system"))
		})
	})

	suite("completePrompt", () => {
		test("should complete prompt successfully", async () => {
			mockCreate.resetBehavior()
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
			} satisfies MockOpenAiCompletion)

			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "Test response")
			assert.ok(mockCreate.calledOnce)

			const callArgs = mockCreate.firstCall.args[0]
			assert.strictEqual(callArgs.stream, false)
		})

		test("should handle empty response", async () => {
			mockCreate.resetBehavior()
			mockCreate.resolves({
				id: "test-completion",
				choices: [{
					message: { role: "assistant", content: "", refusal: null },
					finish_reason: "stop",
					index: 0,
				}],
			} satisfies MockOpenAiCompletion)

			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "")
		})
	})

	suite("getModel", () => {
		test("should return model information", () => {
			const model = handler.getModel()
			assert.strictEqual(model.id, "deepseek-chat")
			assert.notStrictEqual(model.info, undefined)
			assert.ok(model.info.maxTokens > 0)
			assert.ok(model.info.contextWindow > 0)
		})

		test("should return correct properties for deepseek models", () => {
			const model = handler.getModel()
			assert.strictEqual(model.info.supportsImages, false)
			assert.strictEqual(model.info.supportsPromptCache, false)
		})
	})

	suite("countTokens", () => {
		test("should count tokens for text content", async () => {
			mockCreate.resetBehavior()
			mockCreate.resolves({
				id: "usage-test",
				choices: [{
					message: { role: "assistant", content: "", refusal: null },
					finish_reason: "stop",
					index: 0,
				}],
				usage: { prompt_tokens: 42, completion_tokens: 0, total_tokens: 42 },
			} satisfies MockOpenAiCompletion)
			const neutralContent: NeutralMessageContent = [{ type: "text", text: "Test message with some tokens" }]
			const result = await handler.countTokens(neutralContent)
			
			// Should return a reasonable token count
			assert.ok(typeof result === "number")
			assert.ok(result > 0)
		})

		test("should handle empty content", async () => {
			mockCreate.resetBehavior()
			mockCreate.resolves({
				id: "usage-empty",
				choices: [{
					message: { role: "assistant", content: "", refusal: null },
					finish_reason: "stop",
					index: 0,
				}],
				usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
			} satisfies MockOpenAiCompletion)
			const neutralContent: NeutralMessageContent = []
			const result = await handler.countTokens(neutralContent)
			
			assert.strictEqual(result, 0)
		})
	})
// Mock cleanup
})
