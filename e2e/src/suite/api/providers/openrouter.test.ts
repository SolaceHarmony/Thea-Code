import * as assert from 'assert'
import * as sinon from 'sinon'
import axios from "axios"
import OpenAI from "openai"
import type { ApiStreamChunk } from "../../transform/stream" // Added for chunk typing
import { NeutralMessage } from "../../../shared/neutral-history"
import { ApiHandlerOptions, ModelInfo } from "../../../shared/api"
import { OpenRouterHandler } from "../openrouter"
import { API_REFERENCES } from "../../../shared/config/thea-config"
// Mock dependencies
// Mock needs manual implementation

const mockOpenRouterModelInfo: ModelInfo = {
	maxTokens: 1000,
	contextWindow: 2000,
	supportsPromptCache: true,
	inputPrice: 0.01,
	outputPrice: 0.02,
}

suite("OpenRouterHandler", () => {
	const mockOptions: ApiHandlerOptions = {
		openRouterApiKey: "test-key",
		openRouterModelId: "test-model",
		openRouterModelInfo: mockOpenRouterModelInfo,
	}

	setup(() => {
		sinon.restore()
	})

	test("constructor initializes with correct options", () => {
		const handler = new OpenRouterHandler(mockOptions)
		assert.ok(handler instanceof OpenRouterHandler)
		assert.ok(OpenAI.calledWith({
			baseURL: "https://openrouter.ai/api/v1",
			apiKey: mockOptions.openRouterApiKey,
			defaultHeaders: {
				"HTTP-Referer": API_REFERENCES.HOMEPAGE,
				"X-Title": API_REFERENCES.APP_TITLE,
			},
		})
	})

	test("getModel returns correct model info when options are provided", () => {
		const handler = new OpenRouterHandler(mockOptions)
		const result = handler.getModel()

		assert.deepStrictEqual(result, {
			id: mockOptions.openRouterModelId,
			info: mockOptions.openRouterModelInfo,
			maxTokens: 1000,
			temperature: 0,
			thinking: undefined,
			topP: undefined,
		})
	})

	test("getModel returns default model info when options are not provided", () => {
		const handler = new OpenRouterHandler({})
		const result = handler.getModel()

		assert.strictEqual(result.id, "anthropic/claude-3.7-sonnet")
		assert.strictEqual(result.info.supportsPromptCache, true)
	})

	test("getModel honors custom maxTokens for thinking models", () => {
		const handler = new OpenRouterHandler({
			openRouterApiKey: "test-key",
			openRouterModelId: "test-model",
			openRouterModelInfo: {
				...mockOpenRouterModelInfo,
				maxTokens: 128_000,
				thinking: true,
			},
			modelMaxTokens: 32_768,
			modelMaxThinkingTokens: 16_384,
		})

		const result = handler.getModel()
		assert.strictEqual(result.maxTokens, 32_768)
		assert.deepStrictEqual(result.thinking, { type: "enabled", budget_tokens: 16_384 })
		assert.strictEqual(result.temperature, 1.0)
	})

	test("getModel does not honor custom maxTokens for non-thinking models", () => {
		const handler = new OpenRouterHandler({
			...mockOptions,
			modelMaxTokens: 32_768,
			modelMaxThinkingTokens: 16_384,
		})

		const result = handler.getModel()
		assert.strictEqual(result.maxTokens, 1000)
		assert.strictEqual(result.thinking, undefined)
		assert.strictEqual(result.temperature, 0)
	})

	test("createMessage generates correct stream chunks", async () => {
		const handler = new OpenRouterHandler(mockOptions)
		// eslint-disable-next-line @typescript-eslint/require-await
		const mockStream: AsyncIterable<OpenAI.Chat.Completions.ChatCompletionChunk> = (async function* () {
			yield {
				id: "test-id-1",
				created: 1678886400,
				model: "test-model",
				object: "chat.completion.chunk",
				choices: [
					{
						delta: {
							content: "test response",
						},
						index: 0,
						finish_reason: null,
					},
				],
			}
			// Add usage information in the stream response
			yield {
				id: "test-id-2",
				created: 1678886401,
				model: "test-model",
				object: "chat.completion.chunk",
				choices: [{ delta: {}, index: 0, finish_reason: "stop" }],
				usage: {
					prompt_tokens: 10,
					completion_tokens: 20,
					total_tokens: 30,
				},
			}
		})()

		// Mock OpenAI chat.completions.create
		const mockCreate = sinon
			.stub(OpenAI.prototype.chat.completions, "create")
			.callsFake(() => {
				// Return the mock stream directly, casting to the expected return type
				// This works because the OpenRouter handler casts it to AsyncIterable anyway
				return mockStream as unknown as ReturnType<typeof OpenAI.prototype.chat.completions.create>
			})

		const systemPrompt = "test system prompt"
		const messages: NeutralMessage[] = [{ role: "user", content: "test message" }]

		const generator = handler.createMessage(systemPrompt, messages)
		const chunks: ApiStreamChunk[] = []

		for await (const chunk of generator) {
			chunks.push(chunk)
		}

		// Verify stream chunks
		assert.strictEqual(chunks.length, 2) // One text chunk and one usage chunk
		assert.deepStrictEqual(chunks[0], {
			type: "text",
			text: "test response",
		})
		assert.deepStrictEqual(chunks[1], {
			type: "usage",
			inputTokens: 10,
			outputTokens: 20,
			// totalCost: 0.001, // Removed as cost is no longer in mock usage
		})

		// Verify OpenAI client was called with correct parameters
		assert.ok(mockCreate.calledWith({
				model: mockOptions.openRouterModelId,
				temperature: 0,
				messages: // TODO: Array partial match - [
					{ role: "system", content: systemPrompt },
					{ role: "user", content: "test message" },
				])) as unknown,
				stream: true,
			})
	})

	test("createMessage with middle-out transform enabled", async () => {
		const handler = new OpenRouterHandler({
			...mockOptions,
			openRouterUseMiddleOutTransform: true,
		})
		// eslint-disable-next-line @typescript-eslint/require-await
		const mockStream: AsyncIterable<OpenAI.Chat.Completions.ChatCompletionChunk> = (async function* () {
			yield {
				id: "test-id-3",
				created: 1678886402,
				model: "test-model",
				object: "chat.completion.chunk",
				choices: [
					{
						delta: {
							content: "test response",
						},
						index: 0,
						finish_reason: "stop",
					},
				],
			}
		})()

		const mockCreate = sinon
			.stub(OpenAI.prototype.chat.completions, "create")
			.callsFake(() => {
				return mockStream as unknown as ReturnType<typeof OpenAI.prototype.chat.completions.create>
			})
		;(axios.get as sinon.SinonStub).resolves({ data: { data: {} } })

		await handler.createMessage("test", []).next()

		assert.ok(mockCreate.calledWith({
				transforms: ["middle-out"],
			})),
		)
	})

	test("createMessage with Claude model adds cache control", async () => {
		const handler = new OpenRouterHandler({
			...mockOptions,
			openRouterModelId: "anthropic/claude-3.5-sonnet",
		})
		// eslint-disable-next-line @typescript-eslint/require-await
		const mockStream: AsyncIterable<OpenAI.Chat.Completions.ChatCompletionChunk> = (async function* () {
			yield {
				id: "test-id-4",
				created: 1678886403,
				model: "test-model",
				object: "chat.completion.chunk",
				choices: [
					{
						delta: {
							content: "test response",
						},
						index: 0,
						finish_reason: "stop",
					},
				],
			}
		})()

		const mockCreate = sinon
			.stub(OpenAI.prototype.chat.completions, "create")
			.callsFake(() => {
				return mockStream as unknown as ReturnType<typeof OpenAI.prototype.chat.completions.create>
			})
		;(axios.get as sinon.SinonStub).resolves({ data: { data: {} } })

		const messages: NeutralMessage[] = [
			{ role: "user", content: "message 1" },
			{ role: "assistant", content: "response 1" },
			{ role: "user", content: "message 2" },
		]

		await handler.createMessage("test system", messages).next()

		assert.ok(mockCreate.calledWith({
				messages: // TODO: Array partial match - [
					// TODO: Object partial match - {
						role: "system",
						content: // TODO: Array partial match - [
							// TODO: Object partial match - {
								cache_control: { type: "ephemeral" },
							})),
						]) as unknown,
					}),
				]) as unknown,
			})
	})

	test("createMessage handles API errors", async () => {
		const handler = new OpenRouterHandler(mockOptions)
		// eslint-disable-next-line @typescript-eslint/require-await
		const mockStream: AsyncIterable<OpenAI.Chat.Completions.ChatCompletionChunk> = (async function* () {
			throw new Error("API Error") // Throw error directly from the stream
		})()

		sinon.stub(OpenAI.prototype.chat.completions, "create").callsFake(
			() => {
				return mockStream as unknown as ReturnType<typeof OpenAI.prototype.chat.completions.create>
			}
		)

		const generator = handler.createMessage("test", [])
		try {
			await generator.next()
			assert.fail("Should have thrown an error")
} catch (error) {
			assert.ok(error instanceof Error)
			assert.strictEqual(error.message, "OpenRouter API Error 500: API Error")
		}
	})

	test("completePrompt returns correct response", async () => {
		const handler = new OpenRouterHandler(mockOptions)
		const mockResponse: OpenAI.Chat.Completions.ChatCompletion = {
			id: "chatcmpl-test",
			choices: [
				{
					message: { content: "test completion", role: "assistant", refusal: null },
					finish_reason: "stop",
					index: 0,
					logprobs: null,
				},
			],
			created: 1234567890,
			model: "test-model",
			object: "chat.completion",
		}

		const mockCreate = sinon.stub(OpenAI.prototype.chat.completions, "create").resolves(mockResponse)

		const result = await handler.completePrompt("test prompt")

		assert.strictEqual(result, "test completion")

		assert.ok(mockCreate.calledWith({
			model: mockOptions.openRouterModelId,
			max_tokens: 1000,
			thinking: undefined,
			temperature: 0,
			messages: [{ role: "user", content: "test prompt" }],
			stream: false,
		}))
	})

	test("completePrompt handles API errors", async () => {
		const handler = new OpenRouterHandler(mockOptions)
		const mockError = new OpenAI.APIError(500, { error: { message: "API Error" } }, "API Error", new Headers())

		sinon.stub(OpenAI.prototype.chat.completions, "create").rejects(mockError)

		try {
			await handler.completePrompt("test prompt")
			assert.fail("Should have thrown an error")
} catch (error) {
			assert.ok(error instanceof Error)
			assert.strictEqual(error.message, "OpenRouter API Error 500: API Error")
		}
	})

	test("completePrompt handles unexpected errors", async () => {
		const handler = new OpenRouterHandler(mockOptions)
		sinon.stub(OpenAI.prototype.chat.completions, "create").callsFake(() => {
			throw new Error("Unexpected error")
		})

		try {
			await handler.completePrompt("test prompt")
			assert.fail("Should have thrown an error")
} catch (error) {
			assert.ok(error instanceof Error)
			assert.strictEqual(error.message, "Unexpected error")
		}
	})

	test("createMessage processes OpenAI tool calls", async () => {
		const handler = new OpenRouterHandler(mockOptions)
		// eslint-disable-next-line @typescript-eslint/require-await
		const mockStream: AsyncIterable<OpenAI.Chat.Completions.ChatCompletionChunk> = (async function* () {
			yield {
				id: "test-id-6",
				created: 1678886405,
				model: "test-model",
				object: "chat.completion.chunk",
				choices: [
					{
						delta: {
							tool_calls: [
								{
									index: 0,
									id: "call1",
									function: { name: "testTool", arguments: '{"foo":1}' },
								},
							],
						},
						index: 0,
						finish_reason: null,
					},
				],
			}
		})()

		sinon.stub(OpenAI.prototype.chat.completions, "create").callsFake(
			() => {
				return mockStream as unknown as ReturnType<typeof OpenAI.prototype.chat.completions.create>
			}
		)

		const processSpy = sinon
			.stub(handler as unknown as { processToolUse: (content: string | Record<string, unknown>) => Promise<string> }, "processToolUse")
			.resolves("ok")

		const generator = handler.createMessage("test", [])
		const chunks: ApiStreamChunk[] = []
		for await (const chunk of generator) {
			chunks.push(chunk)
		}

		assert.ok(processSpy.calledWith({ id: "call1", name: "testTool", input: { foo: 1 } }))
		const toolResultChunk = chunks.find(c => c.type === "tool_result" && c.id === "call1")
		assert.notStrictEqual(toolResultChunk, undefined)
		assert.deepStrictEqual(toolResultChunk, { type: "tool_result", id: "call1", content: { result: "ok" } })
	})
// Mock cleanup
