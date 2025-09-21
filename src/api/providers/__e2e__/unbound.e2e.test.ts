import { ApiHandlerOptions } from "../../../shared/api"
import { expect } from 'chai'
import { UnboundHandler } from "../unbound"
import type { NeutralConversationHistory } from "../../../shared/neutral-history"
import type OpenAI from "openai" // Added for types
import type { ApiStreamChunk } from "../../transform/stream" // Added for chunk typing
import { EXTENSION_NAME } from "../../../shared/config/thea-config"
import * as assert from 'assert'
import * as sinon from 'sinon'

// Mock OpenAI client  
const mockCreate = sinon.stub<
	Promise<OpenAI.Chat.Completions.ChatCompletion> | OpenAI.Chat.Completions.ChatCompletion,
	[OpenAI.Chat.Completions.ChatCompletionCreateParams, OpenAI.RequestOptions?]
>()
const mockWithResponse = sinon.stub()

// Mock needs manual implementation
		__esModule: true,
		default: sinon.stub().callsFake(() => ({
			chat: {
				completions: {
					create: (
						options: OpenAI.Chat.Completions.ChatCompletionCreateParams,
						requestOptions?: OpenAI.RequestOptions,
					) => {
						const stream = {
							[Symbol.asyncIterator]: function* () {
								// First chunk with content
								yield {
									choices: [
										{
											delta: { content: "Test response" },
											index: 0,
										},
									],
								}
								// Second chunk with usage data
								yield {
									choices: [{ delta: {}, index: 0 }],
									usage: {
										prompt_tokens: 10,
										completion_tokens: 5,
										total_tokens: 15,
									},
								}
								// Third chunk with cache usage data
								yield {
									choices: [{ delta: {}, index: 0 }],
									usage: {
										prompt_tokens: 8,
										completion_tokens: 4,
										total_tokens: 12,
										cache_creation_input_tokens: 3,
										cache_read_input_tokens: 2,
									},
								}
							},
						}

						const result = mockCreate(options, requestOptions) as (OpenAI.Chat.Completions.ChatCompletion | Promise<OpenAI.Chat.Completions.ChatCompletion>) & { withResponse?: typeof mockWithResponse }
						if (options.stream) {
							mockWithResponse.returns(
								Promise.resolve({
									data: stream,
									response: { headers: new Map() },
								})
							result.withResponse = mockWithResponse
						}
						return result
					},
				},
			},
		})),
	}
// Mock cleanup
suite("UnboundHandler", () => {
	let handler: UnboundHandler
	let mockOptions: ApiHandlerOptions

	setup(() => {
		mockOptions = {
			apiModelId: "anthropic/claude-3-5-sonnet-20241022",
			unboundApiKey: "test-api-key",
			unboundModelId: "anthropic/claude-3-5-sonnet-20241022",
			unboundModelInfo: {
				description: "Anthropic's Claude 3 Sonnet model",
				maxTokens: 8192,
				contextWindow: 200000,
				supportsPromptCache: true,
				inputPrice: 0.01,
				outputPrice: 0.02,
			},
		}
		handler = new UnboundHandler(mockOptions)
		mockCreate.resetHistory()
		mockWithResponse.resetHistory()

		// Default mock implementation for non-streaming responses
		mockCreate.resolves({
			id: "test-completion",
			created: Date.now(),
			model: "test-model",
			object: "chat.completion",
			choices: [			{
				message: { role: "assistant", content: "Test response", refusal: null },
				finish_reason: "stop",
				index: 0,
				logprobs: null,
			},
			],
		})
	})

	suite("constructor", () => {
		test("should initialize with provided options", () => {
			assert.ok(handler instanceof UnboundHandler)
			expect(handler.getModel().id).to.equal(mockOptions.apiModelId)
		})
	})

	suite("createMessage", () => {
		const systemPrompt = "You are a helpful assistant."
		const messages: NeutralConversationHistory = [
			{
				role: "user",
				content: "Hello!",
			},
		]

		test("should handle streaming responses with text and usage data", async () => {
			const stream = handler.createMessage(systemPrompt, messages)
			const chunks: ApiStreamChunk[] = []
			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			assert.strictEqual(chunks.length, 3)

			// Verify text chunk
			assert.deepStrictEqual(chunks[0], {
				type: "text",
				text: "Test response",
			})

			// Verify regular usage data
			assert.deepStrictEqual(chunks[1], {
				type: "usage",
				inputTokens: 10,
				outputTokens: 5,
			})

			// Verify usage data with cache information
			assert.deepStrictEqual(chunks[2], {
				type: "usage",
				inputTokens: 8,
				outputTokens: 4,
				cacheWriteTokens: 3,
				cacheReadTokens: 2,
			})

			assert.ok(mockCreate.calledWith({
					model: "claude-3-5-sonnet-20241022",
					messages: sinon.match.instanceOf(Array)), // eslint-disable-line @typescript-eslint/no-unsafe-assignment
					stream: true,
				}),
				// TODO: Object partial match - {
					headers: {
						"X-Unbound-Metadata": // TODO: String contains check - EXTENSION_NAME), // eslint-disable-line @typescript-eslint/no-unsafe-assignment
					},
				})
		})

		test("should handle API errors", async () => {
			mockCreate.mockImplementationOnce(() => {
				throw new Error("API Error")
			})

			const stream = handler.createMessage(systemPrompt, messages)
			const chunks = []

			try {
				for await (const chunk of stream) {
					chunks.push(chunk)
				} catch (error) {
			assert.fail('Unexpected error: ' + error.message)
		}assert.fail("Expected error to be thrown")
} catch (e) {
				const error = e as Error
				assert.ok(error instanceof Error)
				assert.strictEqual(error.message, "API Error")
			} catch (error) {
			assert.fail("Unexpected error: " + error.message)
		}
		})
	})

	suite("completePrompt", () => {
		test("should complete prompt successfully", async () => {
			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "Test response")
			// Verify the call was made with correct parameters
			assert.strictEqual(mockCreate.callCount, 1)
			const [params, options] = mockCreate.mock.calls[0]
			
			// Verify request parameters
			expect(params).toMatchObject({
				model: "claude-3-5-sonnet-20241022",
				messages: [{ role: "user", content: "Test prompt" }],
				temperature: 0,
				max_tokens: 8192,
			})
			
			// Verify request options and headers
			assert.notStrictEqual(options?.headers, undefined)
			const headers = options?.headers as Record<string, string>
			assert.ok(headers["X-Unbound-Metadata"].includes(EXTENSION_NAME))
		})

		test("should handle API errors", async () => {
			mockCreate.mockRejectedValueOnce(new Error("API Error"))
			await expect(handler.completePrompt("Test prompt")).rejects.toThrow("Unbound completion error: API Error")
		})

		test("should handle empty response", async () => {
			mockCreate.mockResolvedValueOnce({
				id: "test-completion",
				created: Date.now(),
				model: "test-model",
				object: "chat.completion",
				choices: [{ message: { role: "assistant", content: "", refusal: null }, finish_reason: "stop", index: 0, logprobs: null }],
			})
			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "")
		})

		test("should not set max_tokens for non-Anthropic models", async () => {
			mockCreate.resetHistory()

			const nonAnthropicOptions = {
				apiModelId: "openai/gpt-4o",
				unboundApiKey: "test-key",
				unboundModelId: "openai/gpt-4o",
				unboundModelInfo: {
					description: "OpenAI's GPT-4",
					maxTokens: undefined,
					contextWindow: 128000,
					supportsPromptCache: true,
					inputPrice: 0.01,
					outputPrice: 0.03,
				},
			}
			const nonAnthropicHandler = new UnboundHandler(nonAnthropicOptions)

			await nonAnthropicHandler.completePrompt("Test prompt")
			// Verify the call was made with correct parameters
			assert.strictEqual(mockCreate.callCount, 1)
			const [params, options] = mockCreate.mock.calls[0]
			
			// Verify request parameters
			expect(params).toMatchObject({
				model: "gpt-4o",
				messages: [{ role: "user", content: "Test prompt" }],
				temperature: 0,
			})
			
			// Verify request options and headers
			assert.notStrictEqual(options?.headers, undefined)
			const headers = options?.headers as Record<string, string>
			assert.ok(headers["X-Unbound-Metadata"].includes(EXTENSION_NAME))
			assert.ok(!mockCreate.mock.calls[0][0].hasOwnProperty('max_tokens'))
		})

		test("should not set temperature for openai/o3-mini", async () => {
			mockCreate.resetHistory()

			const openaiOptions = {
				apiModelId: "openai/o3-mini",
				unboundApiKey: "test-key",
				unboundModelId: "openai/o3-mini",
				unboundModelInfo: {
					maxTokens: undefined,
					contextWindow: 128000,
					supportsPromptCache: true,
					inputPrice: 0.01,
					outputPrice: 0.03,
				},
			}
			const openaiHandler = new UnboundHandler(openaiOptions)

			await openaiHandler.completePrompt("Test prompt")
			// Verify the call was made with correct parameters
			assert.strictEqual(mockCreate.callCount, 1)
			const [params, options] = mockCreate.mock.calls[0]
			
			// Verify request parameters
			expect(params).toMatchObject({
				model: "o3-mini",
				messages: [{ role: "user", content: "Test prompt" }],
			})
			
			// Verify request options and headers
			assert.notStrictEqual(options?.headers, undefined)
			const headers = options?.headers as Record<string, string>
			assert.ok(headers["X-Unbound-Metadata"].includes(EXTENSION_NAME))
			assert.ok(!mockCreate.mock.calls[0][0].hasOwnProperty('temperature'))
		})
	})

	suite("getModel", () => {
		test("should return model info", () => {
			const modelInfo = handler.getModel()
			assert.strictEqual(modelInfo.id, mockOptions.apiModelId)
			assert.notStrictEqual(modelInfo.info, undefined)
		})

		test("should return default model when invalid model provided", () => {
			const handlerWithInvalidModel = new UnboundHandler({
				...mockOptions,
				unboundModelId: "invalid/model",
				unboundModelInfo: undefined,
			})
			const modelInfo = handlerWithInvalidModel.getModel()
			assert.strictEqual(modelInfo.id, "anthropic/claude-3-5-sonnet-20241022") // Default model
			assert.notStrictEqual(modelInfo.info, undefined)
		})
	})
// Mock cleanup
