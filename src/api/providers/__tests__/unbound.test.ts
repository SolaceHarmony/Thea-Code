import { UnboundHandler } from "../unbound"
import { ApiHandlerOptions } from "../../../shared/api"
import type { NeutralConversationHistory } from "../../../shared/neutral-history"
import type OpenAI from "openai" // Added for types
import { EXTENSION_NAME } from "../../../../dist/thea-config" // Import branded constant
import type { ApiStreamChunk } from "../../transform/stream" // Added for chunk typing

// Mock OpenAI client  
const mockCreate = jest.fn<
	Promise<OpenAI.Chat.Completions.ChatCompletion> | OpenAI.Chat.Completions.ChatCompletion,
	[OpenAI.Chat.Completions.ChatCompletionCreateParams, OpenAI.RequestOptions?]
>()
const mockWithResponse = jest.fn()

jest.mock("openai", () => {
	return {
		__esModule: true,
		default: jest.fn().mockImplementation(() => ({
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
							mockWithResponse.mockReturnValue(
								Promise.resolve({
									data: stream,
									response: { headers: new Map() },
								}),
							)
							result.withResponse = mockWithResponse
						}
						return result
					},
				},
			},
		})),
	}
})

describe("UnboundHandler", () => {
	let handler: UnboundHandler
	let mockOptions: ApiHandlerOptions

	beforeEach(() => {
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
		mockCreate.mockClear()
		mockWithResponse.mockClear()

		// Default mock implementation for non-streaming responses
		mockCreate.mockResolvedValue({
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

	describe("constructor", () => {
		it("should initialize with provided options", () => {
			expect(handler).toBeInstanceOf(UnboundHandler)
			expect(handler.getModel().id).toBe(mockOptions.apiModelId)
		})
	})

	describe("createMessage", () => {
		const systemPrompt = "You are a helpful assistant."
		const messages: NeutralConversationHistory = [
			{
				role: "user",
				content: "Hello!",
			},
		]

		it("should handle streaming responses with text and usage data", async () => {
			const stream = handler.createMessage(systemPrompt, messages)
			const chunks: ApiStreamChunk[] = []
			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			expect(chunks.length).toBe(3)

			// Verify text chunk
			expect(chunks[0]).toEqual({
				type: "text",
				text: "Test response",
			})

			// Verify regular usage data
			expect(chunks[1]).toEqual({
				type: "usage",
				inputTokens: 10,
				outputTokens: 5,
			})

			// Verify usage data with cache information
			expect(chunks[2]).toEqual({
				type: "usage",
				inputTokens: 8,
				outputTokens: 4,
				cacheWriteTokens: 3,
				cacheReadTokens: 2,
			})

			expect(mockCreate).toHaveBeenCalledWith(
				expect.objectContaining({
					model: "claude-3-5-sonnet-20241022",
					messages: expect.any(Array), // eslint-disable-line @typescript-eslint/no-unsafe-assignment
					stream: true,
				}),
				expect.objectContaining({
					headers: {
						"X-Unbound-Metadata": expect.stringContaining(EXTENSION_NAME), // eslint-disable-line @typescript-eslint/no-unsafe-assignment
					},
				}),
			)
		})

		it("should handle API errors", async () => {
			mockCreate.mockImplementationOnce(() => {
				throw new Error("API Error")
			})

			const stream = handler.createMessage(systemPrompt, messages)
			const chunks = []

			try {
				for await (const chunk of stream) {
					chunks.push(chunk)
				}
				fail("Expected error to be thrown")
			} catch (e) {
				const error = e as Error
				expect(error).toBeInstanceOf(Error)
				expect(error.message).toBe("API Error")
			}
		})
	})

	describe("completePrompt", () => {
		it("should complete prompt successfully", async () => {
			const result = await handler.completePrompt("Test prompt")
			expect(result).toBe("Test response")
			// Verify the call was made with correct parameters
			expect(mockCreate).toHaveBeenCalledTimes(1)
			const [params, options] = mockCreate.mock.calls[0]
			
			// Verify request parameters
			expect(params).toMatchObject({
				model: "claude-3-5-sonnet-20241022",
				messages: [{ role: "user", content: "Test prompt" }],
				temperature: 0,
				max_tokens: 8192,
			})
			
			// Verify request options and headers
			expect(options?.headers).toBeDefined()
			const headers = options?.headers as Record<string, string>
			expect(headers["X-Unbound-Metadata"]).toContain(EXTENSION_NAME)
		})

		it("should handle API errors", async () => {
			mockCreate.mockRejectedValueOnce(new Error("API Error"))
			await expect(handler.completePrompt("Test prompt")).rejects.toThrow("Unbound completion error: API Error")
		})

		it("should handle empty response", async () => {
			mockCreate.mockResolvedValueOnce({
				id: "test-completion",
				created: Date.now(),
				model: "test-model",
				object: "chat.completion",
				choices: [{ message: { role: "assistant", content: "", refusal: null }, finish_reason: "stop", index: 0, logprobs: null }],
			})
			const result = await handler.completePrompt("Test prompt")
			expect(result).toBe("")
		})

		it("should not set max_tokens for non-Anthropic models", async () => {
			mockCreate.mockClear()

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
			expect(mockCreate).toHaveBeenCalledTimes(1)
			const [params, options] = mockCreate.mock.calls[0]
			
			// Verify request parameters
			expect(params).toMatchObject({
				model: "gpt-4o",
				messages: [{ role: "user", content: "Test prompt" }],
				temperature: 0,
			})
			
			// Verify request options and headers
			expect(options?.headers).toBeDefined()
			const headers = options?.headers as Record<string, string>
			expect(headers["X-Unbound-Metadata"]).toContain(EXTENSION_NAME)
			expect(mockCreate.mock.calls[0][0]).not.toHaveProperty("max_tokens")
		})

		it("should not set temperature for openai/o3-mini", async () => {
			mockCreate.mockClear()

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
			expect(mockCreate).toHaveBeenCalledTimes(1)
			const [params, options] = mockCreate.mock.calls[0]
			
			// Verify request parameters
			expect(params).toMatchObject({
				model: "o3-mini",
				messages: [{ role: "user", content: "Test prompt" }],
			})
			
			// Verify request options and headers
			expect(options?.headers).toBeDefined()
			const headers = options?.headers as Record<string, string>
			expect(headers["X-Unbound-Metadata"]).toContain(EXTENSION_NAME)
			expect(mockCreate.mock.calls[0][0]).not.toHaveProperty("temperature")
		})
	})

	describe("getModel", () => {
		it("should return model info", () => {
			const modelInfo = handler.getModel()
			expect(modelInfo.id).toBe(mockOptions.apiModelId)
			expect(modelInfo.info).toBeDefined()
		})

		it("should return default model when invalid model provided", () => {
			const handlerWithInvalidModel = new UnboundHandler({
				...mockOptions,
				unboundModelId: "invalid/model",
				unboundModelInfo: undefined,
			})
			const modelInfo = handlerWithInvalidModel.getModel()
			expect(modelInfo.id).toBe("anthropic/claude-3-5-sonnet-20241022") // Default model
			expect(modelInfo.info).toBeDefined()
		})
	})
})
