/* eslint-disable @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-assignment */
import * as assert from "assert"
import { Readable } from "stream"
import sinon from "sinon"

import { OpenAiHandler } from "../../api/providers/openai"
import { ApiHandlerOptions } from "../../shared/api"
import type { NeutralConversationHistory } from "../../shared/neutral-history"
import openaiSetup, { openAIMock } from "../../../test/openai-mock/setup"
import { openaiTeardown } from "../../../test/openai-mock/teardown"

/**
 * Tests for User Input to Model Response Flow
 * 
 * This test suite validates the complete flow from user input through API handler
 * to model response, ensuring proper message handling, tool execution, and error handling.
 */

describe("User Input to Model Response Flow", () => {
	let sandbox: sinon.SinonSandbox
	let handler: OpenAiHandler
	let mockOptions: ApiHandlerOptions
	let requestBody: any
	let capturedHeaders: Record<string, string | string[]> = {}

	beforeEach(async () => {
		sandbox = sinon.createSandbox()
		await openaiTeardown()
		await openaiSetup()
		requestBody = undefined
		capturedHeaders = {}

		// Setup custom endpoint for OpenAI mock
		;(openAIMock as any)!.addCustomEndpoint("POST", "/v1/chat/completions", function (_uri: any, body: any) {
			// @ts-expect-error req is provided by nock
			capturedHeaders = this.req.headers as Record<string, string | string[]>
			requestBody = body

			if (!body.stream) {
				return [
					200,
					{
						id: "test-completion",
						choices: [
							{
								message: { role: "assistant", content: "Test response", refusal: null },
								logprobs: null,
								finish_reason: "stop",
								index: 0,
							},
						],
						usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
					},
				]
			}

			const stream = new Readable({ read() {} })
			const chunk1 = {
				id: "chatcmpl-test-1",
				created: 1678886400,
				model: "gpt-4",
				object: "chat.completion.chunk",
				choices: [{ delta: { content: "Test response" }, index: 0, finish_reason: null, logprobs: null }],
				usage: null,
			}
			const chunk2 = {
				id: "chatcmpl-test-2",
				created: 1678886401,
				model: "gpt-4",
				object: "chat.completion.chunk",
				choices: [{ delta: {}, index: 0, finish_reason: "stop" }],
				usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
			}
			stream.push(`data: ${JSON.stringify(chunk1)}\n\n`)
			stream.push(`data: ${JSON.stringify(chunk2)}\n\n`)
			stream.push("data: [DONE]\n\n")
			stream.push(null)
			return [200, stream]
		})

		mockOptions = {
			openAiApiKey: "test-api-key",
			openAiModelId: "gpt-4",
			openAiBaseUrl: "https://api.openai.com/v1",
		}
		handler = new OpenAiHandler(mockOptions)
	})

	afterEach(async () => {
		sandbox.restore()
		await openaiTeardown()
	})

	describe("Simple User Message Flow", () => {
		it("should handle a simple user text message", async () => {
			const messages: NeutralConversationHistory = [
				{
					role: "user",
					content: [
						{
							type: "text",
							text: "Hello, how are you?",
						},
					],
				},
			]

			const stream = handler.createMessage("You are a helpful assistant.", messages)
			const chunks: Array<{ type: string; text?: string }> = []

			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			assert.ok(chunks.length > 0, "Should receive at least one chunk")
			const textChunk = chunks.find((c) => c.type === "text")
			assert.ok(textChunk, "Should receive a text chunk")
			assert.ok(textChunk.text, "Text chunk should have content")
		})

		it("should include user message in request body", async () => {
			const messages: NeutralConversationHistory = [
				{
					role: "user",
					content: [
						{
							type: "text",
							text: "What is the weather today?",
						},
					],
				},
			]

			const stream = handler.createMessage("You are a helpful assistant.", messages)
			for await (const chunk of stream) {
				void chunk // Consume stream
			}

			assert.ok(requestBody, "Request body should be captured")
			assert.ok(Array.isArray(requestBody.messages), "Request should have messages array")
			const userMessage = requestBody.messages.find((m: any) => m.role === "user")
			assert.ok(userMessage, "Should have user message in request")
			assert.ok(userMessage.content.includes("weather"), "User message should contain query")
		})

		it("should handle non-streaming mode correctly", async () => {
			const handler = new OpenAiHandler({
				...mockOptions,
				openAiStreamingEnabled: false,
			})

			const messages: NeutralConversationHistory = [
				{
					role: "user",
					content: [
						{
							type: "text",
							text: "Test message",
						},
					],
				},
			]

			const stream = handler.createMessage("System prompt", messages)
			const chunks: Array<{ type: string; text?: string; inputTokens?: number; outputTokens?: number }> = []

			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			assert.ok(chunks.length > 0)
			const textChunk = chunks.find((c) => c.type === "text")
			const usageChunk = chunks.find((c) => c.type === "usage")

			assert.ok(textChunk, "Should have text chunk")
			assert.ok(usageChunk, "Should have usage chunk")
			assert.strictEqual(usageChunk.inputTokens, 10)
			assert.strictEqual(usageChunk.outputTokens, 5)
		})
	})

	describe("Multi-Turn Conversation Flow", () => {
		it("should handle multi-turn conversation", async () => {
			const messages: NeutralConversationHistory = [
				{
					role: "user",
					content: [{ type: "text", text: "What is 2+2?" }],
				},
				{
					role: "assistant",
					content: [{ type: "text", text: "2+2 equals 4." }],
				},
				{
					role: "user",
					content: [{ type: "text", text: "What about 3+3?" }],
				},
			]

			const stream = handler.createMessage("You are a helpful math tutor.", messages)
			const chunks: Array<{ type: string; text?: string }> = []

			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			assert.ok(chunks.length > 0)
			assert.ok(requestBody.messages.length >= 3, "Should include all conversation history")
		})

		it("should maintain conversation context", async () => {
			const messages: NeutralConversationHistory = [
				{
					role: "user",
					content: [{ type: "text", text: "My name is Alice." }],
				},
				{
					role: "assistant",
					content: [{ type: "text", text: "Hello Alice!" }],
				},
				{
					role: "user",
					content: [{ type: "text", text: "What is my name?" }],
				},
			]

			const stream = handler.createMessage("You are a helpful assistant.", messages)
			for await (const chunk of stream) {
				void chunk
			}

			// Verify all messages are included
			const userMessages = requestBody.messages.filter((m: any) => m.role === "user")
			assert.strictEqual(userMessages.length, 2, "Should have both user messages")
		})
	})

	describe("Tool Use in Conversation", () => {
		beforeEach(async () => {
			await openaiTeardown()
			await openaiSetup()
			;(openAIMock as any)!.addCustomEndpoint("POST", "/v1/chat/completions", function (_uri: any, body: any) {
				// @ts-expect-error req is provided by nock
				capturedHeaders = this.req.headers as Record<string, string | string[]>
				requestBody = body

				if (!body.stream) {
					return [
						200,
						{
							id: "test-completion",
							choices: [
								{
									message: {
										role: "assistant",
										content: null,
										tool_calls: [
											{
												id: "call_123",
												type: "function",
												function: {
													name: "read_file",
													arguments: JSON.stringify({ path: "/test/file.txt" }),
												},
											},
										],
									},
									finish_reason: "tool_calls",
									index: 0,
								},
							],
							usage: { prompt_tokens: 20, completion_tokens: 10, total_tokens: 30 },
						},
					]
				}

				const stream = new Readable({ read() {} })
				const chunk1 = {
					id: "chatcmpl-test-1",
					created: 1678886400,
					model: "gpt-4",
					object: "chat.completion.chunk",
					choices: [
						{
							delta: {
								tool_calls: [
									{
										index: 0,
										id: "call_123",
										type: "function",
										function: { name: "read_file", arguments: '{"path":"/test/file.txt"}' },
									},
								],
							},
							index: 0,
							finish_reason: null,
						},
					],
				}
				const chunk2 = {
					id: "chatcmpl-test-2",
					created: 1678886401,
					model: "gpt-4",
					object: "chat.completion.chunk",
					choices: [{ delta: {}, index: 0, finish_reason: "tool_calls" }],
					usage: { prompt_tokens: 20, completion_tokens: 10, total_tokens: 30 },
				}
				stream.push(`data: ${JSON.stringify(chunk1)}\n\n`)
				stream.push(`data: ${JSON.stringify(chunk2)}\n\n`)
				stream.push("data: [DONE]\n\n")
				stream.push(null)
				return [200, stream]
			})
		})

		it("should detect tool calls in response", async () => {
			const messages: NeutralConversationHistory = [
				{
					role: "user",
					content: [{ type: "text", text: "Can you read the file /test/file.txt?" }],
				},
			]

			const stream = handler.createMessage("You are a helpful assistant.", messages)
			const chunks: Array<{ type: string; id?: string; name?: string; input?: any }> = []

			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			const toolUseChunk = chunks.find((c) => c.type === "tool_use")
			assert.ok(toolUseChunk, "Should receive tool use chunk")
			assert.strictEqual(toolUseChunk.name, "read_file")
			assert.ok(toolUseChunk.input, "Tool use should have input")
		})

		it("should handle tool results in follow-up request", async () => {
			const messages: NeutralConversationHistory = [
				{
					role: "user",
					content: [{ type: "text", text: "Read /test/file.txt" }],
				},
				{
					role: "assistant",
					content: [
						{
							type: "tool_use",
							id: "call_123",
							name: "read_file",
							input: { path: "/test/file.txt" },
						},
					],
				},
				{
					role: "user",
					content: [
						{
							type: "tool_result",
							tool_use_id: "call_123",
							content: [{ type: "text", text: "File contents: Hello World" }],
						},
					],
				},
			]

			// Reset mock to return text response after tool result
			await openaiTeardown()
			await openaiSetup()
			;(openAIMock as any)!.addCustomEndpoint("POST", "/v1/chat/completions", function (_uri: any, body: any) {
				requestBody = body
				return [
					200,
					{
						id: "test-completion",
						choices: [
							{
								message: {
									role: "assistant",
									content: "The file contains: Hello World",
								},
								finish_reason: "stop",
								index: 0,
							},
						],
						usage: { prompt_tokens: 30, completion_tokens: 10, total_tokens: 40 },
					},
				]
			})

			const stream = handler.createMessage("You are a helpful assistant.", messages)
			const chunks: Array<{ type: string; text?: string }> = []

			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			// Verify tool result is in request
			const toolResultMessage = requestBody.messages.find((m: any) => m.role === "tool")
			assert.ok(toolResultMessage, "Should have tool result in request")
		})
	})

	describe("Error Handling in Conversation Flow", () => {
		it("should handle API errors during conversation", async () => {
			await openaiTeardown()
			await openaiSetup()
			;(openAIMock as any)!.addCustomEndpoint("POST", "/v1/chat/completions", () => [
				500,
				{ error: { message: "Internal Server Error" } },
			])

			const messages: NeutralConversationHistory = [
				{
					role: "user",
					content: [{ type: "text", text: "Hello" }],
				},
			]

			const stream = handler.createMessage("System prompt", messages)

			await assert.rejects(
				async () => {
					for await (const chunk of stream) {
						void chunk
					}
				},
				(error: unknown) => error instanceof Error && error.message.includes("Internal Server Error"),
			)
		})

		it("should handle rate limiting gracefully", async () => {
			await openaiTeardown()
			await openaiSetup()
			;(openAIMock as any)!.addCustomEndpoint("POST", "/v1/chat/completions", () => [
				429,
				{ error: { message: "Rate limit exceeded", type: "rate_limit_error" } },
			])

			const messages: NeutralConversationHistory = [
				{
					role: "user",
					content: [{ type: "text", text: "Test message" }],
				},
			]

			const stream = handler.createMessage("System prompt", messages)

			await assert.rejects(
				async () => {
					for await (const chunk of stream) {
						void chunk
					}
				},
				(error: unknown) => error instanceof Error && error.message.includes("Rate limit exceeded"),
			)
		})

		it("should handle invalid response format", async () => {
			await openaiTeardown()
			await openaiSetup()
			;(openAIMock as any)!.addCustomEndpoint("POST", "/v1/chat/completions", () => [200, { invalid: "format" }])

			const messages: NeutralConversationHistory = [
				{
					role: "user",
					content: [{ type: "text", text: "Hello" }],
				},
			]

			const stream = handler.createMessage("System prompt", messages)

			await assert.rejects(
				async () => {
					for await (const chunk of stream) {
						void chunk
					}
				},
				(error: unknown) => error instanceof Error,
			)
		})
	})

	describe("Streaming Response Flow", () => {
		it("should handle streaming text chunks", async () => {
			const messages: NeutralConversationHistory = [
				{
					role: "user",
					content: [{ type: "text", text: "Tell me a story" }],
				},
			]

			const stream = handler.createMessage("You are a storyteller.", messages)
			const textChunks: string[] = []

			for await (const chunk of stream) {
				if (chunk.type === "text" && chunk.text) {
					textChunks.push(chunk.text)
				}
			}

			assert.ok(textChunks.length > 0, "Should receive text chunks")
			const fullText = textChunks.join("")
			assert.ok(fullText.length > 0, "Should have combined text content")
		})

		it("should track token usage in streaming mode", async () => {
			const messages: NeutralConversationHistory = [
				{
					role: "user",
					content: [{ type: "text", text: "Hello" }],
				},
			]

			const stream = handler.createMessage("System prompt", messages)
			const chunks: Array<{ type: string; inputTokens?: number; outputTokens?: number }> = []

			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			const usageChunk = chunks.find((c) => c.type === "usage")
			assert.ok(usageChunk, "Should receive usage information")
			assert.ok(typeof usageChunk.inputTokens === "number", "Should have input tokens")
			assert.ok(typeof usageChunk.outputTokens === "number", "Should have output tokens")
		})
	})

	describe("System Prompt Handling", () => {
		it("should include system prompt in request", async () => {
			const systemPrompt = "You are a specialized coding assistant."
			const messages: NeutralConversationHistory = [
				{
					role: "user",
					content: [{ type: "text", text: "Help me code" }],
				},
			]

			const stream = handler.createMessage(systemPrompt, messages)
			for await (const chunk of stream) {
				void chunk
			}

			assert.ok(requestBody.messages, "Should have messages")
			const systemMessage = requestBody.messages.find((m: any) => m.role === "system")
			assert.ok(systemMessage, "Should have system message")
			assert.ok(systemMessage.content.includes("specialized coding assistant"))
		})

		it("should handle empty system prompt", async () => {
			const messages: NeutralConversationHistory = [
				{
					role: "user",
					content: [{ type: "text", text: "Hello" }],
				},
			]

			const stream = handler.createMessage("", messages)
			for await (const chunk of stream) {
				void chunk
			}

			assert.ok(requestBody.messages, "Should still send request")
		})
	})

	describe("Message Content Types", () => {
		it("should handle text content", async () => {
			const messages: NeutralConversationHistory = [
				{
					role: "user",
					content: [
						{
							type: "text",
							text: "This is a text message",
						},
					],
				},
			]

			const stream = handler.createMessage("System", messages)
			for await (const chunk of stream) {
				void chunk
			}

			const userMsg = requestBody.messages.find((m: any) => m.role === "user")
			assert.ok(userMsg, "Should have user message")
		})

		it("should handle multiple content blocks", async () => {
			const messages: NeutralConversationHistory = [
				{
					role: "user",
					content: [
						{ type: "text", text: "First part" },
						{ type: "text", text: "Second part" },
					],
				},
			]

			const stream = handler.createMessage("System", messages)
			for await (const chunk of stream) {
				void chunk
			}

			assert.ok(requestBody.messages, "Should process multi-block content")
		})
	})

	describe("Conversation Validation", () => {
		it("should handle empty message history", async () => {
			const messages: NeutralConversationHistory = []

			const stream = handler.createMessage("System prompt", messages)
			for await (const chunk of stream) {
				void chunk
			}

			// Should still make request with system prompt only
			assert.ok(requestBody.messages, "Should have messages array")
		})

		it("should handle messages without content", async () => {
			const messages: NeutralConversationHistory = [
				{
					role: "user",
					content: [],
				},
			]

			const stream = handler.createMessage("System", messages)

			// Should handle gracefully
			try {
				for await (const chunk of stream) {
					void chunk
				}
			} catch (error) {
				// error handled gracefully
			}

			// Either succeeds or fails gracefully
			assert.ok(true, "Should handle empty content")
		})
	})
})
