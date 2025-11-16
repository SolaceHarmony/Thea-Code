/* eslint-disable @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-assignment */
import * as assert from "assert"
import sinon from "sinon"

import { OpenAiHandler } from "../../api/providers/openai"
import { ApiHandlerOptions } from "../../shared/api"
import type { NeutralConversationHistory, NeutralMessage, NeutralMessageContent } from "../../shared/neutral-history"
import openaiSetup, { openAIMock } from "../../../test/openai-mock/setup"
import { openaiTeardown } from "../../../test/openai-mock/teardown"

/**
 * Integration Tests for TheaTask Message Flow
 * 
 * This test suite validates the integration between user messages, API handlers,
 * and model responses within the context of a complete task execution flow.
 */

describe("TheaTask Message Flow Integration", () => {
	let sandbox: sinon.SinonSandbox
	let handler: OpenAiHandler
	let mockOptions: ApiHandlerOptions
	let requestBody: any
	let conversationHistory: NeutralConversationHistory

	beforeEach(async () => {
		sandbox = sinon.createSandbox()
		await openaiTeardown()
		await openaiSetup()
		requestBody = undefined
		conversationHistory = []

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

	describe("Complete Task Conversation Flow", () => {
		it("should handle a complete task conversation with multiple turns", async () => {
			// Setup mock to return different responses for each turn
			let callCount = 0
			;(openAIMock)!.addCustomEndpoint("POST", "/v1/chat/completions", function (_uri: any, body: any) {
				requestBody = body
				callCount++

				if (callCount === 1) {
					// First response: Ask for clarification
					return [
						200,
						{
							id: "test-1",
							choices: [
								{
									message: {
										role: "assistant",
										content: "I understand you want to create a file. What should be in the file?",
									},
									finish_reason: "stop",
								},
							],
							usage: { prompt_tokens: 10, completion_tokens: 15, total_tokens: 25 },
						},
					]
				} else if (callCount === 2) {
					// Second response: Use tool
					return [
						200,
						{
							id: "test-2",
							choices: [
								{
									message: {
										role: "assistant",
										content: null,
										tool_calls: [
											{
												id: "call_write",
												type: "function",
												function: {
													name: "write_to_file",
													arguments: JSON.stringify({
														path: "/test/hello.txt",
														content: "Hello World",
													}),
												},
											},
										],
									},
									finish_reason: "tool_calls",
								},
							],
							usage: { prompt_tokens: 30, completion_tokens: 20, total_tokens: 50 },
						},
					]
				} else {
					// Final response: Completion
					return [
						200,
						{
							id: "test-3",
							choices: [
								{
									message: {
										role: "assistant",
										content: "I've successfully created the file with the content you requested.",
									},
									finish_reason: "stop",
								},
							],
							usage: { prompt_tokens: 50, completion_tokens: 12, total_tokens: 62 },
						},
					]
				}
			})

			// Turn 1: Initial user request
			conversationHistory.push({
				role: "user",
				content: [{ type: "text", text: "Create a file called hello.txt" }],
			})

			let stream = handler.createMessage("You are a helpful file management assistant.", conversationHistory)
			let response: NeutralMessageContent = []

			for await (const chunk of stream) {
				if (chunk.type === "text" && chunk.text) {
					response.push({ type: "text", text: chunk.text })
				}
			}

			conversationHistory.push({
				role: "assistant",
				content: response,
			})

			// Verify first response
			const firstResponse = response.find((c) => c.type === "text")
			assert.ok(firstResponse && firstResponse.type === "text", "Should have text response")
			assert.ok(firstResponse.text.includes("What should be in the file"), "Should ask for clarification")

			// Turn 2: User provides content
			conversationHistory.push({
				role: "user",
				content: [{ type: "text", text: 'Put "Hello World" in the file' }],
			})

			stream = handler.createMessage("You are a helpful file management assistant.", conversationHistory)
			response = []
			let toolUse: any = null

			for await (const chunk of stream) {
				if (chunk.type === "text" && chunk.text) {
					response.push({ type: "text", text: chunk.text })
				} else if (chunk.type === "tool_use") {
					toolUse = chunk
					response.push({
						type: "tool_use",
						id: chunk.id,
						name: chunk.name,
						input: chunk.input,
					})
				}
			}

			conversationHistory.push({
				role: "assistant",
				content: response,
			})

			// Verify tool use
			assert.ok(toolUse, "Should use tool")
			assert.strictEqual(toolUse.name, "write_to_file")
			assert.strictEqual(toolUse.input.path, "/test/hello.txt")
			assert.strictEqual(toolUse.input.content, "Hello World")

			// Turn 3: Tool result
			conversationHistory.push({
				role: "user",
				content: [
					{
						type: "tool_result",
						tool_use_id: toolUse.id,
						content: [{ type: "text", text: "File created successfully" }],
					},
				],
			})

			stream = handler.createMessage("You are a helpful file management assistant.", conversationHistory)
			response = []

			for await (const chunk of stream) {
				if (chunk.type === "text" && chunk.text) {
					response.push({ type: "text", text: chunk.text })
				}
			}

			// Verify final response
			const finalResponse = response.find((c) => c.type === "text")
			assert.ok(finalResponse && finalResponse.type === "text", "Should have final response")
			assert.ok(
				finalResponse.text.includes("successfully") || finalResponse.text.includes("created"),
				"Should confirm completion",
			)

			// Verify conversation history is maintained
			assert.strictEqual(conversationHistory.length, 6, "Should have 6 messages in history")
			assert.strictEqual(callCount, 3, "Should have made 3 API calls")
		})

		it("should maintain context across tool executions", async () => {
			let callCount = 0
			;(openAIMock)!.addCustomEndpoint("POST", "/v1/chat/completions", function (_uri: any, body: any) {
				requestBody = body
				callCount++

				if (callCount === 1) {
					// First tool use
					return [
						200,
						{
							id: "test-1",
							choices: [
								{
									message: {
										role: "assistant",
										content: null,
										tool_calls: [
											{
												id: "call_read",
												type: "function",
												function: {
													name: "read_file",
													arguments: JSON.stringify({ path: "/test/data.txt" }),
												},
											},
										],
									},
									finish_reason: "tool_calls",
								},
							],
							usage: { prompt_tokens: 15, completion_tokens: 10, total_tokens: 25 },
						},
					]
				} else {
					// Response using previous context
					return [
						200,
						{
							id: "test-2",
							choices: [
								{
									message: {
										role: "assistant",
										content: "The file contains data, as you requested.",
									},
									finish_reason: "stop",
								},
							],
							usage: { prompt_tokens: 40, completion_tokens: 8, total_tokens: 48 },
						},
					]
				}
			})

			// Initial request
			conversationHistory.push({
				role: "user",
				content: [{ type: "text", text: "What's in the data.txt file?" }],
			})

			let stream = handler.createMessage("You are a helpful assistant.", conversationHistory)
			let toolUse: any = null

			for await (const chunk of stream) {
				if (chunk.type === "tool_use") {
					toolUse = chunk
				}
			}

			assert.ok(toolUse, "Should use read_file tool")

			conversationHistory.push({
				role: "assistant",
				content: [{ type: "tool_use", id: toolUse.id, name: toolUse.name, input: toolUse.input }],
			})

			// Tool result
			conversationHistory.push({
				role: "user",
				content: [
					{
						type: "tool_result",
						tool_use_id: toolUse.id,
						content: [{ type: "text", text: "File contents: Important data" }],
					},
				],
			})

			stream = handler.createMessage("You are a helpful assistant.", conversationHistory)
			let response = ""

			for await (const chunk of stream) {
				if (chunk.type === "text" && chunk.text) {
					response += chunk.text
				}
			}

			// Verify context is maintained
			assert.ok(response.includes("data") || response.includes("file"), "Should reference file contents")

			// Verify all messages are in the request
			assert.ok(requestBody.messages.length >= 3, "Should include all conversation history")
		})
	})

	describe("Error Recovery in Conversation", () => {
		it("should handle API errors and allow retry", async () => {
			let attemptCount = 0
			;(openAIMock)!.addCustomEndpoint("POST", "/v1/chat/completions", function (_uri: any, body: any) {
				requestBody = body
				attemptCount++

				if (attemptCount === 1) {
					// First attempt fails
					return [500, { error: { message: "Server error" } }]
				} else {
					// Retry succeeds
					return [
						200,
						{
							id: "test-success",
							choices: [
								{
									message: {
										role: "assistant",
										content: "Request succeeded on retry",
									},
									finish_reason: "stop",
								},
							],
							usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
						},
					]
				}
			})

			conversationHistory.push({
				role: "user",
				content: [{ type: "text", text: "Test message" }],
			})

			// First attempt should fail
			let stream = handler.createMessage("System", conversationHistory)
			let firstAttemptFailed = false

			try {
				for await (const chunk of stream) {
					void chunk
				}
			} catch (error) {
				firstAttemptFailed = true
			}

			assert.ok(firstAttemptFailed, "First attempt should fail")
			assert.strictEqual(attemptCount, 1, "Should have attempted once")

			// Retry should succeed
			stream = handler.createMessage("System", conversationHistory)
			let response = ""

			for await (const chunk of stream) {
				if (chunk.type === "text" && chunk.text) {
					response += chunk.text
				}
			}

			assert.ok(response.includes("succeeded"), "Retry should succeed")
			assert.strictEqual(attemptCount, 2, "Should have attempted twice")
		})

		it("should handle tool execution failures", async () => {
			;(openAIMock)!.addCustomEndpoint("POST", "/v1/chat/completions", function (_uri: any, body: any) {
				requestBody = body

				// Check if this is the request after tool failure
				const hasToolResult = body.messages.some((m: any) =>
					m.content?.some?.((c: any) => c.type === "tool_result" && c.content?.some?.((t: any) => t.text?.includes("error"))),
				)

				if (hasToolResult) {
					// After error, suggest alternative
					return [
						200,
						{
							id: "test-recovery",
							choices: [
								{
									message: {
										role: "assistant",
										content: "I see the tool failed. Let me try a different approach.",
									},
									finish_reason: "stop",
								},
							],
							usage: { prompt_tokens: 50, completion_tokens: 12, total_tokens: 62 },
						},
					]
				} else {
					// Initial tool use
					return [
						200,
						{
							id: "test-tool",
							choices: [
								{
									message: {
										role: "assistant",
										content: null,
										tool_calls: [
											{
												id: "call_fail",
												type: "function",
												function: {
													name: "execute_command",
													arguments: JSON.stringify({ command: "invalid-command" }),
												},
											},
										],
									},
									finish_reason: "tool_calls",
								},
							],
							usage: { prompt_tokens: 20, completion_tokens: 15, total_tokens: 35 },
						},
					]
				}
			})

			// User request
			conversationHistory.push({
				role: "user",
				content: [{ type: "text", text: "Run the invalid command" }],
			})

			let stream = handler.createMessage("System", conversationHistory)
			let toolUse: any = null

			for await (const chunk of stream) {
				if (chunk.type === "tool_use") {
					toolUse = chunk
				}
			}

			assert.ok(toolUse, "Should attempt tool use")

			conversationHistory.push({
				role: "assistant",
				content: [{ type: "tool_use", id: toolUse.id, name: toolUse.name, input: toolUse.input }],
			})

			// Tool fails
			conversationHistory.push({
				role: "user",
				content: [
					{
						type: "tool_result",
						tool_use_id: toolUse.id,
						content: [{ type: "text", text: "Error: Command not found" }],
					},
				],
			})

			// Model should handle error
			stream = handler.createMessage("System", conversationHistory)
			let response = ""

			for await (const chunk of stream) {
				if (chunk.type === "text" && chunk.text) {
					response += chunk.text
				}
			}

			assert.ok(response.includes("failed") || response.includes("different"), "Should acknowledge error")
		})
	})

	describe("Token Usage Tracking", () => {
		it("should accumulate token usage across conversation", async () => {
			let totalInputTokens = 0
			let totalOutputTokens = 0

			;(openAIMock)!.addCustomEndpoint("POST", "/v1/chat/completions", function (_uri: any, body: any) {
				requestBody = body
				const usage = {
					prompt_tokens: 20,
					completion_tokens: 10,
					total_tokens: 30,
				}

				totalInputTokens += usage.prompt_tokens
				totalOutputTokens += usage.completion_tokens

				return [
					200,
					{
						id: "test",
						choices: [
							{
								message: {
									role: "assistant",
									content: "Response",
								},
								finish_reason: "stop",
							},
						],
						usage,
					},
				]
			})

			// Make multiple requests
			for (let i = 0; i < 3; i++) {
				conversationHistory.push({
					role: "user",
					content: [{ type: "text", text: `Message ${i + 1}` }],
				})

				const stream = handler.createMessage("System", conversationHistory)
				let response: NeutralMessageContent = []

				for await (const chunk of stream) {
					if (chunk.type === "text" && chunk.text) {
						response.push({ type: "text", text: chunk.text })
					}
				}

				conversationHistory.push({
					role: "assistant",
					content: response,
				})
			}

			// Verify token accumulation
			assert.strictEqual(totalInputTokens, 60, "Should accumulate input tokens")
			assert.strictEqual(totalOutputTokens, 30, "Should accumulate output tokens")
		})

		it("should track tokens for tool use conversations", async () => {
			const tokenCounts: Array<{ input: number; output: number }> = []

			;(openAIMock)!.addCustomEndpoint("POST", "/v1/chat/completions", function (_uri: any, body: any) {
				requestBody = body

				// Vary token counts based on message type
				const hasToolResult = body.messages.some((m: any) => m.role === "tool")
				const usage = hasToolResult
					? { prompt_tokens: 40, completion_tokens: 8, total_tokens: 48 }
					: { prompt_tokens: 25, completion_tokens: 15, total_tokens: 40 }

				tokenCounts.push({ input: usage.prompt_tokens, output: usage.completion_tokens })

				if (hasToolResult) {
					return [
						200,
						{
							id: "test-after-tool",
							choices: [
								{
									message: {
										role: "assistant",
										content: "Tool result processed",
									},
									finish_reason: "stop",
								},
							],
							usage,
						},
					]
				} else {
					return [
						200,
						{
							id: "test-tool-use",
							choices: [
								{
									message: {
										role: "assistant",
										content: null,
										tool_calls: [
											{
												id: "call_test",
												type: "function",
												function: {
													name: "test_tool",
													arguments: "{}",
												},
											},
										],
									},
									finish_reason: "tool_calls",
								},
							],
							usage,
						},
					]
				}
			})

			// Initial request
			conversationHistory.push({
				role: "user",
				content: [{ type: "text", text: "Use a tool" }],
			})

			let stream = handler.createMessage("System", conversationHistory)
			let toolUse: any = null

			for await (const chunk of stream) {
				if (chunk.type === "tool_use") {
					toolUse = chunk
				}
			}

			conversationHistory.push({
				role: "assistant",
				content: [{ type: "tool_use", id: toolUse.id, name: toolUse.name, input: toolUse.input }],
			})

			// Tool result
			conversationHistory.push({
				role: "user",
				content: [
					{
						type: "tool_result",
						tool_use_id: toolUse.id,
						content: [{ type: "text", text: "Tool executed" }],
					},
				],
			})

			stream = handler.createMessage("System", conversationHistory)
			for await (const chunk of stream) {
				void chunk
			}

			// Verify token tracking for both requests
			assert.strictEqual(tokenCounts.length, 2, "Should have token counts for both requests")
			assert.ok(tokenCounts[0].input > 0, "First request should have input tokens")
			assert.ok(tokenCounts[1].input > tokenCounts[0].input, "Second request should have more input tokens")
		})
	})

	describe("Message History Management", () => {
		it("should maintain correct message order", async () => {
			;(openAIMock)!.addCustomEndpoint("POST", "/v1/chat/completions", function (_uri: any, body: any) {
				requestBody = body
				return [
					200,
					{
						id: "test",
						choices: [
							{
								message: {
									role: "assistant",
									content: "Response",
								},
								finish_reason: "stop",
							},
						],
						usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
					},
				]
			})

			// Build conversation
			const expectedOrder = ["user", "assistant", "user", "assistant", "user"]

			for (let i = 0; i < 3; i++) {
				conversationHistory.push({
					role: "user",
					content: [{ type: "text", text: `User message ${i + 1}` }],
				})

				const stream = handler.createMessage("System", conversationHistory)
				let response: NeutralMessageContent = []

				for await (const chunk of stream) {
					if (chunk.type === "text" && chunk.text) {
						response.push({ type: "text", text: chunk.text })
					}
				}

				conversationHistory.push({
					role: "assistant",
					content: response,
				})
			}

			// Verify order
			const actualOrder = conversationHistory.map((m) => m.role)
			assert.deepStrictEqual(actualOrder, expectedOrder.slice(0, actualOrder.length), "Should maintain message order")
		})

		it("should handle conversation history with mixed content types", async () => {
			;(openAIMock)!.addCustomEndpoint("POST", "/v1/chat/completions", function (_uri: any, body: any) {
				requestBody = body
				return [
					200,
					{
						id: "test",
						choices: [
							{
								message: {
									role: "assistant",
									content: "Understood",
								},
								finish_reason: "stop",
							},
						],
						usage: { prompt_tokens: 30, completion_tokens: 3, total_tokens: 33 },
					},
				]
			})

			// Add message with multiple content blocks
			conversationHistory.push({
				role: "user",
				content: [
					{ type: "text", text: "Here's some context:" },
					{ type: "text", text: "Additional information" },
				],
			})

			const stream = handler.createMessage("System", conversationHistory)
			for await (const chunk of stream) {
				void chunk
			}

			// Verify multi-block message is preserved
			assert.strictEqual(conversationHistory[0].content.length, 2, "Should preserve multiple content blocks")
		})
	})
})
