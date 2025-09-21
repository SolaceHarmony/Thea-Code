import * as assert from 'assert'
import { expect } from 'chai'
import * as sinon from 'sinon'/* eslint-disable @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-assignment */
import { ApiHandlerOptions } from "../../../shared/api"
import { OpenAiHandler } from "../openai"
import type { NeutralConversationHistory } from "../../../shared/neutral-history"
import { Readable } from "stream"
import { API_REFERENCES } from "../../../shared/config/thea-config"
import openaiSetup, { openAIMock } from "../../../../test/openai-mock/setup"
import { openaiTeardown } from "../../../../test/openai-mock/teardown"

let requestBody: any
let capturedHeaders: Record<string, string | string[]> = {}

setup(async () => {
	await openaiTeardown()
	await openaiSetup()
	requestBody = undefined
	capturedHeaders = {};
	(openAIMock as any)!.addCustomEndpoint("POST", "/v1/chat/completions", function (_uri: any, body: any) {
		// `this` is the nock request
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
			choices: [{ delta: { content: "Test response" }, index: 0, finish_reason: "stop", logprobs: null }],
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
// Mock cleanup
teardown(async () => {
	await openaiTeardown()
// Mock cleanup
suite("OpenAiHandler", () => {
	let handler: OpenAiHandler
	let mockOptions: ApiHandlerOptions

	setup(() => {
		mockOptions = {
			openAiApiKey: "test-api-key",
			openAiModelId: "gpt-4",
			openAiBaseUrl: "https://api.openai.com/v1",
		}
		handler = new OpenAiHandler(mockOptions)
	})

	suite("constructor", () => {
		test("should initialize with provided options", () => {
			assert.ok(handler instanceof OpenAiHandler)
			expect(handler.getModel().id).to.equal(mockOptions.openAiModelId)
		})

		test("should use custom base URL if provided", () => {
			const customBaseUrl = "https://custom.openai.com/v1"
			const handlerWithCustomUrl = new OpenAiHandler({
				...mockOptions,
				openAiBaseUrl: customBaseUrl,
			})
			assert.ok(handlerWithCustomUrl instanceof OpenAiHandler)
		})

		test("should set default headers correctly", async () => {
			await handler.completePrompt("Hi")
			assert.strictEqual(capturedHeaders["http-referer"], API_REFERENCES.HOMEPAGE)
			assert.strictEqual(capturedHeaders["x-title"], API_REFERENCES.APP_TITLE)
		})
	})

	suite("createMessage", () => {
		const systemPrompt = "You are a helpful assistant."
		const messages: NeutralConversationHistory = [
			{
				role: "user",
				content: [
					{
						type: "text",
						text: "Hello!",
					},
				],
			},
		]

		test("should handle non-streaming mode", async () => {
			const handler = new OpenAiHandler({
				...mockOptions,
				openAiStreamingEnabled: false,
			})

			const stream = handler.createMessage(systemPrompt, messages)
			const chunks: Array<{ type: string; text?: string; inputTokens?: number; outputTokens?: number }> = []
			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			assert.ok(chunks.length > 0)
			const textChunk = chunks.find((chunk) => chunk.type === "text")
			const usageChunk = chunks.find((chunk) => chunk.type === "usage")

			assert.notStrictEqual(textChunk, undefined)
			assert.strictEqual(textChunk?.text, "Test response")
			assert.notStrictEqual(usageChunk, undefined)
			assert.strictEqual(usageChunk?.inputTokens, 10)
			assert.strictEqual(usageChunk?.outputTokens, 5)
		})

		test("should handle streaming responses", async () => {
			const stream = handler.createMessage(systemPrompt, messages)
			const chunks: Array<{ type: string; text?: string; inputTokens?: number; outputTokens?: number }> = []
			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			assert.ok(chunks.length > 0)
			const textChunks = chunks.filter((chunk) => chunk.type === "text")
			assert.strictEqual(textChunks.length, 1)
			assert.strictEqual(textChunks[0].text, "Test response")
		})
	})

	suite("error handling", () => {
		const testMessages: NeutralConversationHistory = [
			{
				role: "user",
				content: [
					{
						type: "text",
						text: "Hello",
					},
				],
			},
		]

	test("should handle API errors", async () => {
		await openaiTeardown()
		await openaiSetup();
		(openAIMock as any)!.addCustomEndpoint("POST", "/v1/chat/completions", () => [
			500,
			{ error: { message: "API Error" } },
		])

		const stream = handler.createMessage("system prompt", testMessages)

			await expect(async () => {
				for await (const chunk of stream) {
					void chunk
				}
			}).rejects.toThrow("API Error")
		})

	test("should handle rate limiting", async () => {
		await openaiTeardown()
		await openaiSetup();
		(openAIMock as any)!.addCustomEndpoint("POST", "/v1/chat/completions", () => [
			429,
			{ error: { message: "Rate limit exceeded" } },
		])

		const stream = handler.createMessage("system prompt", testMessages)

			await expect(async () => {
				for await (const chunk of stream) {
					void chunk
				}
			}).rejects.toThrow("Rate limit exceeded")
		})
	})

	suite("completePrompt", () => {
		test("should complete prompt successfully", async () => {
			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "Test response")
			assert.deepStrictEqual(requestBody, {
					model: mockOptions.openAiModelId,
					messages: [{ role: "user", content: "Test prompt" }],
					max_tokens: sinon.match.instanceOf(Number),
					temperature: sinon.match.instanceOf(Number),
					stream: false,
				})
		})

	test("should handle API errors", async () => {
		await openaiTeardown()
		await openaiSetup();
		(openAIMock as any)!.addCustomEndpoint("POST", "/v1/chat/completions", () => [
			500,
			{ error: { message: "API Error" } },
		])

		await expect(handler.completePrompt("Test prompt")).rejects.toThrow("OpenAI completion error: API Error")
	})

	test("should handle empty response", async () => {
		await openaiTeardown()
		await openaiSetup();
		(openAIMock as any)!.addCustomEndpoint("POST", "/v1/chat/completions", () => [
			200,
			{ choices: [{ message: { content: "" } }] },
		])
		const result = await handler.completePrompt("Test prompt")
		assert.strictEqual(result, "")
	})
	})

	suite("getModel", () => {
		test("should return model info with sane defaults", () => {
			const model = handler.getModel()
			assert.strictEqual(model.id, mockOptions.openAiModelId)
			assert.notStrictEqual(model.info, undefined)
			assert.strictEqual(model.info.contextWindow, 128_000)
			assert.strictEqual(model.info.supportsImages, true)
		})

		test("should handle undefined model ID", () => {
			const handlerWithoutModel = new OpenAiHandler({
				...mockOptions,
				openAiModelId: undefined,
			})
			const model = handlerWithoutModel.getModel()
			assert.strictEqual(model.id, "")
			assert.notStrictEqual(model.info, undefined)
		})
	})

	suite("Tool Use Detection", () => {
		test("should extract tool calls from delta", () => {
			const delta = {
				tool_calls: [
					{
						index: 0,
						id: "call_123",
						function: {
							name: "test_tool",
							arguments: '{"param":"value"}',
						},
					},
				],
			}

			const toolCalls = handler.extractToolCalls(delta)

			assert.deepStrictEqual(toolCalls, delta.tool_calls)
		})

		test("should return empty array if no tool calls", () => {
			const delta = {
				content: "Hello",
			}

			const toolCalls = handler.extractToolCalls(delta)

			assert.deepStrictEqual(toolCalls, [])
		})

		test("should detect if delta has tool calls", () => {
			const delta = {
				tool_calls: [
					{
						index: 0,
						id: "call_123",
						function: {
							name: "test_tool",
							arguments: '{"param":"value"}',
						},
					},
				],
			}

			const hasToolCalls = handler.hasToolCalls(delta)

			assert.strictEqual(hasToolCalls, true)
		})

		test("should detect if delta has no tool calls", () => {
			const delta = {
				content: "Hello",
			}

			const hasToolCalls = handler.hasToolCalls(delta)

			assert.strictEqual(hasToolCalls, false)
		})
	})
// Mock cleanup
