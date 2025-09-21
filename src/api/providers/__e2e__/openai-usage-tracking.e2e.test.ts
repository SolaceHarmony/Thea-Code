import * as assert from 'assert'
import * as sinon from 'sinon'/* eslint-disable @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-assignment */
import { ApiHandlerOptions } from "../../../shared/api"
import { OpenAiHandler } from "../openai"
import { Readable } from "stream"
import { NeutralMessage } from "../../../shared/neutral-history"
import openaiSetup, { openAIMock } from "../../../../test/openai-mock/setup"
import { openaiTeardown } from "../../../../test/openai-mock/teardown"

let requestBody: any

setup(async () => {
	await openaiTeardown()
	await openaiSetup()
	requestBody = undefined
	;(openAIMock as any)!.addCustomEndpoint("POST", "/v1/chat/completions", function (_uri: any, body: any) {
		requestBody = body
		if (!body.stream) {
			return [
				200,
				{
					id: "test-completion",
					choices: [
						{ message: { role: "assistant", content: "Test response" }, finish_reason: "stop", index: 0 },
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
			choices: [{ delta: { content: "Test " }, index: 0, finish_reason: null }],
			usage: { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 },
		}
		const chunk2 = {
			id: "chatcmpl-test-2",
			created: 1678886401,
			model: "gpt-4",
			object: "chat.completion.chunk",
			choices: [{ delta: { content: "response" }, index: 0, finish_reason: null }],
			usage: { prompt_tokens: 10, completion_tokens: 4, total_tokens: 14 },
		}
		const chunk3 = {
			id: "chatcmpl-test-3",
			created: 1678886402,
			model: "gpt-4",
			object: "chat.completion.chunk",
			choices: [{ delta: {}, index: 0, finish_reason: "stop" }],
			usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
		}
		stream.push(`data: ${JSON.stringify(chunk1)}\n\n`)
		stream.push(`data: ${JSON.stringify(chunk2)}\n\n`)
		stream.push(`data: ${JSON.stringify(chunk3)}\n\n`)
		stream.push("data: [DONE]\n\n")
		stream.push(null)
		return [200, stream]
	})
// Mock cleanup
teardown(async () => {
	await openaiTeardown()
// Mock cleanup
suite("OpenAiHandler with usage tracking fix", () => {
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

	suite("usage metrics with streaming", () => {
		const systemPrompt = "You are a helpful assistant."
		const messages: NeutralMessage[] = [
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

		test("should only yield usage metrics once at the end of the stream", async () => {
			const stream = handler.createMessage(systemPrompt, messages)
			const chunks: Array<{ type: string; text?: string; inputTokens?: number; outputTokens?: number }> = []
			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			// Check we have text chunks
			const textChunks = chunks.filter((chunk) => chunk.type === "text")
			assert.strictEqual(textChunks.length, 2)
			assert.strictEqual(textChunks[0].text, "Test ")
			assert.strictEqual(textChunks[1].text, "response")

			// Check we only have one usage chunk and it's the last one
			const usageChunks = chunks.filter((chunk) => chunk.type === "usage")
			assert.strictEqual(usageChunks.length, 1)
			assert.deepStrictEqual(usageChunks[0], {
				type: "usage",
				inputTokens: 10,
				outputTokens: 5,
			})

			// Check the usage chunk is the last one reported from the API
			const lastChunk = chunks[chunks.length - 1]
			assert.strictEqual(lastChunk.type, "usage")
			assert.strictEqual(lastChunk.inputTokens, 10)
			assert.strictEqual(lastChunk.outputTokens, 5)

			assert.deepStrictEqual(requestBody, {
					model: mockOptions.openAiModelId,
					messages: [
						{ role: "system", content: systemPrompt },
						{ role: "user", content: "Hello!" },
					],
					stream: true,
				})
		})

		test("should handle case where usage is only in the final chunk", async () => {
			await openaiTeardown()
			await openaiSetup()
			;(openAIMock as any)!.addCustomEndpoint("POST", "/v1/chat/completions", () => {
				const stream = new Readable({ read() {} })
				const chunk1 = {
					id: "chatcmpl-test-4",
					created: 1678886403,
					model: "gpt-4",
					object: "chat.completion.chunk",
					choices: [{ delta: { content: "Test " }, index: 0, finish_reason: null }],
					usage: null,
				}
				const chunk2 = {
					id: "chatcmpl-test-5",
					created: 1678886404,
					model: "gpt-4",
					object: "chat.completion.chunk",
					choices: [{ delta: { content: "response" }, index: 0, finish_reason: null }],
					usage: null,
				}
				const chunk3 = {
					id: "chatcmpl-test-6",
					created: 1678886405,
					model: "gpt-4",
					object: "chat.completion.chunk",
					choices: [{ delta: {}, index: 0, finish_reason: "stop" }],
					usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
				}
				stream.push(`data: ${JSON.stringify(chunk1)}\n\n`)
				stream.push(`data: ${JSON.stringify(chunk2)}\n\n`)
				stream.push(`data: ${JSON.stringify(chunk3)}\n\n`)
				stream.push("data: [DONE]\n\n")
				stream.push(null)
				return [200, stream]
			})

			const stream = handler.createMessage(systemPrompt, messages)
			const chunks: Array<{ type: string; text?: string; inputTokens?: number; outputTokens?: number }> = []
			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			// Check usage metrics
			const usageChunks = chunks.filter((chunk) => chunk.type === "usage")
			assert.strictEqual(usageChunks.length, 1)
			assert.deepStrictEqual(usageChunks[0], {
				type: "usage",
				inputTokens: 10,
				outputTokens: 5,
			})
		})

		test("should handle case where no usage is provided", async () => {
			await openaiTeardown()
			await openaiSetup()
			;(openAIMock as any)!.addCustomEndpoint("POST", "/v1/chat/completions", () => {
				const stream = new Readable({ read() {} })
				const chunk1 = {
					id: "chatcmpl-test-7",
					created: 1678886406,
					model: "gpt-4",
					object: "chat.completion.chunk",
					choices: [{ delta: { content: "Test response" }, index: 0, finish_reason: null }],
					usage: null,
				}
				const chunk2 = {
					id: "chatcmpl-test-8",
					created: 1678886407,
					model: "gpt-4",
					object: "chat.completion.chunk",
					choices: [{ delta: {}, index: 0, finish_reason: "stop" }],
					usage: null,
				}
				stream.push(`data: ${JSON.stringify(chunk1)}\n\n`)
				stream.push(`data: ${JSON.stringify(chunk2)}\n\n`)
				stream.push("data: [DONE]\n\n")
				stream.push(null)
				return [200, stream]
			})

			const stream = handler.createMessage(systemPrompt, messages)
			const chunks: Array<{ type: string; text?: string; inputTokens?: number; outputTokens?: number }> = []
			for await (const chunk of stream) {
				chunks.push(chunk)
			}

			// Check we don't have any usage chunks
			const usageChunks = chunks.filter((chunk) => chunk.type === "usage")
			assert.strictEqual(usageChunks.length, 0)
		})
	})
// Mock cleanup
