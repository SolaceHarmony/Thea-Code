import * as assert from 'assert'
import { expect } from 'chai'
import * as sinon from 'sinon'// import type { Anthropic } from "@anthropic-ai/sdk" // Unused
import type { NeutralConversationHistory } from "../../../shared/neutral-history" // NeutralMessageContent was unused
import type { ApiStreamChunk } from "../../transform/stream"
import OpenAI from "openai"
import { RequestyHandler } from "../requesty"
import { ApiHandlerOptions } from "../../../shared/api" // ModelInfo, requestyDefaultModelInfo were unused
import { convertToR1Format } from "../../transform/r1-format"
import { convertToOpenAiHistory } from "../../transform/neutral-openai-format"
import { API_REFERENCES } from "../../../shared/config/thea-config"
// Mock OpenAI and transform functions
// Mock needs manual implementation
// TODO: Mock setup needs manual migration for "../../transform/r1-format"

suite("RequestyHandler", () => {
	let handler: RequestyHandler
	let mockCreate: sinon.SinonStub

	const defaultOptions: ApiHandlerOptions = {
		requestyApiKey: "test-key",
		requestyModelId: "test-model",
		requestyModelInfo: {
			maxTokens: 8192,
			contextWindow: 200_000,
			supportsImages: true,
			supportsComputerUse: true,
			supportsPromptCache: true,
			inputPrice: 3.0,
			outputPrice: 15.0,
			cacheWritesPrice: 3.75,
			cacheReadsPrice: 0.3,
			description:
				"Claude 3.7 Sonnet is an advanced large language model with improved reasoning, coding, and problem-solving capabilities. It introduces a hybrid reasoning approach, allowing users to choose between rapid responses and extended, step-by-step processing for complex tasks. The model demonstrates notable improvements in coding, particularly in front-end development and full-stack updates, and excels in agentic workflows, where it can autonomously navigate multi-step processes. Claude 3.7 Sonnet maintains performance parity with its predecessor in standard mode while offering an extended reasoning mode for enhanced accuracy in math, coding, and instruction-following tasks. Read more at the [blog post here](https://www.anthropic.com/news/claude-3-7-sonnet)",
		},
		openAiStreamingEnabled: true,
		includeMaxTokens: true, // Add this to match the implementation
	}

	setup(() => {
		// Clear mocks
		sinon.restore()

		// Setup mock create function
		mockCreate = sinon.stub()

		// Mock OpenAI constructor
		;(OpenAI as sinon.SinonStubbedInstanceClass<typeof OpenAI>).callsFake(
			() =>
				({
					chat: {
						completions: {
							create: mockCreate,
						},
					},
				}) as unknown as OpenAI,
		)

		// Mock transform functions
		;(convertToOpenAiHistory as sinon.SinonStub).callsFake((messages: any) => messages) // eslint-disable-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-return
		;(convertToR1Format as sinon.SinonStub).callsFake((messages: any) => messages) // eslint-disable-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-return

		// Create handler instance
		handler = new RequestyHandler(defaultOptions)
	})

	suite("constructor", () => {
		test("should initialize with correct options", () => {
			assert.ok(OpenAI.calledWith({
				baseURL: "https://router.requesty.ai/v1",
				apiKey: defaultOptions.requestyApiKey,
				defaultHeaders: {
					"HTTP-Referer": API_REFERENCES.HOMEPAGE,
					"X-Title": API_REFERENCES.APP_TITLE,
				},
			})
		})
	})

	suite("createMessage", () => {
		const systemPrompt = "You are a helpful assistant"
		const messages: NeutralConversationHistory = [{ role: "user", content: [{ type: "text", text: "Hello" }] }]

		suite("with streaming enabled", () => {
			setup(() => {
				const stream = {
					[Symbol.asyncIterator]: function* () {
						yield {
							choices: [{ delta: { content: "Hello" } }],
						}
						yield {
							choices: [{ delta: { content: " world" } }],
							usage: {
								prompt_tokens: 30,
								completion_tokens: 10,
								prompt_tokens_details: {
									cached_tokens: 15,
									caching_tokens: 5,
								},
							},
						}
					},
				}
				mockCreate.resolves(stream)
			})

			test("should handle streaming response correctly", async () => {
				const stream = handler.createMessage(systemPrompt, messages)
				const results: ApiStreamChunk[] = []

				for await (const chunk of stream) {
					results.push(chunk)
				}

				assert.deepStrictEqual(results, [
					{ type: "text", text: "Hello" },
					{ type: "text", text: " world" },
					{
						type: "usage",
						inputTokens: 30,
						outputTokens: 10,
						cacheWriteTokens: 5,
						cacheReadTokens: 15,
						totalCost: 0.00020325000000000003, // (10 * 3 / 1,000,000) + (5 * 3.75 / 1,000,000) + (15 * 0.3 / 1,000,000) + (10 * 15 / 1,000,000) (the ...0 is a fp skew)
					},
				])

				assert.ok(mockCreate.calledWith({
					model: defaultOptions.requestyModelId,
					temperature: 0,
					messages: [
						{ role: "system", content: systemPrompt },
						{ role: "user", content: "Hello" },
					],
					stream: true,
					stream_options: { include_usage: true },
					max_tokens: defaultOptions.requestyModelInfo?.maxTokens,
				}))
			})

			test("should not include max_tokens when includeMaxTokens is false", async () => {
				handler = new RequestyHandler({
					...defaultOptions,
					includeMaxTokens: false,
				})

				await handler.createMessage(systemPrompt, messages).next()

				assert.ok(mockCreate.calledWith(
					expect.not.objectContaining({
						max_tokens: sinon.match.instanceOf(Number)), // eslint-disable-line @typescript-eslint/no-unsafe-assignment
					})
			})
		})

		suite("with streaming disabled", () => {
			setup(() => {
				handler = new RequestyHandler({
					...defaultOptions,
					openAiStreamingEnabled: false,
				})

				mockCreate.resolves({
					choices: [{ message: { content: "Hello world" } }],
					usage: {
						prompt_tokens: 10,
						completion_tokens: 5,
					},
				})
			})

			test("should handle non-streaming response correctly", async () => {
				const stream = handler.createMessage(systemPrompt, messages)
				const results: ApiStreamChunk[] = []

				for await (const chunk of stream) {
					results.push(chunk)
				}

				assert.deepStrictEqual(results, [
					{ type: "text", text: "Hello world" },
					{
						type: "usage",
						inputTokens: 10,
						outputTokens: 5,
						cacheWriteTokens: 0,
						cacheReadTokens: 0,
						totalCost: 0.000105, // (10 * 3 / 1,000,000) + (5 * 15 / 1,000,000)
					},
				])

				assert.ok(mockCreate.calledWith({
					model: defaultOptions.requestyModelId,
					messages: [
						{ role: "system", content: systemPrompt },
						{ role: "user", content: "Hello" },
					],
				}))
			})
		})
	})

	suite("getModel", () => {
		test("should return correct model information", () => {
			const result = handler.getModel()
			assert.deepStrictEqual(result, {
				id: defaultOptions.requestyModelId,
				info: defaultOptions.requestyModelInfo,
			})
		})

		test("should use sane defaults when no model info provided", () => {
			handler = new RequestyHandler({
				...defaultOptions,
				requestyModelInfo: undefined,
			})

			const result = handler.getModel()
			assert.deepStrictEqual(result, {
				id: defaultOptions.requestyModelId,
				info: defaultOptions.requestyModelInfo,
			})
		})
	})

	suite("completePrompt", () => {
		setup(() => {
			mockCreate.resolves({
				choices: [{ message: { content: "Completed response" } }],
			})
		})

		test("should complete prompt successfully", async () => {
			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "Completed response")

			assert.ok(mockCreate.calledWith({
				model: defaultOptions.requestyModelId,
				messages: [{ role: "user", content: "Test prompt" }],
				max_tokens: sinon.match.instanceOf(Number)), // eslint-disable-line @typescript-eslint/no-unsafe-assignment
				temperature: sinon.match.instanceOf(Number), // eslint-disable-line @typescript-eslint/no-unsafe-assignment
				stream: false, // Expect stream to be false
			})
		})

		test("should handle errors correctly", async () => {
			const errorMessage = "API error"
			mockCreate.rejects(new Error(errorMessage))

			await expect(handler.completePrompt("Test prompt")).rejects.toThrow(
				errorMessage, // OpenAiHandler.completePrompt throws the original error message
			)
		})
	})
// Mock cleanup
