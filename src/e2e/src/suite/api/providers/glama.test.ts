// npx jest src/api/providers/__tests__/glama.test.ts

// import { Anthropic } from "@anthropic-ai/sdk" // Unused import
import axios from "axios"

import { GlamaHandler } from "../glama"
import { ApiHandlerOptions } from "../../../shared/api"
import type { NeutralConversationHistory } from "../../../shared/neutral-history" // Fixed import path
import type OpenAI from "openai"
import type { ApiStreamChunk } from "../../transform/stream"
// Mock OpenAI client
import * as assert from 'assert'
import * as sinon from 'sinon'
const mockCreate = sinon.stub()
const mockWithResponse = sinon.stub()

// TODO: Mock setup needs manual migration
// TODO: Fix mock - needs proxyquire
/*
=> {
	return {
		__esModule: true,
		default: sinon.stub().callsFake(() => ({
			chat: {
				completions: {
					create: (options: OpenAI.Chat.Completions.ChatCompletionCreateParams) => {
						const stream = {
							[Symbol.asyncIterator]: function* () {
								yield {
									choices: [
										{
											delta: { content: "Test response" },
											index: 0,
										},
									],
									usage: null,

								yield {
									choices: [
										{
											delta: {},
											index: 0,
										},
									],
									usage: {
										prompt_tokens: 10,
										completion_tokens: 5,
										total_tokens: 15,
									},

							},

						// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
						const result = mockCreate(options) // Changed from ...args
						if (options.stream) {
							mockWithResponse.returns(
								Promise.resolve({
									data: stream,
									response: {
										headers: {
											get: (name: string) =>
												name === "x-completion-request-id" ? "test-request-id" : null,
										},
									},
								}),

							// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
							result.withResponse = mockWithResponse

						// eslint-disable-next-line @typescript-eslint/no-unsafe-return
						return result
					},
				},
			},
		})),

})*/

suite("GlamaHandler", () => {
	let handler: GlamaHandler
	let mockOptions: ApiHandlerOptions

	setup(() => {
		mockOptions = {
			apiModelId: "anthropic/claude-3-7-sonnet",
			glamaModelId: "anthropic/claude-3-7-sonnet",
			glamaApiKey: "test-api-key",

		handler = new GlamaHandler(mockOptions)
		mockCreate.mockClear()
		mockWithResponse.mockClear()

		// Default mock implementation for non-streaming responses
		mockCreate.resolves({
			id: "test-completion",
			choices: [
				{
					message: { role: "assistant", content: "Test response" },
					finish_reason: "stop",
					index: 0,
				},
			],
			usage: {
				prompt_tokens: 10,
				completion_tokens: 5,
				total_tokens: 15,
			},

	suite("constructor", () => {
		test("should initialize with provided options", () => {
			assert.ok(handler instanceof GlamaHandler)
			expect(handler.getModel().id).toBe(mockOptions.apiModelId)

	suite("createMessage", () => {
		const systemPrompt = "You are a helpful assistant."
		const messages: NeutralConversationHistory = [
			// Explicitly type as NeutralConversationHistory
			{
				role: "user",
				content: "Hello!",
			},

		test("should handle streaming responses", async () => {
			// Mock axios for token usage request
			const mockAxios = sinon.spy(axios, "get").resolvesOnce({
				data: {
					tokenUsage: {
						promptTokens: 10,
						completionTokens: 5,
						cacheCreationInputTokens: 0,
						cacheReadInputTokens: 0,
					},
					totalCostUsd: "0.00",
				},

			const stream = handler.createMessage(systemPrompt, messages)
			const chunks: ApiStreamChunk[] = []
			for await (const chunk of stream) {
				chunks.push(chunk)

			assert.strictEqual(chunks.length, 2) // Text chunk and usage chunk
			assert.deepStrictEqual(chunks[0], {
				type: "text",
				text: "Test response",
			assert.deepStrictEqual(chunks[1], {
				type: "usage",
				inputTokens: 10,
				outputTokens: 5,
				cacheWriteTokens: 0,
				cacheReadTokens: 0,
				totalCost: 0,

			mockAxios.restore()

		test("should handle API errors", async () => {
			mockCreate.onFirstCall().callsFake(() => {
				throw new Error("API Error")

			const stream = handler.createMessage(systemPrompt, messages)
			const chunks = []

			try {
				for await (const chunk of stream) {
					chunks.push(chunk)

				fail("Expected error to be thrown")
			} catch (e) {
				const error = e as Error // Type assertion
				assert.ok(error instanceof Error)
				assert.strictEqual(error.message, "API Error") // Now safe

	suite("completePrompt", () => {
		test("should complete prompt successfully", async () => {
			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "Test response")
			assert.ok(mockCreate.calledWith(
				sinon.match({
					model: mockOptions.apiModelId,
					messages: [{ role: "user", content: "Test prompt" }],
					temperature: 0,
					max_tokens: 8192,
				})),

		test("should handle API errors", async () => {
			mockCreate.rejectsOnce(new Error("API Error"))
			await expect(handler.completePrompt("Test prompt")).rejects.toThrow("Glama completion error: API Error")

		test("should handle empty response", async () => {
			mockCreate.resolvesOnce({
				choices: [{ message: { content: "" } }],
			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "")

		test("should not set max_tokens for non-Anthropic models", async () => {
			// Reset mock to clear any previous calls
			mockCreate.mockClear()

			const nonAnthropicOptions = {
				apiModelId: "openai/gpt-4",
				glamaModelId: "openai/gpt-4",
				glamaApiKey: "test-key",
				glamaModelInfo: {
					maxTokens: 4096,
					contextWindow: 8192,
					supportsImages: true,
					supportsPromptCache: false,
				},

			const nonAnthropicHandler = new GlamaHandler(nonAnthropicOptions)

			await nonAnthropicHandler.completePrompt("Test prompt")
			assert.ok(mockCreate.calledWith(
				sinon.match({
					model: "openai/gpt-4",
					messages: [{ role: "user", content: "Test prompt" }],
					temperature: 0,
				})),

			// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
			expect(mockCreate.args[0][0]).not.toHaveProperty("max_tokens")

	suite("getModel", () => {
		test("should return model info", () => {
			const modelInfo = handler.getModel()
			assert.strictEqual(modelInfo.id, mockOptions.apiModelId)
			assert.ok(modelInfo.info !== undefined)
			assert.strictEqual(modelInfo.info.maxTokens, 8192)
			assert.strictEqual(modelInfo.info.contextWindow, 200_000)
