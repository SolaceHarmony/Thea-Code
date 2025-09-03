import { DeepSeekHandler } from "../deepseek"
import { ApiHandlerOptions, deepSeekDefaultModelId } from "../../../shared/api"
import OpenAI from "openai"
import { NeutralConversationHistory } from "../../../shared/neutral-history"
import type { ApiStreamChunk } from "../../transform/stream"
// Mock OpenAI client
import * as assert from 'assert'
import * as sinon from 'sinon'
const mockCreate = sinon.stub()
// TODO: Mock setup needs manual migration
// TODO: Fix mock - needs proxyquire
/*
=> {
	return {
		__esModule: true,
		default: sinon.stub().callsFake(() => ({
			chat: {
				completions: {
					create: mockCreate.callsFake(
						(options: OpenAI.Chat.Completions.ChatCompletionCreateParams) => {
							if (!options.stream) {
								return {
									id: "test-completion",
									choices: [
										{
											message: { role: "assistant", content: "Test response", refusal: null },
											finish_reason: "stop",
											index: 0,
										},
									],
									usage: {
										prompt_tokens: 10,
										completion_tokens: 5,
										total_tokens: 15,
										prompt_tokens_details: {
											cache_miss_tokens: 8,
											cached_tokens: 2,
										},
									},

							// Return async iterator for streaming
							return {
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
											prompt_tokens_details: {
												cache_miss_tokens: 8,
												cached_tokens: 2,
											},
										},

								},

						},
					),
				},
			},
		})),

})*/

suite("DeepSeekHandler", () => {
	let handler: DeepSeekHandler
	let mockOptions: ApiHandlerOptions

	setup(() => {
		mockOptions = {
			deepSeekApiKey: "test-api-key",
			apiModelId: "deepseek-chat",
			deepSeekBaseUrl: "https://api.deepseek.com",

		handler = new DeepSeekHandler(mockOptions)
		mockCreate.mockClear()

	suite("constructor", () => {
		test("should initialize with provided options", () => {
			assert.ok(handler instanceof DeepSeekHandler)
			expect(handler.getModel().id).toBe(mockOptions.apiModelId)

		test.skip("should throw error if API key is missing", () => {
			expect(() => {
				new DeepSeekHandler({
					...mockOptions,
					deepSeekApiKey: undefined,
			}).toThrow("DeepSeek API key is required")

		test("should use default model ID if not provided", () => {
			const handlerWithoutModel = new DeepSeekHandler({
				...mockOptions,
				apiModelId: undefined,
			expect(handlerWithoutModel.getModel().id).toBe(deepSeekDefaultModelId)

		test("should use default base URL if not provided", () => {
			const handlerWithoutBaseUrl = new DeepSeekHandler({
				...mockOptions,
				deepSeekBaseUrl: undefined,
			assert.ok(handlerWithoutBaseUrl instanceof DeepSeekHandler)
			// The base URL is passed to OpenAI client internally
			assert.ok(OpenAI.calledWith(
				sinon.match({
					baseURL: "https://api.deepseek.com",
				})),

		test("should use custom base URL if provided", () => {
			const customBaseUrl = "https://custom.deepseek.com/v1"
			const handlerWithCustomUrl = new DeepSeekHandler({
				...mockOptions,
				deepSeekBaseUrl: customBaseUrl,
			assert.ok(handlerWithCustomUrl instanceof DeepSeekHandler)
			// The custom base URL is passed to OpenAI client
			assert.ok(OpenAI.calledWith(
				sinon.match({
					baseURL: customBaseUrl,
				})),

		test("should set includeMaxTokens to true", () => {
			// Create a new handler and verify OpenAI client was called with includeMaxTokens
			new DeepSeekHandler(mockOptions)
			assert.ok(OpenAI.calledWith(
				sinon.match({
					apiKey: mockOptions.deepSeekApiKey,
				})),

	suite("getModel", () => {
		test("should return model info for valid model ID", () => {
			const model = handler.getModel()
			assert.strictEqual(model.id, mockOptions.apiModelId)
			assert.ok(model.info !== undefined)
			assert.strictEqual(model.info.maxTokens, 8192)
			assert.strictEqual(model.info.contextWindow, 64_000)
			assert.strictEqual(model.info.supportsImages, false)
			assert.strictEqual(model.info.supportsPromptCache, true) // Should be true now

		test("should return provided model ID with default model info if model does not exist", () => {
			const handlerWithInvalidModel = new DeepSeekHandler({
				...mockOptions,
				apiModelId: "invalid-model",
			const model = handlerWithInvalidModel.getModel()
			assert.strictEqual(model.id, "invalid-model") // Returns provided ID
			assert.ok(model.info !== undefined)
			// With the current implementation, it's the same object reference when using default model info
			assert.strictEqual(model.info, handler.getModel().info)
			// Should have the same base properties
			assert.strictEqual(model.info.contextWindow, handler.getModel().info.contextWindow)
			// And should have supportsPromptCache set to true
			assert.strictEqual(model.info.supportsPromptCache, true)

		test("should return default model if no model ID is provided", () => {
			const handlerWithoutModel = new DeepSeekHandler({
				...mockOptions,
				apiModelId: undefined,
			const model = handlerWithoutModel.getModel()
			assert.strictEqual(model.id, deepSeekDefaultModelId)
			assert.ok(model.info !== undefined)
			assert.strictEqual(model.info.supportsPromptCache, true)

		test("should include model parameters from getModelParams", () => {
			const model = handler.getModel()
			expect(model).toHaveProperty("temperature")
			expect(model).toHaveProperty("maxTokens")

	suite("createMessage", () => {
		const systemPrompt = "You are a helpful assistant."
		const messages: NeutralConversationHistory = [
			// Explicitly type as NeutralConversationHistory
			{
				role: "user",
				content: "Hello!",
			},

		test("should handle streaming responses", async () => {
			const stream = handler.createMessage(systemPrompt, messages)
			const chunks: ApiStreamChunk[] = []
			for await (const chunk of stream) {
				chunks.push(chunk)

			assert.ok(chunks.length > 0)
			const textChunks = chunks.filter((chunk) => chunk.type === "text")
			assert.strictEqual(textChunks.length, 1)
			assert.strictEqual(textChunks[0].text, "Test response")

		test("should include usage information", async () => {
			const stream = handler.createMessage(systemPrompt, messages)
			const chunks: ApiStreamChunk[] = []
			for await (const chunk of stream) {
				chunks.push(chunk)

			const usageChunks = chunks.filter((chunk) => chunk.type === "usage")
			assert.ok(usageChunks.length > 0)
			assert.strictEqual(usageChunks[0].inputTokens, 10)
			assert.strictEqual(usageChunks[0].outputTokens, 5)

		test("should include cache metrics in usage information", async () => {
			const stream = handler.createMessage(systemPrompt, messages)
			const chunks: ApiStreamChunk[] = []
			for await (const chunk of stream) {
				chunks.push(chunk)

			const usageChunks = chunks.filter((chunk) => chunk.type === "usage")
			assert.ok(usageChunks.length > 0)
			assert.strictEqual(usageChunks[0].cacheWriteTokens, 8)
			assert.strictEqual(usageChunks[0].cacheReadTokens, 2)

	suite("processUsageMetrics", () => {
		test("should correctly process usage metrics including cache information", () => {
			// We need to access the protected method, so we'll create a test subclass
			class TestDeepSeekHandler extends DeepSeekHandler {
				public testProcessUsageMetrics(usage: OpenAI.CompletionUsage) {
					return this.processUsageMetrics(usage)

			const testHandler = new TestDeepSeekHandler(mockOptions)

			const usage = {
				prompt_tokens: 100,
				completion_tokens: 50,
				total_tokens: 150,
				prompt_tokens_details: {
					cache_miss_tokens: 80,
					cached_tokens: 20,
				},

			const result = testHandler.testProcessUsageMetrics(usage)

			assert.strictEqual(result.type, "usage")
			assert.strictEqual(result.inputTokens, 100)
			assert.strictEqual(result.outputTokens, 50)
			assert.strictEqual(result.cacheWriteTokens, 80)
			assert.strictEqual(result.cacheReadTokens, 20)

		test("should handle missing cache metrics gracefully", () => {
			class TestDeepSeekHandler extends DeepSeekHandler {
				public testProcessUsageMetrics(usage: OpenAI.CompletionUsage) {
					return this.processUsageMetrics(usage)

			const testHandler = new TestDeepSeekHandler(mockOptions)

			const usage = {
				prompt_tokens: 100,
				completion_tokens: 50,
				total_tokens: 150,
				// No prompt_tokens_details

			const result = testHandler.testProcessUsageMetrics(usage)

			assert.strictEqual(result.type, "usage")
			assert.strictEqual(result.inputTokens, 100)
			assert.strictEqual(result.outputTokens, 50)
			assert.strictEqual(result.cacheWriteTokens, undefined)
			assert.strictEqual(result.cacheReadTokens, undefined)
