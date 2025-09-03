// npx jest src/api/providers/__tests__/anthropic.test.ts

import { AnthropicHandler } from "../anthropic"
import { ApiHandlerOptions } from "../../../shared/api"
import type { NeutralConversationHistory } from "../../../shared/neutral-history"

// Mock NeutralAnthropicClient instead of the direct SDK
import * as assert from 'assert'
import * as sinon from 'sinon'
const mockCreateMessage = sinon.stub()
const mockCountTokens = sinon.stub()

// TODO: Mock setup needs manual migration
// TODO: Fix mock - needs proxyquire
/*
=> {
	return {
		NeutralAnthropicClient: sinon.stub().callsFake(() => ({
			createMessage: mockCreateMessage,
			countTokens: mockCountTokens,
		})),

})*/

suite("AnthropicHandler", () => {
	let handler: AnthropicHandler
	let mockOptions: ApiHandlerOptions

	setup(() => {
		mockOptions = {
			apiKey: "test-api-key",
			apiModelId: "claude-3-5-sonnet-20241022",

		handler = new AnthropicHandler(mockOptions)
		mockCreateMessage.mockClear()
		mockCountTokens.mockClear()

	suite("constructor", () => {
		test("should initialize with provided options", () => {
			assert.ok(handler instanceof AnthropicHandler)
			expect(handler.getModel().id).toBe(mockOptions.apiModelId)

		test("should initialize with undefined API key", () => {
			// The SDK will handle API key validation, so we just verify it initializes
			const handlerWithoutKey = new AnthropicHandler({
				...mockOptions,
				apiKey: undefined,
			assert.ok(handlerWithoutKey instanceof AnthropicHandler)

		test("should use custom base URL if provided", () => {
			const customBaseUrl = "https://custom.anthropic.com"
			const handlerWithCustomUrl = new AnthropicHandler({
				...mockOptions,
				anthropicBaseUrl: customBaseUrl,
			assert.ok(handlerWithCustomUrl instanceof AnthropicHandler)

	suite("createMessage", () => {
		const systemPrompt = "You are a helpful assistant."

		setup(() => {
			// Setup a default mock for createMessage that returns expected chunks
			mockCreateMessage.callsFake(async function* () {
				yield {
					type: "usage",
					inputTokens: 100,
					outputTokens: 50,
					cacheWriteTokens: 20,
					cacheReadTokens: 10,

				// Add await to satisfy async requirement
				await Promise.resolve()
				yield { type: "text", text: "Hello" }
				yield { type: "text", text: " world" }

		test("should handle prompt caching for supported models", async () => {
			// Use neutral format for messages
			const neutralMessages: NeutralConversationHistory = [
				{
					role: "user",
					content: [{ type: "text", text: "First message" }],
				},
				{
					role: "assistant",
					content: [{ type: "text", text: "Response" }],
				},
				{
					role: "user",
					content: [{ type: "text", text: "Second message" }],
				},

			const stream = handler.createMessage(systemPrompt, neutralMessages)

			const chunks: Array<{
				type: string
				inputTokens?: number
				outputTokens?: number
				cacheWriteTokens?: number
				cacheReadTokens?: number
				text?: string
			}> = []
			for await (const chunk of stream) {
				chunks.push(chunk)

			// Verify usage information
			const usageChunk = chunks.find((chunk) => chunk.type === "usage")
			assert.ok(usageChunk !== undefined)
			assert.strictEqual(usageChunk?.inputTokens, 100)
			assert.strictEqual(usageChunk?.outputTokens, 50)
			assert.strictEqual(usageChunk?.cacheWriteTokens, 20)
			assert.strictEqual(usageChunk?.cacheReadTokens, 10)

			// Verify text content
			const textChunks = chunks.filter((chunk) => chunk.type === "text")
			assert.strictEqual(textChunks.length, 2)
			assert.strictEqual(textChunks[0].text, "Hello")
			assert.strictEqual(textChunks[1].text, " world")

			// Verify the neutral client was called
			assert.ok(mockCreateMessage.calledWith({
				model: mockOptions.apiModelId,
				systemPrompt,
				messages: neutralMessages,
				maxTokens: 8192,
				temperature: 0,

	suite("completePrompt", () => {
		test("should complete prompt successfully", async () => {
			// Setup mock to return a simple text stream
			mockCreateMessage.callsFake(async function* () {
				yield { type: "text", text: "Test response" }
				await Promise.resolve() // Add await to satisfy async requirement

			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "Test response")
			assert.ok(mockCreateMessage.calledWith({
				model: mockOptions.apiModelId,
				systemPrompt: "",
				messages: [{ role: "user", content: "Test prompt" }],
				maxTokens: 8192,
				temperature: 0,

		test("should handle non-text content", async () => {
			// Setup mock to return a stream with different text chunks
			mockCreateMessage.callsFake(async function* () {
				yield { type: "text", text: "Hello" }
				yield { type: "text", text: " world" }
				await Promise.resolve() // Add await to satisfy async requirement

			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "Hello world")

		test("should handle empty response", async () => {
			// Setup mock to return empty stream
			mockCreateMessage.callsFake(async function* () {
				// No yields, empty stream

			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "")

	suite("getModel", () => {
		test("should return default model if no model ID is provided", () => {
			const handlerWithoutModel = new AnthropicHandler({
				...mockOptions,
				apiModelId: undefined,
			const model = handlerWithoutModel.getModel()
			assert.ok(model.id !== undefined)
			assert.ok(model.info !== undefined)

		test("should return specified model if valid model ID is provided", () => {
			const model = handler.getModel()
			assert.strictEqual(model.id, mockOptions.apiModelId)
			assert.ok(model.info !== undefined)
			assert.strictEqual(model.info.maxTokens, 8192)
			assert.strictEqual(model.info.contextWindow, 200_000)
			assert.strictEqual(model.info.supportsImages, true)
			assert.strictEqual(model.info.supportsPromptCache, true)

		test("honors custom maxTokens for thinking models", () => {
			const handler = new AnthropicHandler({
				apiKey: "test-api-key",
				apiModelId: "claude-3-7-sonnet-20250219:thinking",
				modelMaxTokens: 32_768,
				modelMaxThinkingTokens: 16_384,

			const result = handler.getModel()
			assert.strictEqual(result.maxTokens, 32_768)
			assert.deepStrictEqual(result.thinking, { type: "enabled", budget_tokens: 16_384 })
			assert.strictEqual(result.temperature, 1.0)

		test("does not honor custom maxTokens for non-thinking models", () => {
			const handler = new AnthropicHandler({
				apiKey: "test-api-key",
				apiModelId: "claude-3-7-sonnet-20250219",
				modelMaxTokens: 32_768,
				modelMaxThinkingTokens: 16_384,

			const result = handler.getModel()
			assert.strictEqual(result.maxTokens, 8192)
			assert.strictEqual(result.thinking, undefined)
			assert.strictEqual(result.temperature, 0)

	suite("countTokens", () => {
		test("should count tokens using NeutralAnthropicClient", async () => {
			// Setup the mock to return token count
			mockCountTokens.resolves(42)

			// Create neutral content for testing
			const neutralContent = [{ type: "text" as const, text: "Test message" }]

			// Call the method
			const result = await handler.countTokens(neutralContent)

			// Verify the result
			assert.strictEqual(result, 42)

			// Verify the NeutralAnthropicClient countTokens was called
			assert.ok(mockCountTokens.calledWith("claude-3-5-sonnet-20241022", neutralContent))

		test("should fall back to base provider implementation on error", async () => {
			// Mock the countTokens to throw an error
			mockCountTokens.rejects(new Error("Failed to count tokens: API Error"))

			// Mock the base provider's countTokens method
			const mockBaseCountTokens = jest
				.spyOn(Object.getPrototypeOf(Object.getPrototypeOf(handler)), "countTokens")
				.resolves(24)

			// Create neutral content for testing
			const neutralContent = [{ type: "text" as const, text: "Test message" }]

			// Call the method
			const result = await handler.countTokens(neutralContent)

			// Verify the result comes from the base implementation
			assert.strictEqual(result, 24)

			// Verify the base method was called with the original neutral content
			assert.ok(mockBaseCountTokens.calledWith(neutralContent))
