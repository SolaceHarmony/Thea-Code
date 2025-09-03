import { LmStudioHandler } from "../lmstudio"
import { ApiHandlerOptions } from "../../../shared/api"
// Unused OpenAI import removed
// Unused Anthropic import removed
import type OpenAI from "openai" // Added for types
import type { ApiStreamChunk } from "../../transform/stream" // Added for types
// import type { Anthropic } from "@anthropic-ai/sdk"; // No longer needed directly in this test file for messages
import type { NeutralConversationHistory } from "../../../shared/neutral-history" // Import for messages type

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
										},

								},

						},
					),
				},
			},
		})),

})*/

suite("LmStudioHandler", () => {
	let handler: LmStudioHandler
	let mockOptions: ApiHandlerOptions

	setup(() => {
		mockOptions = {
			apiModelId: "local-model",
			lmStudioModelId: "local-model",
			lmStudioBaseUrl: "http://localhost:1234/v1",

		handler = new LmStudioHandler(mockOptions)
		mockCreate.mockClear()

	suite("constructor", () => {
		test("should initialize with provided options", () => {
			assert.ok(handler instanceof LmStudioHandler)
			expect(handler.getModel().id).toBe(mockOptions.lmStudioModelId)

		test("should use default base URL if not provided", () => {
			const handlerWithoutUrl = new LmStudioHandler({
				apiModelId: "local-model",
				lmStudioModelId: "local-model",
			assert.ok(handlerWithoutUrl instanceof LmStudioHandler)

	suite("createMessage", () => {
		const systemPrompt = "You are a helpful assistant."
		const messages: NeutralConversationHistory = [
			{
				role: "user",
				content: [{ type: "text", text: "Hello!" }],
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

		test("should handle API errors", async () => {
			mockCreate.rejectsOnce(new Error("API Error"))

			const stream = handler.createMessage(systemPrompt, messages)

			await expect(async () => {
				// eslint-disable-next-line @typescript-eslint/no-unused-vars
				for await (const _chunk of stream) {
					// Should not reach here

			}).rejects.toThrow("Please check the LM Studio developer logs to debug what went wrong")

	suite("completePrompt", () => {
		test("should complete prompt successfully", async () => {
			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "Test response")
			assert.ok(mockCreate.calledWith({
				model: mockOptions.lmStudioModelId,
				messages: [{ role: "user", content: "Test prompt" }],
				temperature: 0,
				stream: false,

		test("should handle API errors", async () => {
			mockCreate.rejectsOnce(new Error("API Error"))
			await expect(handler.completePrompt("Test prompt")).rejects.toThrow(
				"Please check the LM Studio developer logs to debug what went wrong",

		test("should handle empty response", async () => {
			mockCreate.resolvesOnce({
				choices: [{ message: { content: "" } }],
			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "")

	suite("getModel", () => {
		test("should return model info", () => {
			const modelInfo = handler.getModel()
			assert.strictEqual(modelInfo.id, mockOptions.lmStudioModelId)
			assert.ok(modelInfo.info !== undefined)
			assert.strictEqual(modelInfo.info.maxTokens, -1)
			assert.strictEqual(modelInfo.info.contextWindow, 128_000)
