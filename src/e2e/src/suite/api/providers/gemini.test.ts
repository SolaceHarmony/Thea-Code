import { GeminiHandler } from "../gemini"
import type { NeutralConversationHistory, NeutralMessageContent } from "../../../shared/neutral-history"

// Mock the Google Generative AI SDK
// TODO: Mock setup needs manual migration
import * as assert from 'assert'
import * as sinon from 'sinon'
// TODO: Mock needs manual migration
// TODO: Implement proper mock with proxyquire,

suite("GeminiHandler", () => {
	let handler: GeminiHandler

	setup(() => {
		handler = new GeminiHandler({
			apiKey: "test-key",
			apiModelId: "gemini-2.0-flash-thinking-exp-1219",
			geminiApiKey: "test-key",

	suite("constructor", () => {
		test("should initialize with provided config", () => {
			assert.strictEqual(handler["options"].geminiApiKey, "test-key")
			assert.strictEqual(handler["options"].apiModelId, "gemini-2.0-flash-thinking-exp-1219")

		test.skip("should throw if API key is missing", () => {
			expect(() => {
				new GeminiHandler({
					apiModelId: "gemini-2.0-flash-thinking-exp-1219",
					geminiApiKey: "",

			}).toThrow("API key is required for Google Gemini")

	suite("createMessage", () => {
		const mockMessages: NeutralConversationHistory = [
			{
				role: "user",
				content: [{ type: "text", text: "Hello" }],
			},
			{
				role: "assistant",
				content: [{ type: "text", text: "Hi there!" }],
			},

		const systemPrompt = "You are a helpful assistant"

		test("should handle text messages correctly", async () => {
			// Mock the stream response
			const mockStream = {
				stream: [{ text: () => "Hello" }, { text: () => " world!" }],
				response: {
					usageMetadata: {
						promptTokenCount: 10,
						candidatesTokenCount: 5,
					},
				},

		// Setup the mock implementation
		const mockGenerateContentStream = sinon.stub().resolves(mockStream)

		const mockClient = {
			getGenerativeModel: sinon.stub().returns({
				generateContentStream: mockGenerateContentStream,
			}),

		// eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
		;(handler as any)["client"] = mockClient

			const stream = handler.createMessage(systemPrompt, mockMessages)
			const chunks = []

			for await (const chunk of stream) {
				chunks.push(chunk)

			// Should have 3 chunks: 'Hello', ' world!', and usage info
			assert.strictEqual(chunks.length, 3)
			assert.deepStrictEqual(chunks[0], {
				type: "text",
				text: "Hello",

			assert.deepStrictEqual(chunks[1], {
				type: "text",
				text: " world!",

			assert.deepStrictEqual(chunks[2], {
				type: "usage",
				inputTokens: 10,
				outputTokens: 5,

		// Verify the model configuration
		// eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
		expect((handler as any)["client"].getGenerativeModel).toHaveBeenCalledWith(
			{
				model: "gemini-2.0-flash-thinking-exp-1219",
				systemInstruction: systemPrompt,
			},
				{
					baseUrl: undefined,
				},

			// Verify generation config
			assert.ok(mockGenerateContentStream.calledWith(
				sinon.match({
					generationConfig: {
						temperature: 0,
					},
				})),

		test("should handle API errors", async () => {
			const mockError = new Error("Gemini API error")
			const mockGenerateContentStream = sinon.stub().rejects(mockError)

			;(handler["client"] as unknown as { getGenerativeModel: sinon.SinonStub }).getGenerativeModel = sinon.stub().returns({
				generateContentStream: mockGenerateContentStream,

			const stream = handler.createMessage(systemPrompt, mockMessages)

			await expect(async () => {
				// eslint-disable-next-line @typescript-eslint/no-unused-vars
				for await (const _chunk of stream) {
					// Should throw before yielding any chunks

			}).rejects.toThrow("Gemini API error")

	suite("completePrompt", () => {
		test("should complete prompt successfully", async () => {
			const mockGenerateContent = sinon.stub().resolves({
				response: {
					text: () => "Test response",
				},

			;(handler["client"] as unknown as { getGenerativeModel: sinon.SinonStub }).getGenerativeModel = sinon.stub().returns({
				generateContent: mockGenerateContent,

			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "Test response")

			expect((handler["client"] as unknown as { getGenerativeModel: sinon.SinonStub }).getGenerativeModel).toHaveBeenCalledWith(
				{
					model: "gemini-2.0-flash-thinking-exp-1219",
				},
				{
					baseUrl: undefined,
				},

			assert.ok(mockGenerateContent.calledWith({
				contents: [{ role: "user", parts: [{ text: "Test prompt" }] }],
				generationConfig: {
					temperature: 0,
				},

		test("should handle API errors", async () => {
			const mockError = new Error("Gemini API error")
			const mockGenerateContent = sinon.stub().rejects(mockError)

			;(handler["client"] as unknown as { getGenerativeModel: sinon.SinonStub }).getGenerativeModel = sinon.stub().returns({
				generateContent: mockGenerateContent,

			await expect(handler.completePrompt("Test prompt")).rejects.toThrow(
				"Gemini completion error: Gemini API error",

		test("should handle empty response", async () => {
			const mockGenerateContent = sinon.stub().resolves({
				response: {
					text: () => "",
				},

			;(handler["client"] as unknown as { getGenerativeModel: sinon.SinonStub }).getGenerativeModel = sinon.stub().returns({
				generateContent: mockGenerateContent,

			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "")

	suite("getModel", () => {
		test("should return correct model info", () => {
			const modelInfo = handler.getModel()
			assert.strictEqual(modelInfo.id, "gemini-2.0-flash-thinking-exp-1219")
			assert.ok(modelInfo.info !== undefined)
			assert.strictEqual(modelInfo.info.maxTokens, 8192)
			assert.strictEqual(modelInfo.info.contextWindow, 32_767)

		test("should return default model if invalid model specified", () => {
			const invalidHandler = new GeminiHandler({
				apiModelId: "invalid-model",
				geminiApiKey: "test-key",

			const modelInfo = invalidHandler.getModel()
			assert.strictEqual(modelInfo.id, "gemini-2.0-flash-001") // Default model

	suite("countTokens", () => {
		test("should count tokens for text content", async () => {
			// Mock the base provider's countTokens method
			const mockBaseCountTokens = jest
				.spyOn(Object.getPrototypeOf(Object.getPrototypeOf(handler)), "countTokens")
				.resolves(10)

			// Create neutral content for testing
			const neutralContent = [{ type: "text" as const, text: "Test message" }]

			// Call the method
			const result = await handler.countTokens(neutralContent)

			// Verify the result
			assert.strictEqual(result, 10)

			// Verify the base method was called with the original neutral content
			assert.ok(mockBaseCountTokens.calledWith(neutralContent))

			// Restore the original implementation
			mockBaseCountTokens.restore()

		test("should handle mixed content including images", async () => {
			// Mock the base provider's countTokens method
			const mockBaseCountTokens = jest
				.spyOn(Object.getPrototypeOf(Object.getPrototypeOf(handler)), "countTokens")
				// eslint-disable-next-line @typescript-eslint/no-unsafe-argument
				.callsFake(((content: NeutralMessageContent) => {
					// Return 5 tokens for text content
					if (Array.isArray(content) && content.length === 1 && content[0].type === "text") {
						return 5

					return 0
				}) as any) // eslint-disable-line @typescript-eslint/no-explicit-any

			// Create mixed content with text and image
			const mixedContent: NeutralMessageContent = [
				// Explicitly type as NeutralMessageContent
				{ type: "text" as const, text: "Test message" },
				{
					type: "image_base64" as const, // Changed from "image" to "image_base64"
					source: {
						type: "base64" as const,
						media_type: "image/png",
						data: "base64data",
					},
				},

			// Call the method
			const result = await handler.countTokens(mixedContent)

			// Verify the result (5 for text + 258 for image)
			assert.strictEqual(result, 263)

			// Restore the original implementation
			mockBaseCountTokens.restore()

		test("should handle tool use content", async () => {
			// Mock the base provider's countTokens method
			const mockBaseCountTokens = jest
				.spyOn(Object.getPrototypeOf(Object.getPrototypeOf(handler)), "countTokens")
				// eslint-disable-next-line @typescript-eslint/no-unsafe-argument
				.callsFake(((content: NeutralMessageContent) => {
					// Return 15 tokens for the JSON string representation
					if (Array.isArray(content) && content.length === 1 && content[0].type === "text") {
						return 15

					return 0
				}) as any) // eslint-disable-line @typescript-eslint/no-explicit-any

			// Create tool use content
			const toolUseContent = [
				{
					type: "tool_use" as const,
					id: "calculator-123",
					name: "calculator",
					input: { a: 5, b: 10, operation: "add" },
				},

			// Call the method
			const result = await handler.countTokens(toolUseContent)

			// Verify the result
			assert.strictEqual(result, 15)

			// Restore the original implementation
			mockBaseCountTokens.restore()

		test("should handle errors by falling back to base implementation", async () => {
			// Mock the implementation to throw an error first time, then succeed second time
			const mockBaseCountTokens = jest
				.spyOn(Object.getPrototypeOf(Object.getPrototypeOf(handler)), "countTokens")
				.resolves(8)

			// Create a spy on console.warn
			const consoleWarnSpy = sinon.stub(console, "warn")()

			// Create content that will cause an error in our custom logic
			const content = [{ type: "text" as const, text: "Test content" }]

			// Force an error in the try block
			const mockError = new Error("Test error")
			mockBaseCountTokens.onFirstCall().callsFake(() => {
				throw mockError

			// Call the method (this will throw and then call the original)
			const result = await handler.countTokens(content)

			// Verify the warning was logged
			assert.ok(consoleWarnSpy.calledWith("Gemini token counting error, using fallback", mockError))

			// Verify the result from the fallback
			assert.strictEqual(result, 8)

			// Restore the original implementations
			mockBaseCountTokens.restore()
			consoleWarnSpy.restore()
