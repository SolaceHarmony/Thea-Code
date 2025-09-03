// TODO: test.each patterns need manual migration
import { OllamaHandler } from "../ollama"
import { NeutralConversationHistory } from "../../../shared/neutral-history"
import OpenAI from "openai"

// Mock the OpenAI client
// TODO: Mock setup needs manual migration
import * as assert from 'assert'
import * as sinon from 'sinon'
// TODO: Fix mock - needs proxyquire
/*
=> {
	// Create a mock that captures the messages sent to the API
	const mockCreate = jest
		.fn()
		.callsFake(({ messages }: { messages: OpenAI.Chat.ChatCompletionMessageParam[] }) => {
			// Store the messages for later inspection
			// Assign to the mock function instance itself. Requires mockCreate to be typed to allow this.
			;(mockCreate as sinon.SinonStub & { lastMessages?: OpenAI.Chat.ChatCompletionMessageParam[] }).lastMessages =
				messages

			// Return a simple response
			return {
				[Symbol.asyncIterator]: function* () {
					yield {
						choices: [
							{
								delta: { content: "Response" },
							},
						],

				},

	return {
		__esModule: true,
		default: sinon.stub().callsFake(() => ({
			chat: {
				completions: {
					create: mockCreate, // Use the outer mockCreate here
				},
			},
		})),

})*/

suite("Ollama System Role Handling", () => {
	let handler: OllamaHandler
	// Define availableModels as a const directly for .each
	const availableModels: string[] = ["llama2", "mistral", "gemma"]

	setup(() => {
		sinon.restore()

		// Create handler with test options
		handler = new OllamaHandler({
			ollamaBaseUrl: "http://localhost:10000",
			ollamaModelId: "llama2", // Default model for tests

	it.each(availableModels)("should use system role for system prompts with %s model", async (modelId) => {
		// Update handler to use the current model
		handler = new OllamaHandler({
			ollamaBaseUrl: "http://localhost:10000",
			ollamaModelId: modelId,
		// Create neutral history
		const neutralHistory: NeutralConversationHistory = [
			{ role: "user", content: [{ type: "text", text: "Hello" }] },

		// System prompt that contains tool information
		const systemPrompt = "You are a helpful assistant with access to the following tools: tool1, tool2, tool3"

		// Call createMessage
		const stream = handler.createMessage(systemPrompt, neutralHistory)

		// Consume the stream to ensure the API is called
		// eslint-disable-next-line @typescript-eslint/no-unused-vars
		for await (const _chunk of stream) {
			// Do nothing with the chunks

		// Get the messages that were sent to the API

		// const mockApiCreate = (handler['client'].chat.completions.create as sinon.SinonStub); // mockApiCreate is unused if we use the global mockCreate
		const sentMessages = (
			handler["client"].chat.completions.create as sinon.SinonStub & {
				lastMessages?: OpenAI.Chat.ChatCompletionMessageParam[]

		).lastMessages

		// Verify that the system prompt was sent with the system role
		assert.ok(sentMessages !== undefined)
		assert.ok(sentMessages!.length >= 2)

		// The first message should be the system prompt with role 'system'
		assert.deepStrictEqual(sentMessages![0], {
			role: "system",
			content: systemPrompt,

		// The second message should be the user message
		assert.deepStrictEqual(sentMessages![1], {
			role: "user",
			content: "Hello",

	it.each(availableModels)(
		"should preserve existing system messages from neutral history with %s model",
		async (modelId) => {
			// Update handler to use the current model
			handler = new OllamaHandler({
				ollamaBaseUrl: "http://localhost:10000",
				ollamaModelId: modelId,
			// Create neutral history with a system message
			const neutralHistory: NeutralConversationHistory = [
				{ role: "system", content: [{ type: "text", text: "Existing system message" }] },
				{ role: "user", content: [{ type: "text", text: "Hello" }] },

			// Additional system prompt
			const systemPrompt = "Additional system prompt"

			// Call createMessage
			const stream = handler.createMessage(systemPrompt, neutralHistory)

			// Consume the stream to ensure the API is called
			// eslint-disable-next-line @typescript-eslint/no-unused-vars
			for await (const _chunk of stream) {
				// Do nothing with the chunks

			// Get the messages that were sent to the API

			// const mockApiCreate = (handler['client'].chat.completions.create as sinon.SinonStub); // mockApiCreate is unused
			const sentMessages = (
				handler["client"].chat.completions.create as sinon.SinonStub & {
					lastMessages?: OpenAI.Chat.ChatCompletionMessageParam[]

			).lastMessages

			// Verify that both system messages were preserved
			assert.ok(sentMessages !== undefined)
			assert.ok(sentMessages!.length >= 2)

			// The first message should be the existing system message
			assert.deepStrictEqual(sentMessages![0], {
				role: "system",
				content: "Existing system message",

			// The second message should be the user message
			assert.deepStrictEqual(sentMessages![1], {
				role: "user",
				content: "Hello",

			// The additional system prompt should not be added since there's already a system message
			const systemMessages = sentMessages!.filter(
				(msg: OpenAI.Chat.ChatCompletionMessageParam) => msg.role === "system",

			assert.strictEqual(systemMessages.length, 1)
		},

	it.each(availableModels)(
		"should handle multiple system messages if they come from neutral history with %s model",
		async (modelId) => {
			// Update handler to use the current model
			handler = new OllamaHandler({
				ollamaBaseUrl: "http://localhost:10000",
				ollamaModelId: modelId,
			// Create neutral history with multiple system messages
			const neutralHistory: NeutralConversationHistory = [
				{ role: "system", content: [{ type: "text", text: "System message 1" }] },
				{ role: "system", content: [{ type: "text", text: "System message 2" }] },
				{ role: "user", content: [{ type: "text", text: "Hello" }] },

			// Call createMessage with empty system prompt
			const stream = handler.createMessage("", neutralHistory)

			// Consume the stream to ensure the API is called
			// eslint-disable-next-line @typescript-eslint/no-unused-vars
			for await (const _chunk of stream) {
				// Do nothing with the chunks

			// Get the messages that were sent to the API

			// const mockApiCreate = (handler['client'].chat.completions.create as sinon.SinonStub); // mockApiCreate is unused
			const sentMessages = (
				handler["client"].chat.completions.create as sinon.SinonStub & {
					lastMessages?: OpenAI.Chat.ChatCompletionMessageParam[]

			).lastMessages

			// Verify that both system messages were preserved
			assert.ok(sentMessages !== undefined)
			assert.ok(sentMessages!.length >= 3)

			// The first two messages should be the system messages
			assert.deepStrictEqual(sentMessages![0], {
				role: "system",
				content: "System message 1",

			assert.deepStrictEqual(sentMessages![1], {
				role: "system",
				content: "System message 2",

			// The third message should be the user message
			assert.deepStrictEqual(sentMessages![2], {
				role: "user",
				content: "Hello",
		},

	it.each(availableModels)("should not convert tool information to user messages with %s model", async (modelId) => {
		// Update handler to use the current model
		handler = new OllamaHandler({
			ollamaBaseUrl: "http://localhost:10000",
			ollamaModelId: modelId,
		// Create neutral history with tool-related messages
		const neutralHistory: NeutralConversationHistory = [
			{
				role: "user",
				content: [{ type: "text", text: "Use a tool" }],
			},
			{
				role: "assistant",
				content: [
					{ type: "text", text: "I will use a tool" },
					{
						type: "tool_use",
						id: "tool1",
						name: "calculator",
						input: { expression: "2+2" },
					},
				],
			},
			{
				role: "tool",
				content: [
					{
						type: "tool_result",
						tool_use_id: "tool1",
						content: [{ type: "text", text: "4" }],
					},
				],
			},

		// System prompt with tool definitions
		const systemPrompt = "You have access to the following tools: calculator"

		// Call createMessage
		const stream = handler.createMessage(systemPrompt, neutralHistory)

		// Consume the stream to ensure the API is called
		// eslint-disable-next-line @typescript-eslint/no-unused-vars
		for await (const _chunk of stream) {
			// Do nothing with the chunks

		// Get the messages that were sent to the API

		// const mockApiCreate = (handler['client'].chat.completions.create as sinon.SinonStub); // mockApiCreate is unused
		const sentMessages = (
			handler["client"].chat.completions.create as sinon.SinonStub & {
				lastMessages?: OpenAI.Chat.ChatCompletionMessageParam[]

		).lastMessages

		// Verify that the system prompt was sent with the system role
		assert.ok(sentMessages !== undefined)
		assert.deepStrictEqual(sentMessages![0], {
			role: "system",
			content: systemPrompt,

		// Verify that no tool information was converted to user messages
		const userMessages = sentMessages!.filter(
			(msg: OpenAI.Chat.ChatCompletionMessageParam) =>
				msg.role === "user" && typeof msg.content === "string" && msg.content.includes("calculator"),

		assert.strictEqual(userMessages.length, 0)
