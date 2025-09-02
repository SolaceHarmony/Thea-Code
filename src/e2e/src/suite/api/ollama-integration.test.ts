import { OllamaHandler } from "../providers/ollama"
import { NeutralConversationHistory } from "../../shared/neutral-history"
import OpenAI from "openai"
// Note: This test uses port 10000 which is for Msty, a service that uses Ollama on the backend

// Mock the McpIntegration to avoid initialization issues
// TODO: Mock setup needs manual migration
import * as assert from 'assert'
import * as sinon from 'sinon'
// TODO: Fix mock - needs proxyquire
/*
=> {
	const mockInstance = {
		initialize: sinon.stub().resolves(undefined),
		registerTool: sinon.stub(),
		routeToolUse: sinon.stub().resolves("{}"),

	class MockMcpIntegration {
		initialize = sinon.stub().resolves(undefined)
		registerTool = sinon.stub()
		routeToolUse = sinon.stub().resolves("{}")

		static getInstance = sinon.stub().returns(mockInstance)

	return {
		McpIntegration: MockMcpIntegration,

})*/

// Mock the HybridMatcher
// TODO: Mock setup needs manual migration
// TODO: Fix mock - needs proxyquire
/*
=> {
	return {
		HybridMatcher: sinon.stub().callsFake(() => ({
			update: sinon.stub().callsFake((text: string) => {
				if (text.includes("<think>")) {
					return [{ matched: true, type: "reasoning", data: text.replace(/<\/?think>/g, "") }]

				if (text.includes('{"type":"thinking"')) {
					try {
						// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
						const jsonObj = JSON.parse(text)
						// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
						if (jsonObj.type === "thinking") {
							// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
							return [{ matched: true, type: "reasoning", data: String(jsonObj.content) }]

						// eslint-disable-next-line @typescript-eslint/no-unused-vars
					} catch (_e: unknown) {
						// Not valid JSON, treat as text

				return [{ matched: false, type: "text", data: text, text: text }]
			}),

			final: sinon.stub().callsFake((text: string) => {
				if (text) {
					return [{ matched: false, type: "text", data: text, text: text }]

				return []
			}),
		})),

})*/

// Mock the XmlMatcher
// TODO: Mock setup needs manual migration
// TODO: Fix mock - needs proxyquire
/*
=> {
	return {
		XmlMatcher: sinon.stub().callsFake(() => ({
			update: sinon.stub().callsFake((text: string) => {
				if (text.includes("<think>")) {
					return [{ matched: true, type: "reasoning", data: text.replace(/<\/?think>/g, "") }]

				return [{ matched: false, type: "text", data: text, text: text }]
			}),
			final: sinon.stub().callsFake((text: string) => {
				if (text) {
					return [{ matched: false, type: "text", data: text, text: text }]

				return []
			}),
		})),

})*/

// Mock the OpenAI client for integration testing
// TODO: Mock setup needs manual migration
// TODO: Fix mock - needs proxyquire
/*
=> {
	// Create a more realistic mock that simulates Ollama's behavior
	const mockCreate = jest
		.fn()
		.callsFake(
			({ messages, stream }: { messages: OpenAI.Chat.ChatCompletionMessageParam[]; stream: boolean }) => {
				// Simulate streaming response
				if (stream) {
					return {
						[Symbol.asyncIterator]: function* () {
							// eslint-disable-next-line @typescript-eslint/no-unused-vars
							const hasSystemMessage = messages.some(
								(msg: OpenAI.Chat.ChatCompletionMessageParam) => msg.role === "system",

							// Check for specific test cases based on user message content

							const userMessage =
								messages.find((msg: OpenAI.Chat.ChatCompletionMessageParam) => msg.role === "user")
									?.content || ""

							if (typeof userMessage === "string" && userMessage.includes("reasoning")) {
								// Test case for reasoning/thinking
								yield {
									choices: [
										{
											delta: { content: "I need to think about this. " },
											index: 0,
											finish_reason: null,
										},
									],
									id: "chatcmpl-reasoning-1",
									created: 1678886400,
									model: "llama2",
									object: "chat.completion.chunk" as const,

								yield {
									choices: [
										{
											delta: {
												content:
													"<think>This is a reasoning block where I analyze the problem.</think>",
											},
											index: 0,
											finish_reason: null,
										},
									],
									id: "chatcmpl-reasoning-2",
									created: 1678886401,
									model: "llama2",
									object: "chat.completion.chunk" as const,

								yield {
									choices: [
										{
											delta: { content: " After thinking, my answer is 42." },
											index: 0,
											finish_reason: "stop",
										},
									],
									id: "chatcmpl-reasoning-3",
									created: 1678886402,
									model: "llama2",
									object: "chat.completion.chunk" as const,

							} else if (typeof userMessage === "string" && userMessage.includes("multi-turn")) {
								// Test case for multi-turn conversations
								// Return a response that acknowledges previous messages

								const assistantMsgContent = messages.find(
									(msg: OpenAI.Chat.ChatCompletionMessageParam) => msg.role === "assistant",
								)?.content
								const previousAssistantMessage =
									typeof assistantMsgContent === "string" ? assistantMsgContent : ""
								yield {
									choices: [
										{
											delta: {
												content: `I see our previous conversation where I said "${previousAssistantMessage}". `,
											},
											index: 0,
											finish_reason: null,
										},
									],
									id: "chatcmpl-multi-turn-1",
									created: 1678886403,
									model: "llama2",
									object: "chat.completion.chunk" as const,

								yield {
									choices: [
										{
											delta: { content: "Now I can continue from there." },
											index: 0,
											finish_reason: "stop",
										},
									],
									id: "chatcmpl-multi-turn-2",
									created: 1678886404,
									model: "llama2",
									object: "chat.completion.chunk" as const,

							} else if (typeof userMessage === "string" && userMessage.includes("system prompt")) {
								// Test case for system prompt

								const systemMsgContent = messages.find(
									(msg: OpenAI.Chat.ChatCompletionMessageParam) => msg.role === "system",
								)?.content
								const systemMessage =
									typeof systemMsgContent === "string" ? systemMsgContent : "No system prompt"
								yield {
									choices: [
										{
											delta: { content: `I'm following the system prompt: "${systemMessage}". ` },
											index: 0,
											finish_reason: null,
										},
									],
									id: "chatcmpl-system-prompt-1",
									created: 1678886405,
									model: "llama2",
									object: "chat.completion.chunk" as const,

								yield {
									choices: [
										{
											delta: { content: "This shows I received it correctly." },
											index: 0,
											finish_reason: "stop",
										},
									],
									id: "chatcmpl-system-prompt-2",
									created: 1678886406,
									model: "llama2",
									object: "chat.completion.chunk" as const,

							} else {
								// Default response
								yield {
									choices: [
										{
											delta: { content: "Hello! " },
											index: 0,
											finish_reason: null,
										},
									],
									id: "chatcmpl-default-1",
									created: 1678886407,
									model: "llama2",
									object: "chat.completion.chunk" as const,

								yield {
									choices: [
										{
											delta: { content: "This is a response from the Ollama API." },
											index: 0,
											finish_reason: "stop",
										},
									],
									id: "chatcmpl-default-2",
									created: 1678886408,
									model: "llama2",
									object: "chat.completion.chunk" as const,

						},

				// Non-streaming response
				return {
					choices: [
						{
							message: { content: "Hello! This is a response from the Ollama API.", refusal: null },
						},
					],

			},

	return {
		__esModule: true,
		default: sinon.stub().callsFake(() => ({
			chat: {
				completions: {
					create: mockCreate,
				},
			},
		})),

})*/

suite("Ollama Integration", () => {
	let handler: OllamaHandler

	setup(() => {
		sinon.restore()

		// Create handler with test options
		// Note: Using port 10000 for Msty which uses Ollama on the backend
		handler = new OllamaHandler({
			ollamaBaseUrl: "http://localhost:10000",
			ollamaModelId: "llama2",

	test("should handle basic text messages", async () => {
		// Create neutral history with a simple user message
		const neutralHistory: NeutralConversationHistory = [
			{ role: "user", content: [{ type: "text", text: "Hello" }] },

		// Call createMessage
		const stream = handler.createMessage("You are helpful.", neutralHistory)

		// Collect stream chunks
		const chunks = []
		for await (const chunk of stream) {
			chunks.push(chunk)

		// Verify stream chunks
		assert.ok(chunks.some(item => JSON.stringify(item) === JSON.stringify({ type: "text", text: "Hello! " })))
		assert.ok(chunks.some(item => JSON.stringify(item) === JSON.stringify({ type: "text", text: "This is a response from the Ollama API." })))

	test("should handle reasoning/thinking with XML tags", async () => {
		// Create neutral history with a message that triggers reasoning
		const neutralHistory: NeutralConversationHistory = [
			{ role: "user", content: [{ type: "text", text: "Please use reasoning to solve this problem." }] },

		// Call createMessage
		const stream = handler.createMessage("You are helpful.", neutralHistory)

		// Collect stream chunks
		const chunks = []
		for await (const chunk of stream) {
			chunks.push(chunk)

		// Verify stream chunks
		assert.ok(chunks.some(item => JSON.stringify(item) === JSON.stringify({ type: "text", text: "I need to think about this. " })))
		assert.ok(chunks.some(item => JSON.stringify(item) === JSON.stringify({
			type: "reasoning",
			text: "This is a reasoning block where I analyze the problem.",
		})))
		assert.ok(chunks.some(item => JSON.stringify(item) === JSON.stringify({ type: "text", text: " After thinking, my answer is 42." })))

	test("should handle multi-turn conversations", async () => {
		// Create neutral history with multiple turns
		const neutralHistory: NeutralConversationHistory = [
			{ role: "user", content: [{ type: "text", text: "Hello" }] },
			{ role: "assistant", content: [{ type: "text", text: "Hi there!" }] },
			{ role: "user", content: [{ type: "text", text: "Let's continue our multi-turn conversation." }] },

		// Mock the OpenAI client's create method for this specific test
		sinon.stub(handler["client"].chat.completions, "create").onFirstCall().callsFake(
			// eslint-disable-next-line @typescript-eslint/no-unused-vars
			((_body: OpenAI.Chat.ChatCompletionCreateParams, _options?: OpenAI.RequestOptions) => {
				const chunks = [
					{
						choices: [
							{
								delta: { content: 'I see our previous conversation where I said "Hi there!". ' },
								index: 0,
								finish_reason: null,
							},
						],
						id: "chatcmpl-multi-turn-mock-1",
						created: 1678886409,
						model: "llama2",
						object: "chat.completion.chunk" as const,
					},
					{
						choices: [
							{
								delta: { content: "Now I can continue from there." },
								index: 0,
								finish_reason: "stop" as const,
							},
						],
						id: "chatcmpl-multi-turn-mock-2",
						created: 1678886410,
						model: "llama2",
						object: "chat.completion.chunk" as const,

				// Create a proper async iterator that matches OpenAI's Stream interface
				function* generateChunks() {
					for (const chunk of chunks) {
						yield chunk

				// eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-return
				return generateChunks() as any
			}),

		// Call createMessage
		const stream = handler.createMessage("You are helpful.", neutralHistory)

		// Collect stream chunks
		const chunks = []
		for await (const chunk of stream) {
			chunks.push(chunk)

		// Verify stream chunks
		assert.ok(chunks.some(item => JSON.stringify(item) === JSON.stringify({
			type: "text",
			text: 'I see our previous conversation where I said "Hi there!". ',
		})))
		assert.ok(chunks.some(item => JSON.stringify(item) === JSON.stringify({ type: "text", text: "Now I can continue from there." })))

	test("should handle system prompts", async () => {
		// Create neutral history
		const neutralHistory: NeutralConversationHistory = [
			{ role: "user", content: [{ type: "text", text: "Tell me about the system prompt." }] },

		// Call createMessage with a specific system prompt
		const stream = handler.createMessage(
			"You are a helpful assistant that provides concise answers.",
			neutralHistory,

		// Collect stream chunks
		const chunks = []
		for await (const chunk of stream) {
			chunks.push(chunk)

		// Verify stream chunks
		assert.ok(chunks.some(item => JSON.stringify(item) === JSON.stringify({
			type: "text",
			text: 'I\'m following the system prompt: "You are a helpful assistant that provides concise answers.". ',
		})))
		assert.ok(chunks.some(item => JSON.stringify(item) === JSON.stringify({ type: "text", text: "This shows I received it correctly." })))

	test("should handle multiple content blocks", async () => {
		// Create neutral history with multiple content blocks
		const neutralHistory: NeutralConversationHistory = [
			{
				role: "user",
				content: [
					{ type: "text", text: "First paragraph." },
					{ type: "text", text: "Second paragraph." },
				],
			},

		// Call createMessage
		const stream = handler.createMessage("You are helpful.", neutralHistory)

		// Collect stream chunks
		const chunks = []
		for await (const chunk of stream) {
			chunks.push(chunk)

		// Verify stream chunks (should be the default response)
		assert.ok(chunks.some(item => JSON.stringify(item) === JSON.stringify({ type: "text", text: "Hello! " })))
		assert.ok(chunks.some(item => JSON.stringify(item) === JSON.stringify({ type: "text", text: "This is a response from the Ollama API." })))

	test("should handle non-text content blocks by ignoring them", async () => {
		// Create neutral history with mixed content blocks
		const neutralHistory: NeutralConversationHistory = [
			{
				role: "user",
				content: [
					{ type: "text", text: "Look at this image:" },
					{
						type: "image",
						source: {
							type: "base64",
							media_type: "image/png",
							data: "base64data",
						},
					},
				],
			},

		// Call createMessage
		const stream = handler.createMessage("You are helpful.", neutralHistory)

		// Collect stream chunks
		const chunks = []
		for await (const chunk of stream) {
			chunks.push(chunk)

		// Verify stream chunks (should be the default response)
		assert.ok(chunks.some(item => JSON.stringify(item) === JSON.stringify({ type: "text", text: "Hello! " })))
		assert.ok(chunks.some(item => JSON.stringify(item) === JSON.stringify({ type: "text", text: "This is a response from the Ollama API." })))

	test("should handle completePrompt method", async () => {
		// Call completePrompt
		const result = await handler.completePrompt("Hello")

		// Verify result
		assert.strictEqual(result, "Hello! This is a response from the Ollama API.")

	test("should handle countTokens method", async () => {
		// Mock the base provider's countTokens method
		const mockSuperCountTokens = sinon.spy(Object.getPrototypeOf(OllamaHandler.prototype), "countTokens")
		mockSuperCountTokens.resolves(5) // 5 tokens for "Hello! This is a test."

		// Call countTokens
		const tokenCount = await handler.countTokens([{ type: "text", text: "Hello! This is a test." }])

		// Verify token count
		assert.strictEqual(tokenCount, 5)

		// Clean up
		mockSuperCountTokens.restore()

	test("should handle reasoning/thinking with JSON format", async () => {
		// Create neutral history with a message that triggers JSON reasoning
		const neutralHistory: NeutralConversationHistory = [
			{ role: "user", content: [{ type: "text", text: "Let me think about reasoning in JSON format" }] },

		// Mock the OpenAI client's create method for this specific test
		sinon.stub(handler["client"].chat.completions, "create").onFirstCall().callsFake(
			// eslint-disable-next-line @typescript-eslint/no-unused-vars
			((_body: OpenAI.Chat.ChatCompletionCreateParams, _options?: OpenAI.RequestOptions) => {
				const chunks = [
					{
						choices: [
							{
								delta: { content: "I need to think about this. " },
								index: 0,
								finish_reason: null,
							},
						],
						id: "chatcmpl-json-reasoning-1",
						created: 1678886411,
						model: "llama2",
						object: "chat.completion.chunk" as const,
					},
					{
						choices: [
							{
								delta: {
									content: '{"type":"thinking","content":"This is a reasoning block in JSON format"}',
								},
								index: 0,
								finish_reason: null,
							},
						],
						id: "chatcmpl-json-reasoning-2",
						created: 1678886412,
						model: "llama2",
						object: "chat.completion.chunk" as const,
					},
					{
						choices: [
							{
								delta: { content: " After thinking, my answer is 42." },
								index: 0,
								finish_reason: "stop" as const,
							},
						],
						id: "chatcmpl-json-reasoning-3",
						created: 1678886413,
						model: "llama2",
						object: "chat.completion.chunk" as const,

				// Create a proper async iterator that matches OpenAI's Stream interface
				function* generateChunks() {
					for (const chunk of chunks) {
						yield chunk

				// eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-return
				return generateChunks() as any
			}),

		// Call createMessage
		const stream = handler.createMessage("You are helpful.", neutralHistory)

		// Collect stream chunks
		const chunks = []
		for await (const chunk of stream) {
			chunks.push(chunk)

		// Verify stream chunks
		assert.ok(chunks.some(item => JSON.stringify(item) === JSON.stringify({ type: "text", text: "I need to think about this. " })))
		assert.ok(chunks.some(item => JSON.stringify(item) === JSON.stringify({ type: "reasoning", text: "This is a reasoning block in JSON format" })))
		assert.ok(chunks.some(item => JSON.stringify(item) === JSON.stringify({ type: "text", text: " After thinking, my answer is 42." })))
