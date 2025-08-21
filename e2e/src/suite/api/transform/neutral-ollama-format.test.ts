import {
import * as assert from 'assert'
import * as sinon from 'sinon'
	convertToOllamaHistory,
	convertToOllamaContentBlocks,
	convertToNeutralHistoryFromOllama,
} from "../neutral-ollama-format"
import type {
	NeutralConversationHistory,
	NeutralMessageContent,
	NeutralTextContentBlock,
} from "../../../shared/neutral-history"
import OpenAI from "openai"

suite("neutral-ollama-format", () => {
	suite("convertToOllamaHistory", () => {
		// Test case 1: Simple user message
		test("should convert a simple user message", () => {
			const neutralHistory: NeutralConversationHistory = [
				{
					role: "user",
					content: [{ type: "text", text: "What is the capital of France?" }],
				},
			]

			const expected: OpenAI.Chat.ChatCompletionMessageParam[] = [
				{ role: "user", content: "What is the capital of France?" },
			]

			expect(convertToOllamaHistory(neutralHistory)).toEqual(expected)
		})

		// Test case 2: System + User messages
		test("should convert system and user messages", () => {
			const neutralHistory: NeutralConversationHistory = [
				{
					role: "system",
					content: [{ type: "text", text: "You are a helpful assistant." }],
				},
				{
					role: "user",
					content: [{ type: "text", text: "What is the capital of France?" }],
				},
			]

			const expected: OpenAI.Chat.ChatCompletionMessageParam[] = [
				{ role: "system", content: "You are a helpful assistant." },
				{ role: "user", content: "What is the capital of France?" },
			]

			expect(convertToOllamaHistory(neutralHistory)).toEqual(expected)
		})

		// Test case 3: User + Assistant + User messages
		test("should convert multi-turn conversations", () => {
			const neutralHistory: NeutralConversationHistory = [
				{
					role: "user",
					content: [{ type: "text", text: "What is the capital of France?" }],
				},
				{
					role: "assistant",
					content: [{ type: "text", text: "The capital of France is Paris." }],
				},
				{
					role: "user",
					content: [{ type: "text", text: "What is its population?" }],
				},
			]

			const expected: OpenAI.Chat.ChatCompletionMessageParam[] = [
				{ role: "user", content: "What is the capital of France?" },
				{ role: "assistant", content: "The capital of France is Paris." },
				{ role: "user", content: "What is its population?" },
			]

			expect(convertToOllamaHistory(neutralHistory)).toEqual(expected)
		})

		// Test case 4: System + User + Assistant + User messages
		test("should convert system, user, assistant, and user messages", () => {
			const neutralHistory: NeutralConversationHistory = [
				{
					role: "system",
					content: [{ type: "text", text: "You are a helpful assistant." }],
				},
				{
					role: "user",
					content: [{ type: "text", text: "What is the capital of France?" }],
				},
				{
					role: "assistant",
					content: [{ type: "text", text: "The capital of France is Paris." }],
				},
				{
					role: "user",
					content: [{ type: "text", text: "What is its population?" }],
				},
			]

			const expected: OpenAI.Chat.ChatCompletionMessageParam[] = [
				{ role: "system", content: "You are a helpful assistant." },
				{ role: "user", content: "What is the capital of France?" },
				{ role: "assistant", content: "The capital of France is Paris." },
				{ role: "user", content: "What is its population?" },
			]

			expect(convertToOllamaHistory(neutralHistory)).toEqual(expected)
		})

		// Test case 5: Multiple content blocks in a message
		test("should join multiple text blocks with newlines", () => {
			const neutralHistory: NeutralConversationHistory = [
				{
					role: "user",
					content: [
						{ type: "text", text: "Hello" },
						{ type: "text", text: "World" },
					],
				},
			]

			const expected: OpenAI.Chat.ChatCompletionMessageParam[] = [{ role: "user", content: "Hello\n\nWorld" }]

			expect(convertToOllamaHistory(neutralHistory)).toEqual(expected)
		})

		// Test case 6: String content instead of array
		test("should handle string content", () => {
			const neutralHistory: NeutralConversationHistory = [
				{
					role: "user",
					content: "What is the capital of France?",
				},
			]

			const expected: OpenAI.Chat.ChatCompletionMessageParam[] = [
				{ role: "user", content: "What is the capital of France?" },
			]

			expect(convertToOllamaHistory(neutralHistory)).toEqual(expected)
		})

		// Test case 7: Non-text content blocks (should be ignored with warning)
		test("should ignore non-text content blocks with a warning", () => {
			// Mock console.warn
			const originalWarn = console.warn
			console.warn = sinon.stub()

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
			]

			const expected: OpenAI.Chat.ChatCompletionMessageParam[] = [
				{ role: "user", content: "Look at this image:" },
			]

			expect(convertToOllamaHistory(neutralHistory)).toEqual(expected)
			assert.ok(console.warn.called)

			// Restore console.warn
			console.warn = originalWarn
		})

		// Test case 8: Empty content array
		test("should handle empty content array", () => {
			const neutralHistory: NeutralConversationHistory = [
				{
					role: "user",
					content: [],
				},
			]

			const expected: OpenAI.Chat.ChatCompletionMessageParam[] = [{ role: "user", content: "" }]

			expect(convertToOllamaHistory(neutralHistory)).toEqual(expected)
		})

		// Test case 9: Unknown role
		test("should handle unknown roles", () => {
			// Mock console.warn
			const originalWarn = console.warn
			console.warn = sinon.stub()

			const neutralHistory: NeutralConversationHistory = [
				{
					role: "tool",
					content: [{ type: "text", text: "Tool result" }],
				},
			]

			const result = convertToOllamaHistory(neutralHistory)

			// Should default to user role
			assert.strictEqual(result[0].role, "user")
			assert.strictEqual(result[0].content, "Tool result")
			assert.ok(console.warn.called)

			// Restore console.warn
			console.warn = originalWarn
		})
	})

	suite("convertToOllamaContentBlocks", () => {
		// Test case 1: Array with a single text block
		test("should convert a single text block to string", () => {
			const content: NeutralMessageContent = [{ type: "text", text: "Hello, world!" }]
			expect(convertToOllamaContentBlocks(content)).toBe("Hello, world!")
		})

		// Test case 2: Multiple text blocks
		test("should join multiple text blocks with newlines", () => {
			const content: NeutralMessageContent = [
				{ type: "text", text: "Hello" },
				{ type: "text", text: "World" },
			]
			expect(convertToOllamaContentBlocks(content)).toBe("Hello\n\nWorld")
		})

		// Test case 3: Mixed content blocks (should ignore non-text)
		test("should extract only text from mixed content blocks", () => {
			const content: NeutralMessageContent = [
				{ type: "text", text: "Look at this image:" },
				{
					type: "image",
					source: {
						type: "base64",
						media_type: "image/png",
						data: "base64data",
					},
				},
				{ type: "text", text: "What do you think?" },
			]
			expect(convertToOllamaContentBlocks(content)).toBe("Look at this image:\n\nWhat do you think?")
		})

		// Test case 4: Empty array
		test("should handle empty array", () => {
			const content: NeutralMessageContent = []
			expect(convertToOllamaContentBlocks(content)).toBe("")
		})
	})

	suite("convertToNeutralHistoryFromOllama", () => {
		// Test case 1: Simple user message
		test("should convert a simple user message", () => {
			const ollamaHistory: OpenAI.Chat.ChatCompletionMessageParam[] = [
				{ role: "user", content: "What is the capital of France?" },
			]

			const expected: NeutralConversationHistory = [
				{
					role: "user",
					content: [{ type: "text", text: "What is the capital of France?" }],
				},
			]

			expect(convertToNeutralHistoryFromOllama(ollamaHistory)).toEqual(expected)
		})

		// Test case 2: System + User messages
		test("should convert system and user messages", () => {
			const ollamaHistory: OpenAI.Chat.ChatCompletionMessageParam[] = [
				{ role: "system", content: "You are a helpful assistant." },
				{ role: "user", content: "What is the capital of France?" },
			]

			const expected: NeutralConversationHistory = [
				{
					role: "system",
					content: [{ type: "text", text: "You are a helpful assistant." }],
				},
				{
					role: "user",
					content: [{ type: "text", text: "What is the capital of France?" }],
				},
			]

			expect(convertToNeutralHistoryFromOllama(ollamaHistory)).toEqual(expected)
		})

		// Test case 3: User + Assistant + User messages
		test("should convert multi-turn conversations", () => {
			const ollamaHistory: OpenAI.Chat.ChatCompletionMessageParam[] = [
				{ role: "user", content: "What is the capital of France?" },
				{ role: "assistant", content: "The capital of France is Paris." },
				{ role: "user", content: "What is its population?" },
			]

			const expected: NeutralConversationHistory = [
				{
					role: "user",
					content: [{ type: "text", text: "What is the capital of France?" }],
				},
				{
					role: "assistant",
					content: [{ type: "text", text: "The capital of France is Paris." }],
				},
				{
					role: "user",
					content: [{ type: "text", text: "What is its population?" }],
				},
			]

			expect(convertToNeutralHistoryFromOllama(ollamaHistory)).toEqual(expected)
		})

		// Test case 4: Array content (unlikely but should be handled)
		test("should handle array content", () => {
			const ollamaHistory: OpenAI.Chat.ChatCompletionMessageParam[] = [
				{
					role: "user",
					// @ts-expect-error Testing invalid content format
					content: ["Hello", "World"], // This is not valid OpenAI format but we should handle it
				},
			]

			const result = convertToNeutralHistoryFromOllama(ollamaHistory)

			// Should convert each string to a text block
			assert.strictEqual(result[0].role, "user")
			expect(Array.isArray(result[0].content)).toBe(true)

			const content = result[0].content as NeutralTextContentBlock[]
			assert.strictEqual(content.length, 2)
			assert.strictEqual(content[0].type, "text")
			assert.strictEqual(content[0].text, "Hello")
			assert.strictEqual(content[1].type, "text")
			assert.strictEqual(content[1].text, "World")
		})

		// Test case 5: Unknown role
		test("should handle unknown roles", () => {
			// Mock console.warn
			const originalWarn = console.warn
			console.warn = sinon.stub()

			const ollamaHistory: OpenAI.Chat.ChatCompletionMessageParam[] = [
				// @ts-expect-error Testing invalid function message - missing name property
				{
					role: "function",
					content: "Function result",
				},
			]

			const result = convertToNeutralHistoryFromOllama(ollamaHistory)

			// Should default to user role
			assert.strictEqual(result[0].role, "user")
			assert.ok(console.warn.called)

			// Restore console.warn
			console.warn = originalWarn
		})

		// Test case 6: Empty content
		test("should handle empty content", () => {
			const ollamaHistory: OpenAI.Chat.ChatCompletionMessageParam[] = [{ role: "user", content: "" }]

			const expected: NeutralConversationHistory = [
				{
					role: "user",
					content: [{ type: "text", text: "" }],
				},
			]

			expect(convertToNeutralHistoryFromOllama(ollamaHistory)).toEqual(expected)
		})

		// Test case 7: Non-string, non-array content (should be stringified)
		test("should handle non-string, non-array content by stringifying", () => {
			const ollamaHistory: OpenAI.Chat.ChatCompletionMessageParam[] = [
				{
					role: "user",
					// @ts-expect-error Testing invalid content format
					content: { foo: "bar" }, // This is not valid OpenAI format but we should handle it
				},
			]

			const result = convertToNeutralHistoryFromOllama(ollamaHistory)

			// Should stringify the object
			assert.strictEqual(result[0].role, "user")
			expect(Array.isArray(result[0].content)).toBe(true)

			const content = result[0].content as NeutralTextContentBlock[]
			assert.strictEqual(content.length, 1)
			assert.strictEqual(content[0].type, "text")
			assert.strictEqual(content[0].text, JSON.stringify({ foo: "bar" }))
		})
	})
})
