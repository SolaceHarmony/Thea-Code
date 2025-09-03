// npx jest src/api/transform/__tests__/mistral-format.test.ts

import type { NeutralConversationHistory } from "../../../shared/neutral-history"

import { convertToMistralMessages } from "../neutral-mistral-format"

import * as assert from 'assert'
suite("convertToMistralMessages", () => {
	test("should convert simple text messages for user and assistant roles", () => {
		const anthropicMessages: NeutralConversationHistory = [
			{
				role: "user",
				content: "Hello",
			},
			{
				role: "assistant",
				content: "Hi there!",
			},

		const mistralMessages = convertToMistralMessages(anthropicMessages)
		assert.strictEqual(mistralMessages.length, 2)
		assert.deepStrictEqual(mistralMessages[0], {
			role: "user",
			content: "Hello",
		assert.deepStrictEqual(mistralMessages[1], {
			role: "assistant",
			content: "Hi there!",

	test("should handle user messages with image content", () => {
		const anthropicMessages: NeutralConversationHistory = [
			{
				role: "user",
				content: [
					{
						type: "text",
						text: "What is in this image?",
					},
					{
						type: "image",
						source: {
							type: "base64",
							media_type: "image/jpeg",
							data: "base64data",
						},
					},
				],
			},

		const mistralMessages = convertToMistralMessages(anthropicMessages)
		assert.strictEqual(mistralMessages.length, 1)
		assert.strictEqual(mistralMessages[0].role, "user")

		const content = mistralMessages[0].content as Array<{
			type: string
			text?: string
			imageUrl?: { url: string }
		}>

		expect(Array.isArray(content)).toBe(true)
		assert.strictEqual(content.length, 2)
		assert.deepStrictEqual(content[0], { type: "text", text: "What is in this image?" })
		assert.deepStrictEqual(content[1], {
			type: "image_url",
			imageUrl: { url: "data:image/jpeg;base64,base64data" },

	test("should handle user messages with only tool results", () => {
		const anthropicMessages: NeutralConversationHistory = [
			{
				role: "user",
				content: [
					{
						type: "tool_result",
						tool_use_id: "weather-123",
						content: [{ type: "text", text: "Current temperature in London: 20°C" }],
					},
				],
			},

		// Based on the implementation, tool results without accompanying text/image
		// don't generate any messages
		const mistralMessages = convertToMistralMessages(anthropicMessages)
		assert.strictEqual(mistralMessages.length, 0)

	test("should handle user messages with mixed content (text, image, and tool results)", () => {
		const anthropicMessages: NeutralConversationHistory = [
			{
				role: "user",
				content: [
					{
						type: "text",
						text: "Here's the weather data and an image:",
					},
					{
						type: "image",
						source: {
							type: "base64",
							media_type: "image/png",
							data: "imagedata123",
						},
					},
					{
						type: "tool_result",
						tool_use_id: "weather-123",
						content: [{ type: "text", text: "Current temperature in London: 20°C" }],
					},
				],
			},

		const mistralMessages = convertToMistralMessages(anthropicMessages)
		// Based on the implementation, only the text and image content is included
		// Tool results are not converted to separate messages
		assert.strictEqual(mistralMessages.length, 1)

		// Message should be the user message with text and image
		assert.strictEqual(mistralMessages[0].role, "user")
		const userContent = mistralMessages[0].content as Array<{
			type: string
			text?: string
			imageUrl?: { url: string }
		}>
		expect(Array.isArray(userContent)).toBe(true)
		assert.strictEqual(userContent.length, 2)
		assert.deepStrictEqual(userContent[0], { type: "text", text: "Here's the weather data and an image:" })
		assert.deepStrictEqual(userContent[1], {
			type: "image_url",
			imageUrl: { url: "data:image/png;base64,imagedata123" },

	test("should handle assistant messages with text content", () => {
		const anthropicMessages: NeutralConversationHistory = [
			{
				role: "assistant",
				content: [
					{
						type: "text",
						text: "I'll help you with that question.",
					},
				],
			},

		const mistralMessages = convertToMistralMessages(anthropicMessages)
		assert.strictEqual(mistralMessages.length, 1)
		assert.strictEqual(mistralMessages[0].role, "assistant")
		assert.strictEqual(mistralMessages[0].content, "I'll help you with that question.")

	test("should handle assistant messages with tool use", () => {
		const anthropicMessages: NeutralConversationHistory = [
			{
				role: "assistant",
				content: [
					{
						type: "text",
						text: "Let me check the weather for you.",
					},
					{
						type: "tool_use",
						id: "weather-123",
						name: "get_weather",
						input: { city: "London" },
					},
				],
			},

		const mistralMessages = convertToMistralMessages(anthropicMessages)
		assert.strictEqual(mistralMessages.length, 1)
		assert.strictEqual(mistralMessages[0].role, "assistant")
		assert.strictEqual(mistralMessages[0].content, "Let me check the weather for you.")

	test("should handle multiple text blocks in assistant messages", () => {
		const anthropicMessages: NeutralConversationHistory = [
			{
				role: "assistant",
				content: [
					{
						type: "text",
						text: "First paragraph of information.",
					},
					{
						type: "text",
						text: "Second paragraph with more details.",
					},
				],
			},

		const mistralMessages = convertToMistralMessages(anthropicMessages)
		assert.strictEqual(mistralMessages.length, 1)
		assert.strictEqual(mistralMessages[0].role, "assistant")
		assert.strictEqual(mistralMessages[0].content, "First paragraph of information.\nSecond paragraph with more details.")

	test("should handle a conversation with mixed message types", () => {
		const anthropicMessages: NeutralConversationHistory = [
			{
				role: "user",
				content: [
					{
						type: "text",
						text: "What's in this image?",
					},
					{
						type: "image",
						source: {
							type: "base64",
							media_type: "image/jpeg",
							data: "imagedata",
						},
					},
				],
			},
			{
				role: "assistant",
				content: [
					{
						type: "text",
						text: "This image shows a landscape with mountains.",
					},
					{
						type: "tool_use",
						id: "search-123",
						name: "search_info",
						input: { query: "mountain types" },
					},
				],
			},
			{
				role: "user",
				content: [
					{
						type: "tool_result",
						tool_use_id: "search-123",
						content: [{ type: "text", text: "Found information about different mountain types." }],
					},
				],
			},
			{
				role: "assistant",
				content: "Based on the search results, I can tell you more about the mountains in the image.",
			},

		const mistralMessages = convertToMistralMessages(anthropicMessages)
		// Based on the implementation, user messages with only tool results don't generate messages
		assert.strictEqual(mistralMessages.length, 3)

		// User message with image
		assert.strictEqual(mistralMessages[0].role, "user")
		const userContent = mistralMessages[0].content as Array<{
			type: string
			text?: string
			imageUrl?: { url: string }
		}>
		expect(Array.isArray(userContent)).toBe(true)
		assert.strictEqual(userContent.length, 2)

		// Assistant message with text (tool_use is not included in Mistral format)
		assert.strictEqual(mistralMessages[1].role, "assistant")
		assert.strictEqual(mistralMessages[1].content, "This image shows a landscape with mountains.")

		// Final assistant message
		assert.deepStrictEqual(mistralMessages[2], {
			role: "assistant",
			content: "Based on the search results, I can tell you more about the mountains in the image.",

	test("should handle empty content in assistant messages", () => {
		const anthropicMessages: NeutralConversationHistory = [
			{
				role: "assistant",
				content: [
					{
						type: "tool_use",
						id: "search-123",
						name: "search_info",
						input: { query: "test query" },
					},
				],
			},

		const mistralMessages = convertToMistralMessages(anthropicMessages)
		assert.strictEqual(mistralMessages.length, 1)
		assert.strictEqual(mistralMessages[0].role, "assistant")
		assert.strictEqual(mistralMessages[0].content, undefined)
