import * as assert from 'assert'
import * as sinon from 'sinon'
import type { NeutralConversationHistory } from "../../../shared/neutral-history"
import OpenAI from "openai"

import { convertToOpenAiHistory } from "../neutral-openai-format"

suite("convertToOpenAiHistory", () => {
	test("should convert simple text messages", () => {
		const anthropicMessages: NeutralConversationHistory = [
			{
				role: "user",
				content: "Hello",
			},
			{
				role: "assistant",
				content: "Hi there!",
			},
		]

		const openAiMessages = convertToOpenAiHistory(anthropicMessages)
		assert.strictEqual(openAiMessages.length, 2)
		assert.deepStrictEqual(openAiMessages[0], {
			role: "user",
			content: "Hello",
		})
		assert.deepStrictEqual(openAiMessages[1], {
			role: "assistant",
			content: "Hi there!",
		})
	})

	test("should handle messages with image content", () => {
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
		]

		const openAiMessages = convertToOpenAiHistory(anthropicMessages)
		assert.strictEqual(openAiMessages.length, 1)
		assert.strictEqual(openAiMessages[0].role, "user")

		const content = openAiMessages[0].content as Array<{
			type: string
			text?: string
			image_url?: { url: string }
		}>

		expect(Array.isArray(content)).toBe(true)
		assert.strictEqual(content.length, 2)
		assert.deepStrictEqual(content[0], { type: "text", text: "What is in this image?" })
		assert.deepStrictEqual(content[1], {
			type: "image_url",
			image_url: { url: "data:image/jpeg;base64,base64data" },
		})
	})

	test("should handle assistant messages with tool use", () => {
		const anthropicMessages: NeutralConversationHistory = [
			{
				role: "assistant",
				content: [
					{
						type: "text",
						text: "Let me check the weather.",
					},
					{
						type: "tool_use",
						id: "weather-123",
						name: "get_weather",
						input: { city: "London" },
					},
				],
			},
		]

		const openAiMessages = convertToOpenAiHistory(anthropicMessages)
		assert.strictEqual(openAiMessages.length, 1)

		const assistantMessage = openAiMessages[0] as OpenAI.Chat.ChatCompletionAssistantMessageParam
		assert.strictEqual(assistantMessage.role, "assistant")
		assert.strictEqual(assistantMessage.content, "Let me check the weather.")
		assert.strictEqual(assistantMessage.tool_calls.length, 1)
		assert.deepStrictEqual(assistantMessage.tool_calls![0], {
			id: "weather-123",
			type: "function",
			function: {
				name: "get_weather",
				arguments: JSON.stringify({ city: "London" }),
			},
		})
	})

	test("should handle user messages with tool results", () => {
		const anthropicMessages: NeutralConversationHistory = [
			{
				role: "user",
				content: [
					{
						type: "tool_result",
						tool_use_id: "weather-123",
						content: [
							{
								type: "text",
								text: "Current temperature in London: 20°C",
							},
						],
					},
				],
			},
		]

		const openAiMessages = convertToOpenAiHistory(anthropicMessages)
		assert.strictEqual(openAiMessages.length, 1)

		const toolMessage = openAiMessages[0] as OpenAI.Chat.ChatCompletionToolMessageParam
		assert.strictEqual(toolMessage.role, "tool")
		assert.strictEqual(toolMessage.tool_call_id, "weather-123")
		assert.strictEqual(toolMessage.content, "Current temperature in London: 20°C")
	})
})
