import { convertToR1Format } from "../r1-format"
import type { NeutralConversationHistory } from "../../../shared/neutral-history"
import OpenAI from "openai"
import * as assert from 'assert'
import * as sinon from 'sinon'

suite("convertToR1Format", () => {
	test("should convert basic text messages", () => {
		const input: NeutralConversationHistory = [
			{ role: "user", content: "Hello" },
			{ role: "assistant", content: "Hi there" },
		]

		const expected: OpenAI.Chat.ChatCompletionMessageParam[] = [
			{ role: "user", content: "Hello" },
			{ role: "assistant", content: "Hi there" },
		]

		expect(convertToR1Format(input)).toEqual(expected)
	})

	test("should merge consecutive messages with same role", () => {
		const input: NeutralConversationHistory = [
			{ role: "user", content: "Hello" },
			{ role: "user", content: "How are you?" },
			{ role: "assistant", content: "Hi!" },
			{ role: "assistant", content: "I'm doing well" },
		]

		const expected: OpenAI.Chat.ChatCompletionMessageParam[] = [
			{ role: "user", content: "Hello\nHow are you?" },
			{ role: "assistant", content: "Hi!\nI'm doing well" },
		]

		expect(convertToR1Format(input)).toEqual(expected)
	})

	test("should handle image content", () => {
		const input: NeutralConversationHistory = [
			{
				role: "user",
				content: [
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

		const expected: OpenAI.Chat.ChatCompletionMessageParam[] = [
			{
				role: "user",
				content: [
					{
						type: "image_url",
						image_url: {
							url: "data:image/jpeg;base64,base64data",
						},
					},
				],
			},
		]

		expect(convertToR1Format(input)).toEqual(expected)
	})

	test("should handle mixed text and image content", () => {
		const input: NeutralConversationHistory = [
			{
				role: "user",
				content: [
					{ type: "text", text: "Check this image:" },
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

		const expected: OpenAI.Chat.ChatCompletionMessageParam[] = [
			{
				role: "user",
				content: [
					{ type: "text", text: "Check this image:" },
					{
						type: "image_url",
						image_url: {
							url: "data:image/jpeg;base64,base64data",
						},
					},
				],
			},
		]

		expect(convertToR1Format(input)).toEqual(expected)
	})

	test("should merge mixed content messages with same role", () => {
		const input: NeutralConversationHistory = [
			{
				role: "user",
				content: [
					{ type: "text", text: "First image:" },
					{
						type: "image",
						source: {
							type: "base64",
							media_type: "image/jpeg",
							data: "image1",
						},
					},
				],
			},
			{
				role: "user",
				content: [
					{ type: "text", text: "Second image:" },
					{
						type: "image",
						source: {
							type: "base64",
							media_type: "image/png",
							data: "image2",
						},
					},
				],
			},
		]

		const expected: OpenAI.Chat.ChatCompletionMessageParam[] = [
			{
				role: "user",
				content: [
					{ type: "text", text: "First image:" },
					{
						type: "image_url",
						image_url: {
							url: "data:image/jpeg;base64,image1",
						},
					},
					{ type: "text", text: "Second image:" },
					{
						type: "image_url",
						image_url: {
							url: "data:image/png;base64,image2",
						},
					},
				],
			},
		]

		expect(convertToR1Format(input)).toEqual(expected)
	})

	test("should handle empty messages array", () => {
		expect(convertToR1Format([])).toEqual([])
	})

	test("should handle messages with empty content", () => {
		const input: NeutralConversationHistory = [
			{ role: "user", content: "" },
			{ role: "assistant", content: "" },
		]

		const expected: OpenAI.Chat.ChatCompletionMessageParam[] = [
			{ role: "user", content: "" },
			{ role: "assistant", content: "" },
		]

		expect(convertToR1Format(input)).toEqual(expected)
	})

	test("preserves unknown parts as text when merging", () => {
		const history: NeutralConversationHistory = [
			{
				role: "user",
				content: [
					{ type: "text", text: "hello" },
					// @ts-expect-error testing unknown part preservation
					{ type: "unknown", foo: "bar" },
				],
			},
			{
				role: "user",
				content: [{ type: "text", text: "world" }],
			},
		]

		const result = convertToR1Format(history)
		assert.strictEqual(result.length, 1)
		const first = result[0]
		assert.strictEqual(first.role, "user")
		if (typeof first.content === "string") {
			assert.ok(first.content.includes("hello"))
			assert.ok(first.content.includes("world"))
			assert.ok(first.content.includes("foo")) // stringified unknown part
} else {
			type TextPart = { type: "text"; text: string }
			const contentArr = first.content as Array<TextPart | { type: string }>
			const textPart = contentArr.find((p): p is TextPart => p.type === "text")
			const text = textPart?.text ?? ""
			assert.ok(text.includes("hello"))
			assert.ok(text.includes("world"))
			assert.ok(text.includes("foo"))
		}
	})
// Mock cleanup
