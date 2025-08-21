import * as assert from 'assert'
import * as sinon from 'sinon'
import { convertToBedrockConverseMessages } from "../bedrock-converse-format"
import type { NeutralConversationHistory } from "../../../shared/neutral-history"
import { ToolResultContentBlock } from "@aws-sdk/client-bedrock-runtime"

suite("convertToBedrockConverseMessages", () => {
	test("converts simple text messages correctly", () => {
		const messages: NeutralConversationHistory = [
			{ role: "user", content: "Hello" },
			{ role: "assistant", content: "Hi there" },
		]

		const result = convertToBedrockConverseMessages(messages)

		assert.deepStrictEqual(result, [
			{
				role: "user",
				content: [{ text: "Hello" }],
			},
			{
				role: "assistant",
				content: [{ text: "Hi there" }],
			},
		])
	})

	test("converts messages with images correctly", () => {
		const messages: NeutralConversationHistory = [
			{
				role: "user",
				content: [
					{
						type: "text",
						text: "Look at this image:",
					},
					{
						type: "image",
						source: {
							type: "base64",
							data: "SGVsbG8=", // "Hello" in base64
							media_type: "image/jpeg" as const,
						},
					},
				],
			},
		]

		const result = convertToBedrockConverseMessages(messages)

		if (!result[0] || !result[0].content) {
			fail("Expected result to have content")
			return
		}

		assert.strictEqual(result[0].role, "user")
		assert.strictEqual(result[0].content.length, 2)
		assert.deepStrictEqual(result[0].content[0], { text: "Look at this image:" })

		const imageBlock = result[0].content[1]
		if ("image" in imageBlock && imageBlock.image && imageBlock.image.source) {
			assert.strictEqual(imageBlock.image.format, "jpeg")
			assert.notStrictEqual(imageBlock.image.source, undefined)
			assert.notStrictEqual(imageBlock.image.source.bytes, undefined)
		} else {
			fail("Expected image block not found")
		}
	})

	test("converts tool use messages correctly", () => {
		const messages: NeutralConversationHistory = [
			{
				role: "assistant",
				content: [
					{
						type: "tool_use",
						id: "test-id",
						name: "read_file",
						input: {
							path: "test.txt",
						},
					},
				],
			},
		]

		const result = convertToBedrockConverseMessages(messages)

		if (!result[0] || !result[0].content) {
			fail("Expected result to have content")
			return
		}

		assert.strictEqual(result[0].role, "assistant")
		const toolBlock = result[0].content[0]
		if ("toolUse" in toolBlock && toolBlock.toolUse) {
			assert.deepStrictEqual(toolBlock.toolUse, {
				toolUseId: "test-id",
				name: "read_file",
				input: "<read_file>\n<path>\ntest.txt\n</path>\n</read_file>",
			})
		} else {
			fail("Expected tool use block not found")
		}
	})

	test("converts tool result messages correctly", () => {
		const messages: NeutralConversationHistory = [
			{
				role: "assistant",
				content: [
					{
						type: "tool_result",
						tool_use_id: "test-id",
						content: [{ type: "text", text: "File contents here" }],
					},
				],
			},
		]

		const result = convertToBedrockConverseMessages(messages)

		if (!result[0] || !result[0].content) {
			fail("Expected result to have content")
			return
		}

		assert.strictEqual(result[0].role, "assistant")
		const resultBlock = result[0].content[0]
		if ("toolResult" in resultBlock && resultBlock.toolResult) {
			const expectedContent: ToolResultContentBlock[] = [{ text: "File contents here" }]
			assert.deepStrictEqual(resultBlock.toolResult, {
				toolUseId: "test-id",
				content: expectedContent,
				status: "success",
			})
		} else {
			fail("Expected tool result block not found")
		}
	})

	test("handles text content correctly", () => {
		const messages: NeutralConversationHistory = [
			{
				role: "user",
				content: [
					{
						type: "text",
						text: "Hello world",
					},
				],
			},
		]

		const result = convertToBedrockConverseMessages(messages)

		if (!result[0] || !result[0].content) {
			fail("Expected result to have content")
			return
		}

		assert.strictEqual(result[0].role, "user")
		assert.strictEqual(result[0].content.length, 1)
		const textBlock = result[0].content[0]
		assert.deepStrictEqual(textBlock, { text: "Hello world" })
	})
})
