import * as assert from 'assert'
import { expect } from 'chai'
import * as sinon from 'sinon'
import type { NeutralConversationHistory, NeutralMessageContent } from "../../../shared/neutral-history"

import { BaseProvider } from "../../../api/providers/base-provider"
import { ModelInfo } from "../../../shared/api"
import { TOKEN_BUFFER_PERCENTAGE } from "../../mentions/index"
import { ApiStream } from "../../../api/transform/stream"
import { estimateTokenCount, truncateConversation, truncateConversationIfNeeded } from "../../mentions/index"

// Create a mock ApiHandler for testing
class MockApiHandler extends BaseProvider {
	// Override the registerTools method to do nothing
	// This prevents the "McpToolExecutor not initialized" error
	protected registerTools(): void {
		// No-op - we don't need to register tools for these tests
	}
	
	createMessage(): ApiStream {
		throw new Error("Method not implemented.")
	}

	getModel(): { id: string; info: ModelInfo } {
// Mock removed - needs manual implementation,
// 		}
	}
}

// Create a singleton instance for tests
const mockApiHandler = new MockApiHandler()

/**
 * Tests for the truncateConversation function
 */
suite("truncateConversation", () => {
	test("should retain the first message", () => {
		const messages: NeutralConversationHistory = [
			{ role: "user", content: "First message" },
			{ role: "assistant", content: "Second message" },
			{ role: "user", content: "Third message" },
		]

		const result = truncateConversation(messages, 0.5)

		// With 2 messages after the first, 0.5 fraction means remove 1 message
		// But 1 is odd, so it rounds down to 0 (to make it even)
		assert.strictEqual(result.length, 3) // First message + 2 remaining messages
		assert.deepStrictEqual(result[0], messages[0])
		assert.deepStrictEqual(result[1], messages[1])
		assert.deepStrictEqual(result[2], messages[2])
	})

	test("should remove the specified fraction of messages (rounded to even number)", () => {
		const messages: NeutralConversationHistory = [
			{ role: "user", content: "First message" },
			{ role: "assistant", content: "Second message" },
			{ role: "user", content: "Third message" },
			{ role: "assistant", content: "Fourth message" },
			{ role: "user", content: "Fifth message" },
		]

		// 4 messages excluding first, 0.5 fraction = 2 messages to remove
		// 2 is already even, so no rounding needed
		const result = truncateConversation(messages, 0.5)

		assert.strictEqual(result.length, 3)
		assert.deepStrictEqual(result[0], messages[0])
		assert.deepStrictEqual(result[1], messages[3])
		assert.deepStrictEqual(result[2], messages[4])
	})

	test("should round to an even number of messages to remove", () => {
		const messages: NeutralConversationHistory = [
			{ role: "user", content: "First message" },
			{ role: "assistant", content: "Second message" },
			{ role: "user", content: "Third message" },
			{ role: "assistant", content: "Fourth message" },
			{ role: "user", content: "Fifth message" },
			{ role: "assistant", content: "Sixth message" },
			{ role: "user", content: "Seventh message" },
		]

		// 6 messages excluding first, 0.3 fraction = 1.8 messages to remove
		// 1.8 rounds down to 1, then to 0 to make it even
		const result = truncateConversation(messages, 0.3)

		assert.strictEqual(result.length, 7) // No messages removed
		assert.deepStrictEqual(result, messages)
	})

	test("should handle edge case with fracToRemove = 0", () => {
		const messages: NeutralConversationHistory = [
			{ role: "user", content: "First message" },
			{ role: "assistant", content: "Second message" },
			{ role: "user", content: "Third message" },
		]

		const result = truncateConversation(messages, 0)

		assert.deepStrictEqual(result, messages)
	})

	test("should handle edge case with fracToRemove = 1", () => {
		const messages: NeutralConversationHistory = [
			{ role: "user", content: "First message" },
			{ role: "assistant", content: "Second message" },
			{ role: "user", content: "Third message" },
			{ role: "assistant", content: "Fourth message" },
		]

		// 3 messages excluding first, 1.0 fraction = 3 messages to remove
		// But 3 is odd, so it rounds down to 2 to make it even
		const result = truncateConversation(messages, 1)

		assert.strictEqual(result.length, 2)
		assert.deepStrictEqual(result[0], messages[0])
		assert.deepStrictEqual(result[1], messages[3])
	})
// Mock cleanup
/**
 * Tests for the estimateTokenCount function
 */
suite("estimateTokenCount", () => {
	test("should return 0 for empty or undefined content", async () => {
		expect(await estimateTokenCount([], mockApiHandler)).toBe(0)
		// @ts-expect-error - Testing with undefined
		expect(await estimateTokenCount(undefined, mockApiHandler)).toBe(0)
	})

	test("should estimate tokens for text blocks", async () => {
		const content: NeutralMessageContent = [{ type: "text", text: "This is a text block with 36 characters" }]

		// With tiktoken, the exact token count may differ from character-based estimation
		// Instead of expecting an exact number, we verify it's a reasonable positive number
		const result = await estimateTokenCount(content, mockApiHandler)
		assert.ok(result > 0)

		// We can also verify that longer text results in more tokens
		const longerContent: NeutralMessageContent = [
			{
				type: "text",
				text: "This is a longer text block with significantly more characters to encode into tokens",
			},
		]
		const longerResult = await estimateTokenCount(longerContent, mockApiHandler)
		assert.ok(longerResult > result)
	})

	test("should estimate tokens for image blocks based on data size", async () => {
		// Small image
		const smallImage: NeutralMessageContent = [
			{ type: "image", source: { type: "base64", media_type: "image/jpeg", data: "small_dummy_data" } },
		]
		// Larger image with more data
		const largerImage: NeutralMessageContent = [
			{ type: "image", source: { type: "base64", media_type: "image/png", data: "X".repeat(1000) } },
		]

		// Verify the token count scales with the size of the image data
		const smallImageTokens = await estimateTokenCount(smallImage, mockApiHandler)
		const largerImageTokens = await estimateTokenCount(largerImage, mockApiHandler)

		// Small image should have some tokens
		assert.ok(smallImageTokens > 0)

		// Larger image should have proportionally more tokens
		assert.ok(largerImageTokens > smallImageTokens)

		// Verify the larger image calculation matches our formula including the 50% fudge factor
		assert.strictEqual(largerImageTokens, 48)
	})

	test("should estimate tokens for mixed content blocks", async () => {
		const content: NeutralMessageContent = [
			{ type: "text", text: "A text block with 30 characters" },
			{ type: "image", source: { type: "base64", media_type: "image/jpeg", data: "dummy_data" } },
			{ type: "text", text: "Another text with 24 chars" },
		]

		// We know image tokens calculation should be consistent
		const imageTokens = Math.ceil(Math.sqrt("dummy_data".length)) * 1.5

		// With tiktoken, we can't predict exact text token counts,
		// but we can verify the total is greater than just the image tokens
		const result = await estimateTokenCount(content, mockApiHandler)
		assert.ok(result > imageTokens)

		// Also test against a version with only the image to verify text adds tokens
		const imageOnlyContent: NeutralMessageContent = [
			{ type: "image", source: { type: "base64", media_type: "image/jpeg", data: "dummy_data" } },
		]
		const imageOnlyResult = await estimateTokenCount(imageOnlyContent, mockApiHandler)
		assert.ok(result > imageOnlyResult)
	})

	test("should handle empty text blocks", async () => {
		const content: NeutralMessageContent = [{ type: "text", text: "" }]
		expect(await estimateTokenCount(content, mockApiHandler)).toBe(0)
	})

	test("should handle plain string messages", async () => {
		const content = "This is a plain text message"
		expect(await estimateTokenCount([{ type: "text", text: content }], mockApiHandler)).toBeGreaterThan(0)
	})
// Mock cleanup
/**
 * Tests for the truncateConversationIfNeeded function
 */
suite("truncateConversationIfNeeded", () => {
	const createModelInfo = (contextWindow: number, maxTokens?: number): ModelInfo => ({
		contextWindow,
		supportsPromptCache: true,
		maxTokens,
	})

	const messages: NeutralConversationHistory = [
		{ role: "user", content: "First message" },
		{ role: "assistant", content: "Second message" },
		{ role: "user", content: "Third message" },
		{ role: "assistant", content: "Fourth message" },
		{ role: "user", content: "Fifth message" },
	]

	test("should not truncate if tokens are below max tokens threshold", async () => {
		const modelInfo = createModelInfo(100000, 30000)
		// eslint-disable-next-line @typescript-eslint/no-unused-vars
		const maxTokens = 100000 - 30000 // 70000
		const dynamicBuffer = modelInfo.contextWindow * TOKEN_BUFFER_PERCENTAGE // 10000
		const totalTokens = 70000 - dynamicBuffer - 1 // Just below threshold - buffer

		// Create messages with very small content in the last one to avoid token overflow
		const messagesWithSmallContent = [...messages.slice(0, -1), { ...messages[messages.length - 1], content: "" }]

		const result = await truncateConversationIfNeeded({
			messages: messagesWithSmallContent,
			totalTokens,
			contextWindow: modelInfo.contextWindow,
			maxTokens: modelInfo.maxTokens,
			apiHandler: mockApiHandler,
		})
		assert.deepStrictEqual(result, messagesWithSmallContent) // No truncation occurs
	})

	test("should truncate if tokens are above max tokens threshold", async () => {
		const modelInfo = createModelInfo(100000, 30000)
		const totalTokens = 70001 // Above threshold

		// Create messages with very small content in the last one to avoid token overflow
		const messagesWithSmallContent = [...messages.slice(0, -1), { ...messages[messages.length - 1], content: "" }]

		// When truncating, always uses 0.5 fraction
		// With 4 messages after the first, 0.5 fraction means remove 2 messages
		const expectedResult = [messagesWithSmallContent[0], messagesWithSmallContent[3], messagesWithSmallContent[4]]

		const result = await truncateConversationIfNeeded({
			messages: messagesWithSmallContent,
			totalTokens,
			contextWindow: modelInfo.contextWindow,
			maxTokens: modelInfo.maxTokens,
			apiHandler: mockApiHandler,
		})
		assert.deepStrictEqual(result, expectedResult)
	})

	test("should work with non-prompt caching models the same as prompt caching models", async () => {
		// The implementation no longer differentiates between prompt caching and non-prompt caching models
		const modelInfo1 = createModelInfo(100000, 30000)
		const modelInfo2 = createModelInfo(100000, 30000)

		// Create messages with very small content in the last one to avoid token overflow
		const messagesWithSmallContent = [...messages.slice(0, -1), { ...messages[messages.length - 1], content: "" }]

		// Test below threshold
		const belowThreshold = 69999
		const result1 = await truncateConversationIfNeeded({
			messages: messagesWithSmallContent,
			totalTokens: belowThreshold,
			contextWindow: modelInfo1.contextWindow,
			maxTokens: modelInfo1.maxTokens,
			apiHandler: mockApiHandler,
		})

		const result2 = await truncateConversationIfNeeded({
			messages: messagesWithSmallContent,
			totalTokens: belowThreshold,
			contextWindow: modelInfo2.contextWindow,
			maxTokens: modelInfo2.maxTokens,
			apiHandler: mockApiHandler,
		})

		assert.deepStrictEqual(result1, result2)

		// Test above threshold
		const aboveThreshold = 70001
		const result3 = await truncateConversationIfNeeded({
			messages: messagesWithSmallContent,
			totalTokens: aboveThreshold,
			contextWindow: modelInfo1.contextWindow,
			maxTokens: modelInfo1.maxTokens,
			apiHandler: mockApiHandler,
		})

		const result4 = await truncateConversationIfNeeded({
			messages: messagesWithSmallContent,
			totalTokens: aboveThreshold,
			contextWindow: modelInfo2.contextWindow,
			maxTokens: modelInfo2.maxTokens,
			apiHandler: mockApiHandler,
		})

		assert.deepStrictEqual(result3, result4)
	})

	test("should consider incoming content when deciding to truncate", async () => {
		const modelInfo = createModelInfo(100000, 30000)
		const maxTokens = 30000
		const availableTokens = modelInfo.contextWindow - maxTokens

		// Test case 1: Small content that won't push us over the threshold
		const smallContent = [{ type: "text" as const, text: "Small content" }]
		const smallContentTokens = await estimateTokenCount(smallContent, mockApiHandler)
		const messagesWithSmallContent: NeutralConversationHistory = [
			...messages.slice(0, -1),
			{ role: messages[messages.length - 1].role, content: smallContent },
		]

		// Set base tokens so total is well below threshold + buffer even with small content added
		const dynamicBuffer = modelInfo.contextWindow * TOKEN_BUFFER_PERCENTAGE
		const baseTokensForSmall = availableTokens - smallContentTokens - dynamicBuffer - 10
		const resultWithSmall = await truncateConversationIfNeeded({
			messages: messagesWithSmallContent,
			totalTokens: baseTokensForSmall,
			contextWindow: modelInfo.contextWindow,
			maxTokens,
			apiHandler: mockApiHandler,
		})
		assert.deepStrictEqual(resultWithSmall, messagesWithSmallContent) // No truncation

		// Test case 2: Large content that will push us over the threshold
		const largeContent = [
			{
				type: "text" as const,
				text: "A very large incoming message that would consume a significant number of tokens and push us over the threshold",
			},
		]
		const largeContentTokens = await estimateTokenCount(largeContent, mockApiHandler)
		const messagesWithLargeContent: NeutralConversationHistory = [
			...messages.slice(0, -1),
			{ role: messages[messages.length - 1].role, content: largeContent },
		]

		// Set base tokens so we're just below threshold without content, but over with content
		const baseTokensForLarge = availableTokens - Math.floor(largeContentTokens / 2)
		const resultWithLarge = await truncateConversationIfNeeded({
			messages: messagesWithLargeContent,
			totalTokens: baseTokensForLarge,
			contextWindow: modelInfo.contextWindow,
			maxTokens,
			apiHandler: mockApiHandler,
		})
		assert.notDeepStrictEqual(resultWithLarge, messagesWithLargeContent) // Should truncate

		// Test case 3: Very large content that will definitely exceed threshold
		const veryLargeContent = [{ type: "text" as const, text: "X".repeat(1000) }]
		const veryLargeContentTokens = await estimateTokenCount(veryLargeContent, mockApiHandler)
		const messagesWithVeryLargeContent: NeutralConversationHistory = [
			...messages.slice(0, -1),
			{ role: messages[messages.length - 1].role, content: veryLargeContent },
		]

		// Set base tokens so we're just below threshold without content
		const baseTokensForVeryLarge = availableTokens - Math.floor(veryLargeContentTokens / 2)
		const resultWithVeryLarge = await truncateConversationIfNeeded({
			messages: messagesWithVeryLargeContent,
			totalTokens: baseTokensForVeryLarge,
			contextWindow: modelInfo.contextWindow,
			maxTokens,
			apiHandler: mockApiHandler,
		})
		assert.notDeepStrictEqual(resultWithVeryLarge, messagesWithVeryLargeContent) // Should truncate
	})

	test("should truncate if tokens are within TOKEN_BUFFER_PERCENTAGE of the threshold", async () => {
		const modelInfo = createModelInfo(100000, 30000)
		const dynamicBuffer = modelInfo.contextWindow * TOKEN_BUFFER_PERCENTAGE // 10% of 100000 = 10000
		const totalTokens = 70000 - dynamicBuffer + 1 // Just within the dynamic buffer of threshold (70000)

		// Create messages with very small content in the last one to avoid token overflow
		const messagesWithSmallContent = [...messages.slice(0, -1), { ...messages[messages.length - 1], content: "" }]

		// When truncating, always uses 0.5 fraction
		// With 4 messages after the first, 0.5 fraction means remove 2 messages
		const expectedResult = [messagesWithSmallContent[0], messagesWithSmallContent[3], messagesWithSmallContent[4]]

		const result = await truncateConversationIfNeeded({
			messages: messagesWithSmallContent,
			totalTokens,
			contextWindow: modelInfo.contextWindow,
			maxTokens: modelInfo.maxTokens,
			apiHandler: mockApiHandler,
		})
		assert.deepStrictEqual(result, expectedResult)
	})
// Mock cleanup
/**
 * Tests for the getMaxTokens function (private but tested through truncateConversationIfNeeded)
 */
suite("getMaxTokens", () => {
	// We'll test this indirectly through truncateConversationIfNeeded
	const createModelInfo = (contextWindow: number, maxTokens?: number): ModelInfo => ({
		contextWindow,
		supportsPromptCache: true, // Not relevant for getMaxTokens
		maxTokens,
	})

	// Reuse across tests for consistency
	const messages: NeutralConversationHistory = [
		{ role: "user", content: "First message" },
		{ role: "assistant", content: "Second message" },
		{ role: "user", content: "Third message" },
		{ role: "assistant", content: "Fourth message" },
		{ role: "user", content: "Fifth message" },
	]

	test("should use maxTokens as buffer when specified", async () => {
		const modelInfo = createModelInfo(100000, 50000)
		// Max tokens = 100000 - 50000 = 50000

		// Create messages with very small content in the last one to avoid token overflow
		const messagesWithSmallContent = [...messages.slice(0, -1), { ...messages[messages.length - 1], content: "" }]

		// Account for the dynamic buffer which is 10% of context window (10,000 tokens)
		// Below max tokens and buffer - no truncation
		const result1 = await truncateConversationIfNeeded({
			messages: messagesWithSmallContent,
			totalTokens: 39999, // Well below threshold + dynamic buffer
			contextWindow: modelInfo.contextWindow,
			maxTokens: modelInfo.maxTokens,
			apiHandler: mockApiHandler,
		})
		assert.deepStrictEqual(result1, messagesWithSmallContent)

		// Above max tokens - truncate
		const result2 = await truncateConversationIfNeeded({
			messages: messagesWithSmallContent,
			totalTokens: 50001, // Above threshold
			contextWindow: modelInfo.contextWindow,
			maxTokens: modelInfo.maxTokens,
			apiHandler: mockApiHandler,
		})
		assert.notDeepStrictEqual(result2, messagesWithSmallContent)
		assert.strictEqual(result2.length, 3) // Truncated with 0.5 fraction
	})

	test("should use 20% of context window as buffer when maxTokens is undefined", async () => {
		const modelInfo = createModelInfo(100000, undefined)
		// Max tokens = 100000 - (100000 * 0.2) = 80000

		// Create messages with very small content in the last one to avoid token overflow
		const messagesWithSmallContent = [...messages.slice(0, -1), { ...messages[messages.length - 1], content: "" }]

		// Account for the dynamic buffer which is 10% of context window (10,000 tokens)
		// Below max tokens and buffer - no truncation
		const result1 = await truncateConversationIfNeeded({
			messages: messagesWithSmallContent,
			totalTokens: 69999, // Well below threshold + dynamic buffer
			contextWindow: modelInfo.contextWindow,
			maxTokens: modelInfo.maxTokens,
			apiHandler: mockApiHandler,
		})
		assert.deepStrictEqual(result1, messagesWithSmallContent)

		// Above max tokens - truncate
		const result2 = await truncateConversationIfNeeded({
			messages: messagesWithSmallContent,
			totalTokens: 80001, // Above threshold
			contextWindow: modelInfo.contextWindow,
			maxTokens: modelInfo.maxTokens,
			apiHandler: mockApiHandler,
		})
		assert.notDeepStrictEqual(result2, messagesWithSmallContent)
		assert.strictEqual(result2.length, 3) // Truncated with 0.5 fraction
	})

	test("should handle small context windows appropriately", async () => {
		const modelInfo = createModelInfo(50000, 10000)
		// Max tokens = 50000 - 10000 = 40000

		// Create messages with very small content in the last one to avoid token overflow
		const messagesWithSmallContent = [...messages.slice(0, -1), { ...messages[messages.length - 1], content: "" }]

		// Below max tokens and buffer - no truncation
		const result1 = await truncateConversationIfNeeded({
			messages: messagesWithSmallContent,
			totalTokens: 34999, // Well below threshold + buffer
			contextWindow: modelInfo.contextWindow,
			maxTokens: modelInfo.maxTokens,
			apiHandler: mockApiHandler,
		})
		assert.deepStrictEqual(result1, messagesWithSmallContent)

		// Above max tokens - truncate
		const result2 = await truncateConversationIfNeeded({
			messages: messagesWithSmallContent,
			totalTokens: 40001, // Above threshold
			contextWindow: modelInfo.contextWindow,
			maxTokens: modelInfo.maxTokens,
			apiHandler: mockApiHandler,
		})
		assert.notDeepStrictEqual(result2, messagesWithSmallContent)
		assert.strictEqual(result2.length, 3) // Truncated with 0.5 fraction
	})

	test("should handle large context windows appropriately", async () => {
		const modelInfo = createModelInfo(200000, 30000)
		// Max tokens = 200000 - 30000 = 170000

		// Create messages with very small content in the last one to avoid token overflow
		const messagesWithSmallContent = [...messages.slice(0, -1), { ...messages[messages.length - 1], content: "" }]

		// Account for the dynamic buffer which is 10% of context window (20,000 tokens for this test)
		// Below max tokens and buffer - no truncation
		const result1 = await truncateConversationIfNeeded({
			messages: messagesWithSmallContent,
			totalTokens: 149999, // Well below threshold + dynamic buffer
			contextWindow: modelInfo.contextWindow,
			maxTokens: modelInfo.maxTokens,
			apiHandler: mockApiHandler,
		})
		assert.deepStrictEqual(result1, messagesWithSmallContent)

		// Above max tokens - truncate
		const result2 = await truncateConversationIfNeeded({
			messages: messagesWithSmallContent,
			totalTokens: 170001, // Above threshold
			contextWindow: modelInfo.contextWindow,
			maxTokens: modelInfo.maxTokens,
			apiHandler: mockApiHandler,
		})
		assert.notDeepStrictEqual(result2, messagesWithSmallContent)
		assert.strictEqual(result2.length, 3) // Truncated with 0.5 fraction
	})
// Mock cleanup
})
