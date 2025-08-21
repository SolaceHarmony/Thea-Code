import { ApiStreamChunk } from "../stream"
import * as assert from 'assert'
import * as sinon from 'sinon'

suite("API Stream Types", () => {
	suite("ApiStreamChunk", () => {
		test("should correctly handle text chunks", () => {
			const textChunk: ApiStreamChunk = {
				type: "text",
				text: "Hello world",
			}

			assert.strictEqual(textChunk.type, "text")
			assert.strictEqual(textChunk.text, "Hello world")
		})

		test("should correctly handle usage chunks with cache information", () => {
			const usageChunk: ApiStreamChunk = {
				type: "usage",
				inputTokens: 100,
				outputTokens: 50,
				cacheWriteTokens: 20,
				cacheReadTokens: 10,
			}

			assert.strictEqual(usageChunk.type, "usage")
			assert.strictEqual(usageChunk.inputTokens, 100)
			assert.strictEqual(usageChunk.outputTokens, 50)
			assert.strictEqual(usageChunk.cacheWriteTokens, 20)
			assert.strictEqual(usageChunk.cacheReadTokens, 10)
		})

		test("should handle usage chunks without cache tokens", () => {
			const usageChunk: ApiStreamChunk = {
				type: "usage",
				inputTokens: 100,
				outputTokens: 50,
			}

			assert.strictEqual(usageChunk.type, "usage")
			assert.strictEqual(usageChunk.inputTokens, 100)
			assert.strictEqual(usageChunk.outputTokens, 50)
			assert.strictEqual(usageChunk.cacheWriteTokens, undefined)
			assert.strictEqual(usageChunk.cacheReadTokens, undefined)
		})

		test("should handle text chunks with empty strings", () => {
			const emptyTextChunk: ApiStreamChunk = {
				type: "text",
				text: "",
			}

			assert.strictEqual(emptyTextChunk.type, "text")
			assert.strictEqual(emptyTextChunk.text, "")
		})

		test("should handle usage chunks with zero tokens", () => {
			const zeroUsageChunk: ApiStreamChunk = {
				type: "usage",
				inputTokens: 0,
				outputTokens: 0,
			}

			assert.strictEqual(zeroUsageChunk.type, "usage")
			assert.strictEqual(zeroUsageChunk.inputTokens, 0)
			assert.strictEqual(zeroUsageChunk.outputTokens, 0)
		})

		test("should handle usage chunks with large token counts", () => {
			const largeUsageChunk: ApiStreamChunk = {
				type: "usage",
				inputTokens: 1000000,
				outputTokens: 500000,
				cacheWriteTokens: 200000,
				cacheReadTokens: 100000,
			}

			assert.strictEqual(largeUsageChunk.type, "usage")
			assert.strictEqual(largeUsageChunk.inputTokens, 1000000)
			assert.strictEqual(largeUsageChunk.outputTokens, 500000)
			assert.strictEqual(largeUsageChunk.cacheWriteTokens, 200000)
			assert.strictEqual(largeUsageChunk.cacheReadTokens, 100000)
		})

		test("should handle text chunks with special characters", () => {
			const specialCharsChunk: ApiStreamChunk = {
				type: "text",
				text: "!@#$%^&*()_+-=[]{}|;:,.<>?`~",
			}

			assert.strictEqual(specialCharsChunk.type, "text")
			assert.strictEqual(specialCharsChunk.text, "!@#$%^&*()_+-=[]{}|;:,.<>?`~")
		})

		test("should handle text chunks with unicode characters", () => {
			const unicodeChunk: ApiStreamChunk = {
				type: "text",
				text: "你好世界👋🌍",
			}

			assert.strictEqual(unicodeChunk.type, "text")
			assert.strictEqual(unicodeChunk.text, "你好世界👋🌍")
		})

		test("should handle text chunks with multiline content", () => {
			const multilineChunk: ApiStreamChunk = {
				type: "text",
				text: "Line 1\nLine 2\nLine 3",
			}

			assert.strictEqual(multilineChunk.type, "text")
			assert.strictEqual(multilineChunk.text, "Line 1\nLine 2\nLine 3")
			expect(multilineChunk.text.split("\n")).toHaveLength(3)
		})
	})
})
