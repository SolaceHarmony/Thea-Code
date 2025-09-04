import { strict as assert } from "node:assert"
import { supportPrompt } from "../support-prompt"

describe("Code Action Prompts", () => {
	const testFilePath = "test/file.ts"
	const testCode = "function test() { return true; }"

	describe("EXPLAIN action", () => {
		it("should format explain prompt correctly", () => {
			const prompt = supportPrompt.create("EXPLAIN", {
				filePath: testFilePath,
				selectedText: testCode,
			})
			assert.ok(prompt.includes(testFilePath))
			assert.ok(prompt.includes(testCode))
			assert.ok(prompt.includes("purpose and functionality"))
			assert.ok(prompt.includes("Key components"))
			assert.ok(prompt.includes("Important patterns"))
		})
	})

	describe("FIX action", () => {
		it("should format fix prompt without diagnostics", () => {
			const prompt = supportPrompt.create("FIX", {
				filePath: testFilePath,
				selectedText: testCode,
			})
			assert.ok(prompt.includes(testFilePath))
			assert.ok(prompt.includes(testCode))
			assert.ok(prompt.includes("Address all detected problems"))
			assert.ok(!prompt.includes("Current problems detected"))
		})

		it("should format fix prompt with diagnostics", () => {
			const diagnostics = [
				{
					source: "eslint",
					message: "Missing semicolon",
					code: "semi",
				},
				{
					message: "Unused variable",
				},
			]

			const prompt = supportPrompt.create("FIX", {
				filePath: testFilePath,
				selectedText: testCode,
				diagnostics,
			})

			assert.ok(prompt.includes("Current problems detected:"))
			assert.ok(prompt.includes("[eslint] Missing semicolon (semi)"))
			assert.ok(prompt.includes("[Error] Unused variable"))
			assert.ok(prompt.includes(testCode))
		})
	})

	describe("IMPROVE action", () => {
		it("should format improve prompt correctly", () => {
			const prompt = supportPrompt.create("IMPROVE", {
				filePath: testFilePath,
				selectedText: testCode,
			})
			assert.ok(prompt.includes(testFilePath))
			assert.ok(prompt.includes(testCode))
			assert.ok(prompt.includes("Code readability"))
			assert.ok(prompt.includes("Performance optimization"))
			assert.ok(prompt.includes("Best practices"))
			assert.ok(prompt.includes("Error handling"))
		})
	})

	describe("ENHANCE action", () => {
		it("should format enhance prompt correctly", () => {
			const prompt = supportPrompt.create("ENHANCE", {
				userInput: "test",
			})

				assert.equal(
					prompt,
					"Generate an enhanced version of this prompt (reply with only the enhanced prompt - no conversation, explanations, lead-in, bullet points, placeholders, or surrounding quotes):\n\ntest",
				)
			// Verify it ignores parameters since ENHANCE template doesn't use any
			assert.ok(!prompt.includes(testFilePath))
			assert.ok(!prompt.includes(testCode))
		})
	})

	describe("get template", () => {
		it("should return default template when no custom prompts provided", () => {
			const template = supportPrompt.get(undefined, "EXPLAIN")
			assert.equal(template, supportPrompt.default.EXPLAIN)
		})

		it("should return custom template when provided", () => {
			const customTemplate = "Custom template for explaining code"
			const customSupportPrompts = {
				EXPLAIN: customTemplate,
			}
			const template = supportPrompt.get(customSupportPrompts, "EXPLAIN")
			assert.equal(template, customTemplate)
		})

		it("should return default template when custom prompts does not include type", () => {
			const customSupportPrompts = {
				SOMETHING_ELSE: "Other template",
			}
			const template = supportPrompt.get(customSupportPrompts, "EXPLAIN")
			assert.equal(template, supportPrompt.default.EXPLAIN)
		})
	})

	describe("create with custom prompts", () => {
		it("should use custom template when provided", () => {
			const customTemplate = "Custom template for ${filePath}"
			const customSupportPrompts = {
				EXPLAIN: customTemplate,
			}

			const prompt = supportPrompt.create(
				"EXPLAIN",
				{
					filePath: testFilePath,
					selectedText: testCode,
				},
				customSupportPrompts,
			)

			assert.ok(prompt.includes(`Custom template for ${testFilePath}`))
			assert.ok(!prompt.includes("purpose and functionality"))
		})

		it("should use default template when custom prompts does not include type", () => {
			const customSupportPrompts = {
				EXPLAIN: "Other template",
			}

			const prompt = supportPrompt.create(
				"EXPLAIN",
				{
					filePath: testFilePath,
					selectedText: testCode,
				},
				customSupportPrompts,
			)

			assert.ok(prompt.includes("Other template"))
		})
	})
})
