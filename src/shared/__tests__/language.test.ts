// npx jest src/shared/__tests__/language.test.ts
import { strict as assert } from "node:assert"
import { formatLanguage } from "../language"

describe("formatLanguage", () => {
	it("should uppercase region code in locale string", () => {
		assert.equal(formatLanguage("pt-br"), "pt-BR")
		assert.equal(formatLanguage("zh-cn"), "zh-CN")
	})

	it("should return original string if no region code present", () => {
		assert.equal(formatLanguage("en"), "en")
		assert.equal(formatLanguage("fr"), "fr")
	})

	it("should handle empty or undefined input", () => {
		assert.equal(formatLanguage(""), "en")
		assert.equal(formatLanguage(undefined as unknown as string), "en")
})
})
