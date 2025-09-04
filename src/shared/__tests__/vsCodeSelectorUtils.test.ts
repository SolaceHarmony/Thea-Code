import { strict as assert } from "node:assert"
import { stringifyVsCodeLmModelSelector, type LanguageModelChatSelectorLike } from "../vsCodeSelectorUtils"

describe("vsCodeSelectorUtils", () => {
	describe("stringifyVsCodeLmModelSelector", () => {
		it("should join all defined selector properties with separator", () => {
			const selector: LanguageModelChatSelectorLike = {
				vendor: "test-vendor",
				family: "test-family",
				version: "v1",
				id: "test-id",
			}

			const result = stringifyVsCodeLmModelSelector(selector)
			assert.equal(result, "test-vendor/test-family/v1/test-id")
		})

		it("should skip undefined properties", () => {
			const selector: LanguageModelChatSelectorLike = {
				vendor: "test-vendor",
				family: "test-family",
			}

			const result = stringifyVsCodeLmModelSelector(selector)
			assert.equal(result, "test-vendor/test-family")
		})

		it("should handle empty selector", () => {
			const selector: LanguageModelChatSelectorLike = {}

			const result = stringifyVsCodeLmModelSelector(selector)
			assert.equal(result, "")
		})

		it("should handle selector with only one property", () => {
			const selector: LanguageModelChatSelectorLike = {
				vendor: "test-vendor",
			}

			const result = stringifyVsCodeLmModelSelector(selector)
			assert.equal(result, "test-vendor")
	})
})
})
