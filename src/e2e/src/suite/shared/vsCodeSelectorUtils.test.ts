import { stringifyVsCodeLmModelSelector } from "../vsCodeSelectorUtils"
import type { LanguageModelChatSelector } from "vscode"

import * as assert from 'assert'
suite("vsCodeSelectorUtils", () => {
	suite("stringifyVsCodeLmModelSelector", () => {
		test("should join all defined selector properties with separator", () => {
			const selector: LanguageModelChatSelector = {
				vendor: "test-vendor",
				family: "test-family",
				version: "v1",
				id: "test-id",

			const result = stringifyVsCodeLmModelSelector(selector)
			assert.strictEqual(result, "test-vendor/test-family/v1/test-id")

		test("should skip undefined properties", () => {
			const selector: LanguageModelChatSelector = {
				vendor: "test-vendor",
				family: "test-family",

			const result = stringifyVsCodeLmModelSelector(selector)
			assert.strictEqual(result, "test-vendor/test-family")

		test("should handle empty selector", () => {
			const selector: LanguageModelChatSelector = {}

			const result = stringifyVsCodeLmModelSelector(selector)
			assert.strictEqual(result, "")

		test("should handle selector with only one property", () => {
			const selector: LanguageModelChatSelector = {
				vendor: "test-vendor",

			const result = stringifyVsCodeLmModelSelector(selector)
			assert.strictEqual(result, "test-vendor")
