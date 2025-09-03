// npx jest src/shared/__tests__/language.test.ts

import { formatLanguage } from "../language"

import * as assert from 'assert'
suite("formatLanguage", () => {
	test("should uppercase region code in locale string", () => {
		expect(formatLanguage("pt-br")).toBe("pt-BR")
		expect(formatLanguage("zh-cn")).toBe("zh-CN")

	test("should return original string if no region code present", () => {
		expect(formatLanguage("en")).toBe("en")
		expect(formatLanguage("fr")).toBe("fr")

	test("should handle empty or undefined input", () => {
		expect(formatLanguage("")).toBe("en")
		expect(formatLanguage(undefined as unknown as string)).toBe("en")
