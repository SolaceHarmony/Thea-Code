import { supportPrompt } from "../support-prompt"
import * as vscode from "vscode"

import * as assert from 'assert'
import * as vscode from 'vscode'
suite("Code Action Prompts", () => {
	const testFilePath = "test/file.ts"
	const testCode = "function test() { return true; }"

	suite("EXPLAIN action", () => {
		test("should format explain prompt correctly", () => {
			const prompt = supportPrompt.create("EXPLAIN", {
				filePath: testFilePath,
				selectedText: testCode,
			assert.ok(prompt.includes(testFilePath))
			assert.ok(prompt.includes(testCode))
			assert.ok(prompt.includes("purpose and functionality"))
			assert.ok(prompt.includes("Key components"))
			assert.ok(prompt.includes("Important patterns"))

	suite("FIX action", () => {
		test("should format fix prompt without diagnostics", () => {
			const prompt = supportPrompt.create("FIX", {
				filePath: testFilePath,
				selectedText: testCode,
			assert.ok(prompt.includes(testFilePath))
			assert.ok(prompt.includes(testCode))
			assert.ok(prompt.includes("Address all detected problems"))
			assert.ok(!prompt.includes("Current problems detected"))

		test("should format fix prompt with diagnostics", () => {
			const diagnostics: vscode.Diagnostic[] = [
				{
					source: "eslint",
					message: "Missing semicolon",
					code: "semi",
					range: new vscode.Range(0, 0, 0, 0), // Add a dummy range
					severity: vscode.DiagnosticSeverity.Error, // Add a dummy severity
				} as vscode.Diagnostic,
				{
					message: "Unused variable",
					severity: vscode.DiagnosticSeverity.Warning, // Use vscode.DiagnosticSeverity
					range: new vscode.Range(0, 0, 0, 0), // Add a dummy range
				} as vscode.Diagnostic,

			const prompt = supportPrompt.create("FIX", {
				filePath: testFilePath,
				selectedText: testCode,
				diagnostics,

			assert.ok(prompt.includes("Current problems detected:"))
			assert.ok(prompt.includes("[eslint] Missing semicolon (semi))")
			assert.ok(prompt.includes("[Error] Unused variable"))
			assert.ok(prompt.includes(testCode))

	suite("IMPROVE action", () => {
		test("should format improve prompt correctly", () => {
			const prompt = supportPrompt.create("IMPROVE", {
				filePath: testFilePath,
				selectedText: testCode,
			assert.ok(prompt.includes(testFilePath))
			assert.ok(prompt.includes(testCode))
			assert.ok(prompt.includes("Code readability"))
			assert.ok(prompt.includes("Performance optimization"))
			assert.ok(prompt.includes("Best practices"))
			assert.ok(prompt.includes("Error handling"))

	suite("ENHANCE action", () => {
		test("should format enhance prompt correctly", () => {
			const prompt = supportPrompt.create("ENHANCE", {
				userInput: "test",

			assert.strictEqual(prompt, 
				"Generate an enhanced version of this prompt (reply with only the enhanced prompt - no conversation, explanations, lead-in, bullet points, placeholders, or surrounding quotes):\n\ntest",

			// Verify it ignores parameters since ENHANCE template doesn't use any
			assert.ok(!prompt.includes(testFilePath))
			assert.ok(!prompt.includes(testCode))

	suite("get template", () => {
		test("should return default template when no custom prompts provided", () => {
			const template = supportPrompt.get(undefined, "EXPLAIN")
			assert.strictEqual(template, supportPrompt.default.EXPLAIN)

		test("should return custom template when provided", () => {
			const customTemplate = "Custom template for explaining code"
			const customSupportPrompts = {
				EXPLAIN: customTemplate,

			const template = supportPrompt.get(customSupportPrompts, "EXPLAIN")
			assert.strictEqual(template, customTemplate)

		test("should return default template when custom prompts does not include type", () => {
			const customSupportPrompts = {
				SOMETHING_ELSE: "Other template",

			const template = supportPrompt.get(customSupportPrompts, "EXPLAIN")
			assert.strictEqual(template, supportPrompt.default.EXPLAIN)

	suite("create with custom prompts", () => {
		test("should use custom template when provided", () => {
			const customTemplate = "Custom template for ${filePath}"
			const customSupportPrompts = {
				EXPLAIN: customTemplate,

			const prompt = supportPrompt.create(
				"EXPLAIN",
				{
					filePath: testFilePath,
					selectedText: testCode,
				},
				customSupportPrompts,

			assert.ok(prompt.includes(`Custom template for ${testFilePath}`))
			assert.ok(!prompt.includes("purpose and functionality"))

		test("should use default template when custom prompts does not include type", () => {
			const customSupportPrompts = {
				EXPLAIN: "Other template",

			const prompt = supportPrompt.create(
				"EXPLAIN",
				{
					filePath: testFilePath,
					selectedText: testCode,
				},
				customSupportPrompts,

			assert.ok(prompt.includes("Other template"))
