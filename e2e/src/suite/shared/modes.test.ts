// Mock setup must come before imports
// TODO: Mock setup needs manual migration
import * as assert from 'assert'
import * as sinon from 'sinon'
const mockAddCustomInstructions = sinon.stub().resolves("Combined instructions")
// Mock needs manual implementation
// TODO: Implement proper mock with proxyquire

import { isToolAllowedForMode, FileRestrictionError, ModeConfig, getFullModeDetails, modes } from "../modes"
import { addCustomInstructions } from "../../core/prompts/sections/custom-instructions"
import { AI_IDENTITY_NAME } from "../../shared/config/thea-config"
suite("isToolAllowedForMode", () => {
	const customModes: ModeConfig[] = [
		{
			slug: "markdown-editor",
			name: "Markdown Editor",
			roleDefinition: "You are a markdown editor",
			groups: ["read", ["edit", { fileRegex: "\\.md$" }], "browser"],
		},
		{
			slug: "css-editor",
			name: "CSS Editor",
			roleDefinition: "You are a CSS editor",
			groups: ["read", ["edit", { fileRegex: "\\.css$" }], "browser"],
		},
		{
			slug: "test-exp-mode",
			name: "Test Exp Mode",
			roleDefinition: "You are an experimental tester",
			groups: ["read", "edit", "browser"],
		},

	test("allows always available tools", () => {
		expect(isToolAllowedForMode("ask_followup_question", "markdown-editor", customModes)).toBe(true)
		expect(isToolAllowedForMode("attempt_completion", "markdown-editor", customModes)).toBe(true)

	test("allows unrestricted tools", () => {
		expect(isToolAllowedForMode("read_file", "markdown-editor", customModes)).toBe(true)
		expect(isToolAllowedForMode("browser_action", "markdown-editor", customModes)).toBe(true)

	suite("file restrictions", () => {
		test("allows editing matching files", () => {
			// Test markdown editor mode
			const mdResult = isToolAllowedForMode("write_to_file", "markdown-editor", customModes, undefined, {
				path: "test.md",
				content: "# Test",

			assert.strictEqual(mdResult, true)

			// Test CSS editor mode
			const cssResult = isToolAllowedForMode("write_to_file", "css-editor", customModes, undefined, {
				path: "styles.css",
				content: ".test { color: red; }",

			assert.strictEqual(cssResult, true)

		test("rejects editing non-matching files", () => {
			// Test markdown editor mode with non-markdown file
			expect(() =>
				isToolAllowedForMode("write_to_file", "markdown-editor", customModes, undefined, {
					path: "test.js",
					content: "console.log('test')",
				}).toThrow(FileRestrictionError)
			expect(() =>
				isToolAllowedForMode("write_to_file", "markdown-editor", customModes, undefined, {
					path: "test.js",
					content: "console.log('test')",
				}).toThrow(/\\.md\$/)

			// Test CSS editor mode with non-CSS file
			expect(() =>
				isToolAllowedForMode("write_to_file", "css-editor", customModes, undefined, {
					path: "test.js",
					content: "console.log('test')",
				}).toThrow(FileRestrictionError)
			expect(() =>
				isToolAllowedForMode("write_to_file", "css-editor", customModes, undefined, {
					path: "test.js",
					content: "console.log('test')",
				}).toThrow(/\\.css\$/)

		test("handles partial streaming cases (path only, no content/diff)", () => {
			// Should allow path-only for matching files (no validation yet since content/diff not provided)
			assert.strictEqual(isToolAllowedForMode("write_to_file", "markdown-editor", customModes, undefined, {
					path: "test.js",
				}, true)

			assert.strictEqual(isToolAllowedForMode("apply_diff", "markdown-editor", customModes, undefined, {
					path: "test.js",
				}, true)

			// Should allow path-only for architect mode too
			assert.strictEqual(isToolAllowedForMode("write_to_file", "architect", [], undefined, {
					path: "test.js",
				}, true)

		test("applies restrictions to both write_to_file and apply_diff", () => {
			// Test write_to_file
			const writeResult = isToolAllowedForMode("write_to_file", "markdown-editor", customModes, undefined, {
				path: "test.md",
				content: "# Test",

			assert.strictEqual(writeResult, true)

			// Test apply_diff
			const diffResult = isToolAllowedForMode("apply_diff", "markdown-editor", customModes, undefined, {
				path: "test.md",
				diff: "- old\n+ new",

			assert.strictEqual(diffResult, true)

			// Test both with non-matching file
			expect(() =>
				isToolAllowedForMode("write_to_file", "markdown-editor", customModes, undefined, {
					path: "test.js",
					content: "console.log('test')",
				}).toThrow(FileRestrictionError)

			expect(() =>
				isToolAllowedForMode("apply_diff", "markdown-editor", customModes, undefined, {
					path: "test.js",
					diff: "- old\n+ new",
				}).toThrow(FileRestrictionError)

		test("uses description in file restriction error for custom modes", () => {
			const customModesWithDescription: ModeConfig[] = [
				{
					slug: "docs-editor",
					name: "Documentation Editor",
					roleDefinition: "You are a documentation editor",
					groups: [
						"read",
						["edit", { fileRegex: "\\.(md|txt)$", description: "Documentation files only" }],
						"browser",
					],
				},

			// Test write_to_file with non-matching file
			expect(() =>
				isToolAllowedForMode("write_to_file", "docs-editor", customModesWithDescription, undefined, {
					path: "test.js",
					content: "console.log('test')",
				}).toThrow(FileRestrictionError)
			expect(() =>
				isToolAllowedForMode("write_to_file", "docs-editor", customModesWithDescription, undefined, {
					path: "test.js",
					content: "console.log('test')",
				}).toThrow(/Documentation files only/)

			// Test apply_diff with non-matching file
			expect(() =>
				isToolAllowedForMode("apply_diff", "docs-editor", customModesWithDescription, undefined, {
					path: "test.js",
					diff: "- old\n+ new",
				}).toThrow(FileRestrictionError)
			expect(() =>
				isToolAllowedForMode("apply_diff", "docs-editor", customModesWithDescription, undefined, {
					path: "test.js",
					diff: "- old\n+ new",
				}).toThrow(/Documentation files only/)

			// Test that matching files are allowed
			assert.strictEqual(isToolAllowedForMode("write_to_file", "docs-editor", customModesWithDescription, undefined, {
					path: "test.md",
					content: "# Test",
				}, true)

			assert.strictEqual(isToolAllowedForMode("write_to_file", "docs-editor", customModesWithDescription, undefined, {
					path: "test.txt",
					content: "Test content",
				}, true)

			// Test partial streaming cases
			assert.strictEqual(isToolAllowedForMode("write_to_file", "docs-editor", customModesWithDescription, undefined, {
					path: "test.js",
				}, true)

		test("allows architect mode to edit markdown files only", () => {
			// Should allow editing markdown files
			assert.strictEqual(isToolAllowedForMode("write_to_file", "architect", [], undefined, {
					path: "test.md",
					content: "# Test",
				}, true)

			// Should allow applying diffs to markdown files
			assert.strictEqual(isToolAllowedForMode("apply_diff", "architect", [], undefined, {
					path: "readme.md",
					diff: "- old\n+ new",
				}, true)

			// Should reject non-markdown files
			expect(() =>
				isToolAllowedForMode("write_to_file", "architect", [], undefined, {
					path: "test.js",
					content: "console.log('test')",
				}).toThrow(FileRestrictionError)
			expect(() =>
				isToolAllowedForMode("write_to_file", "architect", [], undefined, {
					path: "test.js",
					content: "console.log('test')",
				}).toThrow(/Markdown files only/)

			// Should maintain read capabilities
			expect(isToolAllowedForMode("read_file", "architect", [])).toBe(true)
			expect(isToolAllowedForMode("browser_action", "architect", [])).toBe(true)
			expect(isToolAllowedForMode("use_mcp_tool", "architect", [])).toBe(true)

	test("handles non-existent modes", () => {
		expect(isToolAllowedForMode("write_to_file", "non-existent", customModes)).toBe(false)

	test("respects tool requirements", () => {
		const toolRequirements = {
			write_to_file: false,

		expect(isToolAllowedForMode("write_to_file", "markdown-editor", customModes, toolRequirements)).toBe(false)

	suite("experimental tools", () => {
		test("disables tools when experiment is disabled", () => {
			const experiments = {
				search_and_replace: false,
				insert_content: false,

			expect(
				isToolAllowedForMode(
					"search_and_replace",
					"test-exp-mode",
					customModes,
					undefined,
					undefined,
					experiments,
				),
			).toBe(false)

			expect(
				isToolAllowedForMode("insert_content", "test-exp-mode", customModes, undefined, undefined, experiments),
			).toBe(false)

		test("allows tools when experiment is enabled", () => {
			const experiments = {
				search_and_replace: true,
				insert_content: true,

			expect(
				isToolAllowedForMode(
					"search_and_replace",
					"test-exp-mode",
					customModes,
					undefined,
					undefined,
					experiments,
				),
			).toBe(true)

			expect(
				isToolAllowedForMode("insert_content", "test-exp-mode", customModes, undefined, undefined, experiments),
			).toBe(true)

		test("allows non-experimental tools when experiments are disabled", () => {
			const experiments = {
				search_and_replace: false,
				insert_content: false,

			expect(
				isToolAllowedForMode("read_file", "markdown-editor", customModes, undefined, undefined, experiments),
			).toBe(true)
			expect(
				isToolAllowedForMode(
					"write_to_file",
					"markdown-editor",
					customModes,
					undefined,
					{ path: "test.md" },
					experiments,
				),
			).toBe(true)

suite("FileRestrictionError", () => {
	test("formats error message with pattern when no description provided", () => {
		const error = new FileRestrictionError("Markdown Editor", "\\.md$", undefined, "test.js")
		assert.strictEqual(error.message, 
			"This mode (Markdown Editor) can only edit files matching pattern: \\.md$. Got: test.js",

		assert.strictEqual(error.name, "FileRestrictionError")

	suite("debug mode", () => {
		test("is configured correctly", () => {
			const debugMode = modes.find((mode) => mode.slug === "debug")
			assert.ok(debugMode !== undefined)
			expect(debugMode).toMatchObject({
				slug: "debug",
				name: "Debug",
				roleDefinition: `You are ${AI_IDENTITY_NAME}, an expert software debugger specializing in systematic problem diagnosis and resolution.`,
				groups: ["read", "edit", "browser", "command", "mcp"],

			assert.ok(debugMode?.customInstructions.includes(
				"Reflect on 5-7 different possible sources of the problem, distill those down to 1-2 most likely sources, and then add logs to validate your assumptions. Explicitly ask the user to confirm the diagnosis before fixing the problem.",

	suite("getFullModeDetails", () => {
		setup(() => {
			sinon.restore()
			;(addCustomInstructions as sinon.SinonStub).resolves("Combined instructions")

		test("returns base mode when no overrides exist", async () => {
			const result = await getFullModeDetails("debug")
			expect(result).toMatchObject({
				slug: "debug",
				name: "Debug",
				roleDefinition: `You are ${AI_IDENTITY_NAME}, an expert software debugger specializing in systematic problem diagnosis and resolution.`,

		test("applies custom mode overrides", async () => {
			const customModes: ModeConfig[] = [
				{
					slug: "debug",
					name: "Custom Debug",
					roleDefinition: "Custom debug role",
					groups: ["read"],
				},

			const result = await getFullModeDetails("debug", customModes)
			expect(result).toMatchObject({
				slug: "debug",
				name: "Custom Debug",
				roleDefinition: "Custom debug role",
				groups: ["read"],

		test("applies prompt component overrides", async () => {
			const customModePrompts = {
				debug: {
					roleDefinition: "Overridden role",
					customInstructions: "Overridden instructions",
				},

			const result = await getFullModeDetails("debug", undefined, customModePrompts)
			assert.strictEqual(result.roleDefinition, "Overridden role")
			assert.strictEqual(result.customInstructions, "Overridden instructions")

		test("combines custom instructions when cwd provided", async () => {
			const options = {
				cwd: "/test/path",
				globalCustomInstructions: "Global instructions",
				language: "en",

			await getFullModeDetails("debug", undefined, undefined, options)

			assert.ok(addCustomInstructions.calledWith(
				sinon.match.string),
				"Global instructions",
				"/test/path",
				"debug",
				{ language: "en" },

		test("falls back to first mode for non-existent mode", async () => {
			const result = await getFullModeDetails("non-existent")
			expect(result).toMatchObject({
				...modes[0],
				customInstructions: "",

	test("formats error message with description when provided", () => {
		const error = new FileRestrictionError("Markdown Editor", "\\.md$", "Markdown files only", "test.js")
		assert.strictEqual(error.message, 
			"This mode (Markdown Editor) can only edit files matching pattern: \\.md$ (Markdown files only). Got: test.js",

		assert.strictEqual(error.name, "FileRestrictionError")
