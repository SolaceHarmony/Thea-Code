import { strict as assert } from "node:assert"
import { isToolAllowedForMode, FileRestrictionError, ModeConfig } from "../modes"

describe("isToolAllowedForMode", () => {
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
	]

	it("allows always available tools", () => {
		assert.equal(isToolAllowedForMode("ask_followup_question", "markdown-editor", customModes), true)
		assert.equal(isToolAllowedForMode("attempt_completion", "markdown-editor", customModes), true)
	})

	it("allows unrestricted tools", () => {
		assert.equal(isToolAllowedForMode("read_file", "markdown-editor", customModes), true)
		assert.equal(isToolAllowedForMode("browser_action", "markdown-editor", customModes), true)
	})

	describe("file restrictions", () => {
		it("allows editing matching files", () => {
			// Test markdown editor mode
			const mdResult = isToolAllowedForMode("write_to_file", "markdown-editor", customModes, undefined, {
				path: "test.md",
				content: "# Test",
			})
			assert.equal(mdResult, true)

			// Test CSS editor mode
			const cssResult = isToolAllowedForMode("write_to_file", "css-editor", customModes, undefined, {
				path: "styles.css",
				content: ".test { color: red; }",
			})
			assert.equal(cssResult, true)
		})

		it("rejects editing non-matching files", () => {
			// Test markdown editor mode with non-markdown file
			assert.throws(() =>
				isToolAllowedForMode("write_to_file", "markdown-editor", customModes, undefined, {
					path: "test.js",
					content: "console.log('test')",
				}),
				FileRestrictionError
			)

			// Test CSS editor mode with non-CSS file
			assert.throws(() =>
				isToolAllowedForMode("write_to_file", "css-editor", customModes, undefined, {
					path: "test.js",
					content: "console.log('test')",
				}),
				FileRestrictionError
			)
		})

		it("handles partial streaming cases (path only, no content/diff)", () => {
			// Should allow path-only for any files when content not provided yet
			assert.equal(
				isToolAllowedForMode("write_to_file", "markdown-editor", customModes, undefined, {
					path: "test.js",
				}),
				true
			)

			assert.equal(
				isToolAllowedForMode("apply_diff", "markdown-editor", customModes, undefined, {
					path: "test.js",
				}),
				true
			)
		})

		it("applies restrictions to both write_to_file and apply_diff", () => {
			// Test write_to_file
			const writeResult = isToolAllowedForMode("write_to_file", "markdown-editor", customModes, undefined, {
				path: "test.md",
				content: "# Test",
			})
			assert.equal(writeResult, true)

			// Test apply_diff
			const diffResult = isToolAllowedForMode("apply_diff", "markdown-editor", customModes, undefined, {
				path: "test.md",
				diff: "some diff",
			})
			assert.equal(diffResult, true)

			// Test with non-matching file should throw
			assert.throws(() =>
				isToolAllowedForMode("write_to_file", "markdown-editor", customModes, undefined, {
					path: "test.js",
					content: "console.log('test')",
				}),
				FileRestrictionError
			)

			assert.throws(() =>
				isToolAllowedForMode("apply_diff", "markdown-editor", customModes, undefined, {
					path: "test.js",
					diff: "some diff",
				}),
				FileRestrictionError
			)
		})
	})

	it("handles non-existent modes", () => {
		const result = isToolAllowedForMode("read_file", "non-existent-mode", customModes)
		assert.equal(result, false)
	})

	it("respects tool requirements", () => {
		// Test a tool that's not in the groups
		const result = isToolAllowedForMode("execute_command", "markdown-editor", customModes)
		assert.equal(result, false)
	})
})