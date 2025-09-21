import fs from "fs/promises"
import { expect } from 'chai'
import { loadRuleFiles, addCustomInstructions } from "../../../prompts/sections/custom-instructions"
import * as assert from 'assert'
import * as sinon from 'sinon'

// Mock fs/promises
// TODO: Mock setup needs manual migration for "fs/promises"
const mockedFs = fs as typeof fs

suite("loadRuleFiles", () => {
	setup(() => {
		sinon.restore()
	})

	test("should read and trim file content", async () => {
		mockedFs.readFile.resolves("  content with spaces  ")
		const result = await loadRuleFiles("/fake/path")
		assert.ok(mockedFs.readFile.called)
		assert.strictEqual(result, 
			"\n# Rules from .Thearules:\ncontent with spaces\n" +
				"\n# Rules from .cursorrules:\ncontent with spaces\n" +
				"\n# Rules from .windsurfrules:\ncontent with spaces\n",
		)
	})

	test("should handle ENOENT error", async () => {
		const error = new Error("File not found") as Error & { code: string }
		error.code = "ENOENT"
		mockedFs.readFile.rejects(error)
		const result = await loadRuleFiles("/fake/path")
		assert.strictEqual(result, "")
	})

	test("should handle EISDIR error", async () => {
		const error = new Error("Is a directory") as Error & { code: string }
		error.code = "EISDIR"
		mockedFs.readFile.rejects(error)
		const result = await loadRuleFiles("/fake/path")
		assert.strictEqual(result, "")
	})

	test("should combine content from multiple rule files when they exist", async () => {
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		mockedFs.readFile.callsFake((filePath: any) => {
			// eslint-disable-next-line @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access
			if (filePath.toString().endsWith(".Thearules")) {
				return Promise.resolve("cline rules content")
			}
			// eslint-disable-next-line @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access
			if (filePath.toString().endsWith(".cursorrules")) {
				return Promise.resolve("cursor rules content")
			}
			const error = new Error("File not found") as Error & { code: string }
			error.code = "ENOENT"
			return Promise.reject(error)
		})

		const result = await loadRuleFiles("/fake/path")
		assert.strictEqual(result, 
			"\n# Rules from .Thearules:\ncline rules content\n" +
				"\n# Rules from .cursorrules:\ncursor rules content\n",
		)
	})

	test("should handle when no rule files exist", async () => {
		const error = new Error("File not found") as Error & { code: string }
		error.code = "ENOENT"
		mockedFs.readFile.rejects(error)

		const result = await loadRuleFiles("/fake/path")
		assert.strictEqual(result, "")
	})

	test("should throw on unexpected errors", async () => {
		const error = new Error("Permission denied") as NodeJS.ErrnoException
		error.code = "EPERM"
		mockedFs.readFile.rejects(error)

		await expect(async () => {
			await loadRuleFiles("/fake/path")
		}).rejects.toThrow()
	})

	test("should skip directories with same name as rule files", async () => {
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		mockedFs.readFile.callsFake((filePath: any) => {
			// eslint-disable-next-line @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access
			if (filePath.toString().endsWith(".Thearules")) {
				const error = new Error("Directory error") as Error & { code: string }
				error.code = "EISDIR"
				return Promise.reject(error)
			}
			// eslint-disable-next-line @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access
			if (filePath.toString().endsWith(".cursorrules")) {
				return Promise.resolve("cursor rules content")
			}
			const error = new Error("File not found") as Error & { code: string }
			error.code = "ENOENT"
			return Promise.reject(error)
		})

		const result = await loadRuleFiles("/fake/path")
		assert.strictEqual(result, "\n# Rules from .cursorrules:\ncursor rules content\n")
	})
// Mock cleanup
suite("addCustomInstructions", () => {
	setup(() => {
		sinon.restore()
	})

	test("should combine all instruction types when provided", async () => {
		mockedFs.readFile.resolves("mode specific rules")

		const result = await addCustomInstructions(
			"mode instructions",
			"global instructions",
			"/fake/path",
			"test-mode",
			{ language: "es" },
		)

		assert.ok(result.includes("Language Preference:"))
		assert.ok(result.includes("Español")) // Check for language name
		assert.ok(result.includes("(es))") // Check for language code in parentheses
		assert.ok(result.includes("Global Instructions:\nglobal instructions"))
		assert.ok(result.includes("Mode-specific Instructions:\nmode instructions"))
		assert.ok(result.includes("Rules from .Thearules-test-mode:\nmode specific rules"))
	})

	test("should return empty string when no instructions provided", async () => {
		const error = new Error("File not found") as Error & { code: string }
		error.code = "ENOENT"
		mockedFs.readFile.rejects(error)

		const result = await addCustomInstructions("", "", "/fake/path", "", {})
		assert.strictEqual(result, "")
	})

	test("should handle missing mode-specific rules file", async () => {
		const error = new Error("File not found") as Error & { code: string }
		error.code = "ENOENT"
		mockedFs.readFile.rejects(error)

		const result = await addCustomInstructions(
			"mode instructions",
			"global instructions",
			"/fake/path",
			"test-mode",
		)

		assert.ok(result.includes("Global Instructions:"))
		assert.ok(result.includes("Mode-specific Instructions:"))
		assert.ok(!result.includes("Rules from .Thearules-test-mode"))
	})

	test("should handle unknown language codes properly", async () => {
		const error = new Error("File not found") as Error & { code: string }
		error.code = "ENOENT"
		mockedFs.readFile.rejects(error)

		const result = await addCustomInstructions(
			"mode instructions",
			"global instructions",
			"/fake/path",
			"test-mode",
			{ language: "xyz" }, // Unknown language code
		)

		assert.ok(result.includes("Language Preference:"))
		assert.ok(result.includes('"xyz" (xyz)) language') // For unknown codes, the code is used as the name too
		assert.ok(result.includes("Global Instructions:\nglobal instructions"))
	})

	test("should throw on unexpected errors", async () => {
		const error = new Error("Permission denied") as NodeJS.ErrnoException
		error.code = "EPERM"
		mockedFs.readFile.rejects(error)

		await expect(async () => {
			await addCustomInstructions("", "", "/fake/path", "test-mode")
		}).rejects.toThrow()
	})

	test("should skip mode-specific rule files that are directories", async () => {
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		mockedFs.readFile.callsFake((filePath: any) => {
			// eslint-disable-next-line @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access
			if (filePath.toString().includes(".Thearules-test-mode")) {
				const error = new Error("Directory error") as Error & { code: string }
				error.code = "EISDIR"
				return Promise.reject(error)
			}
			const error = new Error("File not found") as Error & { code: string }
			error.code = "ENOENT"
			return Promise.reject(error)
		})

		const result = await addCustomInstructions(
			"mode instructions",
			"global instructions",
			"/fake/path",
			"test-mode",
		)

		assert.ok(result.includes("Global Instructions:\nglobal instructions"))
		assert.ok(result.includes("Mode-specific Instructions:\nmode instructions"))
		assert.ok(!result.includes("Rules from .Thearules-test-mode"))
	})
// Mock cleanup
})
