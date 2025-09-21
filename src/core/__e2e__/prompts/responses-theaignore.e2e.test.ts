import * as assert from 'assert'
import { expect } from 'chai'
import * as sinon from 'sinon'
import { TheaIgnoreController, LOCK_TEXT_SYMBOL } from "../../ignore/TheaIgnoreController" // Keep original path, use renamed class
import { formatResponse } from "../responses"
import { GLOBAL_FILENAMES } from "../../../shared/config/thea-config"
import { fileExistsAtPath } from "../../../utils/fs"
import * as fs from "fs/promises"
import { toPosix } from "./utils"

// Mock dependencies
// Mock needs manual implementation
// Mock needs manual implementation
// Mock removed - needs manual implementation)),
// 		},
// 		RelativePattern: sinon.stub(),
// 	}
// Mock cleanup
suite("TheaIgnore Response Formatting", () => {
	const TEST_CWD = "/test/path"
	let mockFileExists: sinon.SinonStubbedInstanceFunction<typeof fileExistsAtPath>
	let mockReadFile: sinon.SinonStubbedInstanceFunction<typeof fs.readFile>

	setup(() => {
		// Reset mocks
		sinon.restore()

		// Setup fs mocks
		mockFileExists = fileExistsAtPath as sinon.SinonStubbedInstanceFunction<typeof fileExistsAtPath>
		mockReadFile = fs.readFile as sinon.SinonStubbedInstanceFunction<typeof fs.readFile>

		// Default mock implementations
		mockFileExists.resolves(true)
		mockReadFile.resolves("node_modules\n.git\nsecrets/**\n*.log")
	})

	suite("formatResponse.theaIgnoreError", () => {
		/**
		 * Tests the error message format for ignored files
		 */
		test("should format error message for ignored files", () => {
			const errorMessage = formatResponse.theaIgnoreError("secrets/api-keys.json")

			// Verify error message format
			assert.ok(errorMessage.includes(
				`Access to secrets/api-keys.json is blocked by the ${GLOBAL_FILENAMES.IGNORE_FILENAME} file settings`,
			))
			assert.ok(errorMessage.includes("continue in the task without using this file"))
			assert.ok(errorMessage.includes(`ask the user to update the ${GLOBAL_FILENAMES.IGNORE_FILENAME} file`))
		})

		/**
		 * Tests with different file paths
		 */
		test("should include the file path in the error message", () => {
			const paths = ["node_modules/package.json", ".git/HEAD", "secrets/credentials.env", "logs/app.log"]

			// Test each path
			for (const testPath of paths) {
				const errorMessage = formatResponse.theaIgnoreError(testPath)
				assert.ok(errorMessage.includes(`Access to ${testPath} is blocked`))
			}
		})
	})

	suite("formatResponse.formatFilesList with TheaIgnoreController", () => {
		/**
		 * Tests file listing with theaignore controller
		 */
		test("should format files list with lock symbols for ignored files", async () => {
			// Create controller
			const controller = new TheaIgnoreController(TEST_CWD) // Use renamed class
			await controller.initialize()

			// Mock validateAccess to control which files are ignored
			controller.validateAccess = sinon.stub().callsFake((filePath: string) => {
				// Only allow files not matching these patterns
				return (
					!filePath.includes("node_modules") &&
					!filePath.includes(".git") &&
					!toPosix(filePath).includes("secrets/")
				)
			})

			// Files list with mixed allowed/ignored files
			const files = [
				"src/app.ts", // allowed
				"node_modules/package.json", // ignored
				"README.md", // allowed
				".git/HEAD", // ignored
				"secrets/keys.json", // ignored
			]

			// Format with controller
			const result = formatResponse.formatFilesList(
				TEST_CWD,
				files,
				false,
				// eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-argument
				controller as any,
				true,
			)

			// Should contain each file
			assert.ok(result.includes("src/app.ts"))
			assert.ok(result.includes("README.md"))

			// Should contain lock symbols for ignored files - case insensitive check using regex
			expect(result).toMatch(new RegExp(`${LOCK_TEXT_SYMBOL}.*node_modules/package.json`, "i"))
			expect(result).toMatch(new RegExp(`${LOCK_TEXT_SYMBOL}.*\\.git/HEAD`, "i"))
			expect(result).toMatch(new RegExp(`${LOCK_TEXT_SYMBOL}.*secrets/keys.json`, "i"))

			// No lock symbols for allowed files
			assert.ok(!result.includes(`${LOCK_TEXT_SYMBOL} src/app.ts`))
			assert.ok(!result.includes(`${LOCK_TEXT_SYMBOL} README.md`))
		})

		/**
		 * Tests formatFilesList when showTheaIgnoredFiles is set to false
		 */
		test("should hide ignored files when showTheaIgnoredFiles is false", async () => {
			// Create controller
			const controller = new TheaIgnoreController(TEST_CWD) // Use renamed class
			await controller.initialize()

			// Mock validateAccess to control which files are ignored
			controller.validateAccess = sinon.stub().callsFake((filePath: string) => {
				// Only allow files not matching these patterns
				return (
					!filePath.includes("node_modules") &&
					!filePath.includes(".git") &&
					!toPosix(filePath).includes("secrets/")
				)
			})

			// Files list with mixed allowed/ignored files
			const files = [
				"src/app.ts", // allowed
				"node_modules/package.json", // ignored
				"README.md", // allowed
				".git/HEAD", // ignored
				"secrets/keys.json", // ignored
			]

			// Format with controller and showTheaIgnoredFiles = false
			const result = formatResponse.formatFilesList(
				TEST_CWD,
				files,
				false,
				// eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-argument
				controller as any,
				false, // showTheaIgnoredFiles = false
			)

			// Should contain allowed files
			assert.ok(result.includes("src/app.ts"))
			assert.ok(result.includes("README.md"))

			// Should NOT contain ignored files (even with lock symbols)
			assert.ok(!result.includes("node_modules/package.json"))
			assert.ok(!result.includes(".git/HEAD"))
			assert.ok(!result.includes("secrets/keys.json"))

			// Double-check with regex to ensure no form of these filenames appears
			expect(result).not.toMatch(/node_modules\/package\.json/i)
			expect(result).not.toMatch(/\.git\/HEAD/i)
			expect(result).not.toMatch(/secrets\/keys\.json/i)
		})

		/**
		 * Tests formatFilesList handles truncation correctly with TheaIgnoreController
		 */
		test("should handle truncation with TheaIgnoreController", async () => {
			// Create controller
			const controller = new TheaIgnoreController(TEST_CWD) // Use renamed class
			await controller.initialize()

			// Format with controller and truncation flag
			const result = formatResponse.formatFilesList(
				TEST_CWD,
				["file1.txt", "file2.txt"],
				true, // didHitLimit = true
				// eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-argument
				controller as any,
				true,
			)

			// Should contain truncation message (case-insensitive check)
			assert.ok(result.includes("File list truncated"))
			expect(result).toMatch(/use list_files on specific subdirectories/i)
		})

		/**
		 * Tests formatFilesList handles empty results
		 */
		test("should handle empty file list with TheaIgnoreController", async () => {
			// Create controller
			const controller = new TheaIgnoreController(TEST_CWD) // Use renamed class
			await controller.initialize()

			// Format with empty files array
			const result = formatResponse.formatFilesList(
				TEST_CWD,
				[],
				false,
				// eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-argument
				controller as any,
				true,
			)

			// Should show "No files found"
			assert.strictEqual(result, "No files found.")
		})
	})

	suite("getInstructions", () => {
		/**
		 * Tests the instructions format
		 */
		test("should format .theaignore instructions for the LLM", async () => {
			// Create controller
			const controller = new TheaIgnoreController(TEST_CWD) // Use renamed class
			await controller.initialize()

			// Get instructions
			const instructions = controller.getInstructions()

			// Verify format and content
			assert.ok(instructions.includes(`# ${GLOBAL_FILENAMES.IGNORE_FILENAME}`))
			assert.ok(instructions.includes(LOCK_TEXT_SYMBOL))
			assert.ok(instructions.includes("node_modules"))
			assert.ok(instructions.includes(".git"))
			assert.ok(instructions.includes("secrets/**"))
			assert.ok(instructions.includes("*.log"))

			// Should explain what the lock symbol means
			assert.ok(instructions.includes("you'll notice a"))
			assert.ok(instructions.includes("next to files that are blocked"))
		})

		/**
		 * Tests null/undefined case when ignore file doesn't exist
		 */
		test(`should return undefined when no ${GLOBAL_FILENAMES.IGNORE_FILENAME} exists`, async () => {
			// Set up no .theaignore
			mockFileExists.resolves(false) // Mock file not existing

			// Create controller without .theaignore
			const controller = new TheaIgnoreController(TEST_CWD) // Use renamed class
			await controller.initialize()

			// Should return undefined
			expect(controller.getInstructions()).toBeUndefined()
		})
	})
// Mock cleanup
