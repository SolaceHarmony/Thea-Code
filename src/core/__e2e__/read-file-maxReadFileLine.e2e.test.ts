import * as path from "path"
import { readLines } from "../../integrations/misc/read-lines"
import { countFileLines } from "../../integrations/misc/line-counter"
import { parseSourceCodeDefinitionsForFile } from "../../services/tree-sitter"
import { extractTextFromFile, addLineNumbers } from "../../integrations/misc/extract-text"
import { ReadFileToolUse } from "../assistant-message"
import { isBinaryFile } from "isbinaryfile"
import * as assert from 'assert'
import * as sinon from 'sinon'

// Mock dependencies
// Mock needs manual implementation
// Mock needs manual implementation
// Mock needs manual implementation
// 	const originalPath = require<typeof import("path")>("path")
// Mock removed - needs manual implementation
// Mock cleanup
suite("read_file tool with maxReadFileLine setting", () => {
	// Test data
	const testFilePath = "test/file.txt"
	const absoluteFilePath = "/test/file.txt"
	// const fileContent = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5" - Unused variable
	const numberedFileContent = "1 | Line 1\n2 | Line 2\n3 | Line 3\n4 | Line 4\n5 | Line 5"
	const sourceCodeDef = "\n\n# file.txt\n1--5 | Content"

	// Mocked functions with correct types
	const mockedCountFileLines = countFileLines as sinon.SinonStubbedInstanceFunction<typeof countFileLines>
	const mockedReadLines = readLines as sinon.SinonStubbedInstanceFunction<typeof readLines>
	const mockedExtractTextFromFile = extractTextFromFile as sinon.SinonStubbedInstanceFunction<typeof extractTextFromFile>
	const mockedAddLineNumbers = addLineNumbers as sinon.SinonStubbedInstanceFunction<typeof addLineNumbers>
	const mockedParseSourceCodeDefinitionsForFile = parseSourceCodeDefinitionsForFile as sinon.SinonStubbedInstanceFunction<
		typeof parseSourceCodeDefinitionsForFile
	>
	const mockedIsBinaryFile = isBinaryFile as sinon.SinonStubbedInstanceFunction<typeof isBinaryFile>
	// Use bind to avoid unbound method reference
	const mockedPathResolve = path.resolve.bind(path) as sinon.SinonStubbedInstanceFunction<typeof path.resolve>

	// Define interfaces for mock objects to avoid 'any' type issues
	interface MockProvider {
		getState: sinon.SinonStub
		deref: sinon.SinonStub
	}

	interface MockThea {
		cwd: string
		task: string
		providerRef: MockProvider
		theaIgnoreController: {
			validateAccess: sinon.SinonStub
		}
		say: sinon.SinonStub
		ask: sinon.SinonStub
		presentAssistantMessage: sinon.SinonStub
	}

	// Mock instances
	const mockThea = {} as MockThea
	let mockProvider: MockProvider
	let toolResult: string | undefined

	setup(() => {
		sinon.restore()

		// Setup path resolution
		mockedPathResolve.returns(absoluteFilePath)

		// Setup mocks for file operations
		mockedIsBinaryFile.resolves(false)
		mockedAddLineNumbers.callsFake((content: string, startLine = 1) => {
			return content
				.split("\n")
				.map((line, i) => `${i + startLine} | ${line}`)
				.join("\n")
		})

		// Setup mock provider
		mockProvider = {
			getState: sinon.stub(),
			deref: sinon.stub().mockReturnThis(),
		}

		// Setup Thea instance with mock methods
		mockThea.cwd = "/"
		mockThea.task = "Test"
		mockThea.providerRef = mockProvider
		mockThea.theaIgnoreController = {
			validateAccess: sinon.stub().returns(true),
		}
		mockThea.say = sinon.stub().resolves(undefined)
		mockThea.ask = sinon.stub().resolves(true)
		mockThea.presentAssistantMessage = sinon.stub()

		// Reset tool result
		toolResult = undefined
	})

	/**
	 * Helper function to execute the read file tool with different maxReadFileLine settings
	 */
	async function executeReadFileTool(maxReadFileLine: number, totalLines = 5): Promise<string | undefined> {
		// Configure mocks based on test scenario
		mockProvider.getState.resolves({ maxReadFileLine })
		mockedCountFileLines.resolves(totalLines)

		// Create a tool use object
		const toolUse: ReadFileToolUse = {
			type: "tool_use",
			name: "read_file",
			params: { path: testFilePath },
			partial: false,
		}

		// Import the tool implementation dynamically to avoid hoisting issues
		// eslint-disable-next-line @typescript-eslint/no-require-imports, @typescript-eslint/no-unsafe-assignment
		const { readFileTool } = require("../tools/readFileTool")

		// Execute the tool
		// eslint-disable-next-line @typescript-eslint/no-unsafe-call
		await readFileTool(
			mockThea,
			toolUse,
			mockThea.ask,
			sinon.stub(),
			(result: string) => {
				toolResult = result
			},
			(param: string, value: string) => value,
		)

		return toolResult
	}

	suite("when maxReadFileLine is negative", () => {
		test("should read the entire file using extractTextFromFile", async () => {
			// Setup
			mockedExtractTextFromFile.resolves(numberedFileContent)

			// Execute
			const result = await executeReadFileTool(-1)

			// Verify
			assert.ok(mockedExtractTextFromFile.calledWith(absoluteFilePath))
			assert.ok(!mockedReadLines.called)
			assert.ok(!mockedParseSourceCodeDefinitionsForFile.called)
			assert.strictEqual(result, numberedFileContent)
		})
	})

	suite("when maxReadFileLine is 0", () => {
		test("should return an empty content with source code definitions", async () => {
			// Setup - for maxReadFileLine = 0, the implementation won't call readLines
			mockedParseSourceCodeDefinitionsForFile.resolves(sourceCodeDef)

			// Execute
			const result = await executeReadFileTool(0)

			// Verify
			assert.ok(!mockedExtractTextFromFile.called)
			assert.ok(!mockedReadLines.called) // Per implementation line 141
			assert.ok(mockedParseSourceCodeDefinitionsForFile.calledWith(
				absoluteFilePath,
				mockThea.theaIgnoreController,
			))
			assert.ok(result.includes("[Showing only 0 of 5 total lines"))
			assert.ok(result.includes(sourceCodeDef))
		})
	})

	suite("when maxReadFileLine is less than file length", () => {
		test("should read only maxReadFileLine lines and add source code definitions", async () => {
			// Setup
			const content = "Line 1\nLine 2\nLine 3"
			mockedReadLines.resolves(content)
			mockedParseSourceCodeDefinitionsForFile.resolves(sourceCodeDef)

			// Execute
			const result = await executeReadFileTool(3)

			// Verify - check behavior but not specific implementation details
			assert.ok(!mockedExtractTextFromFile.called)
			assert.ok(mockedReadLines.called)
			assert.ok(mockedParseSourceCodeDefinitionsForFile.calledWith(
				absoluteFilePath,
				mockThea.theaIgnoreController,
			))
			assert.ok(result.includes("1 | Line 1"))
			assert.ok(result.includes("2 | Line 2"))
			assert.ok(result.includes("3 | Line 3"))
			assert.ok(result.includes("[Showing only 3 of 5 total lines"))
			assert.ok(result.includes(sourceCodeDef))
		})
	})

	suite("when maxReadFileLine equals or exceeds file length", () => {
		test("should use extractTextFromFile when maxReadFileLine > totalLines", async () => {
			// Setup
			mockedCountFileLines.resolves(5) // File shorter than maxReadFileLine
			mockedExtractTextFromFile.resolves(numberedFileContent)

			// Execute
			const result = await executeReadFileTool(10, 5)

			// Verify
			assert.ok(mockedExtractTextFromFile.calledWith(absoluteFilePath))
			assert.strictEqual(result, numberedFileContent)
		})

		test("should read with extractTextFromFile when file has few lines", async () => {
			// Setup
			mockedCountFileLines.resolves(3) // File shorter than maxReadFileLine
			mockedExtractTextFromFile.resolves(numberedFileContent)

			// Execute
			const result = await executeReadFileTool(5, 3)

			// Verify
			assert.ok(mockedExtractTextFromFile.calledWith(absoluteFilePath))
			assert.ok(!mockedReadLines.called)
			assert.strictEqual(result, numberedFileContent)
		})
	})

	suite("when file is binary", () => {
		test("should always use extractTextFromFile regardless of maxReadFileLine", async () => {
			// Setup
			mockedIsBinaryFile.resolves(true)
			mockedExtractTextFromFile.resolves(numberedFileContent)

			// Execute
			const result = await executeReadFileTool(3)

			// Verify
			assert.ok(mockedExtractTextFromFile.calledWith(absoluteFilePath))
			assert.ok(!mockedReadLines.called)
			assert.strictEqual(result, numberedFileContent)
		})
	})

	suite("with range parameters", () => {
		test("should honor start_line and end_line when provided", async () => {
			// Setup
			const rangeToolUse: ReadFileToolUse = {
				type: "tool_use",
				name: "read_file",
				params: {
					path: testFilePath,
					start_line: "2",
					end_line: "4",
				},
				partial: false,
			}

			mockedReadLines.resolves("Line 2\nLine 3\nLine 4")

			// Import the tool implementation dynamically
			// eslint-disable-next-line @typescript-eslint/no-require-imports, @typescript-eslint/no-unsafe-assignment
			const { readFileTool } = require("../tools/readFileTool")

			// Execute the tool
			// eslint-disable-next-line @typescript-eslint/no-unsafe-call
			await readFileTool(
				mockThea,
				rangeToolUse,
				mockThea.ask,
				sinon.stub(),
				() => {
					// Result callback - not used in this test
				},
				(param: string, value: string) => value,
			)

			// Verify
			assert.ok(mockedReadLines.calledWith(absoluteFilePath, 3, 1)) // end_line - 1, start_line - 1
			assert.ok(mockedAddLineNumbers.calledWith(sinon.match.instanceOf(String)), 2) // start with proper line numbers
		})
	})
// Mock cleanup
})
