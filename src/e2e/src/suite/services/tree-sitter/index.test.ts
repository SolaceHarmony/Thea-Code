import { parseSourceCodeForDefinitionsTopLevel } from "../index"
import { listFiles } from "../../glob/list-files"
import { loadRequiredLanguageParsers } from "../languageParser"
import { fileExistsAtPath } from "../../../utils/fs"
import * as fs from "fs/promises"

// Mock dependencies
// Mock needs manual implementation
// Mock needs manual implementation
import * as assert from 'assert'
import * as sinon from 'sinon'
suite("Tree-sitter Service", () => {
	setup(() => {
		sinon.restore()
		;(fileExistsAtPath as sinon.SinonStub).resolves(true)

	suite("parseSourceCodeForDefinitionsTopLevel", () => {
		test("should handle non-existent directory", async () => {
			;(fileExistsAtPath as sinon.SinonStub).resolves(false)

			const result = await parseSourceCodeForDefinitionsTopLevel("/non/existent/path")
			assert.strictEqual(result, "This directory does not exist or you do not have permission to access it.")

		test("should handle empty directory", async () => {
			;(listFiles as sinon.SinonStub).resolves([[], new Set()])

			const result = await parseSourceCodeForDefinitionsTopLevel("/test/path")
			assert.strictEqual(result, "No source code definitions found.")

		test("should parse TypeScript files correctly", async () => {
			const mockFiles = ["/test/path/file1.ts", "/test/path/file2.tsx", "/test/path/readme.md"]

			;(listFiles as sinon.SinonStub).resolves([mockFiles, new Set()])

			const mockParser = {
				parse: sinon.stub().returns({
					rootNode: "mockNode",
				}),

			const mockQuery = {
				captures: sinon.stub().returns([
					{
						node: {
							startPosition: { row: 0 },
							endPosition: { row: 0 },
							parent: {
								startPosition: { row: 0 },
								endPosition: { row: 0 },
							},
						},
						name: "name.definition",
					},
				]),

			;(loadRequiredLanguageParsers as sinon.SinonStub).resolves({
				ts: { parser: mockParser, query: mockQuery },
				tsx: { parser: mockParser, query: mockQuery },
			;(fs.readFile as sinon.SinonStub).resolves("export class TestClass {\n  constructor() {}\n}")

			const result = await parseSourceCodeForDefinitionsTopLevel("/test/path")

			assert.ok(result.includes("file1.ts"))
			assert.ok(result.includes("file2.tsx"))
			assert.ok(!result.includes("readme.md"))
			assert.ok(result.includes("export class TestClass"))

		test("should handle multiple definition types", async () => {
			const mockFiles = ["/test/path/file.ts"]
			;(listFiles as sinon.SinonStub).resolves([mockFiles, new Set()])

			const mockParser = {
				parse: sinon.stub().returns({
					rootNode: "mockNode",
				}),

			const mockQuery = {
				captures: sinon.stub().returns([
					{
						node: {
							startPosition: { row: 0 },
							endPosition: { row: 0 },
							parent: {
								startPosition: { row: 0 },
								endPosition: { row: 0 },
							},
						},
						name: "name.definition.class",
					},
					{
						node: {
							startPosition: { row: 2 },
							endPosition: { row: 2 },
							parent: {
								startPosition: { row: 0 },
								endPosition: { row: 0 },
							},
						},
						name: "name.definition.function",
					},
				]),

			;(loadRequiredLanguageParsers as sinon.SinonStub).resolves({
				ts: { parser: mockParser, query: mockQuery },

			const fileContent = "class TestClass {\n" + "  constructor() {}\n" + "  testMethod() {}\n" + "}"

			;(fs.readFile as sinon.SinonStub).resolves(fileContent)

			const result = await parseSourceCodeForDefinitionsTopLevel("/test/path")

			assert.ok(result.includes("class TestClass"))
			assert.ok(result.includes("testMethod())")

		test("should handle parsing errors gracefully", async () => {
			const mockFiles = ["/test/path/file.ts"]
			;(listFiles as sinon.SinonStub).resolves([mockFiles, new Set()])

			const mockParser = {
				parse: sinon.stub().callsFake(() => {
					throw new Error("Parsing error")
				}),

			const mockQuery = {
				captures: sinon.stub(),

			;(loadRequiredLanguageParsers as sinon.SinonStub).resolves({
				ts: { parser: mockParser, query: mockQuery },
			;(fs.readFile as sinon.SinonStub).resolves("invalid code")

			const result = await parseSourceCodeForDefinitionsTopLevel("/test/path")
			assert.strictEqual(result, "No source code definitions found.")

		test("should respect file limit", async () => {
			const mockFiles = Array(100)
				.fill(0)
				.map((_, i) => `/test/path/file${i}.ts`)
			;(listFiles as sinon.SinonStub).resolves([mockFiles, new Set()])

			const mockParser = {
				parse: sinon.stub().returns({
					rootNode: "mockNode",
				}),

			const mockQuery = {
				captures: sinon.stub().returns([]),

			;(loadRequiredLanguageParsers as sinon.SinonStub).resolves({
				ts: { parser: mockParser, query: mockQuery },

			await parseSourceCodeForDefinitionsTopLevel("/test/path")

			// Should only process first 50 files
			assert.strictEqual(mockParser.parse.callCount, 50)

		test("should handle various supported file extensions", async () => {
			const mockFiles = [
				"/test/path/script.js",
				"/test/path/app.py",
				"/test/path/main.rs",
				"/test/path/program.cpp",
				"/test/path/code.go",
				"/test/path/app.kt",
				"/test/path/script.kts",

			;(listFiles as sinon.SinonStub).resolves([mockFiles, new Set()])

			const mockParser = {
				parse: sinon.stub().returns({
					rootNode: "mockNode",
				}),

			const mockQuery = {
				captures: sinon.stub().returns([
					{
						node: {
							startPosition: { row: 0 },
							endPosition: { row: 0 },
							parent: {
								startPosition: { row: 0 },
								endPosition: { row: 0 },
							},
						},
						name: "name",
					},
				]),

			;(loadRequiredLanguageParsers as sinon.SinonStub).resolves({
				js: { parser: mockParser, query: mockQuery },
				py: { parser: mockParser, query: mockQuery },
				rs: { parser: mockParser, query: mockQuery },
				cpp: { parser: mockParser, query: mockQuery },
				go: { parser: mockParser, query: mockQuery },
				kt: { parser: mockParser, query: mockQuery },
				kts: { parser: mockParser, query: mockQuery },
			;(fs.readFile as sinon.SinonStub).resolves("function test() {}")

			const result = await parseSourceCodeForDefinitionsTopLevel("/test/path")

			assert.ok(result.includes("script.js"))
			assert.ok(result.includes("app.py"))
			assert.ok(result.includes("main.rs"))
			assert.ok(result.includes("program.cpp"))
			assert.ok(result.includes("code.go"))
			assert.ok(result.includes("app.kt"))
			assert.ok(result.includes("script.kts"))

		test("should normalize paths in output", async () => {
			const mockFiles = ["/test/path/dir\\file.ts"]
			;(listFiles as sinon.SinonStub).resolves([mockFiles, new Set()])

			const mockParser = {
				parse: sinon.stub().returns({
					rootNode: "mockNode",
				}),

			const mockQuery = {
				captures: sinon.stub().returns([
					{
						node: {
							startPosition: { row: 0 },
							endPosition: { row: 0 },
							parent: {
								startPosition: { row: 0 },
								endPosition: { row: 0 },
							},
						},
						name: "name",
					},
				]),

			;(loadRequiredLanguageParsers as sinon.SinonStub).resolves({
				ts: { parser: mockParser, query: mockQuery },
			;(fs.readFile as sinon.SinonStub).resolves("class Test {}")

			const result = await parseSourceCodeForDefinitionsTopLevel("/test/path")

			// Should use forward slashes regardless of platform
			assert.ok(result.includes("dir/file.ts"))
			assert.ok(!result.includes("dir\\file.ts"))
