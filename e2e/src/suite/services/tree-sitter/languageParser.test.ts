import * as assert from 'assert'
import * as sinon from 'sinon'
/* eslint-disable @typescript-eslint/unbound-method */
import { loadRequiredLanguageParsers } from "../languageParser"
import Parser from "web-tree-sitter"

// Mock web-tree-sitter
const mockSetLanguage = sinon.stub()
// TODO: Mock setup needs manual migration
// TODO: Fix mock - needs proxyquire
/*
=> {
	return {
		__esModule: true,
		default: sinon.stub().callsFake(() => ({
			setLanguage: mockSetLanguage,
		})),

})*/

// Add static methods to Parser mock
const ParserMock = Parser as sinon.SinonStubStaticClass<typeof Parser>
ParserMock.init = sinon.stub().resolves(undefined)
ParserMock.Language = {
	load: sinon.stub().resolves({
		query: sinon.stub().returns("mockQuery"),
	}),
	prototype: {}, // Add required prototype property
} as unknown as typeof Parser.Language

suite("Language Parser", () => {
	setup(() => {
		sinon.restore()

	suite("loadRequiredLanguageParsers", () => {
		test("should initialize parser only once", async () => {
			const files = ["test.js", "test2.js"]
			await loadRequiredLanguageParsers(files)
			await loadRequiredLanguageParsers(files)

			assert.strictEqual(ParserMock.init.callCount, 1)

		test("should load JavaScript parser for .js and .jsx files", async () => {
			const files = ["test.js", "test.jsx"]
			const parsers = await loadRequiredLanguageParsers(files)

			assert.ok(ParserMock.Language.load.calledWith(
				sinon.match.string.and(sinon.match("tree-sitter-javascript.wasm"))),

			assert.ok(parsers.js !== undefined)
			assert.ok(parsers.jsx !== undefined)
			assert.ok(parsers.js.query !== undefined)
			assert.ok(parsers.jsx.query !== undefined)

		test("should load TypeScript parser for .ts and .tsx files", async () => {
			const files = ["test.ts", "test.tsx"]
			const parsers = await loadRequiredLanguageParsers(files)

			assert.ok(ParserMock.Language.load.calledWith(
				sinon.match.string.and(sinon.match("tree-sitter-typescript.wasm"))),

			assert.ok(ParserMock.Language.load.calledWith(sinon.match.string.and(sinon.match("tree-sitter-tsx.wasm"))))
			assert.ok(parsers.ts !== undefined)
			assert.ok(parsers.tsx !== undefined)

		test("should load Python parser for .py files", async () => {
			const files = ["test.py"]
			const parsers = await loadRequiredLanguageParsers(files)

			assert.ok(ParserMock.Language.load.calledWith(sinon.match.string.and(sinon.match("tree-sitter-python.wasm"))))
			assert.ok(parsers.py !== undefined)

		test("should load multiple language parsers as needed", async () => {
			const files = ["test.js", "test.py", "test.rs", "test.go"]
			const parsers = await loadRequiredLanguageParsers(files)

			assert.strictEqual(ParserMock.Language.load.callCount, 4)
			assert.ok(parsers.js !== undefined)
			assert.ok(parsers.py !== undefined)
			assert.ok(parsers.rs !== undefined)
			assert.ok(parsers.go !== undefined)

		test("should handle C/C++ files correctly", async () => {
			const files = ["test.c", "test.h", "test.cpp", "test.hpp"]
			const parsers = await loadRequiredLanguageParsers(files)

			assert.ok(ParserMock.Language.load.calledWith(sinon.match.string.and(sinon.match("tree-sitter-c.wasm"))))
			assert.ok(ParserMock.Language.load.calledWith(sinon.match.string.and(sinon.match("tree-sitter-cpp.wasm"))))
			assert.ok(parsers.c !== undefined)
			assert.ok(parsers.h !== undefined)
			assert.ok(parsers.cpp !== undefined)
			assert.ok(parsers.hpp !== undefined)

		test("should handle Kotlin files correctly", async () => {
			const files = ["test.kt", "test.kts"]
			const parsers = await loadRequiredLanguageParsers(files)

			assert.ok(ParserMock.Language.load.calledWith(sinon.match.string.and(sinon.match("tree-sitter-kotlin.wasm"))))
			assert.ok(parsers.kt !== undefined)
			assert.ok(parsers.kts !== undefined)
			assert.ok(parsers.kt.query !== undefined)
			assert.ok(parsers.kts.query !== undefined)

		test("should throw error for unsupported file extensions", async () => {
			const files = ["test.unsupported"]

			await expect(loadRequiredLanguageParsers(files)).rejects.toThrow("Unsupported language: unsupported")

		test("should load each language only once for multiple files", async () => {
			const files = ["test1.js", "test2.js", "test3.js"]
			await loadRequiredLanguageParsers(files)

			assert.strictEqual(ParserMock.Language.load.callCount, 1)
			assert.ok(ParserMock.Language.load.calledWith(
				sinon.match.string.and(sinon.match("tree-sitter-javascript.wasm"))),

		test("should set language for each parser instance", async () => {
			const files = ["test.js", "test.py"]
			await loadRequiredLanguageParsers(files)

			assert.strictEqual(mockSetLanguage.callCount, 2)
