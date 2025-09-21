import * as vscode from "vscode"
import { expect } from 'chai'
import { EditorUtils } from "../../EditorUtils"
import { CodeActionProvider, ACTION_NAMES } from "../../CodeActionProvider"

import * as assert from 'assert'
import * as sinon from 'sinon'
import * as vscode from 'vscode'
interface MockCodeAction extends vscode.CodeAction {
	title: string
	kind: vscode.CodeActionKind
	command: vscode.Command | undefined

interface MockPosition {
	line: number
	character: number

interface MockCodeActionContext extends vscode.CodeActionContext {
	diagnostics: vscode.Diagnostic[]
	triggerKind: vscode.CodeActionTriggerKind
	only: vscode.CodeActionKind | undefined // Made explicitly required

// Mock VSCode API
// Mock needs manual implementation
/*
=> {
	const actualVscode: typeof vscode = // TODO: requireActual("vscode")

	// Mock static methods of Uri directly on the actual Uri class
	actualVscode.Uri.file = sinon.stub().returns((path: string) => actualVscode.Uri.file(path))
	actualVscode.Uri.parse = sinon.stub().returns((value: string) => actualVscode.Uri.parse(value))
	actualVscode.Uri.joinPath = sinon.stub().returns((uri: vscode.Uri, ...paths: string[]) =>
		actualVscode.Uri.joinPath(uri, ...paths),

	actualVscode.Uri.from = sinon.stub().returns(
		(components: { scheme?: string; authority?: string; path?: string; query?: string; fragment?: string }) =>
			actualVscode.Uri.from({
				scheme: components.scheme || "",
				authority: components.authority,
				path: components.path,
				query: components.query,
				fragment: components.fragment,
			}),

	return {
		...actualVscode,
		CodeAction: sinon.stub().callsFake((title: string, kind: vscode.CodeActionKind) => ({
			title,
			kind,
			command: undefined,
		})),
		CodeActionKind: actualVscode.CodeActionKind, // Return the actual CodeActionKind class
		Range: sinon.stub()
			.callsFake((startLine: number, startChar: number, endLine: number, endChar: number) => ({
				start: { line: startLine, character: startChar } satisfies MockPosition,
				end: { line: endLine, character: endChar } satisfies MockPosition,
			})),
		DiagnosticSeverity: {
			Error: 0,
			Warning: 1,
			Information: 2,
			Hint: 3,
		},
		CodeActionTriggerKind: {
			Invoke: 1,
			Automatic: 2,
		},
		Uri: actualVscode.Uri, // Return the actual Uri class with mocked static methods
	} satisfies Partial<typeof vscode> as typeof vscode
})*/

// Mock EditorUtils
// Mock needs manual implementation
// TODO: Implement proper mock with proxyquire

suite("CodeActionProvider", () => {
	let provider: CodeActionProvider
	let mockDocument: vscode.TextDocument
	let mockRange: vscode.Range
	let mockContext: MockCodeActionContext

	setup(() => {
		provider = new CodeActionProvider()

		// Mock document
		mockDocument = {
			uri: vscode.Uri.file("/test/file.ts"),
			fileName: "/test/file.ts",
			isUntitled: false,
			languageId: "typescript",
			encoding: "utf8", // Added missing property
			version: 1,
			isDirty: false,
			isClosed: false,
			eol: vscode.EndOfLine.LF,
			lineCount: 10,
			getText: sinon.stub().returns(() => "mocked text"), // Removed unused 'range' parameter
			lineAt: sinon.stub().returns((line: number | vscode.Position) => {
				const lineNumber = typeof line === "number" ? line : line.line
				return {
					text: `mocked line ${lineNumber}`,
					lineNumber: lineNumber,
					range: new vscode.Range(lineNumber, 0, lineNumber, 0), // Minimal valid range
					rangeIncludingLineBreak: new vscode.Range(lineNumber, 0, lineNumber, 0),
					firstNonWhitespaceCharacterIndex: 0,
					isEmptyOrWhitespace: false,
				} as vscode.TextLine
			}),
			offsetAt: sinon.stub(),
			positionAt: sinon.stub(),
			getWordRangeAtPosition: sinon.stub(),
			validateRange: sinon.stub(),
			validatePosition: sinon.stub(),
			save: sinon.stub(),
		} satisfies Partial<vscode.TextDocument> as sinon.SinonStubStatic<vscode.TextDocument>

		// Mock range
		mockRange = new vscode.Range(0, 0, 0, 10)

		// Mock context
		mockContext = {
			diagnostics: [],
			triggerKind: vscode.CodeActionTriggerKind.Invoke,
			only: undefined, // Added default value for 'only'
		} as MockCodeActionContext

		// Setup default EditorUtils mocks
		;(EditorUtils.getEffectiveRange as sinon.SinonStub).returns({
			range: mockRange,
			text: "test code",
		} satisfies { range: vscode.Range; text: string })
		;(EditorUtils.getFilePath as sinon.SinonStub).returns("/test/file.ts")
		;(EditorUtils.hasIntersectingRange as sinon.SinonStub).returns(true)
		;(EditorUtils.createDiagnosticData as sinon.SinonStub).callsFake((d: vscode.Diagnostic) => d)

	suite("provideCodeActions", () => {
		test("should provide explain, improve, fix logic, and add to context actions by default", () => {
			const actions = provider.provideCodeActions(mockDocument, mockRange, mockContext)
			const typedActions = actions as readonly MockCodeAction[]

			assert.strictEqual(typedActions.length, 7) // 2 explain + 2 fix logic + 2 improve + 1 add to context
			assert.strictEqual(typedActions[0].title, ACTION_NAMES.ADD_TO_CONTEXT)
			assert.strictEqual(typedActions[1].title, `${ACTION_NAMES.EXPLAIN} in New Task`)
			assert.strictEqual(typedActions[2].title, `${ACTION_NAMES.EXPLAIN} in Current Task`)
			assert.strictEqual(typedActions[3].title, `${ACTION_NAMES.FIX_LOGIC} in New Task`)
			assert.strictEqual(typedActions[4].title, `${ACTION_NAMES.FIX_LOGIC} in Current Task`)
			assert.strictEqual(typedActions[5].title, `${ACTION_NAMES.IMPROVE} in New Task`)
			assert.strictEqual(typedActions[6].title, `${ACTION_NAMES.IMPROVE} in Current Task`)

		test("should provide fix action instead of fix logic when diagnostics exist", () => {
			const diagnostics: vscode.Diagnostic[] = [
				{
					message: "test error",
					severity: vscode.DiagnosticSeverity.Error,
					range: mockRange,
				} satisfies Partial<vscode.Diagnostic>,
			] as vscode.Diagnostic[]
			mockContext.diagnostics = diagnostics

			const actions = provider.provideCodeActions(mockDocument, mockRange, mockContext)
			const typedActions = actions as readonly MockCodeAction[]

			assert.strictEqual(typedActions.length, 7) // 2 explain + 2 fix + 2 improve + 1 add to context
			expect(typedActions.some((a) => a.title === `${ACTION_NAMES.FIX} in New Task`)).toBe(true)
			expect(typedActions.some((a) => a.title === `${ACTION_NAMES.FIX} in Current Task`)).toBe(true)
			expect(typedActions.some((a) => a.title === `${ACTION_NAMES.FIX_LOGIC} in New Task`)).toBe(false)
			expect(typedActions.some((a) => a.title === `${ACTION_NAMES.FIX_LOGIC} in Current Task`)).toBe(false)

		test("should return empty array when no effective range", () => {
			;(EditorUtils.getEffectiveRange as sinon.SinonStub).returns(null)

			const actions = provider.provideCodeActions(mockDocument, mockRange, mockContext)

			assert.deepStrictEqual(actions, [])

		test("should handle errors gracefully", () => {
			const consoleErrorSpy = sinon.spy(console, "error").callsFake(() => {})
			;(EditorUtils.getEffectiveRange as sinon.SinonStub).callsFake(() => {
				throw new Error("Test error")

			const actions = provider.provideCodeActions(mockDocument, mockRange, mockContext)

			assert.deepStrictEqual(actions, [])
			assert.ok(consoleErrorSpy.calledWith("Error providing code actions:", sinon.match.instanceOf(Error)))

			consoleErrorSpy.restore()
