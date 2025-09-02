import { DiffViewProvider } from "../DiffViewProvider"
import * as vscode from "vscode"

// Mock vscode
// TODO: Mock setup needs manual migration
import * as assert from 'assert'
import * as sinon from 'sinon'
import * as vscode from 'vscode'
// TODO: Mock needs manual migration
// TODO: Implement proper mock with proxyquire,
	Range: sinon.stub(),
	Position: sinon.stub(),
	Selection: sinon.stub(),
	TextEditorRevealType: {
		InCenter: 2,
	},

// Mock DecorationController
// TODO: Mock setup needs manual migration
// TODO: Mock needs manual migration
// TODO: Implement proper mock with proxyquire,

suite("DiffViewProvider", () => {
	let diffViewProvider: DiffViewProvider
	const mockCwd = "/mock/cwd"
	let mockWorkspaceEdit: { replace: sinon.SinonStub; delete: sinon.SinonStub }

	setup(() => {
		sinon.restore()
		mockWorkspaceEdit = {
			replace: sinon.stub(),
			delete: sinon.stub(),

		;(vscode.WorkspaceEdit as sinon.SinonStub).callsFake(() => mockWorkspaceEdit)

		diffViewProvider = new DiffViewProvider(mockCwd)
		// Mock the necessary properties and methods
		;(diffViewProvider as unknown as { relPath: string }).relPath = "test.txt"
		;(diffViewProvider as unknown as { activeDiffEditor: vscode.TextEditor }).activeDiffEditor = {
			document: {
				uri: { fsPath: `${mockCwd}/test.txt`, scheme: 'file' },
				getText: sinon.stub(),
				lineCount: 10,
			},
			selection: {
				active: { line: 0, character: 0 },
				anchor: { line: 0, character: 0 },
			},
			edit: sinon.stub().resolves(true),
			revealRange: sinon.stub(),

		;(
			diffViewProvider as unknown as { activeLineController: { setActiveLine: sinon.SinonStub; clear: sinon.SinonStub } }
		).activeLineController = { setActiveLine: sinon.stub(), clear: sinon.stub() }
		;(
			diffViewProvider as unknown as {
				fadedOverlayController: { updateOverlayAfterLine: sinon.SinonStub; clear: sinon.SinonStub }

		).fadedOverlayController = { updateOverlayAfterLine: sinon.stub(), clear: sinon.stub() }

	suite("update method", () => {
		test("should preserve empty last line when original content has one", async () => {
			;(diffViewProvider as unknown as { originalContent: string }).originalContent = "Original content\n"
			await diffViewProvider.update("New content", true)

			assert.ok(mockWorkspaceEdit.replace.calledWith(
				expect.anything()),
				expect.anything(),
				"New content\n",

		test("should not add extra newline when accumulated content already ends with one", async () => {
			;(diffViewProvider as unknown as { originalContent: string }).originalContent = "Original content\n"
			await diffViewProvider.update("New content\n", true)

			assert.ok(mockWorkspaceEdit.replace.calledWith(
				expect.anything()),
				expect.anything(),
				"New content\n",

		test("should not add newline when original content does not end with one", async () => {
			;(diffViewProvider as unknown as { originalContent: string }).originalContent = "Original content"
			await diffViewProvider.update("New content", true)

			assert.ok(mockWorkspaceEdit.replace.calledWith(expect.anything()), expect.anything(), "New content")
