import * as assert from 'assert'
import * as sinon from 'sinon'// Create mock vscode module before importing anything
const createMockUri = (scheme: string, path: string) => ({
	scheme,
	authority: "",
	path,
	query: "",
	fragment: "",
	fsPath: path,
	with: sinon.stub(),
	toString: () => path,
	toJSON: () => ({
		scheme,
		authority: "",
		path,
		query: "",
		fragment: "",
	}),
// Mock cleanup
const mockExecuteCommand = sinon.stub()
const mockOpenExternal = sinon.stub()
const mockShowErrorMessage = sinon.stub()

const mockVscode = {
	workspace: {
		workspaceFolders: [
			{
				uri: { fsPath: "/test/workspace" },
			},
		] as { uri: { fsPath: string } }[] | undefined,
		getWorkspaceFolder: sinon.stub().returns("/test/workspace"),
		fs: {
			stat: sinon.stub(),
			writeFile: sinon.stub(),
		},
		openTextDocument: sinon.stub().resolves({}),
	},
	window: {
		showErrorMessage: mockShowErrorMessage,
		showInformationMessage: sinon.stub(),
		showWarningMessage: sinon.stub(),
		createTextEditorDecorationType: sinon.stub(),
		createOutputChannel: sinon.stub(),
		createWebviewPanel: sinon.stub(),
		showTextDocument: sinon.stub().resolves({}),
		activeTextEditor: undefined as
			| undefined
			| {
					document: {
						uri: { fsPath: string }
					}
			  },
	},
	commands: {
		executeCommand: mockExecuteCommand,
	},
	env: {
		openExternal: mockOpenExternal,
	},
	Uri: {
		parse: sinon.stub((url: string) => createMockUri("https", url)),
		file: sinon.stub((path: string) => createMockUri("file", path)),
	},
	Position: sinon.stub(),
	Range: sinon.stub(),
	TextEdit: sinon.stub(),
	WorkspaceEdit: sinon.stub(),
	DiagnosticSeverity: {
		Error: 0,
		Warning: 1,
		Information: 2,
		Hint: 3,
	},
}

// Mock modules
// TODO: Use proxyquire for module mocking - "vscode", () => mockVscode)
// Mock needs manual implementation
// TODO: Mock setup needs manual migration for "../../../utils/path"

// Now import the modules that use the mocks
import { UrlContentFetcher } from "../../../services/browser/UrlContentFetcher"
import { parseMentions, openMention } from "../index"
import * as git from "../../../utils/git"

import { getWorkspacePath } from "../../../utils/path-vscode"
;(getWorkspacePath as sinon.SinonStub).returns("/test/workspace")

suite("mentions", () => {
	const mockCwd = "/test/workspace"
	let mockUrlContentFetcher: UrlContentFetcher

	setup(() => {
		sinon.restore()

		// Create a mock instance with just the methods we need
		mockUrlContentFetcher = {
			launchBrowser: sinon.stub().resolves(undefined),
			closeBrowser: sinon.stub().resolves(undefined),
			urlToMarkdown: sinon.stub().resolves(""),
		} as unknown as UrlContentFetcher

		// Reset all vscode mocks
		mockVscode.workspace.fs.stat.reset()
		mockVscode.workspace.fs.writeFile.reset()
		mockVscode.workspace.openTextDocument.reset().resolves({})
		mockVscode.window.showTextDocument.reset().resolves({})
		mockVscode.window.showErrorMessage.reset()
		mockExecuteCommand.reset()
		mockOpenExternal.reset()
	})

	suite("parseMentions", () => {
		test("should parse git commit mentions", async () => {
			const commitHash = "abc1234"
			const commitInfo = `abc1234 Fix bug in parser

Author: John Doe
Date: Mon Jan 5 23:50:06 2025 -0500

Detailed commit message with multiple lines
- Fixed parsing issue
- Added tests`

			(git.getCommitInfo as sinon.SinonStub).resolves(commitInfo)

			const result = await parseMentions(`Check out this commit @${commitHash}`, mockCwd, mockUrlContentFetcher)

			assert.ok(result.includes(`'${commitHash}' (see below for commit info))`)
			assert.ok(result.includes(`<git_commit hash="${commitHash}">`))
			assert.ok(result.includes(commitInfo))
		})

		test("should handle errors fetching git info", async () => {
			const commitHash = "abc1234"
			const errorMessage = "Failed to get commit info"

			(git.getCommitInfo as sinon.SinonStub).rejects(new Error(errorMessage))

			const result = await parseMentions(`Check out this commit @${commitHash}`, mockCwd, mockUrlContentFetcher)

			assert.ok(result.includes(`'${commitHash}' (see below for commit info))`)
			assert.ok(result.includes(`<git_commit hash="${commitHash}">`))
			assert.ok(result.includes(`Error fetching commit info: ${errorMessage}`))
		})
	})

	suite("openMention", () => {
		test("should handle file paths and problems", async () => {
			// Mock stat to simulate file not existing
			mockVscode.workspace.fs.stat.mockRejectedValueOnce(new Error("File does not exist"))

			// Call openMention and wait for it to complete
			await openMention("/path/to/file")

			// Verify error handling
			assert.ok(!mockExecuteCommand.called)
			assert.ok(!mockOpenExternal.called)
			assert.ok(mockVscode.window.showErrorMessage.calledWith("Could not open file: File does not exist"))

			// Reset mocks for next test
			sinon.restore()

			// Test problems command
			await openMention("problems")
			assert.ok(mockExecuteCommand.calledWith("workbench.actions.view.problems"))
		})

		test("should handle URLs", async () => {
			const url = "https://example.com"
			await openMention(url)
			const mockUri = mockVscode.Uri.parse(url)
			assert.ok(mockVscode.env.openExternal.called)
			// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-member-access
			const calledArg = mockVscode.env.openExternal.mock.calls[0][0]
			assert.deepStrictEqual(calledArg, {
					scheme: mockUri.scheme,
					authority: mockUri.authority,
					path: mockUri.path,
					query: mockUri.query,
					fragment: mockUri.fragment,
				})
		})
	})
// Mock cleanup
