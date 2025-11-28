import * as assert from 'assert'
import * as vscode from "vscode"
import * as path from "path"
import * as sinon from 'sinon'
import WorkspaceTracker from "../WorkspaceTracker"
import { TheaProvider } from "../../../core/webview/TheaProvider"

suite("WorkspaceTracker E2E", () => {
	let workspaceTracker: WorkspaceTracker
	let mockProvider: { postMessageToWebview: sinon.SinonSpy }
	let workspaceUri: vscode.Uri

	setup(async () => {
		// Ensure we have a workspace
		if (!vscode.workspace.workspaceFolders || vscode.workspace.workspaceFolders.length === 0) {
			throw new Error("No workspace folder found. Please run tests with a workspace open.")
		}
		workspaceUri = vscode.workspace.workspaceFolders[0].uri;

		// Create a mock provider that spies on messages
		mockProvider = {
			postMessageToWebview: sinon.spy()
		};

		// Initialize tracker with the mock provider
		// We cast to any/TheaProvider because we only need to satisfy the WeakRef usage
		workspaceTracker = new WorkspaceTracker(mockProvider as unknown as TheaProvider)
	})

	teardown(() => {
		sinon.restore()
		// Dispose tracker if it has a dispose method (it seems it handles disposables internally but doesn't expose dispose?)
		// Looking at the code, it has private disposables. Ideally it should have a dispose method.
		// For now, we just let it be garbage collected or we can try to dispose its watchers if we could access them.
	})

	test("should initialize with workspace files", async () => {
		// Create a test file
		const testFileUri = vscode.Uri.joinPath(workspaceUri, "test-init.txt")
		await vscode.workspace.fs.writeFile(testFileUri, new Uint8Array(Buffer.from("test content")))

		// Wait a bit for FS to settle
		await new Promise(resolve => setTimeout(resolve, 500))

		// Initialize
		await workspaceTracker.initializeFilePaths()

		// Check if postMessageToWebview was called with the file
		assert.ok(mockProvider.postMessageToWebview.called, "postMessageToWebview should be called")
		
		const lastCall = mockProvider.postMessageToWebview.lastCall
		const message = lastCall.args[0]
		assert.strictEqual(message.type, "workspaceUpdated")
		
		const filePaths = message.filePaths as string[]
		const relativePath = path.relative(workspaceUri.fsPath, testFileUri.fsPath)
		// The tracker normalizes paths, likely to relative or absolute. 
		// Based on previous code it seems to store normalized paths.
		// Let's check if our file is in the list.
		// Note: listFiles returns absolute paths, and WorkspaceTracker.normalizeFilePath might change them.
		// But usually it sends what it has.
		
		// We need to check how normalizeFilePath works or just check for presence loosely
		const found = filePaths.some(p => p.includes("test-init.txt"))
		assert.ok(found, `File test-init.txt should be found in ${JSON.stringify(filePaths)}`)

		// Cleanup
		await vscode.workspace.fs.delete(testFileUri)
	})

	test("should detect file creation", async () => {
		await workspaceTracker.initializeFilePaths()
		mockProvider.postMessageToWebview.resetHistory()

		const newFileUri = vscode.Uri.joinPath(workspaceUri, "test-create.txt")
		await vscode.workspace.fs.writeFile(newFileUri, new Uint8Array(Buffer.from("new content")))

		// Wait for watcher to trigger (debounce is likely involved)
		await new Promise(resolve => setTimeout(resolve, 1500))

		assert.ok(mockProvider.postMessageToWebview.called, "postMessageToWebview should be called after file creation")
		const message = mockProvider.postMessageToWebview.lastCall.args[0]
		const filePaths = message.filePaths as string[]
		const found = filePaths.some(p => p.includes("test-create.txt"))
		assert.ok(found, "Newly created file should be in the list")

		// Cleanup
		await vscode.workspace.fs.delete(newFileUri)
	})

	test("should detect file deletion", async () => {
		const fileToDeleteUri = vscode.Uri.joinPath(workspaceUri, "test-delete.txt")
		await vscode.workspace.fs.writeFile(fileToDeleteUri, new Uint8Array(Buffer.from("content")))
		
		await workspaceTracker.initializeFilePaths()
		// Wait for init
		await new Promise(resolve => setTimeout(resolve, 500))
		
		mockProvider.postMessageToWebview.resetHistory()

		await vscode.workspace.fs.delete(fileToDeleteUri)

		// Wait for watcher
		await new Promise(resolve => setTimeout(resolve, 1500))

		assert.ok(mockProvider.postMessageToWebview.called, "postMessageToWebview should be called after file deletion")
		const message = mockProvider.postMessageToWebview.lastCall.args[0]
		const filePaths = message.filePaths as string[]
		const found = filePaths.some(p => p.includes("test-delete.txt"))
		assert.strictEqual(found, false, "Deleted file should NOT be in the list")
	})
})


