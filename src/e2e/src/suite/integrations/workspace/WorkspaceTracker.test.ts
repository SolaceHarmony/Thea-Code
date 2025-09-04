import * as assert from 'assert'
import * as sinon from 'sinon'/* eslint-disable @typescript-eslint/unbound-method */
import * as vscode from "vscode"
import WorkspaceTracker from "../WorkspaceTracker"
import { TheaProvider } from "../../../core/webview/TheaProvider" // Renamed import
import { listFiles } from "../../../services/glob/list-files"
import { getWorkspacePath } from "../../../utils/path-vscode"

// Mock functions - must be defined before module mocking
const mockOnDidCreate = sinon.stub()
const mockOnDidDelete = sinon.stub()
const mockDispose = sinon.stub()

// Store registered tab change callback
let registeredTabChangeCallback: (() => Promise<void>) | null = null

// Mock workspace path
// TODO: Use proxyquire for module mocking - "../../../utils/path", () => ({
	getWorkspacePath: sinon.stub().returns("/test/workspace"),
	toRelativePath: sinon.stub((p: string, cwd: string) => {
		// Simple mock that preserves the original behavior for tests
		const relativePath = p.replace(`${cwd}/`, "")
		// Add trailing slash if original path had one
		return p.endsWith("/") ? relativePath + "/" : relativePath
	}),
// Mock cleanup needed

// Mock watcher - must be defined after mockDispose but before // TODO: Mock setup needs manual migration for "vscode"
const mockWatcher: {
	onDidCreate: sinon.SinonStub
	onDidDelete: sinon.SinonStub
	dispose: sinon.SinonStub
} = {
	onDidCreate: mockOnDidCreate.returns({ dispose: mockDispose }),
	onDidDelete: mockOnDidDelete.returns({ dispose: mockDispose }),
	dispose: mockDispose,
}

// Mock vscode
// TODO: Use proxyquire for module mocking - "vscode", () => ({
	window: {
		tabGroups: {
			onDidChangeTabs: sinon.stub((callback: () => Promise<void>) => {
				registeredTabChangeCallback = callback
// Mock removed - needs manual implementation),
// 			all: [],
// 		},
// 		onDidChangeActiveTextEditor: sinon.stub(() => ({ dispose: sinon.stub() })),
// 	},
// 	workspace: {
// 		workspaceFolders: [
// 			{
// 				uri: { fsPath: "/test/workspace" },
// 				name: "test",
// 				index: 0,
// 			},
// 		],
// 		createFileSystemWatcher: sinon.stub(() => mockWatcher),
// 		fs: {
// 			stat: sinon.stub().resolves({ type: 1 }), // FileType.File = 1
// 		},
// 	},
// 	FileType: { File: 1, Directory: 2 },
// // Mock cleanup needed
// 
// // Mock needs manual implementation
// 	let workspaceTracker: WorkspaceTracker
// 	let mockProvider: TheaProvider // Renamed type
// 
// 	let clock: sinon.SinonFakeTimers
// 	
// 	setup(() => {
// 		sinon.restore()
// 		clock = sinon.useFakeTimers()
// 
// 		// Reset all mock implementations
// 		registeredTabChangeCallback = null
// 
// 		// Reset workspace path mock
// 		;(getWorkspacePath as sinon.SinonStub).returns("/test/workspace")
// 
// 		// Create provider mock
// 		mockProvider = {
// 			postMessageToWebview: sinon.stub().resolves(undefined),
// 		} as unknown as TheaProvider & { postMessageToWebview: sinon.SinonStub } // Renamed type assertion
// 
// 		// Create tracker instance
// 		workspaceTracker = new WorkspaceTracker(mockProvider)
// 
// 		// Ensure the tab change callback was registered
// 		assert.notStrictEqual(registeredTabChangeCallback, null)
// 	})
// 
// 	test("should initialize with workspace files", async () => {
// 		const mockFiles = [["/test/workspace/file1.ts", "/test/workspace/file2.ts"], false]
// 		;(listFiles as sinon.SinonStub).resolves(mockFiles)
// 
// 		await workspaceTracker.initializeFilePaths()
// 		clock.runAll()
// 
// 		assert.ok(mockProvider.postMessageToWebview.calledWith({
// 			type: "workspaceUpdated",
// 			filePaths: // TODO: Array partial match - ["file1.ts", "file2.ts"])) as unknown[],
// 			openedTabs: [],
// 		})
// 		const firstCall = (mockProvider.postMessageToWebview as sinon.SinonStub).mock.calls[0] as [{ filePaths: string[] }]
// 		assert.strictEqual(firstCall[0].filePaths.length, 2)
// 	})
// 
// 	test("should handle file creation events", async () => {
// 		// Get the creation callback and call it
// 		const createCalls = mockOnDidCreate.mock.calls as [[(args: { fsPath: string }) => Promise<void>]]
// 		const callback = createCalls[0][0]
// 		await callback({ fsPath: "/test/workspace/newfile.ts" })
// 		clock.runAll()
// 
// 		assert.ok(mockProvider.postMessageToWebview.calledWith({
// 			type: "workspaceUpdated",
// 			filePaths: ["newfile.ts"],
// 			openedTabs: [],
// 		}))
// 	})
// 
// 	test("should handle file deletion events", async () => {
// 		// First add a file
// 		const createCalls2 = mockOnDidCreate.mock.calls as [[(args: { fsPath: string }) => Promise<void>]]
// 		const createCallback = createCalls2[0][0]
// 		await createCallback({ fsPath: "/test/workspace/file.ts" })
// 		clock.runAll()
// 
// 		// Then delete it
// 		const deleteCalls = mockOnDidDelete.mock.calls as [[(args: { fsPath: string }) => Promise<void>]]
// 		const deleteCallback = deleteCalls[0][0]
// 		await deleteCallback({ fsPath: "/test/workspace/file.ts" })
// 		clock.runAll()
// 
// 		// The last call should have empty filePaths
// 		expect(mockProvider.postMessageToWebview).lastCall.calledWith({
// 			type: "workspaceUpdated",
// 			filePaths: [],
// 			openedTabs: [],
// 		})
// 	})
// 
// 	test("should handle directory paths correctly", async () => {
// 		// Mock stat to return directory type
// 		;(vscode.workspace.fs.stat as sinon.SinonStub).mockResolvedValueOnce({ type: 2 }) // FileType.Directory = 2
// 
// 		const dirCalls = mockOnDidCreate.mock.calls as [[(args: { fsPath: string }) => Promise<void>]]
// 		const callback = dirCalls[0][0]
// 		await callback({ fsPath: "/test/workspace/newdir" })
// 		clock.runAll()
// 
// 		assert.ok(mockProvider.postMessageToWebview.calledWith({
// 			type: "workspaceUpdated",
// 			filePaths: // TODO: Array partial match - ["newdir"])) as unknown[],
// 			openedTabs: [],
// 		})
// 		const lastCall = (mockProvider.postMessageToWebview as sinon.SinonStub).mock.calls.slice(-1)[0] as [
// 			{ filePaths: string[] },
// 		]
// 		assert.strictEqual(lastCall[0].filePaths.length, 1)
// 	})
// 
// 	test("should respect file limits", async () => {
// 		// Create array of unique file paths for initial load
// 		const files = Array.from({ length: 1001 }, (_, i) => `/test/workspace/file${i}.ts`)
// 		;(listFiles as sinon.SinonStub).resolves([files, false])
// 
// 		await workspaceTracker.initializeFilePaths()
// 		clock.runAll()
// 
// 		// Should only have 1000 files initially
// 		const expectedFiles = Array.from({ length: 1000 }, (_, i) => `file${i}.ts`).sort()
// 		const calls = (mockProvider.postMessageToWebview as sinon.SinonStub).mock.calls as [{ filePaths: string[] }[]]
// 
// 		assert.ok(mockProvider.postMessageToWebview.calledWith({
// 			type: "workspaceUpdated",
// 			filePaths: // TODO: Array partial match - expectedFiles)) as unknown[],
// 			openedTabs: [],
// 		})
// 		assert.strictEqual(calls[0][0].filePaths.length, 1000)
// 
// 		// Should allow adding up to 2000 total files
// 		const extraCalls = mockOnDidCreate.mock.calls as [[(args: { fsPath: string }) => Promise<void>]]
// 		const callback = extraCalls[0][0]
// 		for (let i = 0; i < 1000; i++) {
// 			await callback({ fsPath: `/test/workspace/extra${i}.ts` })
// 		}
		clock.runAll()

		const lastCall = (mockProvider.postMessageToWebview as sinon.SinonStub).mock.calls.slice(-1)[0] as [
			{ filePaths: string[] },
		]
		assert.strictEqual(lastCall[0].filePaths.length, 2000)

		// Adding one more file beyond 2000 should not increase the count
		await callback({ fsPath: "/test/workspace/toomany.ts" })
		clock.runAll()

		const finalCall = (mockProvider.postMessageToWebview as sinon.SinonStub).mock.calls.slice(-1)[0] as [
			{ filePaths: string[] },
		]
		assert.strictEqual(finalCall[0].filePaths.length, 2000)
	})

	test("should clean up watchers and timers on dispose", () => {
		// Set up updateTimer
		const disposeCalls = mockOnDidCreate.mock.calls as [[(args: { fsPath: string }) => void]]
		const callback = disposeCalls[0][0]
		callback({ fsPath: "/test/workspace/file.ts" })

		workspaceTracker.dispose()
		assert.ok(mockDispose.called)
		clock.runAll() // Ensure any pending timers are cleared

		// No more updates should happen after dispose
		assert.ok(!mockProvider.postMessageToWebview.called)
	})

	test("should handle workspace path changes when tabs change", async () => {
		expect(registeredTabChangeCallback).not.toBeNull()

		// Set initial workspace path and create tracker
		;(getWorkspacePath as sinon.SinonStub).returns("/test/workspace")
		workspaceTracker = new WorkspaceTracker(mockProvider)

		// Clear any initialization calls
		sinon.restore()

		// Mock listFiles to return some files
		const mockFiles = [["/test/new-workspace/file1.ts"], false]
		;(listFiles as sinon.SinonStub).resolves(mockFiles)

		// Change workspace path
		;(getWorkspacePath as sinon.SinonStub).returns("/test/new-workspace")

		// Simulate tab change event
		await registeredTabChangeCallback!()

		// Run the debounce timer for workspaceDidReset
		clock.tick(300)

		// Should clear file paths and reset workspace
		assert.ok(mockProvider.postMessageToWebview.calledWith({
			type: "workspaceUpdated",
			filePaths: [],
			openedTabs: [],
		}))

		// Run all remaining timers to complete initialization
		await Promise.resolve() // Wait for initializeFilePaths to complete
		clock.runAll()

		// Should initialize file paths for new workspace
		assert.ok(listFiles.calledWith("/test/new-workspace", true, 1000))
		clock.runAll()
	})

	test("should not update file paths if workspace changes during initialization", async () => {
		// Setup initial workspace path
		;(getWorkspacePath as sinon.SinonStub).returns("/test/workspace")
		workspaceTracker = new WorkspaceTracker(mockProvider)

		// Clear any initialization calls
		sinon.restore()
		;(mockProvider.postMessageToWebview as sinon.SinonStub).resetHistory()

		// Create a promise to control listFiles timing
		let resolveListFiles: (value: [string[], boolean]) => void
		const listFilesPromise = new Promise<[string[], boolean]>((resolve) => {
			resolveListFiles = resolve
		})

		// Setup listFiles to use our controlled promise
		;(listFiles as sinon.SinonStub).callsFake(() => {
			// Change workspace path before listFiles resolves
			;(getWorkspacePath as sinon.SinonStub).returns("/test/changed-workspace")
			return listFilesPromise
		})

		// Start initialization
		const initPromise = workspaceTracker.initializeFilePaths()

		// Resolve listFiles after workspace path change
		resolveListFiles!([["/test/workspace/file1.ts", "/test/workspace/file2.ts"], false])

		// Wait for initialization to complete
		await initPromise
		clock.runAll()

		// Should not update file paths because workspace changed during initialization
		assert.ok(mockProvider.postMessageToWebview.calledWith({
			filePaths: ["/test/workspace/file1.ts", "/test/workspace/file2.ts"],
			openedTabs: [],
			type: "workspaceUpdated",
		}))
	})

	test("should clear resetTimer when calling workspaceDidReset multiple times", async () => {
		expect(registeredTabChangeCallback).not.toBeNull()

		// Set initial workspace path
		;(getWorkspacePath as sinon.SinonStub).returns("/test/workspace")

		// Create tracker instance to set initial prevWorkSpacePath
		workspaceTracker = new WorkspaceTracker(mockProvider)

		// Change workspace path to trigger update
		;(getWorkspacePath as sinon.SinonStub).returns("/test/new-workspace")

		// Call workspaceDidReset through tab change event
		await registeredTabChangeCallback!()

		// Call again before timer completes
		await registeredTabChangeCallback!()

		// Advance timer
		clock.tick(300)

		// Should only have one call to postMessageToWebview
		assert.ok(mockProvider.postMessageToWebview.calledWith({
			type: "workspaceUpdated",
			filePaths: [],
			openedTabs: [],
		}))
		assert.strictEqual(mockProvider.postMessageToWebview.callCount, 1)
	})

	test("should handle dispose with active resetTimer", async () => {
		expect(registeredTabChangeCallback).not.toBeNull()

		// Mock workspace path change to trigger resetTimer
		;(getWorkspacePath as sinon.SinonStub)
			.onFirstCall().returns("/test/workspace")
			.onFirstCall().returns("/test/new-workspace")

		// Trigger resetTimer
		await registeredTabChangeCallback!()

		// Dispose before timer completes
		workspaceTracker.dispose()

		// Advance timer
		clock.tick(300)

		// Should have called dispose on all disposables
		assert.ok(mockDispose.called)

		// No postMessage should be called after dispose
		assert.ok(!mockProvider.postMessageToWebview.called)
	})
// Mock cleanup
})})
