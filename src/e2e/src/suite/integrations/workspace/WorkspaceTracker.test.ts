import * as assert from 'assert'
import * as sinon from 'sinon'
import * as vscode from 'vscode'
/* eslint-disable @typescript-eslint/unbound-method */
import * as vscode from "vscode"
import WorkspaceTracker from "../WorkspaceTracker"
import { TheaProvider } from "../../../core/webview/TheaProvider" // Renamed import
import { listFiles } from "../../../services/glob/list-files"
import { getWorkspacePath } from "../../../utils/path"

// Mock functions - must be defined before jest.mock calls
const mockOnDidCreate = sinon.stub()
const mockOnDidDelete = sinon.stub()
const mockDispose = sinon.stub()

// Store registered tab change callback
let registeredTabChangeCallback: (() => Promise<void>) | null = null

// Mock workspace path
// TODO: Mock setup needs manual migration
// TODO: Mock needs manual migration
// TODO: Implement proper mock with proxyquire

// Mock watcher - must be defined after mockDispose but before // TODO: Mock setup needs manual migration
const mockWatcher: {
	onDidCreate: sinon.SinonStub
	onDidDelete: sinon.SinonStub
	dispose: sinon.SinonStub
} = {
	onDidCreate: mockOnDidCreate.returns({ dispose: mockDispose }),
	onDidDelete: mockOnDidDelete.returns({ dispose: mockDispose }),
	dispose: mockDispose,

// Mock vscode
// TODO: Mock setup needs manual migration
// TODO: Mock needs manual migration
// TODO: Implement proper mock with proxyquire,
	},
	workspace: {
		workspaceFolders: [
			{
				uri: { fsPath: "/test/workspace" },
				name: "test",
				index: 0,
			},
		],
		createFileSystemWatcher: sinon.stub().returns(() => mockWatcher),
		fs: {
			stat: sinon.stub().resolves({ type: 1 }), // FileType.File = 1
		},
	},
	FileType: { File: 1, Directory: 2 },

// TODO: Mock setup needs manual migration
suite("WorkspaceTracker", () => {
	let workspaceTracker: WorkspaceTracker
	let mockProvider: TheaProvider // Renamed type

	setup(() => {
		sinon.restore()
		jest.useFakeTimers()

		// Reset all mock implementations
		registeredTabChangeCallback = null

		// Reset workspace path mock
		;(getWorkspacePath as sinon.SinonStub).returns("/test/workspace")

		// Create provider mock
		mockProvider = {
			postMessageToWebview: sinon.stub().resolves(undefined),
		} as unknown as TheaProvider & { postMessageToWebview: sinon.SinonStub } // Renamed type assertion

		// Create tracker instance
		workspaceTracker = new WorkspaceTracker(mockProvider)

		// Ensure the tab change callback was registered
		expect(registeredTabChangeCallback).not.toBeNull()

	test("should initialize with workspace files", async () => {
		const mockFiles = [["/test/workspace/file1.ts", "/test/workspace/file2.ts"], false]
		;(listFiles as sinon.SinonStub).resolves(mockFiles)

		await workspaceTracker.initializeFilePaths()
		jest.runAllTimers()

		assert.ok(mockProvider.postMessageToWebview.calledWith({
			type: "workspaceUpdated",
			filePaths: expect.arrayContaining(["file1.ts", "file2.ts"])) as unknown[],
			openedTabs: [],

		const firstCall = (mockProvider.postMessageToWebview as sinon.SinonStub).args[0] as [{ filePaths: string[] }]
		assert.strictEqual(firstCall[0].filePaths.length, 2)

	test("should handle file creation events", async () => {
		// Get the creation callback and call it
		const createCalls = mockOnDidCreate.args as [[(args: { fsPath: string }) => Promise<void>]]
		const callback = createCalls[0][0]
		await callback({ fsPath: "/test/workspace/newfile.ts" })
		jest.runAllTimers()

		assert.ok(mockProvider.postMessageToWebview.calledWith({
			type: "workspaceUpdated",
			filePaths: ["newfile.ts"],
			openedTabs: [],

	test("should handle file deletion events", async () => {
		// First add a file
		const createCalls2 = mockOnDidCreate.args as [[(args: { fsPath: string }) => Promise<void>]]
		const createCallback = createCalls2[0][0]
		await createCallback({ fsPath: "/test/workspace/file.ts" })
		jest.runAllTimers()

		// Then delete it
		const deleteCalls = mockOnDidDelete.args as [[(args: { fsPath: string }) => Promise<void>]]
		const deleteCallback = deleteCalls[0][0]
		await deleteCallback({ fsPath: "/test/workspace/file.ts" })
		jest.runAllTimers()

		// The last call should have empty filePaths
		expect(mockProvider.postMessageToWebview).toHaveBeenLastCalledWith({
			type: "workspaceUpdated",
			filePaths: [],
			openedTabs: [],

	test("should handle directory paths correctly", async () => {
		// Mock stat to return directory type
		;(vscode.workspace.fs.stat as sinon.SinonStub).resolvesOnce({ type: 2 }) // FileType.Directory = 2

		const dirCalls = mockOnDidCreate.args as [[(args: { fsPath: string }) => Promise<void>]]
		const callback = dirCalls[0][0]
		await callback({ fsPath: "/test/workspace/newdir" })
		jest.runAllTimers()

		assert.ok(mockProvider.postMessageToWebview.calledWith({
			type: "workspaceUpdated",
			filePaths: expect.arrayContaining(["newdir"])) as unknown[],
			openedTabs: [],

		const lastCall = (mockProvider.postMessageToWebview as sinon.SinonStub).args.slice(-1)[0] as [
			{ filePaths: string[] },

		assert.strictEqual(lastCall[0].filePaths.length, 1)

	test("should respect file limits", async () => {
		// Create array of unique file paths for initial load
		const files = Array.from({ length: 1001 }, (_, i) => `/test/workspace/file${i}.ts`)
		;(listFiles as sinon.SinonStub).resolves([files, false])

		await workspaceTracker.initializeFilePaths()
		jest.runAllTimers()

		// Should only have 1000 files initially
		const expectedFiles = Array.from({ length: 1000 }, (_, i) => `file${i}.ts`).sort()
		const calls = (mockProvider.postMessageToWebview as sinon.SinonStub).args as [{ filePaths: string[] }[]]

		assert.ok(mockProvider.postMessageToWebview.calledWith({
			type: "workspaceUpdated",
			filePaths: expect.arrayContaining(expectedFiles)) as unknown[],
			openedTabs: [],

		assert.strictEqual(calls[0][0].filePaths.length, 1000)

		// Should allow adding up to 2000 total files
		const extraCalls = mockOnDidCreate.args as [[(args: { fsPath: string }) => Promise<void>]]
		const callback = extraCalls[0][0]
		for (let i = 0; i < 1000; i++) {
			await callback({ fsPath: `/test/workspace/extra${i}.ts` })

		jest.runAllTimers()

		const lastCall = (mockProvider.postMessageToWebview as sinon.SinonStub).args.slice(-1)[0] as [
			{ filePaths: string[] },

		assert.strictEqual(lastCall[0].filePaths.length, 2000)

		// Adding one more file beyond 2000 should not increase the count
		await callback({ fsPath: "/test/workspace/toomany.ts" })
		jest.runAllTimers()

		const finalCall = (mockProvider.postMessageToWebview as sinon.SinonStub).args.slice(-1)[0] as [
			{ filePaths: string[] },

		assert.strictEqual(finalCall[0].filePaths.length, 2000)

	test("should clean up watchers and timers on dispose", () => {
		// Set up updateTimer
		const disposeCalls = mockOnDidCreate.args as [[(args: { fsPath: string }) => void]]
		const callback = disposeCalls[0][0]
		callback({ fsPath: "/test/workspace/file.ts" })

		workspaceTracker.dispose()
		assert.ok(mockDispose.called)
		jest.runAllTimers() // Ensure any pending timers are cleared

		// No more updates should happen after dispose
		assert.ok(!mockProvider.postMessageToWebview.called)

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
		jest.advanceTimersByTime(300)

		// Should clear file paths and reset workspace
		assert.ok(mockProvider.postMessageToWebview.calledWith({
			type: "workspaceUpdated",
			filePaths: [],
			openedTabs: [],

		// Run all remaining timers to complete initialization
		await Promise.resolve() // Wait for initializeFilePaths to complete
		jest.runAllTimers()

		// Should initialize file paths for new workspace
		assert.ok(listFiles.calledWith("/test/new-workspace", true, 1000))
		jest.runAllTimers()

	test("should not update file paths if workspace changes during initialization", async () => {
		// Setup initial workspace path
		;(getWorkspacePath as sinon.SinonStub).returns("/test/workspace")
		workspaceTracker = new WorkspaceTracker(mockProvider)

		// Clear any initialization calls
		sinon.restore()
		;(mockProvider.postMessageToWebview as sinon.SinonStub).mockClear()

		// Create a promise to control listFiles timing
		let resolveListFiles: (value: [string[], boolean]) => void
		const listFilesPromise = new Promise<[string[], boolean]>((resolve) => {
			resolveListFiles = resolve

		// Setup listFiles to use our controlled promise
		;(listFiles as sinon.SinonStub).callsFake(() => {
			// Change workspace path before listFiles resolves
			;(getWorkspacePath as sinon.SinonStub).returns("/test/changed-workspace")
			return listFilesPromise

		// Start initialization
		const initPromise = workspaceTracker.initializeFilePaths()

		// Resolve listFiles after workspace path change
		resolveListFiles!([["/test/workspace/file1.ts", "/test/workspace/file2.ts"], false])

		// Wait for initialization to complete
		await initPromise
		jest.runAllTimers()

		// Should not update file paths because workspace changed during initialization
		assert.ok(mockProvider.postMessageToWebview.calledWith({
			filePaths: ["/test/workspace/file1.ts", "/test/workspace/file2.ts"],
			openedTabs: [],
			type: "workspaceUpdated",

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
		jest.advanceTimersByTime(300)

		// Should only have one call to postMessageToWebview
		assert.ok(mockProvider.postMessageToWebview.calledWith({
			type: "workspaceUpdated",
			filePaths: [],
			openedTabs: [],

		assert.strictEqual(mockProvider.postMessageToWebview.callCount, 1)

	test("should handle dispose with active resetTimer", async () => {
		expect(registeredTabChangeCallback).not.toBeNull()

		// Mock workspace path change to trigger resetTimer
		;(getWorkspacePath as sinon.SinonStub)
			.returnsOnce("/test/workspace")
			.returnsOnce("/test/new-workspace")

		// Trigger resetTimer
		await registeredTabChangeCallback!()

		// Dispose before timer completes
		workspaceTracker.dispose()

		// Advance timer
		jest.advanceTimersByTime(300)

		// Should have called dispose on all disposables
		assert.ok(mockDispose.called)

		// No postMessage should be called after dispose
		assert.ok(!mockProvider.postMessageToWebview.called)
