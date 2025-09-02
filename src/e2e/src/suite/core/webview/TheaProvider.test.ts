import * as assert from 'assert'
import * as sinon from 'sinon'
import * as vscode from 'vscode'
/* eslint-disable @typescript-eslint/no-unsafe-return, @typescript-eslint/no-unsafe-assignment, @typescript-eslint/unbound-method */
import * as vscode from "vscode"
import { TheaProvider } from "../TheaProvider" // Renamed import
import { TheaTaskStack } from "../thea/TheaTaskStack" // Renamed import and path
import { TheaStateManager } from "../thea/TheaStateManager" // Renamed import and path
import { TheaApiManager } from "../api/TheaApiManager" // Renamed import
import { TheaTaskHistory } from "../history/TheaTaskHistory" // Renamed import
import { TheaCacheManager } from "../cache/TheaCacheManager" // Renamed import
import { TheaMcpManager } from "../mcp/TheaMcpManager" // Renamed import
import { ContextProxy } from "../../config/ContextProxy"
import { ProviderSettingsManager } from "../../config/ProviderSettingsManager"
import { CustomModesManager } from "../../config/CustomModesManager"
import { TheaTask } from "../../TheaTask" // Renamed import
import { McpServerManager } from "../../../services/mcp/management/McpServerManager"
import { defaultModeSlug } from "../../../shared/modes"
import { HistoryItem } from "../../../shared/HistoryItem"
import { t } from "../../../i18n"

// Mock dependencies
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// Updated mock path
// TODO: Mock setup needs manual migration
// Updated mock path
// TODO: Mock setup needs manual migration
// Updated mock path
// TODO: Mock setup needs manual migration
// Updated mock path
// TODO: Mock setup needs manual migration
// Updated mock path
// TODO: Mock setup needs manual migration
// Updated mock path
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// Updated mock path
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// Mock console to prevent test output noise
console.log = sinon.stub()
console.error = sinon.stub()

suite("TheaProvider", () => {
	// Renamed describe block
	// Setup variables
	let theaProvider: TheaProvider // Renamed variable and type
	let mockContext: vscode.ExtensionContext
	let mockOutputChannel: vscode.OutputChannel
	let mockWebview: vscode.Webview
	let mockWebviewView: vscode.WebviewView
	let mockTheaTask: sinon.SinonStubStatic<TheaTask> // Renamed variable and type

	// Create mocks for manager instances
	const mockTheaTaskStack = {
		// Renamed variable
		addTheaTask: sinon.stub(),
		removeCurrentTheaTask: sinon.stub(),
		getCurrentTheaTask: sinon.stub(),
		getSize: sinon.stub(),
		getTaskStack: sinon.stub(),
		finishSubTask: sinon.stub(),

	const mockTheaStateManager = {
		// Renamed variable
		getState: sinon.stub(),
		updateGlobalState: sinon.stub(),
		getGlobalState: sinon.stub(),
		setValue: sinon.stub(),
		getValue: sinon.stub(),
		getValues: sinon.stub(),
		setValues: sinon.stub(),
		getCustomModes: sinon.stub(),

	const mockTheaApiManager = {
		// Renamed variable
		handleModeSwitch: sinon.stub(),
		updateApiConfiguration: sinon.stub(),
		upsertApiConfiguration: sinon.stub(),
		handleGlamaCallback: sinon.stub(),
		handleOpenRouterCallback: sinon.stub(),
		handleRequestyCallback: sinon.stub(),

	const mockTheaTaskHistory = {
		// Renamed variable
		getTaskWithId: sinon.stub(),
		showTaskWithId: sinon.stub(),
		exportTaskWithId: sinon.stub(),
		deleteTaskWithId: sinon.stub(),
		updateTaskHistory: sinon.stub(),

	const mockTheaCacheManager = {
		// Renamed variable
		ensureCacheDirectoryExists: sinon.stub(),
		ensureSettingsDirectoryExists: sinon.stub(),
		readModelsFromCache: sinon.stub(),
		writeModelsToCache: sinon.stub(),

	const mockTheaMcpManager = {
		// Renamed variable
		setMcpHub: sinon.stub(),
		getMcpHub: sinon.stub(),
		getAllServers: sinon.stub(),
		ensureMcpServersDirectoryExists: sinon.stub(),
		dispose: sinon.stub(),

	const mockContextProxy = {
		initialize: sinon.stub(),
		isInitialized: false,
		extensionUri: { fsPath: "/test/path" },
		extensionMode: vscode.ExtensionMode.Production,
		globalStorageUri: { fsPath: "/test/storage" },
		setValue: sinon.stub(),
		getValue: sinon.stub(),
		getValues: sinon.stub(),
		setValues: sinon.stub(),
		resetAllState: sinon.stub(),

	const mockProviderSettingsManager = {
		resetAllConfigs: sinon.stub(),
		loadConfig: sinon.stub(),

	const mockCustomModesManager = {
		getCustomModes: sinon.stub(),
		resetCustomModes: sinon.stub(),
		dispose: sinon.stub(),

	const mockMcpHub = {
		dispose: sinon.stub(),

	setup(() => {
		sinon.restore()

		// Setup mock extension context
		mockContext = {
			subscriptions: [],
			extension: {
				packageJSON: { version: "1.0.0" },
			},
		} as unknown as vscode.ExtensionContext

		// Setup mock output channel
		mockOutputChannel = {
			appendLine: sinon.stub(),
			clear: sinon.stub(),
			dispose: sinon.stub(),
		} as unknown as vscode.OutputChannel

		// Setup mock webview
		mockWebview = {
			onDidReceiveMessage: sinon.stub(),
			postMessage: sinon.stub().resolves(true),
			html: "",
			options: {},
			cspSource: "https://test-source",
			asWebviewUri: sinon.stub().callsFake((uri) => uri),
		} as unknown as vscode.Webview

		// Setup mock webview view
		mockWebviewView = {
			webview: mockWebview,
			onDidDispose: sinon.stub(),
			onDidChangeVisibility: sinon.stub(),
			visible: true,
			description: "Test view",
			title: "Test Title",
			viewType: "test.viewType",
			show: sinon.stub(),
			dispose: sinon.stub(),
		} as unknown as vscode.WebviewView

		// Setup mock TheaTask
		mockTheaTask = {
			// Renamed variable
			taskId: "test-task-id",
			instanceId: "test-instance-id",
			abortTask: sinon.stub(),
			resumePausedTask: sinon.stub(),
			parentTask: undefined,
			rootTask: undefined,
			taskNumber: 1,
			customInstructions: "",
			isStreaming: false,
			didFinishAbortingStream: false,
			isWaitingForFirstChunk: false,
			clineMessages: [],
			abandoned: false,
			api: { getModel: sinon.stub().returns({ id: "test-model-id" }) },
			diffStrategy: { getName: sinon.stub().returns("test-diff-strategy") },
			taskStateManager: {
				// Add mock state manager
				clineMessages: [], // Provide at least clineMessages
				apiConversationHistory: [], // Add other properties if needed by tests
				getTokenUsage: jest
					.fn()
					.returns({ totalTokensIn: 0, totalTokensOut: 0, totalCost: 0, contextTokens: 0 }),
			},
		} as unknown as sinon.SinonStubStatic<TheaTask>

		// Setup mock manager responses
		;(TheaTaskStack as sinon.SinonStub).callsFake(() => mockTheaTaskStack) // Renamed class and variable
		;(TheaStateManager as sinon.SinonStub).callsFake(() => mockTheaStateManager) // Renamed class and variable
		;(TheaApiManager as sinon.SinonStub).callsFake(() => mockTheaApiManager) // Renamed class and variable
		;(TheaTaskHistory as sinon.SinonStub).callsFake(() => mockTheaTaskHistory) // Renamed class and variable
		;(TheaCacheManager as sinon.SinonStub).callsFake(() => mockTheaCacheManager) // Renamed class and variable
		;(TheaMcpManager as sinon.SinonStub).callsFake(() => mockTheaMcpManager) // Renamed class and variable
		;(ContextProxy as sinon.SinonStub).callsFake(() => mockContextProxy)
		;(ProviderSettingsManager as unknown as sinon.SinonStub).callsFake(() => mockProviderSettingsManager)
		;(CustomModesManager as sinon.SinonStub).callsFake(() => mockCustomModesManager)

		// Mock McpServerManager.getInstance to return a mock McpHub - fix the mock implementation
		const mockGetInstance = sinon.stub().resolves(mockMcpHub)
		McpServerManager.getInstance = mockGetInstance

		// Setup i18n translation mock - fix the mock implementation
		jest.mocked(t).callsFake((key: string) => {
			const translations: Record<string, string> = {
				"common:confirmation.reset_state": "Are you sure you want to reset all state?",
				"common:answers.yes": "Yes",

			return translations[key] || key

		// Create instance of TheaProvider
		theaProvider = new TheaProvider(mockContext, mockOutputChannel, "sidebar") // Renamed variable and constructor

	teardown(() => {
		sinon.restore()

	test("constructor initializes dependencies correctly", () => {
		assert.ok(TheaTaskStack.called) // Renamed class
		assert.ok(TheaStateManager.called) // Renamed class
		assert.ok(TheaApiManager.called) // Renamed class
		assert.ok(TheaTaskHistory.called) // Renamed class
		assert.ok(TheaCacheManager.called) // Renamed class
		assert.ok(TheaMcpManager.called) // Renamed class
		assert.ok(ContextProxy.calledWith(mockContext))
		assert.ok(ProviderSettingsManager.calledWith(mockContext))
		assert.ok(CustomModesManager.called)

		// Verify McpServerManager was called
		assert.ok(McpServerManager.getInstance.calledWith(mockContext, theaProvider))

		// Check initial properties
		assert.strictEqual(theaProvider.isViewLaunched, false)
		assert.strictEqual(theaProvider.contextProxy, mockContextProxy)
		assert.strictEqual(theaProvider.providerSettingsManager, mockProviderSettingsManager)
		assert.strictEqual(theaProvider.customModesManager, mockCustomModesManager) // Renamed variable
		assert.strictEqual(theaProvider["theaTaskStackManager"], mockTheaTaskStack) // Renamed property access and variable
		assert.strictEqual(theaProvider["theaStateManager"], mockTheaStateManager) // Renamed property access and variable
		assert.strictEqual(theaProvider["theaApiManager"], mockTheaApiManager) // Renamed property access and variable
		assert.strictEqual(theaProvider["theaTaskHistoryManager"], mockTheaTaskHistory) // Renamed property access and variable
		assert.strictEqual(theaProvider["theaCacheManager"], mockTheaCacheManager) // Renamed property access and variable
		assert.strictEqual(theaProvider["theaMcpManager"], mockTheaMcpManager) // Renamed property access and variable

	test("resolveWebviewView initializes webview correctly", async () => {
		// Setup state manager to return mock state
		mockTheaStateManager.getState.resolves({
			// Renamed variable
			soundEnabled: true,
			terminalShellIntegrationTimeout: 5000,
			ttsEnabled: false,
			ttsSpeed: 1.0,

		// Execute
		await theaProvider.resolveWebviewView(mockWebviewView)

		// Verify
		assert.strictEqual(theaProvider.view, mockWebviewView)
		assert.ok(mockWebview.onDidReceiveMessage.called)
		assert.ok(mockWebviewView.onDidChangeVisibility.called)
		assert.ok(mockWebviewView.onDidDispose.called)
		assert.deepStrictEqual(mockWebview.options, {
			enableScripts: true,
			localResourceRoots: [mockContextProxy.extensionUri],
		assert.ok(mockTheaTaskStack.removeCurrentTheaTask.called) // Updated to new method name

	test("dispose cleans up resources properly", async () => {
		// Setup
		theaProvider.view = mockWebviewView

		// Execute
		await theaProvider.dispose()

		// Verify
		assert.ok(mockTheaTaskStack.removeCurrentTheaTask.called) // Updated to new method name
		// WebviewView doesn't have a dispose method, we should check if unregisterProvider was called
		// and if clineMcpManager.dispose was called, which are the important cleanup steps
		assert.ok(mockTheaMcpManager.dispose.called) // Renamed variable
		assert.ok(McpServerManager.unregisterProvider.calledWith(theaProvider)) // Renamed variable

	test("initClineWithTask creates a new TheaTask instance and adds it to stack", async () => {
		// Updated test description
		// Setup state
		mockTheaStateManager.getState.resolves({
			// Renamed variable
			apiConfiguration: { apiProvider: "test-provider" },
			customModePrompts: {},
			diffEnabled: true,
			enableCheckpoints: true,
			checkpointStorage: "task",
			fuzzyMatchThreshold: 1.0,
			mode: defaultModeSlug,
			customInstructions: "test instructions",
			experiments: {},

		// Mock stack size for task number
		mockTheaTaskStack.getSize.returns(0) // Renamed variable

		// Mock TheaTask constructor
		;(TheaTask as unknown as sinon.SinonStub).callsFake(() => mockTheaTask) // Renamed class and variable

		// Execute
		const result = await theaProvider.initWithTask("test task") // Updated to new method name

		// Verify
		assert.ok(mockTheaStateManager.getState.called) // Renamed variable
		assert.ok(TheaTask.calledWith(
			expect.objectContaining({
				// Renamed class
				provider: theaProvider, // Renamed variable
				task: "test task",
				enableDiff: true,
				enableCheckpoints: true,
				checkpointStorage: "task",
				fuzzyMatchThreshold: 1.0,
				taskNumber: 1,
			})),

		assert.ok(mockTheaTaskStack.addTheaTask.calledWith(mockTheaTask)) // Updated to new method name
		assert.strictEqual(result, mockTheaTask) // Renamed variable

	test("initClineWithHistoryItem initializes TheaTask from history and adds to stack", async () => {
		// Updated test description
		// Setup state
		mockTheaStateManager.getState.resolves({
			// Renamed variable
			apiConfiguration: { apiProvider: "test-provider" },
			customModePrompts: {},
			diffEnabled: true,
			enableCheckpoints: true,
			checkpointStorage: "task",
			fuzzyMatchThreshold: 1.0,
			mode: defaultModeSlug,
			customInstructions: "test instructions",
			experiments: {},

		// Create history item
		const historyItem: HistoryItem & { rootTask?: TheaTask; parentTask?: TheaTask } = {
			// Renamed type
			id: "test-history-id",
			task: "test history task",
			ts: Date.now(),
			number: 2,
			tokensIn: 0,
			tokensOut: 0,
			totalCost: 0,

		// Mock TheaTask constructor
		;(TheaTask as unknown as sinon.SinonStub).callsFake(() => mockTheaTask) // Renamed class and variable

		// Execute
		const result = await theaProvider.initWithHistoryItem(historyItem) // Updated to new method name

		// Verify
		assert.ok(mockTheaTaskStack.removeCurrentTheaTask.called) // Updated to new method name
		assert.ok(mockTheaStateManager.getState.called) // Renamed variable
		assert.ok(TheaTask.calledWith(
			expect.objectContaining({
				// Renamed class
				provider: theaProvider, // Renamed variable
				historyItem,
				enableDiff: true,
				enableCheckpoints: true,
				checkpointStorage: expect.any(String)),
				fuzzyMatchThreshold: 1.0,
				taskNumber: 2,
			}),

		assert.ok(mockTheaTaskStack.addTheaTask.calledWith(mockTheaTask)) // Updated to new method name
		assert.strictEqual(result, mockTheaTask) // Renamed variable

	test("cancelTask aborts current task and reloads from history", async () => {
		// Setup
		const historyItem = {
			id: "test-task-id",
			task: "test task",
			ts: Date.now(),

		mockTheaTaskStack.getCurrentTheaTask.returns(mockTheaTask) // Updated to new method name
		mockTheaTaskHistory.getTaskWithId.resolves({ historyItem }) // Renamed variable

		// Execute
		await theaProvider.cancelTask()

		// Verify
		assert.ok(mockTheaTask.abortTask.called) // Renamed variable
		assert.strictEqual(mockTheaTask.abandoned, true) // Renamed variable
		assert.ok(mockTheaTaskHistory.getTaskWithId.calledWith("test-task-id")) // Renamed variable

	test("postStateToWebview sends correct state to webview", async () => {
		// Setup
		const mockState = {
			apiConfiguration: { apiProvider: "test-provider" },
			mode: "default",
			soundEnabled: true,
			taskHistory: [{ id: "task1", ts: 1000, task: "Task 1" }],

		mockTheaStateManager.getState.resolves(mockState) // Renamed variable
		mockTheaTaskStack.getCurrentTheaTask.returns(undefined) // Updated to new method name
		mockCustomModesManager.getCustomModes.resolves([])
		mockTheaMcpManager.getAllServers.returns([]) // Renamed variable

		theaProvider.view = mockWebviewView

		// Execute
		await theaProvider.postStateToWebview()

		// Verify
		assert.ok(mockTheaStateManager.getState.called) // Renamed variable
		assert.ok(mockWebview.postMessage.calledWith({
			type: "state",
			state: expect.objectContaining({
				version: "1.0.0",
				mode: "default",
				soundEnabled: true,
				taskHistory: expect.arrayContaining([expect.objectContaining({ id: "task1", task: "Task 1" }))]),
				currentTaskItem: undefined,
				clineMessages: [],
			}),

	test("resetState calls appropriate reset methods", async () => {
		// Mock the vscode.window.showInformationMessage to simulate user clicking "Yes"
		// Fix: Use proper type for the mock function to match VS Code API
		jest.mocked(vscode.window.showInformationMessage).callsFake(
			// eslint-disable-next-line @typescript-eslint/no-unused-vars
			(_message: string, _options: vscode.MessageOptions, ..._items: vscode.MessageItem[]) => {
				// Return the "Yes" option to simulate user confirmation
				return Promise.resolve("Yes" as unknown as vscode.MessageItem)
			},

		theaProvider.view = mockWebviewView

		// Execute
		await theaProvider.resetState()

		// Verify
		assert.ok(mockContextProxy.resetAllState.called)
		assert.ok(mockProviderSettingsManager.resetAllConfigs.called)
		assert.ok(mockCustomModesManager.resetCustomModes.called)
		assert.ok(mockTheaTaskStack.removeCurrentTheaTask.called) // Updated to new method name
		assert.strictEqual(mockWebview.postMessage.callCount, 2)

	test("handleModeSwitchAndUpdateCline delegates to managers correctly", async () => {
		// Setup
		const mockApiConfig = { apiProvider: "updated-provider" }
		mockTheaApiManager.handleModeSwitch.resolves(mockApiConfig) // Renamed variable
		mockTheaTaskStack.getCurrentTheaTask.returns(mockTheaTask) // Updated to new method name

		// Execute
		await theaProvider.handleModeSwitchAndUpdate("default") // Updated to new method name

		// Verify
		assert.ok(mockTheaApiManager.handleModeSwitch.calledWith("default")) // Renamed variable
		assert.ok(mockTheaTask.api !== undefined) // Checks that api was updated // Renamed variable

	test("updateCustomInstructions updates context value and current cline", async () => {
		// Setup
		const newInstructions = "New instructions"
		mockTheaTaskStack.getCurrentTheaTask.returns(mockTheaTask) // Updated to new method name

		// Execute
		await theaProvider.updateCustomInstructions(newInstructions) // Renamed variable

		// Verify
		assert.ok(mockContextProxy.setValue.calledWith("customInstructions", newInstructions))
		assert.strictEqual(mockTheaTask.customInstructions, newInstructions) // Renamed variable

	test("getTelemetryProperties returns correct properties", async () => {
		// Setup
		mockTheaStateManager.getState.resolves({
			// Renamed variable
			mode: "default",
			apiConfiguration: { apiProvider: "test-provider" },
			language: "en",
		mockTheaTaskStack.getCurrentTheaTask.returns(mockTheaTask) // Updated to new method name

		// Execute
		const result = await theaProvider.getTelemetryProperties()

		// Verify
		assert.deepStrictEqual(result, 
			expect.objectContaining({
				appVersion: "1.0.0",
				mode: "default",
				apiProvider: "test-provider",
				language: "en",
				modelId: "test-model-id",
				diffStrategy: "test-diff-strategy",
			}),

	test("proxy methods delegate to the appropriate manager instances", async () => {
		// Setup
		const mockApiConfig = { apiProvider: "test-provider" }
		mockTheaStateManager.getValue.returns(mockApiConfig) // Renamed variable
		mockTheaStateManager.getState.resolves({ mode: "default" }) // Renamed variable
		mockProviderSettingsManager.loadConfig.resolves(mockApiConfig)
		mockTheaTaskStack.getCurrentTheaTask.returns(mockTheaTask) // Updated to new method name

		// Test state manager proxy methods
		await theaProvider.setValue("currentApiConfigName", "test-config") // Renamed variable
		assert.ok(mockTheaStateManager.setValue.calledWith("currentApiConfigName", "test-config")) // Renamed variable
		assert.ok(mockProviderSettingsManager.loadConfig.calledWith("test-config"))

		// Use a valid key for ProviderSettings instead of "apiConfig"
		const value = theaProvider.getValue("apiProvider") // Renamed variable
		assert.ok(mockTheaStateManager.getValue.calledWith("apiProvider")) // Renamed variable
		assert.deepStrictEqual(value, mockApiConfig)

		// Test stack manager proxy methods
		await theaProvider.addToStack(mockTheaTask) // Updated to new method name
		assert.ok(mockTheaTaskStack.addTheaTask.calledWith(mockTheaTask)) // Updated to new method name

		const currentTheaTask = theaProvider.getCurrent() // Updated to new method name
		assert.ok(mockTheaTaskStack.getCurrentTheaTask.called) // Updated to new method name
		assert.strictEqual(currentTheaTask, mockTheaTask) // Renamed variable

		// Call getStackSize and verify the method was called
		theaProvider.getStackSize() // Updated to new method name
		assert.ok(mockTheaTaskStack.getSize.called) // Renamed variable
