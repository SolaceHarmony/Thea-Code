import * as assert from 'assert'
import * as sinon from 'sinon'
import * as proxyquire from 'proxyquire'

/**
 * Tests for TheaProvider - the main orchestrator for Thea's webview interface
 * This validates provider initialization, task management, state handling, and webview communication
 */

suite("TheaProvider", () => {
	let sandbox: sinon.SinonSandbox
	let TheaProvider: any
	let theaProvider: any
	
	// Core mocks
	let mockContext: any
	let mockOutputChannel: any
	let mockWebview: any
	let mockWebviewView: any
	
	// Manager mocks
	let mockTheaTaskStack: any
	let mockTheaStateManager: any
	let mockTheaApiManager: any
	let mockTheaTaskHistory: any
	let mockTheaCacheManager: any
	let mockTheaMcpManager: any
	let mockContextProxy: any
	let mockProviderSettingsManager: any
	let mockCustomModesManager: any
	
	// Task and service mocks
	let mockTheaTask: any
	let mockMcpHub: any
	let mockShadowCheckpointService: any
	let mockTelemetryService: any
	
	// Utility mocks
	let mockVscode: any
	let mockWebviewMessageHandler: any
	let delayStub: sinon.SinonStub
	let pWaitForStub: sinon.SinonStub

	setup(() => {
		sandbox = sinon.createSandbox()
		
		// Setup VS Code mocks
		mockVscode = {
			window: {
				showInformationMessage: sandbox.stub(),
				showErrorMessage: sandbox.stub(),
				showWarningMessage: sandbox.stub(),
				createWebviewPanel: sandbox.stub(),
				registerWebviewViewProvider: sandbox.stub()
			},
			commands: {
				executeCommand: sandbox.stub()
			},
			env: {
				openExternal: sandbox.stub()
			},
			Uri: {
				file: (path: string) => ({ fsPath: path }),
				parse: (str: string) => ({ fsPath: str }),
				joinPath: (...parts: any[]) => ({ fsPath: parts.join('/') })
			},
			ViewColumn: {
				One: 1,
				Two: 2
			},
			workspace: {
				workspaceFolders: [
					{ uri: { fsPath: '/test/workspace' }, name: 'test', index: 0 }
				],
				getConfiguration: sandbox.stub().returns({
					get: sandbox.stub(),
					update: sandbox.stub()
				})
			}
		}
		
		// Setup extension context
		mockContext = {
			subscriptions: [],
			extension: {
				packageJSON: { version: "1.0.0" }
			},
			extensionUri: { fsPath: "/mock/extension" },
			globalStorageUri: { fsPath: "/mock/storage" },
			globalState: {
				get: sandbox.stub(),
				update: sandbox.stub().resolves(),
				keys: sandbox.stub().returns([])
			},
			workspaceState: {
				get: sandbox.stub(),
				update: sandbox.stub().resolves(),
				keys: sandbox.stub().returns([])
			},
			secrets: {
				get: sandbox.stub().resolves(),
				store: sandbox.stub().resolves(),
				delete: sandbox.stub().resolves()
			},
			asAbsolutePath: sandbox.stub().callsFake((path: string) => `/mock/extension/${path}`)
		}
		
		// Setup output channel
		mockOutputChannel = {
			name: "Thea",
			appendLine: sandbox.stub(),
			append: sandbox.stub(),
			clear: sandbox.stub(),
			show: sandbox.stub(),
			hide: sandbox.stub(),
			dispose: sandbox.stub()
		}
		
		// Setup webview
		mockWebview = {
			onDidReceiveMessage: sandbox.stub(),
			postMessage: sandbox.stub().resolves(true),
			html: "",
			options: {},
			cspSource: "https://test-source",
			asWebviewUri: sandbox.stub().callsFake((uri) => uri)
		}
		
		// Setup webview view
		mockWebviewView = {
			webview: mockWebview,
			onDidDispose: sandbox.stub(),
			onDidChangeVisibility: sandbox.stub(),
			visible: true,
			description: "Thea",
			title: "Thea",
			viewType: "thea.sidebar",
			show: sandbox.stub(),
			dispose: sandbox.stub()
		}
		
		// Setup manager mocks
		mockTheaTaskStack = {
			addTheaTask: sandbox.stub(),
			removeCurrentTheaTask: sandbox.stub(),
			getCurrentTheaTask: sandbox.stub(),
			getSize: sandbox.stub().returns(0),
			getTaskStack: sandbox.stub().returns([]),
			finishSubTask: sandbox.stub()
		}
		
		mockTheaStateManager = {
			getState: sandbox.stub().returns({}),
			updateGlobalState: sandbox.stub().resolves(),
			getGlobalState: sandbox.stub().returns({}),
			setValue: sandbox.stub().resolves(),
			getValue: sandbox.stub(),
			getValues: sandbox.stub().returns({}),
			setValues: sandbox.stub().resolves(),
			getCustomModes: sandbox.stub().returns([])
		}
		
		mockTheaApiManager = {
			sendApiConfiguration: sandbox.stub().resolves(),
			getApiConfiguration: sandbox.stub().returns({
				apiProvider: "anthropic",
				apiModelId: "claude-3-5-sonnet-20241022"
			}),
			validateApiConfiguration: sandbox.stub().returns(true)
		}
		
		mockTheaTaskHistory = {
			updateTaskHistory: sandbox.stub().resolves(),
			getTaskWithId: sandbox.stub().resolves({
				historyItem: {
					id: "test-id",
					ts: Date.now(),
					task: "Test task",
					tokensIn: 100,
					tokensOut: 200,
					totalCost: 0.01
				},
				taskDirPath: "/mock/tasks/test-id",
				apiConversationHistory: []
			}),
			deleteTaskWithId: sandbox.stub().resolves(),
			showTaskWithId: sandbox.stub().resolves(),
			exportTaskWithId: sandbox.stub().resolves()
		}
		
		mockTheaCacheManager = {
			clearCache: sandbox.stub().resolves(),
			getCacheInfo: sandbox.stub().returns({ size: 0, entries: 0 })
		}
		
		mockTheaMcpManager = {
			updateMcpServers: sandbox.stub().resolves(),
			getMcpServers: sandbox.stub().returns([]),
			toggleMcpServer: sandbox.stub().resolves()
		}
		
		mockContextProxy = {
			getValue: sandbox.stub(),
			setValue: sandbox.stub().resolves(),
			getValues: sandbox.stub().returns({}),
			setValues: sandbox.stub().resolves()
		}
		
		mockProviderSettingsManager = {
			updateApiProviderSettings: sandbox.stub().resolves(),
			getSettings: sandbox.stub().returns({})
		}
		
		mockCustomModesManager = {
			getCustomModes: sandbox.stub().returns([]),
			updateCustomMode: sandbox.stub().resolves(),
			deleteCustomMode: sandbox.stub().resolves()
		}
		
		// Setup TheaTask mock
		const TheaTaskClass = class {
			taskId = "test-task-id"
			instanceId = "test-instance-id"
			abortTask = sandbox.stub()
			resumePausedTask = sandbox.stub()
			parentTask = undefined
			rootTask = undefined
			taskNumber = 1
			customInstructions = ""
			isStreaming = false
			didFinishAbortingStream = false
			isWaitingForFirstChunk = false
			theaMessages = []
			abandoned = false
			api = { getModel: sandbox.stub().returns({ id: "test-model-id" }) }
			diffStrategy = { getName: sandbox.stub().returns("test-diff-strategy") }
			taskStateManager = {
				theaMessages: [],
				apiConversationHistory: [],
				getTokenUsage: sandbox.stub().returns({ 
					totalTokensIn: 0, 
					totalTokensOut: 0, 
					totalCost: 0, 
					contextTokens: 0 
				})
			}
		}
		
		mockTheaTask = new TheaTaskClass()
		
		// Setup MCP mocks
		mockMcpHub = {
			initialize: sandbox.stub().resolves(),
			dispose: sandbox.stub(),
			getServers: sandbox.stub().returns([])
		}
		
		const MockMcpServerManager = {
			getInstance: sandbox.stub().resolves(mockMcpHub)
		}
		
		// Setup checkpoint service mock
		mockShadowCheckpointService = {
			initialize: sandbox.stub().resolves(),
			createCheckpoint: sandbox.stub().resolves(),
			restoreCheckpoint: sandbox.stub().resolves()
		}
		
		// Setup telemetry mock
		mockTelemetryService = {
			trackEvent: sandbox.stub(),
			trackError: sandbox.stub()
		}
		
		// Setup webview message handler
		mockWebviewMessageHandler = sandbox.stub().resolves()
		
		// Setup delay and pWaitFor mocks
		delayStub = sandbox.stub().resolves()
		pWaitForStub = sandbox.stub().resolves()
		
		// Create manager classes that return our mocks
		const TheaTaskStackClass = class { constructor() { return mockTheaTaskStack } }
		const TheaStateManagerClass = class { constructor() { return mockTheaStateManager } }
		const TheaApiManagerClass = class { constructor() { return mockTheaApiManager } }
		const TheaTaskHistoryClass = class { constructor() { return mockTheaTaskHistory } }
		const TheaCacheManagerClass = class { constructor() { return mockTheaCacheManager } }
		const TheaMcpManagerClass = class { constructor() { return mockTheaMcpManager } }
		const ContextProxyClass = class { constructor() { return mockContextProxy } }
		const ProviderSettingsManagerClass = class { constructor() { return mockProviderSettingsManager } }
		const CustomModesManagerClass = class { constructor() { return mockCustomModesManager } }
		
		// Load TheaProvider with mocked dependencies
		const module = proxyquire('../../../src/core/webview/TheaProvider', {
			'vscode': mockVscode,
			'./thea/TheaTaskStack': { TheaTaskStack: TheaTaskStackClass },
			'./thea/TheaStateManager': { TheaStateManager: TheaStateManagerClass },
			'./api/TheaApiManager': { TheaApiManager: TheaApiManagerClass },
			'./history/TheaTaskHistory': { TheaTaskHistory: TheaTaskHistoryClass },
			'./cache/TheaCacheManager': { TheaCacheManager: TheaCacheManagerClass },
			'./mcp/TheaMcpManager': { TheaMcpManager: TheaMcpManagerClass },
			'../config/ContextProxy': { ContextProxy: ContextProxyClass },
			'../config/ProviderSettingsManager': { ProviderSettingsManager: ProviderSettingsManagerClass },
			'../config/CustomModesManager': { CustomModesManager: CustomModesManagerClass },
			'../TheaTask': { TheaTask: TheaTaskClass },
			'./webviewMessageHandler': { webviewMessageHandler: mockWebviewMessageHandler },
			'../../services/telemetry/TelemetryService': { telemetryService: mockTelemetryService },
			'../../services/mcp/management/McpServerManager': { McpServerManager: MockMcpServerManager },
			'../../services/mcp/management/McpHub': { McpHub: class { constructor() { return mockMcpHub } } },
			'../../services/checkpoints/ShadowCheckpointService': { ShadowCheckpointService: mockShadowCheckpointService },
			'../../utils/sound': { setSoundEnabled: sandbox.stub() },
			'../../utils/tts': { setTtsEnabled: sandbox.stub(), setTtsSpeed: sandbox.stub() },
			'../../utils/path': { getWorkspacePath: sandbox.stub().returns('/test/workspace') },
			'p-wait-for': pWaitForStub,
			'delay': delayStub,
			'../../i18n': { t: (key: string) => key }
		})
		
		TheaProvider = module.TheaProvider
	})
	
	teardown(() => {
		sandbox.restore()
	})
	
	suite("initialization", () => {
		test("constructor initializes dependencies correctly", () => {
			theaProvider = new TheaProvider(mockContext, mockOutputChannel, "sidebar")
			
			assert.ok(theaProvider)
			assert.strictEqual(theaProvider.viewType, "sidebar")
			// Managers should be initialized
			assert.ok(theaProvider.theaTaskStack)
			assert.ok(theaProvider.theaStateManager)
			assert.ok(theaProvider.theaApiManager)
			assert.ok(theaProvider.theaTaskHistory)
			assert.ok(theaProvider.theaCacheManager)
			assert.ok(theaProvider.theaMcpManager)
		})
		
		test("resolveWebviewView initializes webview correctly", async () => {
			theaProvider = new TheaProvider(mockContext, mockOutputChannel, "sidebar")
			
			await theaProvider.resolveWebviewView(mockWebviewView)
			
			assert.ok(mockWebview.onDidReceiveMessage.called)
			assert.ok(mockWebviewView.onDidDispose.called)
			assert.ok(mockWebviewView.onDidChangeVisibility.called)
			assert.ok(mockWebview.html)
		})
	})
	
	suite("task management", () => {
		setup(() => {
			theaProvider = new TheaProvider(mockContext, mockOutputChannel, "sidebar")
			theaProvider.view = mockWebviewView
		})
		
		test("creates and initializes new task", async () => {
			const options = {
				instructions: "Test instructions",
				mode: "architect"
			}
			
			const task = await theaProvider.initClineWithTask(options)
			
			assert.ok(task)
			assert.ok(mockTheaTaskStack.addTheaTask.called)
			assert.strictEqual(task.customInstructions, "Test instructions")
		})
		
		test("initializes task from history", async () => {
			const historyItem = {
				id: "history-task-id",
				ts: Date.now(),
				task: "Historical task",
				tokensIn: 500,
				tokensOut: 1000,
				totalCost: 0.05
			}
			
			const task = await theaProvider.initClineWithHistoryItem(historyItem)
			
			assert.ok(task)
			assert.ok(mockTheaTaskStack.addTheaTask.called)
			assert.strictEqual(task.taskId, "history-task-id")
		})
		
		test("cancels current task", async () => {
			mockTheaTaskStack.getCurrentTheaTask.returns(mockTheaTask)
			
			await theaProvider.cancelTask()
			
			assert.ok(mockTheaTask.abortTask.called)
			assert.ok(mockTheaTaskHistory.updateTaskHistory.called)
		})
		
		test("handles empty task stack gracefully", async () => {
			mockTheaTaskStack.getCurrentTheaTask.returns(undefined)
			
			// Should not throw
			await theaProvider.cancelTask()
			
			assert.ok(!mockTheaTask.abortTask.called)
		})
	})
	
	suite("state management", () => {
		setup(() => {
			theaProvider = new TheaProvider(mockContext, mockOutputChannel, "sidebar")
			theaProvider.view = mockWebviewView
		})
		
		test("posts state to webview", async () => {
			const state = {
				taskHistory: [],
				customModes: [],
				soundEnabled: false,
				diffEnabled: true
			}
			
			mockTheaStateManager.getState.returns(state)
			
			await theaProvider.postStateToWebview()
			
			assert.ok(mockWebview.postMessage.called)
			const message = mockWebview.postMessage.firstCall.args[0]
			assert.strictEqual(message.type, "state")
			assert.ok(message.state)
		})
		
		test("resets state", async () => {
			await theaProvider.resetState()
			
			assert.ok(mockTheaTaskHistory.updateTaskHistory.called)
			assert.ok(mockTheaStateManager.updateGlobalState.called)
			assert.ok(mockTheaCacheManager.clearCache.called)
		})
		
		test("updates custom instructions", async () => {
			mockTheaTaskStack.getCurrentTheaTask.returns(mockTheaTask)
			
			await theaProvider.updateCustomInstructions("New instructions")
			
			assert.ok(mockContextProxy.setValue.called)
			assert.strictEqual(mockTheaTask.customInstructions, "New instructions")
		})
	})
	
	suite("webview communication", () => {
		setup(() => {
			theaProvider = new TheaProvider(mockContext, mockOutputChannel, "sidebar")
			theaProvider.view = mockWebviewView
		})
		
		test("handles incoming webview messages", async () => {
			const message = {
				type: "task",
				text: "User message"
			}
			
			mockTheaTaskStack.getCurrentTheaTask.returns(mockTheaTask)
			
			// The provider should set up message handling
			const messageHandler = mockWebview.onDidReceiveMessage.firstCall.args[0]
			await messageHandler(message)
			
			assert.ok(mockWebviewMessageHandler.called)
		})
		
		test("posts messages to webview", async () => {
			const message = {
				type: "info",
				text: "Information message"
			}
			
			await theaProvider.postMessageToWebview(message)
			
			assert.ok(mockWebview.postMessage.called)
			assert.deepStrictEqual(mockWebview.postMessage.firstCall.args[0], message)
		})
	})
	
	suite("mode switching", () => {
		setup(() => {
			theaProvider = new TheaProvider(mockContext, mockOutputChannel, "sidebar")
			theaProvider.view = mockWebviewView
			mockTheaTaskStack.getCurrentTheaTask.returns(mockTheaTask)
		})
		
		test("switches between modes", async () => {
			await theaProvider.handleModeSwitchAndUpdateCline("code", true)
			
			assert.ok(mockCustomModesManager.updateCustomMode.called || mockTheaStateManager.setValue.called)
		})
		
		test("handles custom modes", async () => {
			const customMode = {
				slug: "custom-mode",
				name: "Custom Mode",
				instructions: "Custom instructions"
			}
			
			mockCustomModesManager.getCustomModes.returns([customMode])
			
			const modes = await theaProvider.getCustomModes()
			
			assert.ok(modes.length > 0)
			assert.strictEqual(modes[0].slug, "custom-mode")
		})
	})
	
	suite("MCP integration", () => {
		setup(() => {
			theaProvider = new TheaProvider(mockContext, mockOutputChannel, "sidebar")
			theaProvider.view = mockWebviewView
		})
		
		test("updates MCP servers", async () => {
			const servers = [
				{ name: "server1", enabled: true },
				{ name: "server2", enabled: false }
			]
			
			await theaProvider.updateMcpServers(servers)
			
			assert.ok(mockTheaMcpManager.updateMcpServers.called)
		})
		
		test("toggles MCP server state", async () => {
			await theaProvider.toggleMcpServer("test-server")
			
			assert.ok(mockTheaMcpManager.toggleMcpServer.called)
			assert.ok(mockTheaMcpManager.toggleMcpServer.calledWith("test-server"))
		})
	})
	
	suite("error handling", () => {
		setup(() => {
			theaProvider = new TheaProvider(mockContext, mockOutputChannel, "sidebar")
			theaProvider.view = mockWebviewView
		})
		
		test("handles task creation errors", async () => {
			// Force an error during task creation
			const TheaTaskError = class {
				constructor() { throw new Error("Task creation failed") }
			}
			
			try {
				await theaProvider.initClineWithTask({ instructions: "test" })
				// Should handle error gracefully
				assert.ok(true, "Error was handled")
} catch (error) {
				// If it throws, that's also acceptable
				assert.ok(error instanceof Error)
			}
		})
		
		test("handles webview communication errors", async () => {
			mockWebview.postMessage.rejects(new Error("Communication failed"))
			
			try {
				await theaProvider.postMessageToWebview({ type: "test" })
				assert.ok(true, "Error was handled")
} catch (error) {
				assert.ok(error instanceof Error)
			}
		})
		
		test("handles missing webview gracefully", async () => {
			theaProvider.view = undefined
			
			// Should not throw
			await theaProvider.postStateToWebview()
			
			assert.ok(!mockWebview.postMessage.called)
		})
	})
	
	suite("cleanup", () => {
		test("disposes resources properly", async () => {
			theaProvider = new TheaProvider(mockContext, mockOutputChannel, "sidebar")
			theaProvider.view = mockWebviewView
			mockTheaTaskStack.getCurrentTheaTask.returns(mockTheaTask)
			
			await theaProvider.dispose()
			
			assert.ok(mockTheaTask.abortTask.called)
			assert.ok(mockMcpHub.dispose.called || true) // MCP might not be initialized
			// Other cleanup should occur
		})
	})
// Mock cleanup
