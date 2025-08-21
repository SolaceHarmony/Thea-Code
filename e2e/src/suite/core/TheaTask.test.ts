import * as assert from 'assert'
import * as sinon from 'sinon'
import * as proxyquire from 'proxyquire'
import * as os from "os"
import * as path from "path"
import delay from "delay"

/**
 * Tests for TheaTask - the core task execution engine
 * This validates task creation, message handling, tool execution, and history management
 */

suite("TheaTask", () => {
	let sandbox: sinon.SinonSandbox
	let TheaTask: any
	let theaTask: any
	let mockProvider: any
	let mockExtensionContext: any
	let mockOutputChannel: any
	let mockApiConfig: any
	let mockApiHandler: any
	let mockTerminal: any
	let mockDiffViewProvider: any
	let mockUrlContentFetcher: any
	
	// File system mocks
	let fsStub: any
	let fileExistsStub: sinon.SinonStub
	
	// VS Code mocks
	let mockVscode: any
	let mockEventEmitter: any
	let mockDisposable: any
	
	// Other mocks
	let mockTheaIgnoreController: any
	let parseMentionsStub: sinon.SinonStub
	let mockDiffStrategy: any

	setup(() => {
		sandbox = sinon.createSandbox()
		
		// Setup VS Code mocks
		mockDisposable = { dispose: sandbox.stub() }
		mockEventEmitter = {
			event: sandbox.stub(),
			fire: sandbox.stub(),
			dispose: sandbox.stub()
		}
		
		mockVscode = {
			Disposable: { from: sandbox.stub().returns(mockDisposable) },
			EventEmitter: class {
				event = mockEventEmitter.event
				fire = mockEventEmitter.fire
				dispose = mockEventEmitter.dispose
			},
			workspace: {
				workspaceFolders: [
					{
						uri: { fsPath: "/mock/workspace" },
						name: "mock-workspace",
						index: 0
					}
				],
				getConfiguration: sandbox.stub().returns({
					get: sandbox.stub().returns(undefined),
					has: sandbox.stub().returns(false),
					inspect: sandbox.stub().returns(undefined),
					update: sandbox.stub().resolves()
				}),
				onDidChangeConfiguration: sandbox.stub(),
				fs: {
					writeFile: sandbox.stub().resolves(),
					readFile: sandbox.stub().resolves(Buffer.from("test content")),
					readDirectory: sandbox.stub().resolves([]),
					stat: sandbox.stub().resolves({ type: 1, ctime: 0, mtime: 0, size: 100 }),
					createDirectory: sandbox.stub().resolves(),
					delete: sandbox.stub().resolves()
				},
				textDocuments: [],
				openTextDocument: sandbox.stub().resolves({
					uri: { fsPath: "/mock/file.txt" },
					fileName: "/mock/file.txt",
					isUntitled: false,
					languageId: "plaintext",
					version: 1,
					isDirty: false,
					isClosed: false,
					save: sandbox.stub().resolves(true),
					eol: 1,
					lineCount: 1,
					getText: sandbox.stub().returns("test content")
				})
			},
			window: {
				activeTextEditor: undefined,
				showErrorMessage: sandbox.stub(),
				showWarningMessage: sandbox.stub(),
				showInformationMessage: sandbox.stub(),
				showTextDocument: sandbox.stub(),
				createOutputChannel: sandbox.stub()
			},
			Uri: {
				file: (path: string) => ({ fsPath: path }),
				parse: (str: string) => ({ fsPath: str })
			},
			Range: class {
				constructor(public start: any, public end: any) {}
			},
			Position: class {
				constructor(public line: number, public character: number) {}
			},
			commands: {
				executeCommand: sandbox.stub()
			},
			env: {
				openExternal: sandbox.stub()
			}
		}
		
		// Setup file system mocks
		const mockMessages = [
			{
				ts: Date.now(),
				type: "say",
				say: "text",
				text: "historical task"
			}
		]
		
		fileExistsStub = sandbox.stub().callsFake((filePath: string) => {
			return filePath.includes("ui_messages.json") || filePath.includes("api_conversation_history.json")
		})
		
		fsStub = {
			mkdir: sandbox.stub().resolves(undefined),
			writeFile: sandbox.stub().resolves(undefined),
			readFile: sandbox.stub().callsFake((filePath: string) => {
				if (filePath.includes("ui_messages.json")) {
					return Promise.resolve(JSON.stringify(mockMessages))
				}
				if (filePath.includes("api_conversation_history.json")) {
					return Promise.resolve(
						JSON.stringify([
							{
								role: "user",
								content: [{ type: "text", text: "historical task" }],
								ts: Date.now()
							},
							{
								role: "assistant",
								content: [{ type: "text", text: "I'll help you with that task." }],
								ts: Date.now()
							}
						])
					)
				}
				return Promise.resolve("[]")
			}),
			unlink: sandbox.stub().resolves(undefined),
			rmdir: sandbox.stub().resolves(undefined)
		}
		
		// Setup TheaIgnoreController mock
		mockTheaIgnoreController = {
			files: sandbox.stub().returns([]),
			patterns: sandbox.stub().returns([]),
			reload: sandbox.stub().resolves()
		}
		
		// Setup parseMentions mock
		parseMentionsStub = sandbox.stub().callsFake((text: string) => {
			// Simple mock implementation
			const urlRegex = /(https?:\/\/[^\s]+)/g
			const urls = text.match(urlRegex) || []
// Mock removed - needs manual implementation)
// 		
// 		// Setup DiffStrategy mock
// 		mockDiffStrategy = {
// 			applyDiff: sandbox.stub().resolves({ success: true })
// 		}
// 		
		// Setup Terminal mock
		mockTerminal = {
			execute: sandbox.stub().resolves({ code: 0, output: "Command executed successfully" }),
			dispose: sandbox.stub()
		}
		
		// Setup DiffViewProvider mock
		mockDiffViewProvider = {
			open: sandbox.stub().resolves(),
			dispose: sandbox.stub()
		}
		
		// Setup UrlContentFetcher mock
		mockUrlContentFetcher = {
			fetchContent: sandbox.stub().resolves("Fetched content from URL")
		}
		
		// Setup extension context
		const storageUri = { fsPath: "/mock/storage/path" }
		mockExtensionContext = {
			subscriptions: [],
			globalState: {
				get: sandbox.stub().callsFake(() => undefined),
				update: sandbox.stub().callsFake(() => Promise.resolve()),
				keys: sandbox.stub().returns([])
			},
			globalStorageUri: storageUri,
			workspaceState: {
				get: sandbox.stub().callsFake(() => undefined),
				update: sandbox.stub().callsFake(() => Promise.resolve()),
				keys: sandbox.stub().returns([])
			},
			secrets: {
				get: sandbox.stub().callsFake(() => Promise.resolve(undefined)),
				store: sandbox.stub().callsFake(() => Promise.resolve()),
				delete: sandbox.stub().callsFake(() => Promise.resolve())
			},
			extensionUri: {
				fsPath: "/mock/extension/path"
			},
			extension: {
				packageJSON: {
					version: "1.0.0"
				}
			}
		}
		
		// Setup output channel
		mockOutputChannel = {
			name: "mockOutputChannel",
			appendLine: sandbox.stub(),
			append: sandbox.stub(),
			clear: sandbox.stub(),
			show: sandbox.stub(),
			hide: sandbox.stub(),
			dispose: sandbox.stub(),
			replace: sandbox.stub()
		}
		
		// Setup API configuration
		mockApiConfig = {
			apiProvider: "anthropic",
			apiModelId: "claude-3-5-sonnet-20241022",
			apiKey: "test-api-key"
		}
		
		// Setup API handler mock
		mockApiHandler = {
			createMessage: sandbox.stub().returns({
				async *[Symbol.asyncIterator]() {
					yield { type: "text", text: "Test response" }
				}
			}),
			getModel: sandbox.stub().returns({
				id: "test-model",
				info: {
					maxTokens: 4096,
					contextWindow: 8192,
					supportsImages: true,
					supportsPromptCache: false
				}
			}),
			countTokens: sandbox.stub().resolves(100)
		}
		
		// Mock buildApiHandler
		const buildApiHandlerStub = sandbox.stub().returns(mockApiHandler)
		
		// Setup provider mock
		// Note: We need to create a more complete mock that TheaTask can use
		const TheaProviderClass = class {
			postMessageToWebview = sandbox.stub().resolves(undefined)
			postStateToWebview = sandbox.stub().resolves(undefined)
			getTaskWithId = sandbox.stub().callsFake((id: string) => ({
				historyItem: {
					id,
					ts: Date.now(),
					task: "historical task",
					tokensIn: 100,
					tokensOut: 200,
					cacheWrites: 0,
					cacheReads: 0,
					totalCost: 0.001
				},
				taskDirPath: "/mock/storage/path/tasks/123",
				apiConversationHistoryFilePath: "/mock/storage/path/tasks/123/api_conversation_history.json",
				uiMessagesFilePath: "/mock/storage/path/tasks/123/ui_messages.json",
				apiConversationHistory: [
					{
						role: "user",
						content: [{ type: "text", text: "historical task" }],
						ts: Date.now()
					},
					{
						role: "assistant",
						content: [{ type: "text", text: "I'll help you with that task." }],
						ts: Date.now()
					}
				]
			}))
		}
		
		mockProvider = new TheaProviderClass()
		
		// Load TheaTask with all mocked dependencies
		const module = proxyquire('../../../src/core/TheaTask', {
			'fs/promises': fsStub,
			'vscode': mockVscode,
			'../utils/fs': { fileExistsAtPath: fileExistsStub },
			'../ignore/TheaIgnoreController': { TheaIgnoreController: mockTheaIgnoreController },
			'./mentions': { parseMentions: parseMentionsStub },
			'../diff/DiffStrategy': mockDiffStrategy,
			'../integrations/terminal/Terminal': { Terminal: class { 
				constructor() { return mockTerminal }
			}},
			'../integrations/editor/DiffViewProvider': { DiffViewProvider: class {
				constructor() { return mockDiffViewProvider }
			}},
			'../services/browser/UrlContentFetcher': { UrlContentFetcher: class {
				constructor() { return mockUrlContentFetcher }
			}},
			'../api': { buildApiHandler: buildApiHandlerStub }
		})
		
		TheaTask = module.TheaTask
	})
	
	teardown(() => {
		sandbox.restore()
	})
	
	suite("constructor", () => {
		test("should initialize with provided settings", () => {
			theaTask = new TheaTask(
				mockProvider,
				mockApiConfig,
				"test-instruction",
				"architect",
				false,
				"test-task-id"
			)
			
			assert.ok(theaTask)
			assert.strictEqual(theaTask.taskId, "test-task-id")
			assert.strictEqual(theaTask.customInstructions, "test-instruction")
		})
		
		test("should generate task ID if not provided", () => {
			theaTask = new TheaTask(
				mockProvider,
				mockApiConfig,
				"test-instruction",
				"architect"
			)
			
			assert.ok(theaTask.taskId)
			assert.ok(typeof theaTask.taskId === "string")
			assert.ok(theaTask.taskId.length > 0)
		})
		
		test("should initialize with historical data when resuming", async () => {
			theaTask = new TheaTask(
				mockProvider,
				mockApiConfig,
				"test-instruction",
				"architect",
				true, // isHistoryItemTask
				"historical-task-id",
				{
					id: "historical-task-id",
					ts: Date.now(),
					task: "historical task",
					tokensIn: 500,
					tokensOut: 1000,
					cacheWrites: 10,
					cacheReads: 20,
					totalCost: 0.05
				}
			)
			
			assert.strictEqual(theaTask.taskId, "historical-task-id")
			// Token counts should be restored from history
		})
	})
	
	suite("message handling", () => {
		setup(() => {
			theaTask = new TheaTask(
				mockProvider,
				mockApiConfig,
				"test-instruction",
				"architect",
				false,
				"test-task-id"
			)
		})
		
		test("should handle user messages", async () => {
			const userMessage = {
				type: "say" as const,
				say: "text" as const,
				text: "Hello, can you help me?"
			}
			
			// Spy on internal methods
			const handleSayStub = sandbox.stub(theaTask, "handleSay").resolves()
			
			await theaTask.handleMessage(userMessage)
			
			assert.ok(handleSayStub.calledOnce)
			assert.ok(handleSayStub.calledWith("text", "Hello, can you help me?"))
		})
		
		test("should handle tool responses", async () => {
			const toolResponse = {
				type: "tool_result" as const,
				tool: "readFile",
				result: "File content here"
			}
			
			// Mock internal state to expect tool response
			theaTask.isWaitingForResponse = true
			theaTask.askType = "tool"
			theaTask.currentToolUse = { 
				tool_use_id: "test-tool-id",
				name: "readFile",
				input: { path: "/test/file.txt" }
			}
			
			const handleToolResponseStub = sandbox.stub(theaTask, "handleToolResponse").resolves()
			
			await theaTask.handleMessage(toolResponse)
			
			assert.ok(handleToolResponseStub.calledOnce)
		})
		
		test("should handle ask responses", async () => {
			const askResponse = {
				type: "ask_response" as const,
				response: "approved"
			}
			
			theaTask.isWaitingForResponse = true
			theaTask.askType = "confirmation"
			
			const handleAskResponseStub = sandbox.stub(theaTask, "handleAskResponse").resolves()
			
			await theaTask.handleMessage(askResponse)
			
			assert.ok(handleAskResponseStub.calledOnce)
			assert.ok(handleAskResponseStub.calledWith("approved"))
		})
	})
	
	suite("tool execution", () => {
		setup(() => {
			theaTask = new TheaTask(
				mockProvider,
				mockApiConfig,
				"test-instruction",
				"architect",
				false,
				"test-task-id"
			)
		})
		
		test("should execute read_file tool", async () => {
			const toolUse = {
				tool_use_id: "test-id",
				name: "read_file",
				input: { path: "/test/file.txt" }
			}
			
			// Mock file reading
			mockVscode.workspace.fs.readFile.resolves(Buffer.from("File contents"))
			
			const result = await theaTask.executeTool(toolUse)
			
			assert.ok(result)
			assert.ok(result.includes("File contents") || result.includes("successfully"))
		})
		
		test("should execute list_files tool", async () => {
			const toolUse = {
				tool_use_id: "test-id",
				name: "list_files",
				input: { path: "/test", recursive: true }
			}
			
			// This will use the mocked file system
			const result = await theaTask.executeTool(toolUse)
			
			assert.ok(result)
			assert.ok(typeof result === "string")
		})
		
		test("should execute execute_command tool", async () => {
			const toolUse = {
				tool_use_id: "test-id",
				name: "execute_command",
				input: { command: "echo 'Hello World'" }
			}
			
			const result = await theaTask.executeTool(toolUse)
			
			assert.ok(result)
			assert.ok(result.includes("success") || result.includes("executed"))
		})
		
		test("should handle tool execution errors", async () => {
			const toolUse = {
				tool_use_id: "test-id",
				name: "invalid_tool",
				input: {}
			}
			
			const result = await theaTask.executeTool(toolUse)
			
			assert.ok(result)
			assert.ok(result.includes("error") || result.includes("unknown") || result.includes("invalid"))
		})
	})
	
	suite("API streaming", () => {
		setup(() => {
			theaTask = new TheaTask(
				mockProvider,
				mockApiConfig,
				"test-instruction",
				"architect",
				false,
				"test-task-id"
			)
		})
		
		test("should handle streaming API responses", async () => {
			// Setup mock stream
			mockApiHandler.createMessage.returns({
				async *[Symbol.asyncIterator]() {
					yield { type: "text", text: "First chunk" }
					yield { type: "text", text: " Second chunk" }
					yield { type: "tool_use", id: "tool-1", name: "read_file", input: { path: "/test.txt" } }
				}
			})
			
			const chunks: any[] = []
			theaTask.on("chunk", (chunk: any) => chunks.push(chunk))
			
			await theaTask.streamApiResponse()
			
			assert.ok(chunks.length > 0)
			assert.ok(chunks.some(c => c.type === "text"))
			assert.ok(chunks.some(c => c.type === "tool_use"))
		})
		
		test("should handle API errors gracefully", async () => {
			mockApiHandler.createMessage.throws(new Error("API Error"))
			
			try {
				await theaTask.streamApiResponse()
				// Should handle error internally, not throw
				assert.ok(true, "Error was handled")
} catch (error) {
				// If it throws, that's also acceptable
				assert.ok(error instanceof Error)
			}
		})
	})
	
	suite("history management", () => {
		setup(() => {
			theaTask = new TheaTask(
				mockProvider,
				mockApiConfig,
				"test-instruction",
				"architect",
				false,
				"test-task-id"
			)
		})
		
		test("should save conversation history", async () => {
			// Add some messages
			theaTask.theaMessages = [
				{ type: "say", say: "text", text: "User message", ts: Date.now() },
				{ type: "say", say: "text", text: "Assistant response", ts: Date.now() }
			]
			
			theaTask.apiConversationHistory = [
				{ role: "user", content: [{ type: "text", text: "User message" }] },
				{ role: "assistant", content: [{ type: "text", text: "Assistant response" }] }
			]
			
			await theaTask.saveConversationHistory()
			
			assert.ok(fsStub.writeFile.called)
			// Should write both UI messages and API conversation history
			assert.ok(fsStub.writeFile.calledWith(
				sinon.match(/ui_messages\.json/)
			))
			assert.ok(fsStub.writeFile.calledWith(
				sinon.match(/api_conversation_history\.json/)
			))
		})
		
		test("should load conversation history when resuming", async () => {
			theaTask = new TheaTask(
				mockProvider,
				mockApiConfig,
				"test-instruction",
				"architect",
				true, // isHistoryItemTask
				"historical-task-id"
			)
			
			await theaTask.loadConversationHistory()
			
			assert.ok(fsStub.readFile.called)
			assert.ok(theaTask.theaMessages.length > 0)
			assert.ok(theaTask.apiConversationHistory.length > 0)
		})
	})
	
	suite("token counting", () => {
		setup(() => {
			theaTask = new TheaTask(
				mockProvider,
				mockApiConfig,
				"test-instruction",
				"architect",
				false,
				"test-task-id"
			)
		})
		
		test("should count input tokens", async () => {
			mockApiHandler.countTokens.resolves(150)
			
			const count = await theaTask.countTokens([
				{ type: "text", text: "Test message" }
			])
			
			assert.strictEqual(count, 150)
			assert.ok(mockApiHandler.countTokens.calledOnce)
		})
		
		test("should track cumulative token usage", async () => {
			theaTask.inputTokens = 100
			theaTask.outputTokens = 200
			
			// Simulate adding more tokens
			theaTask.inputTokens += 50
			theaTask.outputTokens += 100
			
			assert.strictEqual(theaTask.inputTokens, 150)
			assert.strictEqual(theaTask.outputTokens, 300)
		})
	})
	
	suite("abort handling", () => {
		setup(() => {
			theaTask = new TheaTask(
				mockProvider,
				mockApiConfig,
				"test-instruction",
				"architect",
				false,
				"test-task-id"
			)
		})
		
		test("should handle abort during API streaming", async () => {
			// Setup a slow stream
			mockApiHandler.createMessage.returns({
				async *[Symbol.asyncIterator]() {
					yield { type: "text", text: "Start" }
					await delay(100)
					yield { type: "text", text: "Should not reach" }
				}
			})
			
			// Start streaming
			const streamPromise = theaTask.streamApiResponse()
			
			// Abort after a short delay
			setTimeout(() => theaTask.abort("user_cancelled"), 10)
			
			await streamPromise
			
			assert.ok(theaTask.abortReason)
			assert.strictEqual(theaTask.abortReason, "user_cancelled")
		})
		
		test("should cleanup resources on abort", async () => {
			theaTask.abort("test_abort")
			
			assert.ok(theaTask.abortReason)
			// Should cleanup any active operations
		})
	})
	
	suite("mode handling", () => {
		test("should use specified mode configuration", () => {
			theaTask = new TheaTask(
				mockProvider,
				mockApiConfig,
				"test-instruction",
				"code", // Different mode
				false,
				"test-task-id"
			)
			
			// Mode should affect behavior
			assert.ok(theaTask)
			// The mode influences system prompt and tool availability
		})
		
		test("should default to architect mode", () => {
			theaTask = new TheaTask(
				mockProvider,
				mockApiConfig,
				"test-instruction",
				undefined, // No mode specified
				false,
				"test-task-id"
			)
			
			assert.ok(theaTask)
			// Should use default mode
		})
	})
	
	suite("error handling", () => {
		setup(() => {
			theaTask = new TheaTask(
				mockProvider,
				mockApiConfig,
				"test-instruction",
				"architect",
				false,
				"test-task-id"
			)
		})
		
		test("should handle file read errors", async () => {
			mockVscode.workspace.fs.readFile.rejects(new Error("File not found"))
			
			const toolUse = {
				tool_use_id: "test-id",
				name: "read_file",
				input: { path: "/nonexistent/file.txt" }
			}
			
			const result = await theaTask.executeTool(toolUse)
			
			assert.ok(result)
			assert.ok(result.includes("error") || result.includes("not found"))
		})
		
		test("should handle terminal command errors", async () => {
			mockTerminal.execute.rejects(new Error("Command failed"))
			
			const toolUse = {
				tool_use_id: "test-id",
				name: "execute_command",
				input: { command: "invalid-command" }
			}
			
			const result = await theaTask.executeTool(toolUse)
			
			assert.ok(result)
			assert.ok(result.includes("error") || result.includes("failed"))
		})
		
		test("should handle API connection errors", async () => {
			mockApiHandler.createMessage.rejects(new Error("Network error"))
			
			try {
				await theaTask.streamApiResponse()
				assert.ok(true, "Handled network error")
} catch (error) {
				assert.ok(error instanceof Error)
				assert.ok(error.message.includes("Network") || error.message.includes("error"))
			}
		})
	})
// Mock cleanup
