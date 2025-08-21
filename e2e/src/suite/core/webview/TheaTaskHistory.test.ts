import * as assert from 'assert'
import * as sinon from 'sinon'
import * as proxyquire from 'proxyquire'
import * as path from 'path'
import { GlobalFileNames } from "../../../../../src/shared/globalFileNames"
import { HistoryItem } from "../../../../../src/shared/HistoryItem"

suite("TheaTaskHistory", () => {
	let sandbox: sinon.SinonSandbox
	let TheaTaskHistory: any
	let taskHistory: any
	let mockContext: any
	let mockContextProxy: any
	
	// Stubs for mocked dependencies
	let fsStub: any
	let fileExistsStub: sinon.SinonStub
	let getWorkspacePathStub: sinon.SinonStub
	let downloadTaskStub: sinon.SinonStub
	let getTaskDirectoryPathStub: sinon.SinonStub
	let ShadowCheckpointServiceStub: any
	let loggerStub: any

	setup(() => {
		sandbox = sinon.createSandbox()
		
		// Create stubs
		fsStub = {
			readFile: sandbox.stub(),
			rm: sandbox.stub()
		}
		
		fileExistsStub = sandbox.stub()
		getWorkspacePathStub = sandbox.stub()
		downloadTaskStub = sandbox.stub()
		getTaskDirectoryPathStub = sandbox.stub()
		
		ShadowCheckpointServiceStub = {
			deleteTask: sandbox.stub()
		}
		
		loggerStub = {
			warn: sandbox.stub(),
			error: sandbox.stub(),
			info: sandbox.stub()
		}
		
		// Set up default stub behaviors
		getWorkspacePathStub.returns("/test/workspace")
		getTaskDirectoryPathStub.callsFake((storagePath: string, id: string) => 
			Promise.resolve(path.join(storagePath, "tasks", id)))
		fileExistsStub.resolves(false)
		fsStub.readFile.resolves("[]")
		fsStub.rm.resolves()
		downloadTaskStub.resolves()
		ShadowCheckpointServiceStub.deleteTask.resolves()
		
		// Mock context
		mockContext = {
			extensionPath: "/test/path",
			extensionUri: {},
			globalStorageUri: {
				fsPath: "/test/storage/path",
			},
		}
		
		// Mock contextProxy
		mockContextProxy = {
			getValue: sandbox.stub().returns(undefined),
			setValue: sandbox.stub().resolves(),
		}
		
		// Load TheaTaskHistory with mocked dependencies
		const module = proxyquire('../history/TheaTaskHistory', {
			'fs/promises': { default: fsStub },
			'../../../utils/fs': { fileExistsAtPath: fileExistsStub },
			'../../../utils/path': { getWorkspacePath: getWorkspacePathStub },
			'../../../integrations/misc/export-markdown': { downloadTask: downloadTaskStub },
			'../../../shared/storagePathManager': { getTaskDirectoryPath: getTaskDirectoryPathStub },
			'../../../services/checkpoints/ShadowCheckpointService': { ShadowCheckpointService: ShadowCheckpointServiceStub },
			'../../../utils/logging': { logger: loggerStub },
			'../../../i18n': { t: (key: string) => key }
		})
		
		TheaTaskHistory = module.TheaTaskHistory
		
		// Create instance
		taskHistory = new TheaTaskHistory(mockContext, mockContextProxy)
	})

	teardown(() => {
		sandbox.restore()
	})

	suite("updateTaskHistory", () => {
		test("adds a new history item", async () => {
			// Setup
			const newHistoryItem: HistoryItem = {
				id: "test-id",
				task: "Test Task",
				ts: Date.now(),
				number: 1,
				tokensIn: 100,
				tokensOut: 200,
				totalCost: 0.01,
			}

			// Execute
			const result = await taskHistory.updateTaskHistory(newHistoryItem)

			// Verify
			assert.ok(mockContextProxy.getValue.calledWith("taskHistory"))
			assert.ok(mockContextProxy.setValue.calledWith("taskHistory", [newHistoryItem]))
			assert.deepStrictEqual(result, [newHistoryItem])
		})

		test("updates an existing history item", async () => {
			// Setup
			const existingItem: HistoryItem = {
				id: "test-task-id",
				task: "Original Task",
				ts: 123456789,
				number: 1,
				tokensIn: 100,
				tokensOut: 200,
				totalCost: 0.01,
			}

			const mockHistory = [existingItem]
			mockContextProxy.getValue.returns(mockHistory)

			const updatedItem: HistoryItem = {
				id: "test-task-id",
				task: "Updated Task",
				ts: 123456789,
				number: 1,
				tokensIn: 150,
				tokensOut: 250,
				totalCost: 0.02,
			}

			// Execute
			const result = await taskHistory.updateTaskHistory(updatedItem)

			// Verify
			assert.ok(mockContextProxy.getValue.calledWith("taskHistory"))
			assert.ok(mockContextProxy.setValue.calledOnce)
			const updatedHistory = mockContextProxy.setValue.firstCall.args[1]
			assert.strictEqual(updatedHistory.length, 1)
			assert.deepStrictEqual(updatedHistory[0], updatedItem)
			assert.deepStrictEqual(result[0], updatedItem)
		})
	})

	suite("getTaskWithId", () => {
		test("returns task data when task exists", async () => {
			// Setup
			const historyItem: HistoryItem = {
				id: "test-id",
				task: "Test Task",
				ts: Date.now(),
				number: 1,
				tokensIn: 100,
				tokensOut: 200,
				totalCost: 0.01,
			}
			
			mockContextProxy.getValue.returns([historyItem])
			
			const mockApiHistory = [
				{ role: "user", content: [{ type: "text", text: "Hello" }] }
			]
			fileExistsStub.resolves(true)
			fsStub.readFile.resolves(JSON.stringify(mockApiHistory))

			// Execute
			const result = await taskHistory.getTaskWithId("test-id")

			// Verify
			assert.strictEqual(result.historyItem, historyItem)
			assert.strictEqual(result.taskDirPath, path.join("/test/storage/path", "tasks", "test-id"))
			assert.ok(result.apiConversationHistoryFilePath.includes(GlobalFileNames.apiConversationHistory))
			assert.ok(result.uiMessagesFilePath.includes(GlobalFileNames.uiMessages))
			assert.deepStrictEqual(result.apiConversationHistory, mockApiHistory)
		})

		test("throws error when task not found", async () => {
			// Setup
			mockContextProxy.getValue.returns([])

			// Execute & Verify
			try {
				await taskHistory.getTaskWithId("non-existent-id")
				assert.fail("Should have thrown an error")
			} catch (error) {
				assert.ok(error instanceof Error)
				assert.ok(error.message.includes("non-existent-id not found"))
				assert.ok(loggerStub.warn.called)
			}
		})

		test("handles missing conversation history file gracefully", async () => {
			// Setup
			const historyItem: HistoryItem = {
				id: "test-id",
				task: "Test Task",
				ts: Date.now(),
				number: 1,
				tokensIn: 100,
				tokensOut: 200,
				totalCost: 0.01,
			}
			
			mockContextProxy.getValue.returns([historyItem])
			fileExistsStub.resolves(false) // File doesn't exist

			// Execute
			const result = await taskHistory.getTaskWithId("test-id")

			// Verify - should return empty conversation history
			assert.deepStrictEqual(result.apiConversationHistory, [])
		})
	})

	suite("deleteTaskWithId", () => {
		test("deletes task completely when it exists", async () => {
			// Setup
			const historyItem: HistoryItem = {
				id: "test-id",
				task: "Test Task",
				ts: Date.now(),
				number: 1,
				tokensIn: 100,
				tokensOut: 200,
				totalCost: 0.01,
			}
			
			mockContextProxy.getValue.returns([historyItem])
			
			const getCurrentCline = sandbox.stub().returns(undefined)
			const finishSubTask = sandbox.stub().resolves()

			// Execute
			await taskHistory.deleteTaskWithId("test-id", getCurrentCline, finishSubTask)

			// Verify
			assert.ok(ShadowCheckpointServiceStub.deleteTask.calledOnce)
			assert.ok(fsStub.rm.calledOnce)
			assert.ok(mockContextProxy.setValue.called)
			
			// Verify the task was removed from history
			const updatedHistory = mockContextProxy.setValue.firstCall.args[1]
			assert.strictEqual(updatedHistory.length, 0)
		})

		test("handles current task deletion", async () => {
			// Setup
			const historyItem: HistoryItem = {
				id: "current-task-id",
				task: "Current Task",
				ts: Date.now(),
				number: 1,
				tokensIn: 100,
				tokensOut: 200,
				totalCost: 0.01,
			}
			
			mockContextProxy.getValue.returns([historyItem])
			
			const getCurrentCline = sandbox.stub().returns({ taskId: "current-task-id" })
			const finishSubTask = sandbox.stub().resolves()

			// Execute
			await taskHistory.deleteTaskWithId("current-task-id", getCurrentCline, finishSubTask)

			// Verify
			assert.ok(finishSubTask.calledWith("common:tasks.deleted"))
			assert.ok(ShadowCheckpointServiceStub.deleteTask.called)
			assert.ok(fsStub.rm.called)
		})

		test("handles missing task gracefully", async () => {
			// Setup
			mockContextProxy.getValue.returns([])
			
			const getCurrentCline = sandbox.stub().returns(undefined)
			const finishSubTask = sandbox.stub().resolves()

			// Execute - should not throw
			await taskHistory.deleteTaskWithId("non-existent-id", getCurrentCline, finishSubTask)

			// Verify - should exit early without calling delete operations
			assert.ok(!ShadowCheckpointServiceStub.deleteTask.called)
			assert.ok(!fsStub.rm.called)
		})
	})

	suite("exportTaskWithId", () => {
		test("exports task when it exists", async () => {
			// Setup
			const historyItem: HistoryItem = {
				id: "test-id",
				task: "Test Task",
				ts: 123456789,
				number: 1,
				tokensIn: 100,
				tokensOut: 200,
				totalCost: 0.01,
			}
			
			const mockApiHistory = [
				{ role: "user", content: [{ type: "text", text: "Hello" }] }
			]
			
			mockContextProxy.getValue.returns([historyItem])
			fileExistsStub.resolves(true)
			fsStub.readFile.resolves(JSON.stringify(mockApiHistory))

			// Execute
			await taskHistory.exportTaskWithId("test-id")

			// Verify
			assert.ok(downloadTaskStub.calledOnce)
			assert.ok(downloadTaskStub.calledWith(123456789, mockApiHistory))
		})
	})

	suite("showTaskWithId", () => {
		test("shows non-current task", async () => {
			// Setup
			const historyItem: HistoryItem = {
				id: "other-task-id",
				task: "Other Task",
				ts: Date.now(),
				number: 1,
				tokensIn: 100,
				tokensOut: 200,
				totalCost: 0.01,
			}
			
			mockContextProxy.getValue.returns([historyItem])
			
			const getCurrentCline = sandbox.stub().returns({ taskId: "current-task-id" })
			const initClineWithHistoryItem = sandbox.stub().resolves({ taskId: "other-task-id" })
			const postWebviewAction = sandbox.stub().resolves()

			// Execute
			await taskHistory.showTaskWithId("other-task-id", getCurrentCline, initClineWithHistoryItem, postWebviewAction)

			// Verify
			assert.ok(initClineWithHistoryItem.calledWith(historyItem))
			assert.ok(postWebviewAction.calledWith("chatButtonClicked"))
		})

		test("shows current task without re-initializing", async () => {
			// Setup
			const historyItem: HistoryItem = {
				id: "current-task-id",
				task: "Current Task",
				ts: Date.now(),
				number: 1,
				tokensIn: 100,
				tokensOut: 200,
				totalCost: 0.01,
			}
			
			mockContextProxy.getValue.returns([historyItem])
			
			const getCurrentCline = sandbox.stub().returns({ taskId: "current-task-id" })
			const initClineWithHistoryItem = sandbox.stub().resolves()
			const postWebviewAction = sandbox.stub().resolves()

			// Execute
			await taskHistory.showTaskWithId("current-task-id", getCurrentCline, initClineWithHistoryItem, postWebviewAction)

			// Verify - should only post action, not re-initialize
			assert.ok(!initClineWithHistoryItem.called)
			assert.ok(postWebviewAction.calledWith("chatButtonClicked"))
		})
	})
// Mock cleanup