// filepath: /Volumes/stuff/Projects/Thea-Code/src/core/webview/__tests__/ClineTaskHistory.test.ts
import * as assert from 'assert'
import * as sinon from 'sinon'
import * as vscode from 'vscode'
/* eslint-disable @typescript-eslint/unbound-method */
import * as vscode from "vscode"
import * as path from "path"
import fs from "fs/promises"
import { TheaTaskHistory } from "../history/TheaTaskHistory" // Updated import
import { ContextProxy } from "../../config/ContextProxy"
import { fileExistsAtPath } from "../../../utils/fs"
import { ShadowCheckpointService } from "../../../services/checkpoints/ShadowCheckpointService"
import { downloadTask } from "../../../integrations/misc/export-markdown"
import { getWorkspacePath } from "../../../utils/path"
import { GlobalFileNames } from "../../../shared/globalFileNames"
import { HistoryItem } from "../../../shared/HistoryItem"

// Mock dependencies
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// Updated mock path
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// TODO: Mock needs manual migration
// TODO: Implement proper mock with proxyquire

suite("TheaTaskHistory", () => {
	// Updated describe block
	let taskHistory: TheaTaskHistory // Updated type
	let mockContext: vscode.ExtensionContext
	let mockContextProxy: sinon.SinonStubStatic<ContextProxy>

	setup(() => {
		// Reset mocks
		sinon.restore()

		// Mock context
		mockContext = {
			extensionPath: "/test/path",
			extensionUri: {} as vscode.Uri,
			globalStorageUri: {
				fsPath: "/test/storage/path",
			},
		} as unknown as vscode.ExtensionContext

		// Mock contextProxy
		mockContextProxy = {
			getValue: sinon.stub().callsFake(() => Promise.resolve(undefined)),
			setValue: sinon.stub().callsFake(() => Promise.resolve()),
		} as unknown as sinon.SinonStubStatic<ContextProxy>

		// Mock getWorkspacePath
		;(getWorkspacePath as sinon.SinonStub).returns("/test/workspace")

		// Create instance of TheaTaskHistory
		taskHistory = new TheaTaskHistory(mockContext, mockContextProxy) // Updated instantiation

		// Mock fs methods
		;(fs.rm as sinon.SinonStub) = sinon.stub().callsFake(() => Promise.resolve())
		;(fs.readFile as sinon.SinonStub).callsFake(() => Promise.resolve("[]"))

		// Mock console to prevent test output noise
		sinon.spy(console, "log").callsFake(() => {})
		sinon.spy(console, "error").callsFake(() => {})
		sinon.spy(console, "warn").callsFake(() => {})

	teardown(() => {
		sinon.restore()

	suite("updateTaskHistory", () => {
		test("adds a new history item when it doesn't exist", async () => {
			// Setup
			const mockHistory: HistoryItem[] = []
			mockContextProxy.getValue.callsFake(() => Promise.resolve(mockHistory))

			const newHistoryItem: HistoryItem = {
				id: "test-task-id",
				task: "Test Task",
				ts: 123456789,
				number: 1,
				tokensIn: 100,
				tokensOut: 200,
				totalCost: 0.01,

			// Execute
			const result = await taskHistory.updateTaskHistory(newHistoryItem)

			// Verify
			assert.ok(mockContextProxy.getValue.calledWith("taskHistory"))
			assert.ok(mockContextProxy.setValue.calledWith("taskHistory", [newHistoryItem]))
			assert.deepStrictEqual(result, [newHistoryItem])

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

			const mockHistory = [existingItem]
			mockContextProxy.getValue.callsFake(() => Promise.resolve(mockHistory))

			const updatedItem: HistoryItem = {
				id: "test-task-id",
				task: "Updated Task",
				ts: 123456789,
				number: 1,
				tokensIn: 150,
				tokensOut: 250,
				totalCost: 0.02,

			// Execute
			const result = await taskHistory.updateTaskHistory(updatedItem)

			// Verify
			assert.ok(mockContextProxy.setValue.calledWith("taskHistory", [updatedItem]))
			assert.deepStrictEqual(result, [updatedItem])

		test("handles empty history list", async () => {
			// Setup
			mockContextProxy.getValue.callsFake(() => Promise.resolve(null))
			const newHistoryItem: HistoryItem = {
				id: "test-task-id",
				task: "Test Task",
				ts: 123456789,
				number: 1,
				tokensIn: 100,
				tokensOut: 200,
				totalCost: 0.01,

			// Execute
			const result = await taskHistory.updateTaskHistory(newHistoryItem)

			// Verify
			assert.ok(mockContextProxy.setValue.calledWith("taskHistory", [newHistoryItem]))
			assert.deepStrictEqual(result, [newHistoryItem])

	suite("getTaskWithId", () => {
		test("returns task data when task exists", async () => {
			// Setup
			const mockHistoryItem: HistoryItem = {
				id: "test-task-id",
				task: "Test Task",
				ts: 123456789,
				number: 1,
				tokensIn: 100,
				tokensOut: 200,
				totalCost: 0.01,

			mockContextProxy.getValue.callsFake(() => Promise.resolve([mockHistoryItem]))

			const expectedTaskDirPath = path.join("/test/storage/path", "tasks", "test-task-id")
			const expectedApiHistoryPath = path.join(expectedTaskDirPath, GlobalFileNames.apiConversationHistory)
			const expectedUiMessagesPath = path.join(expectedTaskDirPath, GlobalFileNames.uiMessages)

			;(fileExistsAtPath as sinon.SinonStub).resolves(true)
			;(fs.readFile as sinon.SinonStub).resolves(JSON.stringify([{ role: "user", content: "Hello" }]))

			// Execute
			const result = await taskHistory.getTaskWithId("test-task-id")

			// Verify
			assert.deepStrictEqual(result, {
				historyItem: mockHistoryItem,
				taskDirPath: expectedTaskDirPath,
				apiConversationHistoryFilePath: expectedApiHistoryPath,
				uiMessagesFilePath: expectedUiMessagesPath,
				apiConversationHistory: [{ role: "user", content: "Hello" }],

		test("handles missing conversation history file", async () => {
			// Setup
			const mockHistoryItem: HistoryItem = {
				id: "test-task-id",
				task: "Test Task",
				ts: 123456789,
				number: 1,
				tokensIn: 100,
				tokensOut: 200,
				totalCost: 0.01,

			mockContextProxy.getValue.callsFake(() => Promise.resolve([mockHistoryItem]))
			;(fileExistsAtPath as sinon.SinonStub).resolves(false)

			// Execute
			const result = await taskHistory.getTaskWithId("test-task-id")

			// Verify
			assert.deepStrictEqual(result.apiConversationHistory, [])

		test("throws error and attempts cleanup when task not found", async () => {
			// Setup
			mockContextProxy.getValue.callsFake(() => Promise.resolve([]))

			// Mock the deleteTaskFromState method
			const deleteTaskFromStateSpy = sinon.spy(taskHistory, "deleteTaskFromState").resolves(undefined)

			// Execute & Verify
			await expect(taskHistory.getTaskWithId("non-existent-id")).rejects.toThrow("Task non-existent-id not found")
			assert.ok(deleteTaskFromStateSpy.calledWith("non-existent-id"))

		test("handles error when reading conversation history", async () => {
			// Setup
			const mockHistoryItem: HistoryItem = {
				id: "test-task-id",
				task: "Test Task",
				ts: 123456789,
				number: 1,
				tokensIn: 100,
				tokensOut: 200,
				totalCost: 0.01,

			mockContextProxy.getValue.callsFake(() => Promise.resolve([mockHistoryItem]))
			;(fileExistsAtPath as sinon.SinonStub).resolves(true)
			;(fs.readFile as sinon.SinonStub).rejects(new Error("Read error"))

			// Execute
			const result = await taskHistory.getTaskWithId("test-task-id")

			// Verify
			assert.deepStrictEqual(result.apiConversationHistory, [])
			assert.ok(console.error.called)

	suite("showTaskWithId", () => {
		test("initializes new Cline when showing a different task", async () => {
			// Setup
			const mockHistoryItem: HistoryItem = {
				id: "different-task-id",
				task: "Different Task",
				ts: 123456789,
				number: 1,
				tokensIn: 100,
				tokensOut: 200,
				totalCost: 0.01,

			const mockGetTaskWithId = jest
				.spyOn(taskHistory, "getTaskWithId")
				.resolves({ historyItem: mockHistoryItem })

			const mockGetCurrentCline = sinon.stub().returns({ taskId: "current-task-id" }) // Keep as is, represents return value
			const mockInitClineWithHistoryItem = sinon.stub().resolves({ taskId: mockHistoryItem.id }) // Keep as is, represents return value
			const mockPostWebviewAction = sinon.stub().resolves(undefined)

			// Execute
			await taskHistory.showTaskWithId(
				"different-task-id",
				mockGetCurrentCline,
				mockInitClineWithHistoryItem,
				mockPostWebviewAction,

			// Verify
			assert.ok(mockGetTaskWithId.calledWith("different-task-id"))
			assert.ok(mockInitClineWithHistoryItem.calledWith(mockHistoryItem))
			assert.ok(mockPostWebviewAction.calledWith("chatButtonClicked"))

		test("doesn't initialize TheaTask when showing current task", async () => {
			// Updated test description
			// Setup
			const mockGetCurrentCline = sinon.stub().returns({ taskId: "current-task-id" })
			const mockInitClineWithHistoryItem = sinon.stub()
			const mockPostWebviewAction = sinon.stub().resolves(undefined)
			const mockGetTaskWithId = sinon.spy(taskHistory, "getTaskWithId")

			// Execute
			await taskHistory.showTaskWithId(
				"current-task-id",
				mockGetCurrentCline,
				mockInitClineWithHistoryItem,
				mockPostWebviewAction,

			// Verify
			assert.ok(!mockGetTaskWithId.called)
			assert.ok(!mockInitClineWithHistoryItem.called)
			assert.ok(mockPostWebviewAction.calledWith("chatButtonClicked"))

	suite("exportTaskWithId", () => {
		test("exports task conversation to markdown", async () => {
			// Setup
			const mockHistoryItem: HistoryItem = {
				id: "test-task-id",
				task: "Test Task",
				ts: 123456789,
				number: 1,
				tokensIn: 100,
				tokensOut: 200,
				totalCost: 0.01,

			const mockApiHistory = [{ role: "user", content: "Hello" }]
			const mockGetTaskWithId = jest
				.spyOn(taskHistory, "getTaskWithId")
				.resolves({ historyItem: mockHistoryItem, apiConversationHistory: mockApiHistory })

			// Execute
			await taskHistory.exportTaskWithId("test-task-id")

			// Verify
			assert.ok(mockGetTaskWithId.calledWith("test-task-id"))
			assert.ok(downloadTask.calledWith(123456789, mockApiHistory))

	suite("deleteTaskWithId", () => {
		test("deletes task data, directory, and shadow checkpoints", async () => {
			// Setup
			const taskId = "test-task-id"
			const taskDirPath = path.join("/test/storage/path", "tasks", taskId)
			sinon.spy(taskHistory, "getTaskWithId").resolves({ taskDirPath })
			sinon.spy(taskHistory, "deleteTaskFromState").resolves(undefined)

			// Mock getCurrentCline to return a different task ID
			const mockGetCurrentCline = sinon.stub().returns({ taskId: "different-task-id" })
			const mockFinishSubTask = sinon.stub()

			// Execute
			await taskHistory.deleteTaskWithId(taskId, mockGetCurrentCline, mockFinishSubTask)

			// Verify
			assert.ok(mockGetTaskWithId.calledWith(taskId))
			assert.ok(mockDeleteTaskFromState.calledWith(taskId))
			assert.ok(ShadowCheckpointService.deleteTask.calledWith({
				taskId,
				globalStorageDir: "/test/storage/path",
				workspaceDir: "/test/workspace",

			assert.ok(fs.rm.calledWith(taskDirPath, { recursive: true, force: true }))
			assert.ok(!mockFinishSubTask.called) // Should not be called for non-current task

		test("finishes subtask when deleting current task", async () => {
			// Setup
			const taskId = "current-task-id"
			const taskDirPath = path.join("/test/storage/path", "tasks", taskId)
			sinon.spy(taskHistory, "getTaskWithId").resolves({ taskDirPath })

			sinon.spy(taskHistory, "deleteTaskFromState").resolves(undefined)

			// Mock getCurrentCline to return the same task ID
			const mockGetCurrentCline = sinon.stub().returns({ taskId })
			const mockFinishSubTask = sinon.stub().resolves(undefined)

			// Execute
			await taskHistory.deleteTaskWithId(taskId, mockGetCurrentCline, mockFinishSubTask)

			// Verify
			assert.ok(mockFinishSubTask.called) // Should be called for current task

		test("handles task not found error", async () => {
			// Setup
			const taskId = "non-existent-id"
			sinon.spy(taskHistory, "getTaskWithId").rejects(new Error("Task non-existent-id not found"))

			const mockGetCurrentCline = sinon.stub()
			const mockFinishSubTask = sinon.stub()

			// Execute
			await taskHistory.deleteTaskWithId(taskId, mockGetCurrentCline, mockFinishSubTask)

			// Verify - should handle gracefully
			assert.ok(console.log.called)

		test("handles shadow checkpoint deletion error", async () => {
			// Setup
			const taskId = "test-task-id"
			const taskDirPath = path.join("/test/storage/path", "tasks", taskId)
			sinon.spy(taskHistory, "getTaskWithId").resolves({ taskDirPath })
			sinon.spy(taskHistory, "deleteTaskFromState").resolves(undefined)

			// Mock ShadowCheckpointService to throw an error
			const mockError = new Error("Shadow deletion error")
			;(ShadowCheckpointService.deleteTask as sinon.SinonStub).rejects(mockError)

			const mockGetCurrentCline = sinon.stub().returns({ taskId: "different-task-id" })
			const mockFinishSubTask = sinon.stub()

			// Execute
			await taskHistory.deleteTaskWithId(taskId, mockGetCurrentCline, mockFinishSubTask)

			// Verify - should continue and log error
			assert.ok(console.error.called)
			assert.ok(fs.rm.called) // Should still try to delete directory

	suite("deleteTaskFromState", () => {
		test("removes task from history list", async () => {
			// Setup
			const taskId = "test-task-id"
			const mockHistory = [
				{
					id: "other-task-id",
					task: "Other Task",
					number: 1,
					ts: 123456789,
					tokensIn: 100,
					tokensOut: 200,
					totalCost: 0.01,
				},
				{
					id: taskId,
					task: "Test Task",
					number: 2,
					ts: 123456789,
					tokensIn: 100,
					tokensOut: 200,
					totalCost: 0.01,
				},

			mockContextProxy.getValue.callsFake(() => Promise.resolve(mockHistory))

			// Execute
			await taskHistory.deleteTaskFromState(taskId)

			// Verify
			assert.ok(mockContextProxy.setValue.calledWith("taskHistory", [
				{
					id: "other-task-id",
					task: "Other Task",
					number: 1,
					ts: 123456789,
					tokensIn: 100,
					tokensOut: 200,
					totalCost: 0.01,
				},
			]))

		test("does nothing when task is not in history list", async () => {
			// Setup
			const taskId = "non-existent-id"
			const mockHistory = [
				{
					id: "other-task-id",
					task: "Other Task",
					number: 1,
					ts: 123456789,
					tokensIn: 100,
					tokensOut: 200,
					totalCost: 0.01,
				},

			mockContextProxy.getValue.callsFake(() => Promise.resolve(mockHistory))

			// Execute
			await taskHistory.deleteTaskFromState(taskId)

			// Verify
			assert.ok(!mockContextProxy.setValue.called)
