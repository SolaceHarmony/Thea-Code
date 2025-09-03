import * as assert from 'assert'
import * as sinon from 'sinon'
import * as vscode from 'vscode'
/**
 * TheaTaskHistory IO and cleanup tests
 * Tests file I/O operations, cleanup order, export paths, and error handling
 */

import * as vscode from "vscode"
import { TheaTaskHistory } from "../TheaTaskHistory"
import * as fs from "fs/promises"
import * as path from "path"
import { ShadowCheckpointService } from "../../../../services/checkpoints/ShadowCheckpointService"
import { ContextProxy } from "../../../config/ContextProxy"
import { getTaskDirectoryPath } from "../../../../shared/storagePathManager"
import { fileExistsAtPath } from "../../../../utils/fs"
import type { HistoryItem } from "../../../../shared/HistoryItem"
import { downloadTask } from "../../../../integrations/misc/export-markdown"
import type { NeutralConversationHistory } from "../../../../shared/neutral-history"

// Mock dependencies
// Mock needs manual implementation
// TODO: Implement proper mock with proxyquire
// Mock needs manual implementation
// Mock needs manual implementation
// Mock needs manual implementation
// TODO: Implement proper mock with proxyquire
// Mock needs manual implementation
// TODO: Implement proper mock with proxyquire

suite("TheaTaskHistory - IO and Cleanup Operations", () => {
	let taskHistory: TheaTaskHistory
	let mockContext: sinon.SinonStubStatic<vscode.ExtensionContext>
	let mockContextProxy: sinon.SinonStubStatic<ContextProxy>
	let mockFs: sinon.SinonStubStatic<typeof fs>
	let mockFileExists: sinon.SinonStub
	let mockGetTaskDirectoryPath: sinon.SinonStub
	let mockDownloadTask: sinon.SinonStub
	let consoleErrorSpy: sinon.SinonSpy
	let consoleLogSpy: sinon.SinonSpy

	const testTaskId = "task-123"
	const testTaskDir = "/storage/tasks/task-123"
	const testHistoryItem: HistoryItem = {
		id: testTaskId,
		ts: Date.now(),
		task: "Test task",
		totalCost: 0.1

	setup(() => {
		// Reset mocks
		sinon.restore()

		// Setup console spies
		consoleErrorSpy = sinon.stub(console, 'error')
		consoleLogSpy = sinon.stub(console, 'log')

		// Setup context mock
		mockContext = {
			globalStorageUri: {
				fsPath: "/global/storage"

		} as any

		// Setup context proxy mock
		mockContextProxy = {
			getValue: sinon.stub(),
			setValue: sinon.stub()
		} as any

		// Setup fs mocks
		mockFs = fs as sinon.SinonStubStatic<typeof fs>

		// Setup utility mocks
		mockFileExists = fileExistsAtPath as sinon.SinonStub
		mockGetTaskDirectoryPath = getTaskDirectoryPath as sinon.SinonStub
		mockDownloadTask = downloadTask as sinon.SinonStub

		// Default mock implementations
		mockContextProxy.getValue.returns([testHistoryItem])
		mockGetTaskDirectoryPath.resolves(testTaskDir)
		mockFileExists.resolves(true)
		mockFs.readFile.resolves(JSON.stringify([]))

		// Create instance
		taskHistory = new TheaTaskHistory(mockContext, mockContextProxy)

	teardown(() => {
		sinon.restore()

	suite("File I/O Operations", () => {
		test("should read conversation history from file system", async () => {
			const mockConversation: NeutralConversationHistory = [
				{
					role: "user",
					content: "Test message"
				},
				{
					role: "assistant",
					content: "Test response"

			mockFs.readFile.resolves(JSON.stringify(mockConversation))

			const result = await taskHistory.getTaskWithId(testTaskId)

			// Should read from correct path
			assert.ok(mockFs.readFile.calledWith(
				path.join(testTaskDir, "api_conversation_history.json")),
				"utf8"

			// Should parse JSON correctly
			assert.deepStrictEqual(result.apiConversationHistory, mockConversation)

		test("should handle missing conversation history file gracefully", async () => {
			mockFileExists.resolves(false)

			const result = await taskHistory.getTaskWithId(testTaskId)

			// Should not attempt to read non-existent file
			assert.ok(!mockFs.readFile.called)

			// Should return empty conversation history
			assert.deepStrictEqual(result.apiConversationHistory, [])

		test("should handle JSON parse errors in conversation history", async () => {
			mockFs.readFile.resolves("invalid json{")

			const result = await taskHistory.getTaskWithId(testTaskId)

			// Should return empty history on parse error
			assert.deepStrictEqual(result.apiConversationHistory, [])

			// Should log error (check console.error was called)
			assert.ok(console.error.calledWith(
				sinon.match.string.and(sinon.match(`Error reading conversation history for task ${testTaskId}:`))),
				sinon.match.instanceOf(Error)

		test("should handle file read errors gracefully", async () => {
			mockFs.readFile.rejects(new Error("Permission denied"))

			const result = await taskHistory.getTaskWithId(testTaskId)

			// Should return empty history on read error
			assert.deepStrictEqual(result.apiConversationHistory, [])

			// Should not throw
			assert.deepStrictEqual(result.historyItem, testHistoryItem)

		test("should construct correct file paths", async () => {
			await taskHistory.getTaskWithId(testTaskId)

			// Should get task directory path with correct parameters
			assert.ok(mockGetTaskDirectoryPath.calledWith(
				"/global/storage",
				testTaskId

			// Should check for file existence at correct path
			assert.ok(mockFileExists.calledWith(
				path.join(testTaskDir, "api_conversation_history.json"))

	suite("Cleanup Operations", () => {
		test("should delete task in correct order", async () => {
			const mockGetCurrentCline = sinon.stub().returns({ taskId: testTaskId })
			const mockFinishSubTask = sinon.stub()
			const callOrder: string[] = []

			// Track call order
			mockFinishSubTask.callsFake(async () => {
				callOrder.push("finishSubTask")

			mockContextProxy.setValue.callsFake(async () => {
				callOrder.push("deleteFromState")

			sinon.spy(ShadowCheckpointService, "deleteTask").callsFake(async () => {
				callOrder.push("deleteShadow")

			mockFs.rm.callsFake(async () => {
				callOrder.push("deleteDirectory")

			await taskHistory.deleteTaskWithId(testTaskId, mockGetCurrentCline, mockFinishSubTask)

			// Verify correct cleanup order
			assert.deepStrictEqual(callOrder, [
				"finishSubTask",      // 1. Remove from stack if current
				"deleteFromState",    // 2. Update state
				"deleteShadow",       // 3. Delete shadow repo
				"deleteDirectory"     // 4. Delete task directory
			])

		test("should handle deletion of non-current task", async () => {
			const mockGetCurrentCline = sinon.stub().returns({ taskId: "other-task" })
			const mockFinishSubTask = sinon.stub()

			await taskHistory.deleteTaskWithId(testTaskId, mockGetCurrentCline, mockFinishSubTask)

			// Should not call finishSubTask for non-current task
			assert.ok(!mockFinishSubTask.called)

			// Should still delete from state and filesystem
			assert.ok(mockContextProxy.setValue.called)
			assert.ok(mockFs.rm.called)

		test("should handle task not found in history", async () => {
			mockContextProxy.getValue.returns([]) // Empty history

			const mockGetCurrentCline = sinon.stub().returns(undefined)
			const mockFinishSubTask = sinon.stub()

			// Should not throw, but handle gracefully
			await expect(
				taskHistory.deleteTaskWithId(testTaskId, mockGetCurrentCline, mockFinishSubTask)
			).resolves.not.toThrow()

			// Should log that task was already removed
			assert.ok(console.log.calledWith(
				sinon.match.string.and(sinon.match("already removed or data inaccessible")))

		test("should continue cleanup even if shadow deletion fails", async () => {
			const mockGetCurrentCline = sinon.stub().returns(undefined)
			const mockFinishSubTask = sinon.stub()

			// Make shadow deletion fail
			sinon.spy(ShadowCheckpointService, "deleteTask").rejects(
				new Error("Shadow deletion failed")

			await taskHistory.deleteTaskWithId(testTaskId, mockGetCurrentCline, mockFinishSubTask)

			// Should log error but continue
			assert.ok(console.error.calledWith(
				sinon.match.string.and(sinon.match("[deleteTaskWithId task-123] failed to delete associated shadow repository or branch: Shadow deletion failed")))

			// Should still attempt directory deletion
			assert.ok(mockFs.rm.calledWith(testTaskDir, {
				recursive: true,
				force: true

		test("should handle directory deletion failure", async () => {
			const mockGetCurrentCline = sinon.stub().returns(undefined)
			const mockFinishSubTask = sinon.stub()

			mockFs.rm.rejects(new Error("Directory locked"))

			await taskHistory.deleteTaskWithId(testTaskId, mockGetCurrentCline, mockFinishSubTask)

			// Should log error
			assert.ok(console.error.calledWith(
				sinon.match.string.and(sinon.match(`[deleteTaskWithId task-123] failed to remove task directory ${testTaskDir}: Directory locked`)))

			// Should still update state
			assert.ok(mockContextProxy.setValue.called)

		test("should use force and recursive options for directory deletion", async () => {
			const mockGetCurrentCline = sinon.stub().returns(undefined)
			const mockFinishSubTask = sinon.stub()

			await taskHistory.deleteTaskWithId(testTaskId, mockGetCurrentCline, mockFinishSubTask)

			assert.ok(mockFs.rm.calledWith(testTaskDir, {
				recursive: true,
				force: true

	suite("Export Operations", () => {
		test("should export task with correct parameters", async () => {
			const mockConversation: NeutralConversationHistory = [
				{ role: "user", content: "Test" }

			mockFs.readFile.resolves(JSON.stringify(mockConversation))

			await taskHistory.exportTaskWithId(testTaskId)

			// Should call downloadTask with timestamp and conversation
			assert.ok(mockDownloadTask.calledWith(
				testHistoryItem.ts,
				mockConversation

		test("should handle export of task with empty conversation", async () => {
			mockFileExists.resolves(false)

			await taskHistory.exportTaskWithId(testTaskId)

			// Should still export with empty conversation
			assert.ok(mockDownloadTask.calledWith(
				testHistoryItem.ts,
				[]

		test("should throw if task not found for export", async () => {
			mockContextProxy.getValue.returns([]) // No tasks

			await expect(taskHistory.exportTaskWithId(testTaskId)).rejects.toThrow(
				`Task ${testTaskId} not found`

	suite("State Management", () => {
		test("should update history list when updating task", async () => {
			const newItem: HistoryItem = {
				id: "new-task",
				ts: Date.now(),
				task: "New task",
				totalCost: 0

			await taskHistory.updateTaskHistory(newItem)

			// Should save updated list
			assert.ok(mockContextProxy.setValue.calledWith(
				"taskHistory",
				[testHistoryItem, newItem]

		test("should replace existing task when updating", async () => {
			const updatedItem: HistoryItem = {
				...testHistoryItem,
				task: "Updated task"

			await taskHistory.updateTaskHistory(updatedItem)

			// Should replace, not duplicate
			assert.ok(mockContextProxy.setValue.calledWith(
				"taskHistory",
				[updatedItem]

		test("should handle empty initial history", async () => {
			mockContextProxy.getValue.returns(undefined)

			const newItem: HistoryItem = {
				id: "first-task",
				ts: Date.now(),
				task: "First task",
				totalCost: 0

			await taskHistory.updateTaskHistory(newItem)

			assert.ok(mockContextProxy.setValue.calledWith(
				"taskHistory",
				[newItem]

	suite("Show Task Operations", () => {
		test("should initialize new task when showing different task", async () => {
			const mockGetCurrentCline = sinon.stub().returns({ taskId: "other-task" })
			const mockInitClineWithHistoryItem = sinon.stub()
			const mockPostWebviewAction = sinon.stub()

			await taskHistory.showTaskWithId(
				testTaskId,
				mockGetCurrentCline,
				mockInitClineWithHistoryItem,
				mockPostWebviewAction

			// Should initialize with history item
			assert.ok(mockInitClineWithHistoryItem.calledWith(testHistoryItem))

			// Should post chat button action
			assert.ok(mockPostWebviewAction.calledWith("chatButtonClicked"))

		test("should not reinitialize when showing current task", async () => {
			const mockGetCurrentCline = sinon.stub().returns({ taskId: testTaskId })
			const mockInitClineWithHistoryItem = sinon.stub()
			const mockPostWebviewAction = sinon.stub()

			await taskHistory.showTaskWithId(
				testTaskId,
				mockGetCurrentCline,
				mockInitClineWithHistoryItem,
				mockPostWebviewAction

			// Should not initialize again
			assert.ok(!mockInitClineWithHistoryItem.called)

			// Should still post action
			assert.ok(mockPostWebviewAction.calledWith("chatButtonClicked"))

	suite("Error Handling", () => {
		test("should cleanup state when task not found", async () => {
			mockContextProxy.getValue.returns([]) // Task not in history

			// First call to getValue returns empty array
			// The getTaskWithId method will then call deleteTaskFromState which gets history and updates it
			await expect(taskHistory.getTaskWithId(testTaskId)).rejects.toThrow(
				`Task ${testTaskId} not found`

			// Should attempt to delete from state (though list is already empty)
			// The setValue won't be called because the list didn't change (was already empty)
			assert.ok(mockContextProxy.getValue.called)

		test("should handle general errors in deleteTaskWithId", async () => {
			const mockGetCurrentCline = sinon.stub().callsFake(() => {
				throw new Error("Unexpected error")

			const mockFinishSubTask = sinon.stub()

			// Should throw the error after logging
			await expect(
				taskHistory.deleteTaskWithId(testTaskId, mockGetCurrentCline, mockFinishSubTask)
			).rejects.toThrow("Unexpected error")

			assert.ok(console.error.calledWith(
				sinon.match.string.and(sinon.match("General error in deleteTaskWithId"))),
				sinon.match.instanceOf(Error)

		test("should provide all required paths in getTaskWithId result", async () => {
			const result = await taskHistory.getTaskWithId(testTaskId)

			assert.ok(result.hasOwnProperty('historyItem'))
			assert.ok(result.hasOwnProperty('taskDirPath'))
			assert.ok(result.hasOwnProperty('apiConversationHistoryFilePath'))
			assert.ok(result.hasOwnProperty('uiMessagesFilePath'))
			assert.ok(result.hasOwnProperty('apiConversationHistory'))

			// Verify paths are constructed correctly
			assert.strictEqual(result.taskDirPath, testTaskDir)
			assert.strictEqual(result.apiConversationHistoryFilePath, 
				path.join(testTaskDir, "api_conversation_history.json")

			assert.strictEqual(result.uiMessagesFilePath, 
				path.join(testTaskDir, "ui_messages.json")
