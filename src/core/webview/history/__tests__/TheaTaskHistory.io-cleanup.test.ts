/**
 * TheaTaskHistory IO and cleanup tests
 * Tests file I/O operations, cleanup order, export paths, and error handling
 */

import { describe, expect, it, jest, beforeEach, afterEach } from "@jest/globals"
import { TheaTaskHistory } from "../TheaTaskHistory"
import * as vscode from "vscode"
import * as fs from "fs/promises"
import * as path from "path"
import { ContextProxy } from "../../../config/ContextProxy"
import { ShadowCheckpointService } from "../../../../services/checkpoints/ShadowCheckpointService"
import { fileExistsAtPath } from "../../../../utils/fs"
import { getTaskDirectoryPath } from "../../../../shared/storagePathManager"
import { downloadTask } from "../../../../integrations/misc/export-markdown"
import type { HistoryItem } from "../../../../shared/HistoryItem"
import type { NeutralConversationHistory } from "../../../../shared/neutral-history"

// Mock dependencies
jest.mock("fs/promises", () => ({
	readFile: jest.fn(),
	rm: jest.fn(),
	writeFile: jest.fn(),
	mkdir: jest.fn()
}))
jest.mock("../../../../utils/fs")
jest.mock("../../../../shared/storagePathManager")
jest.mock("../../../../integrations/misc/export-markdown")
jest.mock("../../../../services/checkpoints/ShadowCheckpointService")
jest.mock("../../../../utils/path", () => ({
	getWorkspacePath: jest.fn().mockReturnValue("/workspace")
}))
jest.mock("../../../../utils/logging", () => ({
	logger: {
		warn: jest.fn(),
		error: jest.fn(),
		info: jest.fn()
	}
}))

describe("TheaTaskHistory - IO and Cleanup Operations", () => {
	let taskHistory: TheaTaskHistory
	let mockContext: jest.Mocked<vscode.ExtensionContext>
	let mockContextProxy: jest.Mocked<ContextProxy>
	let mockFs: jest.Mocked<typeof fs>
	let mockFileExists: jest.MockedFunction<typeof fileExistsAtPath>
	let mockGetTaskDirectoryPath: jest.MockedFunction<typeof getTaskDirectoryPath>
	let mockDownloadTask: jest.MockedFunction<typeof downloadTask>
	let consoleErrorSpy: jest.SpyInstance
	let consoleLogSpy: jest.SpyInstance

	const testTaskId = "task-123"
	const testTaskDir = "/storage/tasks/task-123"
	const testHistoryItem: HistoryItem = {
		id: testTaskId,
		ts: Date.now(),
		task: "Test task",
		totalCost: 0.1
	}

	beforeEach(() => {
		// Reset mocks
		jest.clearAllMocks()

		// Setup console spies
		consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation()
		consoleLogSpy = jest.spyOn(console, 'log').mockImplementation()

		// Setup context mock
		mockContext = {
			globalStorageUri: {
				fsPath: "/global/storage"
			}
		} as any

		// Setup context proxy mock
		mockContextProxy = {
			getValue: jest.fn(),
			setValue: jest.fn()
		} as any

		// Setup fs mocks
		mockFs = fs as jest.Mocked<typeof fs>

		// Setup utility mocks
		mockFileExists = fileExistsAtPath as jest.MockedFunction<typeof fileExistsAtPath>
		mockGetTaskDirectoryPath = getTaskDirectoryPath as jest.MockedFunction<typeof getTaskDirectoryPath>
		mockDownloadTask = downloadTask as jest.MockedFunction<typeof downloadTask>

		// Default mock implementations
		mockContextProxy.getValue.mockReturnValue([testHistoryItem])
		mockGetTaskDirectoryPath.mockResolvedValue(testTaskDir)
		mockFileExists.mockResolvedValue(true)
		mockFs.readFile.mockResolvedValue(JSON.stringify([]))

		// Create instance
		taskHistory = new TheaTaskHistory(mockContext, mockContextProxy)
	})

	afterEach(() => {
		jest.restoreAllMocks()
	})

	describe("File I/O Operations", () => {
		it("should read conversation history from file system", async () => {
			const mockConversation: NeutralConversationHistory = [
				{
					role: "user",
					content: "Test message"
				},
				{
					role: "assistant",
					content: "Test response"
				}
			]

			mockFs.readFile.mockResolvedValue(JSON.stringify(mockConversation))

			const result = await taskHistory.getTaskWithId(testTaskId)

			// Should read from correct path
			expect(mockFs.readFile).toHaveBeenCalledWith(
				path.join(testTaskDir, "api_conversation_history.json"),
				"utf8"
			)

			// Should parse JSON correctly
			expect(result.apiConversationHistory).toEqual(mockConversation)
		})

		it("should handle missing conversation history file gracefully", async () => {
			mockFileExists.mockResolvedValue(false)

			const result = await taskHistory.getTaskWithId(testTaskId)

			// Should not attempt to read non-existent file
			expect(mockFs.readFile).not.toHaveBeenCalled()

			// Should return empty conversation history
			expect(result.apiConversationHistory).toEqual([])
		})

		it("should handle JSON parse errors in conversation history", async () => {
			mockFs.readFile.mockResolvedValue("invalid json{")

			const result = await taskHistory.getTaskWithId(testTaskId)

			// Should return empty history on parse error
			expect(result.apiConversationHistory).toEqual([])

			// Should log error (check console.error was called)
			expect(console.error).toHaveBeenCalledWith(
				expect.stringContaining(`Error reading conversation history for task ${testTaskId}:`),
				expect.any(Error)
			)
		})

		it("should handle file read errors gracefully", async () => {
			mockFs.readFile.mockRejectedValue(new Error("Permission denied"))

			const result = await taskHistory.getTaskWithId(testTaskId)

			// Should return empty history on read error
			expect(result.apiConversationHistory).toEqual([])

			// Should not throw
			expect(result.historyItem).toEqual(testHistoryItem)
		})

		it("should construct correct file paths", async () => {
			await taskHistory.getTaskWithId(testTaskId)

			// Should get task directory path with correct parameters
			expect(mockGetTaskDirectoryPath).toHaveBeenCalledWith(
				"/global/storage",
				testTaskId
			)

			// Should check for file existence at correct path
			expect(mockFileExists).toHaveBeenCalledWith(
				path.join(testTaskDir, "api_conversation_history.json")
			)
		})
	})

	describe("Cleanup Operations", () => {
		it("should delete task in correct order", async () => {
			const mockGetCurrentCline = jest.fn().mockReturnValue({ taskId: testTaskId })
			const mockFinishSubTask = jest.fn()
			const callOrder: string[] = []

			// Track call order
			mockFinishSubTask.mockImplementation(async () => {
				callOrder.push("finishSubTask")
			})
			mockContextProxy.setValue.mockImplementation(async () => {
				callOrder.push("deleteFromState")
			})
			jest.spyOn(ShadowCheckpointService, "deleteTask").mockImplementation(async () => {
				callOrder.push("deleteShadow")
			})
			mockFs.rm.mockImplementation(async () => {
				callOrder.push("deleteDirectory")
			})

			await taskHistory.deleteTaskWithId(testTaskId, mockGetCurrentCline, mockFinishSubTask)

			// Verify correct cleanup order
			expect(callOrder).toEqual([
				"finishSubTask",      // 1. Remove from stack if current
				"deleteFromState",    // 2. Update state
				"deleteShadow",       // 3. Delete shadow repo
				"deleteDirectory"     // 4. Delete task directory
			])
		})

		it("should handle deletion of non-current task", async () => {
			const mockGetCurrentCline = jest.fn().mockReturnValue({ taskId: "other-task" })
			const mockFinishSubTask = jest.fn()

			await taskHistory.deleteTaskWithId(testTaskId, mockGetCurrentCline, mockFinishSubTask)

			// Should not call finishSubTask for non-current task
			expect(mockFinishSubTask).not.toHaveBeenCalled()

			// Should still delete from state and filesystem
			expect(mockContextProxy.setValue).toHaveBeenCalled()
			expect(mockFs.rm).toHaveBeenCalled()
		})

		it("should handle task not found in history", async () => {
			mockContextProxy.getValue.mockReturnValue([]) // Empty history

			const mockGetCurrentCline = jest.fn().mockReturnValue(undefined)
			const mockFinishSubTask = jest.fn()

			// Should not throw, but handle gracefully
			await expect(
				taskHistory.deleteTaskWithId(testTaskId, mockGetCurrentCline, mockFinishSubTask)
			).resolves.not.toThrow()

			// Should log that task was already removed
			expect(console.log).toHaveBeenCalledWith(
				expect.stringContaining("already removed or data inaccessible")
			)
		})

		it("should continue cleanup even if shadow deletion fails", async () => {
			const mockGetCurrentCline = jest.fn().mockReturnValue(undefined)
			const mockFinishSubTask = jest.fn()

			// Make shadow deletion fail
			jest.spyOn(ShadowCheckpointService, "deleteTask").mockRejectedValue(
				new Error("Shadow deletion failed")
			)

			await taskHistory.deleteTaskWithId(testTaskId, mockGetCurrentCline, mockFinishSubTask)

			// Should log error but continue
			expect(console.error).toHaveBeenCalledWith(
				expect.stringContaining("[deleteTaskWithId task-123] failed to delete associated shadow repository or branch: Shadow deletion failed")
			)

			// Should still attempt directory deletion
			expect(mockFs.rm).toHaveBeenCalledWith(testTaskDir, {
				recursive: true,
				force: true
			})
		})

		it("should handle directory deletion failure", async () => {
			const mockGetCurrentCline = jest.fn().mockReturnValue(undefined)
			const mockFinishSubTask = jest.fn()

			mockFs.rm.mockRejectedValue(new Error("Directory locked"))

			await taskHistory.deleteTaskWithId(testTaskId, mockGetCurrentCline, mockFinishSubTask)

			// Should log error
			expect(console.error).toHaveBeenCalledWith(
				expect.stringContaining(`[deleteTaskWithId task-123] failed to remove task directory ${testTaskDir}: Directory locked`)
			)

			// Should still update state
			expect(mockContextProxy.setValue).toHaveBeenCalled()
		})

		it("should use force and recursive options for directory deletion", async () => {
			const mockGetCurrentCline = jest.fn().mockReturnValue(undefined)
			const mockFinishSubTask = jest.fn()

			await taskHistory.deleteTaskWithId(testTaskId, mockGetCurrentCline, mockFinishSubTask)

			expect(mockFs.rm).toHaveBeenCalledWith(testTaskDir, {
				recursive: true,
				force: true
			})
		})
	})

	describe("Export Operations", () => {
		it("should export task with correct parameters", async () => {
			const mockConversation: NeutralConversationHistory = [
				{ role: "user", content: "Test" }
			]
			mockFs.readFile.mockResolvedValue(JSON.stringify(mockConversation))

			await taskHistory.exportTaskWithId(testTaskId)

			// Should call downloadTask with timestamp and conversation
			expect(mockDownloadTask).toHaveBeenCalledWith(
				testHistoryItem.ts,
				mockConversation
			)
		})

		it("should handle export of task with empty conversation", async () => {
			mockFileExists.mockResolvedValue(false)

			await taskHistory.exportTaskWithId(testTaskId)

			// Should still export with empty conversation
			expect(mockDownloadTask).toHaveBeenCalledWith(
				testHistoryItem.ts,
				[]
			)
		})

		it("should throw if task not found for export", async () => {
			mockContextProxy.getValue.mockReturnValue([]) // No tasks

			await expect(taskHistory.exportTaskWithId(testTaskId)).rejects.toThrow(
				`Task ${testTaskId} not found`
			)
		})
	})

	describe("State Management", () => {
		it("should update history list when updating task", async () => {
			const newItem: HistoryItem = {
				id: "new-task",
				ts: Date.now(),
				task: "New task",
				totalCost: 0
			}

			await taskHistory.updateTaskHistory(newItem)

			// Should save updated list
			expect(mockContextProxy.setValue).toHaveBeenCalledWith(
				"taskHistory",
				[testHistoryItem, newItem]
			)
		})

		it("should replace existing task when updating", async () => {
			const updatedItem: HistoryItem = {
				...testHistoryItem,
				task: "Updated task"
			}

			await taskHistory.updateTaskHistory(updatedItem)

			// Should replace, not duplicate
			expect(mockContextProxy.setValue).toHaveBeenCalledWith(
				"taskHistory",
				[updatedItem]
			)
		})

		it("should handle empty initial history", async () => {
			mockContextProxy.getValue.mockReturnValue(undefined)

			const newItem: HistoryItem = {
				id: "first-task",
				ts: Date.now(),
				task: "First task",
				totalCost: 0
			}

			await taskHistory.updateTaskHistory(newItem)

			expect(mockContextProxy.setValue).toHaveBeenCalledWith(
				"taskHistory",
				[newItem]
			)
		})
	})

	describe("Show Task Operations", () => {
		it("should initialize new task when showing different task", async () => {
			const mockGetCurrentCline = jest.fn().mockReturnValue({ taskId: "other-task" })
			const mockInitClineWithHistoryItem = jest.fn()
			const mockPostWebviewAction = jest.fn()

			await taskHistory.showTaskWithId(
				testTaskId,
				mockGetCurrentCline,
				mockInitClineWithHistoryItem,
				mockPostWebviewAction
			)

			// Should initialize with history item
			expect(mockInitClineWithHistoryItem).toHaveBeenCalledWith(testHistoryItem)

			// Should post chat button action
			expect(mockPostWebviewAction).toHaveBeenCalledWith("chatButtonClicked")
		})

		it("should not reinitialize when showing current task", async () => {
			const mockGetCurrentCline = jest.fn().mockReturnValue({ taskId: testTaskId })
			const mockInitClineWithHistoryItem = jest.fn()
			const mockPostWebviewAction = jest.fn()

			await taskHistory.showTaskWithId(
				testTaskId,
				mockGetCurrentCline,
				mockInitClineWithHistoryItem,
				mockPostWebviewAction
			)

			// Should not initialize again
			expect(mockInitClineWithHistoryItem).not.toHaveBeenCalled()

			// Should still post action
			expect(mockPostWebviewAction).toHaveBeenCalledWith("chatButtonClicked")
		})
	})

	describe("Error Handling", () => {
		it("should cleanup state when task not found", async () => {
			mockContextProxy.getValue.mockReturnValue([]) // Task not in history

			// First call to getValue returns empty array
			// The getTaskWithId method will then call deleteTaskFromState which gets history and updates it
			await expect(taskHistory.getTaskWithId(testTaskId)).rejects.toThrow(
				`Task ${testTaskId} not found`
			)

			// Should attempt to delete from state (though list is already empty)
			// The setValue won't be called because the list didn't change (was already empty)
			expect(mockContextProxy.getValue).toHaveBeenCalled()
		})

		it("should handle general errors in deleteTaskWithId", async () => {
			const mockGetCurrentCline = jest.fn().mockImplementation(() => {
				throw new Error("Unexpected error")
			})
			const mockFinishSubTask = jest.fn()

			// Should throw the error after logging
			await expect(
				taskHistory.deleteTaskWithId(testTaskId, mockGetCurrentCline, mockFinishSubTask)
			).rejects.toThrow("Unexpected error")

			expect(console.error).toHaveBeenCalledWith(
				expect.stringContaining("General error in deleteTaskWithId"),
				expect.any(Error)
			)
		})

		it("should provide all required paths in getTaskWithId result", async () => {
			const result = await taskHistory.getTaskWithId(testTaskId)

			expect(result).toHaveProperty("historyItem")
			expect(result).toHaveProperty("taskDirPath")
			expect(result).toHaveProperty("apiConversationHistoryFilePath")
			expect(result).toHaveProperty("uiMessagesFilePath")
			expect(result).toHaveProperty("apiConversationHistory")

			// Verify paths are constructed correctly
			expect(result.taskDirPath).toBe(testTaskDir)
			expect(result.apiConversationHistoryFilePath).toBe(
				path.join(testTaskDir, "api_conversation_history.json")
			)
			expect(result.uiMessagesFilePath).toBe(
				path.join(testTaskDir, "ui_messages.json")
			)
		})
	})
})