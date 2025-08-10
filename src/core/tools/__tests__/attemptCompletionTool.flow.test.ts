/**
 * attemptCompletionTool partial/final flow tests
 * Tests partial completion, final completion, approval flows, and telemetry
 */

import { describe, expect, it, jest, beforeEach, afterEach } from "@jest/globals"
import { attemptCompletionTool } from "../attemptCompletionTool"
import { TheaTask } from "../../TheaTask"
import { telemetryService } from "../../../services/telemetry/TelemetryService"
import type { ToolUse } from "../../assistant-message"
import type { 
	AskApproval, 
	HandleError, 
	PushToolResult, 
	RemoveClosingTag, 
	ToolDescription,
	AskFinishSubTaskApproval 
} from "../types"

// Mock dependencies
jest.mock("../../../services/telemetry/TelemetryService")
jest.mock("../../TheaTask")

describe("attemptCompletionTool - Partial/Final Flow Tests", () => {
	let mockTheaTask: jest.Mocked<TheaTask>
	let mockAskApproval: jest.MockedFunction<AskApproval>
	let mockHandleError: jest.MockedFunction<HandleError>
	let mockPushToolResult: jest.MockedFunction<PushToolResult>
	let mockRemoveClosingTag: jest.MockedFunction<RemoveClosingTag>
	let mockToolDescription: jest.MockedFunction<ToolDescription>
	let mockAskFinishSubTaskApproval: jest.MockedFunction<AskFinishSubTaskApproval>
	let mockTelemetryService: jest.Mocked<typeof telemetryService>

	beforeEach(() => {
		// Reset all mocks
		jest.clearAllMocks()

		// Create mock TheaTask
		mockTheaTask = {
			taskId: "test-task-123",
			consecutiveMistakeCount: 0,
			didRejectTool: false,
			parentTask: null,
			providerRef: { deref: jest.fn() },
			userMessageContent: [],
			taskStateManager: {
				theaTaskMessages: [],
				getTokenUsage: jest.fn().mockReturnValue({ input: 100, output: 50 })
			},
			webviewCommunicator: {
				say: jest.fn().mockResolvedValue(undefined),
				ask: jest.fn().mockResolvedValue({ 
					response: "messageResponse", 
					text: "User feedback", 
					images: [] 
				})
			},
			emit: jest.fn(),
			sayAndCreateMissingParamError: jest.fn().mockResolvedValue("Missing parameter: result"),
			executeCommandTool: jest.fn().mockResolvedValue([false, "Command executed successfully"])
		} as any

		// Create mock functions
		mockAskApproval = jest.fn().mockResolvedValue(true)
		mockHandleError = jest.fn().mockResolvedValue(undefined)
		mockPushToolResult = jest.fn()
		mockRemoveClosingTag = jest.fn((tag, content) => content || "")
		mockToolDescription = jest.fn().mockReturnValue("Attempt Completion")
		mockAskFinishSubTaskApproval = jest.fn().mockResolvedValue(true)

		// Mock telemetry service
		mockTelemetryService = telemetryService as jest.Mocked<typeof telemetryService>
		mockTelemetryService.captureTaskCompleted = jest.fn()
	})

	afterEach(() => {
		jest.restoreAllMocks()
	})

	describe("Partial Flow", () => {
		it("should handle partial result without command", async () => {
			const block: ToolUse = {
				tool: "attempt_completion",
				partial: true,
				params: {
					result: "Partial result in progress..."
				}
			}

			await attemptCompletionTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag,
				mockToolDescription,
				mockAskFinishSubTaskApproval
			)

			// Should say partial completion result
			expect(mockTheaTask.webviewCommunicator.say).toHaveBeenCalledWith(
				"completion_result",
				"Partial result in progress...",
				undefined,
				true // partial flag
			)

			// Should not capture telemetry for partial
			expect(mockTelemetryService.captureTaskCompleted).not.toHaveBeenCalled()
			
			// Should not emit taskCompleted for partial
			expect(mockTheaTask.emit).not.toHaveBeenCalled()
		})

		it("should handle partial result with command when last message is command ask", async () => {
			// Set up last message as command ask
			mockTheaTask.taskStateManager.theaTaskMessages = [
				{ ask: "command", text: "Previous command" } as any
			]

			const block: ToolUse = {
				tool: "attempt_completion",
				partial: true,
				params: {
					result: "Task complete",
					command: "npm test"
				}
			}

			await attemptCompletionTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag,
				mockToolDescription,
				mockAskFinishSubTaskApproval
			)

			// Should update the command ask
			expect(mockTheaTask.webviewCommunicator.ask).toHaveBeenCalledWith(
				"command",
				"npm test",
				true // partial
			)

			// Should not say completion_result since we're updating command
			expect(mockTheaTask.webviewCommunicator.say).not.toHaveBeenCalled()
		})

		it("should handle partial result with command when last message is not command ask", async () => {
			// Set up last message as something else
			mockTheaTask.taskStateManager.theaTaskMessages = [
				{ say: "completion_result", text: "Result" } as any
			]

			const block: ToolUse = {
				tool: "attempt_completion",
				partial: true,
				params: {
					result: "Task complete",
					command: "npm run build"
				}
			}

			await attemptCompletionTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag,
				mockToolDescription,
				mockAskFinishSubTaskApproval
			)

			// Should say completion result first
			expect(mockTheaTask.webviewCommunicator.say).toHaveBeenCalledWith(
				"completion_result",
				"Task complete",
				undefined,
				false
			)

			// Should capture telemetry
			expect(mockTelemetryService.captureTaskCompleted).toHaveBeenCalledWith("test-task-123")

			// Should emit taskCompleted
			expect(mockTheaTask.emit).toHaveBeenCalledWith(
				"taskCompleted",
				"test-task-123",
				{ input: 100, output: 50 }
			)

			// Then ask for command
			expect(mockTheaTask.webviewCommunicator.ask).toHaveBeenCalledWith(
				"command",
				"npm run build",
				true
			)
		})

		it("should handle errors in ask calls gracefully", async () => {
			// Make ask throw an error
			mockTheaTask.webviewCommunicator.ask = jest.fn().mockRejectedValue(new Error("Ask failed"))

			const block: ToolUse = {
				tool: "attempt_completion",
				partial: true,
				params: {
					result: "Result",
					command: "command"
				}
			}

			// Should not throw
			await expect(attemptCompletionTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag,
				mockToolDescription,
				mockAskFinishSubTaskApproval
			)).resolves.not.toThrow()
		})
	})

	describe("Final Flow", () => {
		it("should handle missing result parameter", async () => {
			const block: ToolUse = {
				tool: "attempt_completion",
				partial: false,
				params: {} // Missing result
			}

			await attemptCompletionTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag,
				mockToolDescription,
				mockAskFinishSubTaskApproval
			)

			// Should increment mistake count
			expect(mockTheaTask.consecutiveMistakeCount).toBe(1)

			// Should create missing param error
			expect(mockTheaTask.sayAndCreateMissingParamError).toHaveBeenCalledWith(
				"attempt_completion",
				"result"
			)

			// Should push error result
			expect(mockPushToolResult).toHaveBeenCalledWith("Missing parameter: result")
		})

		it("should handle final completion without command", async () => {
			const block: ToolUse = {
				tool: "attempt_completion",
				partial: false,
				params: {
					result: "Task completed successfully"
				}
			}

			await attemptCompletionTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag,
				mockToolDescription,
				mockAskFinishSubTaskApproval
			)

			// Should reset mistake count
			expect(mockTheaTask.consecutiveMistakeCount).toBe(0)

			// Should say completion result
			expect(mockTheaTask.webviewCommunicator.say).toHaveBeenCalledWith(
				"completion_result",
				"Task completed successfully",
				undefined,
				false
			)

			// Should capture telemetry
			expect(mockTelemetryService.captureTaskCompleted).toHaveBeenCalledWith("test-task-123")

			// Should emit taskCompleted
			expect(mockTheaTask.emit).toHaveBeenCalledWith(
				"taskCompleted",
				"test-task-123",
				{ input: 100, output: 50 }
			)

			// Should ask for user feedback
			expect(mockTheaTask.webviewCommunicator.ask).toHaveBeenCalledWith(
				"completion_result",
				"",
				false
			)
		})

		it("should handle final completion with command approval", async () => {
			const block: ToolUse = {
				tool: "attempt_completion",
				partial: false,
				params: {
					result: "Task complete",
					command: "git push"
				}
			}

			await attemptCompletionTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag,
				mockToolDescription,
				mockAskFinishSubTaskApproval
			)

			// Should ask for command approval
			expect(mockAskApproval).toHaveBeenCalledWith("command", "git push")

			// Should execute command
			expect(mockTheaTask.executeCommandTool).toHaveBeenCalledWith("git push")

			// Should ask for completion feedback
			expect(mockTheaTask.webviewCommunicator.ask).toHaveBeenCalledWith(
				"completion_result",
				"",
				false
			)
		})

		it("should handle command rejection", async () => {
			// Set up command rejection
			mockAskApproval.mockResolvedValue(false)

			const block: ToolUse = {
				tool: "attempt_completion",
				partial: false,
				params: {
					result: "Task complete",
					command: "rm -rf /"
				}
			}

			await attemptCompletionTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag,
				mockToolDescription,
				mockAskFinishSubTaskApproval
			)

			// Should ask for approval
			expect(mockAskApproval).toHaveBeenCalledWith("command", "rm -rf /")

			// Should not execute command
			expect(mockTheaTask.executeCommandTool).not.toHaveBeenCalled()

			// Should not ask for completion feedback
			expect(mockTheaTask.webviewCommunicator.ask).not.toHaveBeenCalledWith(
				"completion_result",
				"",
				false
			)
		})

		it("should handle user rejection during command execution", async () => {
			// Set up user rejection in executeCommandTool
			mockTheaTask.executeCommandTool.mockResolvedValue([true, "User rejected command"])

			const block: ToolUse = {
				tool: "attempt_completion",
				partial: false,
				params: {
					result: "Task complete",
					command: "dangerous command"
				}
			}

			await attemptCompletionTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag,
				mockToolDescription,
				mockAskFinishSubTaskApproval
			)

			// Should set didRejectTool flag
			expect(mockTheaTask.didRejectTool).toBe(true)

			// Should push rejection result
			expect(mockPushToolResult).toHaveBeenCalledWith("User rejected command")

			// Should not continue to ask for feedback
			expect(mockTheaTask.webviewCommunicator.ask).not.toHaveBeenCalledWith(
				"completion_result",
				"",
				false
			)
		})
	})

	describe("Sub-task Completion", () => {
		it("should handle sub-task completion approval", async () => {
			// Set up parent task
			mockTheaTask.parentTask = { id: "parent-task" } as any
			mockTheaTask.taskStateManager.theaTaskMessages = [
				{ text: "Sub-task complete message" } as any
			]
			
			const mockProvider = { finishSubTask: jest.fn() }
			mockTheaTask.providerRef.deref = jest.fn().mockReturnValue(mockProvider)

			const block: ToolUse = {
				tool: "attempt_completion",
				partial: false,
				params: {
					result: "Sub-task completed"
				}
			}

			await attemptCompletionTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag,
				mockToolDescription,
				mockAskFinishSubTaskApproval
			)

			// Should ask for sub-task finish approval
			expect(mockAskFinishSubTaskApproval).toHaveBeenCalled()

			// Should finish sub-task with provider
			expect(mockProvider.finishSubTask).toHaveBeenCalledWith(
				"Task complete: Sub-task complete message"
			)

			// Should not ask for regular completion feedback
			expect(mockTheaTask.webviewCommunicator.ask).not.toHaveBeenCalledWith(
				"completion_result",
				"",
				false
			)
		})

		it("should handle sub-task completion rejection", async () => {
			// Set up parent task and rejection
			mockTheaTask.parentTask = { id: "parent-task" } as any
			mockAskFinishSubTaskApproval.mockResolvedValue(false)

			const block: ToolUse = {
				tool: "attempt_completion",
				partial: false,
				params: {
					result: "Sub-task completed"
				}
			}

			await attemptCompletionTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag,
				mockToolDescription,
				mockAskFinishSubTaskApproval
			)

			// Should ask for approval
			expect(mockAskFinishSubTaskApproval).toHaveBeenCalled()

			// Should not finish sub-task
			expect(mockTheaTask.providerRef.deref).not.toHaveBeenCalled()
		})
	})

	describe("User Feedback Flow", () => {
		it("should handle user feedback with text and images", async () => {
			// Set up user feedback response
			mockTheaTask.webviewCommunicator.ask.mockResolvedValue({
				response: "messageResponse",
				text: "Please add error handling",
				images: ["image1.png", "image2.png"]
			})

			const block: ToolUse = {
				tool: "attempt_completion",
				partial: false,
				params: {
					result: "Task done"
				}
			}

			await attemptCompletionTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag,
				mockToolDescription,
				mockAskFinishSubTaskApproval
			)

			// Should say user feedback
			expect(mockTheaTask.webviewCommunicator.say).toHaveBeenCalledWith(
				"user_feedback",
				"Please add error handling",
				["image1.png", "image2.png"]
			)

			// The tool modifies userMessageContent array by pushing content
			// Since we're using a mock, we just verify the say was called with feedback
			// The actual content manipulation is an implementation detail
		})

		it("should handle yesButtonClicked response", async () => {
			// Set up yes button response
			mockTheaTask.webviewCommunicator.ask.mockResolvedValue({
				response: "yesButtonClicked",
				text: null,
				images: []
			})

			const block: ToolUse = {
				tool: "attempt_completion",
				partial: false,
				params: {
					result: "Task done"
				}
			}

			await attemptCompletionTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag,
				mockToolDescription,
				mockAskFinishSubTaskApproval
			)

			// Should push empty result to signal completion
			expect(mockPushToolResult).toHaveBeenCalledWith("")

			// Should not say user feedback
			expect(mockTheaTask.webviewCommunicator.say).not.toHaveBeenCalledWith(
				"user_feedback",
				expect.anything(),
				expect.anything()
			)
		})

		it("should include command result in feedback flow", async () => {
			// Set up command with result
			mockTheaTask.executeCommandTool.mockResolvedValue([
				false, 
				[
					{ type: "text", text: "Tests passed" },
					{ type: "image", data: "test-results.png" }
				]
			])

			const block: ToolUse = {
				tool: "attempt_completion",
				partial: false,
				params: {
					result: "Ran tests",
					command: "npm test"
				}
			}

			await attemptCompletionTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag,
				mockToolDescription,
				mockAskFinishSubTaskApproval
			)

			// Should include command result in user message content
			expect(mockTheaTask.userMessageContent).toContainEqual({
				type: "text",
				text: "Tests passed"
			})

			expect(mockTheaTask.userMessageContent).toContainEqual({
				type: "image",
				data: "test-results.png"
			})
		})
	})

	describe("Error Handling", () => {
		it("should handle errors during execution", async () => {
			// Make webviewCommunicator.say throw an error
			const testError = new Error("Communication failed")
			mockTheaTask.webviewCommunicator.say.mockRejectedValue(testError)

			const block: ToolUse = {
				tool: "attempt_completion",
				partial: false,
				params: {
					result: "Task done"
				}
			}

			await attemptCompletionTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag,
				mockToolDescription,
				mockAskFinishSubTaskApproval
			)

			// Should handle error
			expect(mockHandleError).toHaveBeenCalledWith("inspecting site", testError)
		})

		it("should handle non-Error exceptions", async () => {
			// Make something throw a non-Error
			mockTheaTask.webviewCommunicator.say.mockRejectedValue("String error")

			const block: ToolUse = {
				tool: "attempt_completion",
				partial: false,
				params: {
					result: "Task done"
				}
			}

			await attemptCompletionTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag,
				mockToolDescription,
				mockAskFinishSubTaskApproval
			)

			// Should convert to Error
			expect(mockHandleError).toHaveBeenCalledWith(
				"inspecting site",
				expect.objectContaining({
					message: "String error"
				})
			)
		})
	})

	describe("Telemetry", () => {
		it("should capture task completion telemetry for final completion", async () => {
			const block: ToolUse = {
				tool: "attempt_completion",
				partial: false,
				params: {
					result: "Task completed"
				}
			}

			await attemptCompletionTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag,
				mockToolDescription,
				mockAskFinishSubTaskApproval
			)

			expect(mockTelemetryService.captureTaskCompleted).toHaveBeenCalledWith("test-task-123")
		})

		it("should emit taskCompleted event with token usage", async () => {
			const block: ToolUse = {
				tool: "attempt_completion",
				partial: false,
				params: {
					result: "Task completed"
				}
			}

			await attemptCompletionTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag,
				mockToolDescription,
				mockAskFinishSubTaskApproval
			)

			expect(mockTheaTask.emit).toHaveBeenCalledWith(
				"taskCompleted",
				"test-task-123",
				{ input: 100, output: 50 }
			)
		})

		it("should not capture telemetry for partial completions", async () => {
			const block: ToolUse = {
				tool: "attempt_completion",
				partial: true,
				params: {
					result: "Partial..."
				}
			}

			await attemptCompletionTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag,
				mockToolDescription,
				mockAskFinishSubTaskApproval
			)

			expect(mockTelemetryService.captureTaskCompleted).not.toHaveBeenCalled()
		})
	})
})