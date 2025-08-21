import * as assert from 'assert'
import * as sinon from 'sinon'
/**
 * attemptCompletionTool partial/final flow tests
 * Tests partial completion, final completion, approval flows, and telemetry
 */

import { TheaTask } from "../../TheaTask"
import { attemptCompletionTool } from "../attemptCompletionTool"
import type { ToolUse } from "../../assistant-message"
import { telemetryService } from "../../../services/telemetry/TelemetryService"
import type { 
	AskApproval, 
	HandleError, 
	PushToolResult, 
	RemoveClosingTag, 
	ToolDescription,
	AskFinishSubTaskApproval 
} from "../types"

// Mock dependencies
// TODO: Mock setup needs manual migration for "../../../services/telemetry/TelemetryService"
// TODO: Mock setup needs manual migration for "../../TheaTask"

suite("attemptCompletionTool - Partial/Final Flow Tests", () => {
	let mockTheaTask: sinon.SinonStubbedInstance<TheaTask>
	let mockAskApproval: sinon.SinonStubbedInstanceFunction<AskApproval>
	let mockHandleError: sinon.SinonStubbedInstanceFunction<HandleError>
	let mockPushToolResult: sinon.SinonStubbedInstanceFunction<PushToolResult>
	let mockRemoveClosingTag: sinon.SinonStubbedInstanceFunction<RemoveClosingTag>
	let mockToolDescription: sinon.SinonStubbedInstanceFunction<ToolDescription>
	let mockAskFinishSubTaskApproval: sinon.SinonStubbedInstanceFunction<AskFinishSubTaskApproval>
	let mockTelemetryService: sinon.SinonStubbedInstance<typeof telemetryService>

	setup(() => {
		// Reset all mocks
		sinon.restore()

		// Create mock TheaTask
		mockTheaTask = {
			taskId: "test-task-123",
			consecutiveMistakeCount: 0,
			didRejectTool: false,
			parentTask: null,
			providerRef: { deref: sinon.stub() },
			userMessageContent: [],
			taskStateManager: {
				theaTaskMessages: [],
				getTokenUsage: sinon.stub().returns({ input: 100, output: 50 })
			},
			webviewCommunicator: {
				say: sinon.stub().resolves(undefined),
				ask: sinon.stub().resolves({ 
					response: "messageResponse", 
					text: "User feedback", 
					images: [] 
				})
			},
			emit: sinon.stub(),
			sayAndCreateMissingParamError: sinon.stub().resolves("Missing parameter: result"),
			executeCommandTool: sinon.stub().resolves([false, "Command executed successfully"])
		} as any

		// Create mock functions
		mockAskApproval = sinon.stub().resolves(true)
		mockHandleError = sinon.stub().resolves(undefined)
		mockPushToolResult = sinon.stub()
		mockRemoveClosingTag = sinon.stub((tag, content) => content || "")
		mockToolDescription = sinon.stub().returns("Attempt Completion")
		mockAskFinishSubTaskApproval = sinon.stub().resolves(true)

		// Mock telemetry service
		mockTelemetryService = telemetryService as sinon.SinonStubbedInstance<typeof telemetryService>
		mockTelemetryService.captureTaskCompleted = sinon.stub()
	})

	teardown(() => {
		sinon.restore()
	})

	suite("Partial Flow", () => {
		test("should handle partial result without command", async () => {
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
			assert.ok(mockTheaTask.webviewCommunicator.say.calledWith(
				"completion_result",
				"Partial result in progress...",
				undefined,
				true // partial flag
			))

			// Should not capture telemetry for partial
			assert.ok(!mockTelemetryService.captureTaskCompleted.called)
			
			// Should not emit taskCompleted for partial
			assert.ok(!mockTheaTask.emit.called)
		})

		test("should handle partial result with command when last message is command ask", async () => {
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
			assert.ok(mockTheaTask.webviewCommunicator.ask.calledWith(
				"command",
				"npm test",
				true // partial
			))

			// Should not say completion_result since we're updating command
			assert.ok(!mockTheaTask.webviewCommunicator.say.called)
		})

		test("should handle partial result with command when last message is not command ask", async () => {
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
			assert.ok(mockTheaTask.webviewCommunicator.say.calledWith(
				"completion_result",
				"Task complete",
				undefined,
				false
			))

			// Should capture telemetry
			assert.ok(mockTelemetryService.captureTaskCompleted.calledWith("test-task-123"))

			// Should emit taskCompleted
			assert.ok(mockTheaTask.emit.calledWith(
				"taskCompleted",
				"test-task-123",
				{ input: 100, output: 50 }
			))

			// Then ask for command
			assert.ok(mockTheaTask.webviewCommunicator.ask.calledWith(
				"command",
				"npm run build",
				true
			))
		})

		test("should handle errors in ask calls gracefully", async () => {
			// Make ask throw an error
			mockTheaTask.webviewCommunicator.ask = sinon.stub().rejects(new Error("Ask failed"))

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

	suite("Final Flow", () => {
		test("should handle missing result parameter", async () => {
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
			assert.strictEqual(mockTheaTask.consecutiveMistakeCount, 1)

			// Should create missing param error
			assert.ok(mockTheaTask.sayAndCreateMissingParamError.calledWith(
				"attempt_completion",
				"result"
			))

			// Should push error result
			assert.ok(mockPushToolResult.calledWith("Missing parameter: result"))
		})

		test("should handle final completion without command", async () => {
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
			assert.strictEqual(mockTheaTask.consecutiveMistakeCount, 0)

			// Should say completion result
			assert.ok(mockTheaTask.webviewCommunicator.say.calledWith(
				"completion_result",
				"Task completed successfully",
				undefined,
				false
			))

			// Should capture telemetry
			assert.ok(mockTelemetryService.captureTaskCompleted.calledWith("test-task-123"))

			// Should emit taskCompleted
			assert.ok(mockTheaTask.emit.calledWith(
				"taskCompleted",
				"test-task-123",
				{ input: 100, output: 50 }
			))

			// Should ask for user feedback
			assert.ok(mockTheaTask.webviewCommunicator.ask.calledWith(
				"completion_result",
				"",
				false
			))
		})

		test("should handle final completion with command approval", async () => {
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
			assert.ok(mockAskApproval.calledWith("command", "git push"))

			// Should execute command
			assert.ok(mockTheaTask.executeCommandTool.calledWith("git push"))

			// Should ask for completion feedback
			assert.ok(mockTheaTask.webviewCommunicator.ask.calledWith(
				"completion_result",
				"",
				false
			))
		})

		test("should handle command rejection", async () => {
			// Set up command rejection
			mockAskApproval.resolves(false)

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
			assert.ok(mockAskApproval.calledWith("command", "rm -rf /"))

			// Should not execute command
			assert.ok(!mockTheaTask.executeCommandTool.called)

			// Should not ask for completion feedback
			assert.ok(!mockTheaTask.webviewCommunicator.ask.calledWith(
				"completion_result",
				"",
				false
			)
		})

		test("should handle user rejection during command execution", async () => {
			// Set up user rejection in executeCommandTool
			mockTheaTask.executeCommandTool.resolves([true, "User rejected command"])

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
			assert.strictEqual(mockTheaTask.didRejectTool, true)

			// Should push rejection result
			assert.ok(mockPushToolResult.calledWith("User rejected command"))

			// Should not continue to ask for feedback
			assert.ok(!mockTheaTask.webviewCommunicator.ask.calledWith(
				"completion_result",
				"",
				false
			)
		})
	})

	suite("Sub-task Completion", () => {
		test("should handle sub-task completion approval", async () => {
			// Set up parent task
			mockTheaTask.parentTask = { id: "parent-task" } as any
			mockTheaTask.taskStateManager.theaTaskMessages = [
				{ text: "Sub-task complete message" } as any
			]
			
			const mockProvider = { finishSubTask: sinon.stub() }
			mockTheaTask.providerRef.deref = sinon.stub().returns(mockProvider)

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
			assert.ok(mockAskFinishSubTaskApproval.called)

			// Should finish sub-task with provider
			assert.ok(mockProvider.finishSubTask.calledWith(
				"Task complete: Sub-task complete message"
			))

			// Should not ask for regular completion feedback
			assert.ok(!mockTheaTask.webviewCommunicator.ask.calledWith(
				"completion_result",
				"",
				false
			)
		})

		test("should handle sub-task completion rejection", async () => {
			// Set up parent task and rejection
			mockTheaTask.parentTask = { id: "parent-task" } as any
			mockAskFinishSubTaskApproval.resolves(false)

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
			assert.ok(mockAskFinishSubTaskApproval.called)

			// Should not finish sub-task
			assert.ok(!mockTheaTask.providerRef.deref.called)
		})
	})

	suite("User Feedback Flow", () => {
		test("should handle user feedback with text and images", async () => {
			// Set up user feedback response
			mockTheaTask.webviewCommunicator.ask.resolves({
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
			assert.ok(mockTheaTask.webviewCommunicator.say.calledWith(
				"user_feedback",
				"Please add error handling",
				["image1.png", "image2.png"]
			))

			// The tool modifies userMessageContent array by pushing content
			// Since we're using a mock, we just verify the say was called with feedback
			// The actual content manipulation is an implementation detail
		})

		test("should handle yesButtonClicked response", async () => {
			// Set up yes button response
			mockTheaTask.webviewCommunicator.ask.resolves({
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
			assert.ok(mockPushToolResult.calledWith(""))

			// Should not say user feedback
			assert.ok(!mockTheaTask.webviewCommunicator.say.calledWith(
				"user_feedback",
				expect.anything(),
				expect.anything()
			)
		})

		test("should include command result in feedback flow", async () => {
			// Set up command with result
			mockTheaTask.executeCommandTool.resolves([
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

	suite("Error Handling", () => {
		test("should handle errors during execution", async () => {
			// Make webviewCommunicator.say throw an error
			const testError = new Error("Communication failed")
			mockTheaTask.webviewCommunicator.say.rejects(testError)

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
			assert.ok(mockHandleError.calledWith("inspecting site", testError))
		})

		test("should handle non-Error exceptions", async () => {
			// Make something throw a non-Error
			mockTheaTask.webviewCommunicator.say.rejects("String error")

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
			assert.ok(mockHandleError.calledWith(
				"inspecting site",
				// TODO: Object partial match - {
					message: "String error"
				}))
		})
	})

	suite("Telemetry", () => {
		test("should capture task completion telemetry for final completion", async () => {
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

			assert.ok(mockTelemetryService.captureTaskCompleted.calledWith("test-task-123"))
		})

		test("should emit taskCompleted event with token usage", async () => {
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

			assert.ok(mockTheaTask.emit.calledWith(
				"taskCompleted",
				"test-task-123",
				{ input: 100, output: 50 }
			))
		})

		test("should not capture telemetry for partial completions", async () => {
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

			assert.ok(!mockTelemetryService.captureTaskCompleted.called)
		})
	})
// Mock cleanup