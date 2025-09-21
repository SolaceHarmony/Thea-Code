
import * as assert from 'assert'
import * as sinon from 'sinon'
import { TheaTask } from "../../TheaTask" // Renamed import
import { executeCommandTool } from "../../tools/executeCommandTool"
import type { TheaIgnoreController } from "../../ignore/TheaIgnoreController"
import type { TaskWebviewCommunicator } from "../../TaskWebviewCommunicator"
import type { TaskStateManager } from "../../TaskStateManager"
// TODO: Mock setup needs manual migration for "../../ignore/TheaIgnoreController"
import type { TheaMessage } from "../../../shared/ExtensionMessage"
import { TheaProvider } from "../../webview/TheaProvider"
import { formatResponse } from "../../prompts/responses"
import { ToolUse } from "../../assistant-message"
import { TheaAskResponse } from "../../../shared/WebviewMessage" // Import response type
import { AskApproval, HandleError, PushToolResult, RemoveClosingTag } from "../../diff/types"

// Mock dependencies
// Mock needs manual implementation
// TODO: Mock setup needs manual migration for "../../ignore/TheaIgnoreController"

suite("executeCommandTool", () => {
	// Setup common test variables
	let mockTheaTask: Partial<TheaTask> & {
		consecutiveMistakeCount: number
		didRejectTool: boolean
		webviewCommunicator: Partial<TaskWebviewCommunicator> & { say: sinon.SinonStub }
		taskStateManager: Partial<TaskStateManager> & { getTokenUsage: sinon.SinonStub }
		theaIgnoreController: Partial<TheaIgnoreController>
		ask: sinon.SinonStub
		say: sinon.SinonStub
		sayAndCreateMissingParamError: sinon.SinonStub
		executeCommandTool: sinon.SinonStub
	}
	let mockAskApproval: sinon.SinonStub
	let mockHandleError: sinon.SinonStub
	let mockPushToolResult: sinon.SinonStub
	let mockRemoveClosingTag: sinon.SinonStub
	let mockToolUse: ToolUse

	setup(() => {
		// Reset mocks
		sinon.restore()

		// Create mock implementations with eslint directives to handle the type issues
		mockTheaTask = ({
			ask: sinon.stub().resolves(undefined as never),
			say: sinon.stub().resolves(undefined as never),
			sayAndCreateMissingParamError: sinon.stub().resolves("Missing parameter error" as never),
			executeCommandTool: sinon.stub().resolves([false, "Command executed"] as never),
			consecutiveMistakeCount: 0,
			didRejectTool: false,
			theaIgnoreController: { 
				validateCommand: sinon.stub<(command: string) => string | undefined>().returns(undefined),
				validateAccess: sinon.stub<(filePath: string) => boolean>().returns(true),
				filterPaths: sinon.stub<(paths: string[]) => string[]>().callsFake((paths) => paths),
				getInstructions: sinon.stub<() => string | undefined>().returns(undefined)
			} as Partial<TheaIgnoreController>,
			webviewCommunicator: {
				// minimal TaskWebviewCommunicator shape used by code
				saveMessages: sinon.stub<() => Promise<void>>().resolves(undefined),
				isTaskAborted: sinon.stub<() => boolean>().returns(false),
				taskId: "test-task-id",
				instanceId: "test-instance-id",
				onAskResponded: sinon.stub(),
				handleWebviewAskResponse: sinon.stub().resolves(undefined as never),
				say: sinon.stub<() => Promise<void>>().resolves(undefined),
				ask: sinon.stub<() => Promise<{ response: TheaAskResponse; text?: string; images?: string[] }>>()
					.resolves({ response: "yesButtonClicked" }),
				providerRef: new WeakRef({} as unknown as TheaProvider),
				getMessages: sinon.stub<() => TheaMessage[]>().returns([]),
				addMessage: sinon.stub<(m: TheaMessage) => Promise<void>>().resolves(undefined),
				updateMessageUi: sinon.stub<(m: TheaMessage) => Promise<void>>().resolves(undefined),
				// optional UI helpers are not required here
			} as Partial<TaskWebviewCommunicator> & { say: sinon.SinonStub },
			taskStateManager: {
				// Add state manager mock setup
				getTokenUsage: sinon.stub<
					() => {
						totalTokensIn: number
						totalTokensOut: number
						totalCost: number
						contextTokens: number
					}
				>()
				.returns({ totalTokensIn: 0, totalTokensOut: 0, totalCost: 0, contextTokens: 0 }),
				providerRef: new WeakRef({} as unknown as TheaProvider),
				taskId: "test",
				taskNumber: 1,
				apiConversationHistory: [],
				setTaskState: sinon.stub(),
				updateLatestUiMessage: sinon.stub(),
				markTaskComplete: sinon.stub(),
				updateTokenUsage: sinon.stub(),
			} as Partial<TaskStateManager> & { getTokenUsage: sinon.SinonStub },
		} as unknown) as Partial<TheaTask> & {
			consecutiveMistakeCount: number
			didRejectTool: boolean
			webviewCommunicator: Partial<TaskWebviewCommunicator> & { say: sinon.SinonStub }
			taskStateManager: Partial<TaskStateManager> & { getTokenUsage: sinon.SinonStub }
			theaIgnoreController: Partial<TheaIgnoreController>
			ask: sinon.SinonStub
			say: sinon.SinonStub
			sayAndCreateMissingParamError: sinon.SinonStub
			executeCommandTool: sinon.SinonStub
		}

		// @ts-expect-error - Mock function type issues
		mockAskApproval = sinon.stub().resolves(true)
		// @ts-expect-error - Mock function type issues
		mockHandleError = sinon.stub().resolves(undefined)
		mockPushToolResult = sinon.stub()
		mockRemoveClosingTag = sinon.stub().returns("command")

		// Create a mock tool use object
		mockToolUse = {
			type: "tool_use",
			name: "execute_command",
			params: {
				command: "echo test",
			},
			partial: false,
		}
	})

	/**
	 * Tests for HTML entity unescaping in commands
	 * This verifies that HTML entities are properly converted to their actual characters
	 * before the command is executed
	 */
	suite("HTML entity unescaping", () => {
		test("should unescape &lt; to < character in commands", async () => {
			// Setup
			mockToolUse.params.command = "echo &lt;test&gt;"

			// Execute
			await executeCommandTool(
				mockTheaTask as unknown as TheaTask,
				mockToolUse,
				mockAskApproval as unknown as AskApproval,
				mockHandleError as unknown as HandleError,
				mockPushToolResult as unknown as PushToolResult,
				mockRemoveClosingTag as unknown as RemoveClosingTag,
			)

			// Verify
			assert.ok(mockAskApproval.calledWith("command", "echo <test>"))
			assert.ok(mockTheaTask.executeCommandTool.calledWith("echo <test>", undefined))
		})

		test("should unescape &gt; to > character in commands", async () => {
			// Setup
			mockToolUse.params.command = "echo test &gt; output.txt"

			// Execute
			await executeCommandTool(
				mockTheaTask as unknown as TheaTask,
				mockToolUse,
				mockAskApproval as unknown as AskApproval,
				mockHandleError as unknown as HandleError,
				mockPushToolResult as unknown as PushToolResult,
				mockRemoveClosingTag as unknown as RemoveClosingTag,
			)

			// Verify
			assert.ok(mockAskApproval.calledWith("command", "echo test > output.txt"))
			assert.ok(mockTheaTask.executeCommandTool.calledWith("echo test > output.txt", undefined))
		})

		test("should unescape &amp; to & character in commands", async () => {
			// Setup
			mockToolUse.params.command = "echo foo &amp;&amp; echo bar"

			// Execute
			await executeCommandTool(
				mockTheaTask as unknown as TheaTask,
				mockToolUse,
				mockAskApproval as unknown as AskApproval,
				mockHandleError as unknown as HandleError,
				mockPushToolResult as unknown as PushToolResult,
				mockRemoveClosingTag as unknown as RemoveClosingTag,
			)

			// Verify
			assert.ok(mockAskApproval.calledWith("command", "echo foo && echo bar"))
			assert.ok(mockTheaTask.executeCommandTool.calledWith("echo foo && echo bar", undefined))
		})

		test("should handle multiple mixed HTML entities in commands", async () => {
			// Setup
			mockToolUse.params.command = "grep -E 'pattern' &lt;file.txt &gt;output.txt 2&gt;&amp;1"

			// Execute
			await executeCommandTool(
				mockTheaTask as unknown as TheaTask,
				mockToolUse,
				mockAskApproval as unknown as AskApproval,
				mockHandleError as unknown as HandleError,
				mockPushToolResult as unknown as PushToolResult,
				mockRemoveClosingTag as unknown as RemoveClosingTag,
			)

			// Verify
			const expectedCommand = "grep -E 'pattern' <file.txt >output.txt 2>&1"
			assert.ok(mockAskApproval.calledWith("command", expectedCommand))
			assert.ok(mockTheaTask.executeCommandTool.calledWith(expectedCommand, undefined))
		})
	})

	// Other functionality tests
	suite("Basic functionality", () => {
		test("should execute a command normally without HTML entities", async () => {
			// Setup
			mockToolUse.params.command = "echo test"

			// Execute
			await executeCommandTool(
				mockTheaTask as unknown as TheaTask,
				mockToolUse,
				mockAskApproval as unknown as AskApproval,
				mockHandleError as unknown as HandleError,
				mockPushToolResult as unknown as PushToolResult,
				mockRemoveClosingTag as unknown as RemoveClosingTag,
			)

			// Verify
			assert.ok(mockAskApproval.calledWith("command", "echo test"))
			assert.ok(mockTheaTask.executeCommandTool.calledWith("echo test", undefined))
			assert.ok(mockPushToolResult.calledWith("Command executed"))
		})

		test("should pass along custom working directory if provided", async () => {
			// Setup
			mockToolUse.params.command = "echo test"
			mockToolUse.params.cwd = "/custom/path"

			// Execute
			await executeCommandTool(
				mockTheaTask as unknown as TheaTask,
				mockToolUse,
				mockAskApproval as unknown as AskApproval,
				mockHandleError as unknown as HandleError,
				mockPushToolResult as unknown as PushToolResult,
				mockRemoveClosingTag as unknown as RemoveClosingTag,
			)

			// Verify
			assert.ok(mockTheaTask.executeCommandTool.calledWith("echo test", "/custom/path"))
		})
	})

	suite("Error handling", () => {
		test("should handle missing command parameter", async () => {
			// Setup
			mockToolUse.params.command = undefined

			// Execute
			await executeCommandTool(
				mockTheaTask as unknown as TheaTask,
				mockToolUse,
				mockAskApproval as unknown as AskApproval,
				mockHandleError as unknown as HandleError,
				mockPushToolResult as unknown as PushToolResult,
				mockRemoveClosingTag as unknown as RemoveClosingTag,
			)

			// Verify
			assert.strictEqual(mockTheaTask.consecutiveMistakeCount, 1)
			assert.ok(mockTheaTask.sayAndCreateMissingParamError.calledWith("execute_command", "command"))
			assert.ok(mockPushToolResult.calledWith("Missing parameter error"))
			assert.ok(!mockAskApproval.called)
			assert.ok(!mockTheaTask.executeCommandTool.called)
		})

		test("should handle command rejection", async () => {
			// Setup
			mockToolUse.params.command = "echo test"
			// @ts-expect-error - Mock function type issues
			mockAskApproval.resolves(false)

			// Execute
			await executeCommandTool(
				mockTheaTask as unknown as TheaTask,
				mockToolUse,
				mockAskApproval as unknown as AskApproval,
				mockHandleError as unknown as HandleError,
				mockPushToolResult as unknown as PushToolResult,
				mockRemoveClosingTag as unknown as RemoveClosingTag,
			)

			// Verify
			assert.ok(mockAskApproval.calledWith("command", "echo test"))
			assert.ok(!mockTheaTask.executeCommandTool.called)
			assert.ok(!mockPushToolResult.called)
		})

		test("should handle theaignore validation failures", async () => {
			// Setup
			mockToolUse.params.command = "cat .env"
			// Override the validateCommand mock to return a filename
			const validateCommandMock = sinon.stub().returns(".env")
			// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
			mockTheaTask.theaIgnoreController = {
				validateCommand: validateCommandMock, // Simplified mock
				// eslint-disable-next-line @typescript-eslint/no-explicit-any
			} as any // Use 'as any' to bypass strict type check for mock

			const mockTheaIgnoreError = "TheaIgnore error"
			;(formatResponse.theaIgnoreError as sinon.SinonStub).returns(mockTheaIgnoreError)
			;(formatResponse.toolError as sinon.SinonStub).returns("Tool error")

			// Execute
			await executeCommandTool(
				mockTheaTask as unknown as TheaTask,
				mockToolUse,
				mockAskApproval as unknown as AskApproval,
				mockHandleError as unknown as HandleError,
				mockPushToolResult as unknown as PushToolResult,
				mockRemoveClosingTag as unknown as RemoveClosingTag,
			)

			// Verify
			assert.ok(validateCommandMock.calledWith("cat .env"))
			// Add check to ensure communicator exists
			assert.notStrictEqual(mockTheaTask.webviewCommunicator, undefined)
			assert.ok(mockTheaTask.webviewCommunicator.say.calledWith("theaignore_error", ".env"))
			assert.ok(formatResponse.theaIgnoreError.calledWith(".env"))
			assert.ok(formatResponse.toolError.calledWith(mockTheaIgnoreError))
			assert.ok(mockPushToolResult.called)
			assert.ok(!mockAskApproval.called)
			assert.ok(!mockTheaTask.executeCommandTool.called)
		})
	})
// Mock cleanup
})
