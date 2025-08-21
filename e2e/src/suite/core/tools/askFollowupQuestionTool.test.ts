
import * as assert from 'assert'
import * as sinon from 'sinon'
import { askFollowupQuestionTool } from "../askFollowupQuestionTool"
import type { TheaTask } from "../../TheaTask"
import type { ToolUse } from "../../assistant-message"
import { AskApproval, HandleError, PushToolResult, RemoveClosingTag } from "../types"
import { formatResponse } from "../../prompts/responses"
import type { TheaAskResponse } from "../../../shared/WebviewMessage"

// TODO: Use proxyquire for module mocking - "../../prompts/responses")

suite("askFollowupQuestionTool", () => {
	let mockAsk: sinon.SinonStub
	let mockSay: sinon.SinonStub
	let mockSayAndCreateMissingParamError: sinon.SinonStub
	let mockTheaTask: TheaTask
	let mockAskApproval: AskApproval
	let mockHandleError: HandleError
	let mockPushToolResult: PushToolResult
	let mockRemoveClosingTag: RemoveClosingTag
	const mockedFormatResponse = formatResponse as sinon.SinonStubbedFunction<typeof formatResponse>

	setup(() => {
		sinon.restore()
		
		// Create mock functions
		mockAsk = sinon.stub()
		mockSay = sinon.stub()
		mockSayAndCreateMissingParamError = sinon.stub()
		
		// Create mock TheaTask with proper type casting
		mockTheaTask = {
			consecutiveMistakeCount: 0,
			webviewCommunicator: {
				ask: mockAsk,
				say: mockSay,
			},
			sayAndCreateMissingParamError: mockSayAndCreateMissingParamError,
		} as unknown as TheaTask

		// Set up mock return values - Sinon mocks have complex typing
		// @ts-expect-error - Mock setup requires bypassing strict typing
		mockAsk.resolves({ 
			response: "yesButtonClicked" as TheaAskResponse, 
			text: "", 
			images: [] 
		})
		// @ts-expect-error - Mock setup requires bypassing strict typing
		mockSay.resolves(undefined)
		// @ts-expect-error - Mock setup requires bypassing strict typing
		mockSayAndCreateMissingParamError.resolves("Missing param error")
		
		// @ts-expect-error - Mock setup requires bypassing strict typing
		mockAskApproval = sinon.stub().resolves(true)
		// @ts-expect-error - Mock setup requires bypassing strict typing
		mockHandleError = sinon.stub().resolves(undefined)
		mockPushToolResult = sinon.stub() as PushToolResult
		mockRemoveClosingTag = sinon.stub((tag: string, content?: string) => content ?? "") as RemoveClosingTag
	})

	test("handles partial blocks by sending a progress update", async () => {
		const block: ToolUse = {
			type: "tool_use",
			name: "ask_followup_question",
			params: { question: "Test?" },
			partial: true,
		}

		await askFollowupQuestionTool(
			mockTheaTask,
			block,
			mockAskApproval,
			mockHandleError,
			mockPushToolResult,
			mockRemoveClosingTag,
		)

		assert.ok(mockRemoveClosingTag.calledWith("question", "Test?"))
		assert.ok(mockAsk.calledWith("followup", "Test?", true))
		assert.ok(!mockPushToolResult.called)
	})

	test("handles missing question parameter", async () => {
		const block: ToolUse = {
			type: "tool_use",
			name: "ask_followup_question",
			params: {},
			partial: false,
		}

		await askFollowupQuestionTool(
			mockTheaTask,
			block,
			mockAskApproval,
			mockHandleError,
			mockPushToolResult,
			mockRemoveClosingTag,
		)

		assert.ok(mockSayAndCreateMissingParamError.calledWith("ask_followup_question", "question"))
		assert.ok(mockPushToolResult.calledWith("Missing param error"))
		assert.strictEqual(mockTheaTask.consecutiveMistakeCount, 1)
		assert.ok(!mockAsk.called)
	})

	test("sends followup question and pushes tool result", async () => {
		const block: ToolUse = {
			type: "tool_use",
			name: "ask_followup_question",
			params: { question: "What?", follow_up: "<suggest><answer>Yes</answer></suggest>" },
			partial: false,
		}
		// @ts-expect-error - Mock setup requires bypassing strict typing
		mockAsk.resolves({ 
			response: "messageResponse" as TheaAskResponse,
			text: "Sure", 
			images: ["img"] 
		})
		mockedFormatResponse.toolResult.returns("tool result")

		await askFollowupQuestionTool(
			mockTheaTask,
			block,
			mockAskApproval,
			mockHandleError,
			mockPushToolResult,
			mockRemoveClosingTag,
		)

		assert.ok(mockAsk.calledWith(
			"followup",
			JSON.stringify({ question: "What?", suggest: ["<answer>Yes</answer>"] })),
			false,
		)
		assert.ok(mockSay.calledWith("user_feedback", "Sure", ["img"]))
		assert.ok(mockPushToolResult.calledWith("tool result"))
		assert.strictEqual(mockTheaTask.consecutiveMistakeCount, 0)
	})

	test("handles invalid follow_up XML", async () => {
		const block: ToolUse = {
			type: "tool_use",
			name: "ask_followup_question",
			params: { question: "Q", follow_up: "<invalid" },
			partial: false,
		}
		mockedFormatResponse.toolError.returns("tool error")

		await askFollowupQuestionTool(
			mockTheaTask,
			block,
			mockAskApproval,
			mockHandleError,
			mockPushToolResult,
			mockRemoveClosingTag,
		)

		assert.ok(mockSay.calledWith(
			"error",
			// TODO: String contains check - "Failed to parse operations")),
		)
		assert.ok(mockPushToolResult.calledWith("tool error"))
		assert.strictEqual(mockTheaTask.consecutiveMistakeCount, 1)
	})
})
