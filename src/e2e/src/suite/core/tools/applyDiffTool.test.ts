// npx jest src/core/tools/__tests__/applyDiffTool.test.ts

import { applyDiffTool } from "../applyDiffTool"
import { TheaTask } from "../../TheaTask"
import type { ToolUse } from "../../assistant-message"
import { AskApproval, HandleError, PushToolResult, RemoveClosingTag } from "../types"
import { fileExistsAtPath } from "../../../utils/fs"
import fs from "fs/promises"

// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
import * as assert from 'assert'
import * as sinon from 'sinon'
suite("applyDiffTool", () => {
	type MockTheaTask = {
		cwd: string
		consecutiveMistakeCount: number
		consecutiveMistakeCountForApplyDiff: Map<string, number>
		webviewCommunicator: { 
			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			ask: sinon.SinonStub Promise<any>>
			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			say: sinon.SinonStub Promise<any>>

		diffViewProvider: {
			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			open: sinon.SinonStub any>
			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			update: sinon.SinonStub any>
			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			scrollToFirstDiff: sinon.SinonStub
			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			revertChanges: sinon.SinonStub
			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			saveChanges: sinon.SinonStub
			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			reset: sinon.SinonStub

		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		diffStrategy: { applyDiff: sinon.SinonStub; getProgressStatus: sinon.SinonStub }
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		theaIgnoreController: { validateAccess: sinon.SinonStub }
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		sayAndCreateMissingParamError: sinon.SinonStub
		didEditFile?: boolean

	let mockTheaTask: MockTheaTask
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	let mockAskApproval: sinon.SinonStub
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	let mockHandleError: sinon.SinonStub
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	let mockPushToolResult: sinon.SinonStub
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	let mockRemoveClosingTag: sinon.SinonStub
	const mockedFs = fs as sinon.SinonStubStatic<typeof fs>

	setup(() => {
		sinon.restore()
		mockTheaTask = {
			cwd: "/test",
			consecutiveMistakeCount: 0,
			consecutiveMistakeCountForApplyDiff: new Map(),
			webviewCommunicator: {
				// @ts-expect-error - Jest mock setup requires bypassing strict typing
				ask: sinon.stub().resolves(undefined),
				// @ts-expect-error - Jest mock setup requires bypassing strict typing
				say: sinon.stub().resolves(undefined),
			},
			diffViewProvider: {
				open: sinon.stub(),
				update: sinon.stub(),
				scrollToFirstDiff: sinon.stub(),
				revertChanges: sinon.stub(),
				// @ts-expect-error - Jest mock setup requires bypassing strict typing
				saveChanges: sinon.stub().resolves({ newProblemsMessage: "", userEdits: undefined, finalContent: "" }),
				reset: sinon.stub(),
			},
			diffStrategy: {
				// @ts-expect-error - Jest mock setup requires bypassing strict typing
				applyDiff: sinon.stub().resolves({ success: true, content: "" }),
				getProgressStatus: sinon.stub(),
			},
			theaIgnoreController: { validateAccess: sinon.stub().returns(true) },
			// @ts-expect-error - Jest mock setup requires bypassing strict typing
			sayAndCreateMissingParamError: sinon.stub().resolves("Missing parameter error"),
		} as MockTheaTask
		// @ts-expect-error - Jest mock setup requires bypassing strict typing
		mockAskApproval = sinon.stub().resolves(true)
		// @ts-expect-error - Jest mock setup requires bypassing strict typing
		mockHandleError = sinon.stub().resolves(undefined)
		mockPushToolResult = sinon.stub()
		mockRemoveClosingTag = sinon.stub().returns("")

	test("handles partial blocks by sending a progress update", async () => {
		const block: ToolUse = {
			type: "tool_use",
			name: "apply_diff",
			params: { path: "file.txt", diff: "d" },
			partial: true,

		await applyDiffTool(
			mockTheaTask as unknown as TheaTask,
			block,
			mockAskApproval as unknown as AskApproval,
			mockHandleError as unknown as HandleError,
			mockPushToolResult as unknown as PushToolResult,
			mockRemoveClosingTag as unknown as RemoveClosingTag,

		assert.ok(mockTheaTask.webviewCommunicator.ask.called)
		assert.ok(!mockAskApproval.called)
		assert.ok(!mockTheaTask.diffViewProvider.open.called)

	test("handles missing path parameter", async () => {
		const block: ToolUse = { type: "tool_use", name: "apply_diff", params: { diff: "d" }, partial: false }

		await applyDiffTool(
			mockTheaTask as unknown as TheaTask,
			block,
			mockAskApproval as unknown as AskApproval,
			mockHandleError as unknown as HandleError,
			mockPushToolResult as unknown as PushToolResult,
			mockRemoveClosingTag as unknown as RemoveClosingTag,

		assert.ok(mockTheaTask.sayAndCreateMissingParamError.calledWith("apply_diff", "path"))
		assert.ok(mockPushToolResult.calledWith("Missing parameter error"))
		assert.strictEqual(mockTheaTask.consecutiveMistakeCount, 1)
		assert.ok(!mockAskApproval.called)

	test("handles non-existent files", async () => {
		// eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access
		;(fileExistsAtPath as any).resolves(false)
		const block: ToolUse = {
			type: "tool_use",
			name: "apply_diff",
			params: { path: "file.txt", diff: "d" },
			partial: false,

		await applyDiffTool(
			mockTheaTask as unknown as TheaTask,
			block,
			mockAskApproval as unknown as AskApproval,
			mockHandleError as unknown as HandleError,
			mockPushToolResult as unknown as PushToolResult,
			mockRemoveClosingTag as unknown as RemoveClosingTag,

		assert.ok(fileExistsAtPath.called)
		assert.ok(mockTheaTask.webviewCommunicator.say.called)
		assert.ok(mockPushToolResult.called)
		assert.strictEqual(mockTheaTask.consecutiveMistakeCount, 1)
		// ensure no attempt to read file or apply diff
		assert.ok(!mockedFs.readFile.called)
		assert.ok(!mockTheaTask.diffStrategy.applyDiff.called)
