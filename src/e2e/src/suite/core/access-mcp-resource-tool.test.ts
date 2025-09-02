import type { ToolUse } from "../assistant-message"
import type { TheaTask } from "../TheaTask"
import { accessMcpResourceTool } from "../tools/accessMcpResourceTool"
import type { RemoveClosingTag } from "../tools/types"

import * as assert from 'assert'
import * as sinon from 'sinon'
suite("accessMcpResourceTool", () => {
	const askApproval = sinon.stub()
	const handleError = sinon.stub()
	const removeClosingTag: RemoveClosingTag = (_: string, val?: string) => val ?? ""

	function createMockTask(): sinon.SinonStubStatic<TheaTask> {
		return {
			consecutiveMistakeCount: 0,
			sayAndCreateMissingParamError: sinon.stub().resolves("err"),
			webviewCommunicator: {
				ask: sinon.stub(),
				say: sinon.stub(),
				handleWebviewAskResponse: sinon.stub(),
			} as unknown as sinon.SinonStubStatic<TheaTask["webviewCommunicator"]>,
			providerRef: {
				deref: sinon.stub(),
			} as unknown as sinon.SinonStubStatic<TheaTask["providerRef"]>,
		} as unknown as sinon.SinonStubStatic<TheaTask>

	test("increments mistake count and reports missing server_name", async () => {
		const theaTask = createMockTask()
		const pushToolResult = sinon.stub()
		const block: ToolUse = {
			type: "tool_use",
			name: "access_mcp_resource",
			params: { uri: "/res" },
			partial: false,

		await accessMcpResourceTool(theaTask, block, askApproval, handleError, pushToolResult, removeClosingTag)

		assert.strictEqual(theaTask.consecutiveMistakeCount, 1)
		// eslint-disable-next-line @typescript-eslint/unbound-method
		assert.ok(theaTask.sayAndCreateMissingParamError.calledWith("access_mcp_resource", "server_name"))
		assert.ok(pushToolResult.calledWith("err"))

	test("increments mistake count and reports missing uri", async () => {
		const theaTask = createMockTask()
		const pushToolResult = sinon.stub()
		const block: ToolUse = {
			type: "tool_use",
			name: "access_mcp_resource",
			params: { server_name: "srv" },
			partial: false,

		await accessMcpResourceTool(theaTask, block, askApproval, handleError, pushToolResult, removeClosingTag)

		assert.strictEqual(theaTask.consecutiveMistakeCount, 1)
		// eslint-disable-next-line @typescript-eslint/unbound-method
		assert.ok(theaTask.sayAndCreateMissingParamError.calledWith("access_mcp_resource", "uri"))
		assert.ok(pushToolResult.calledWith("err"))
