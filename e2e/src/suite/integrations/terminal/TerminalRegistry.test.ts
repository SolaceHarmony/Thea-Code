
import { TerminalRegistry } from "../TerminalRegistry"
import { EXTENSION_DISPLAY_NAME } from "../../../shared/config/thea-config"

// Mock vscode.window.createTerminal
import * as assert from 'assert'
import * as sinon from 'sinon'
import * as vscode from 'vscode'
const mockCreateTerminal = sinon.stub()
// TODO: Mock setup needs manual migration
// TODO: Mock needs manual migration
// TODO: Implement proper mock with proxyquire

suite("TerminalRegistry", () => {
	setup(() => {
		mockCreateTerminal.mockClear()

	suite("createTerminal", () => {
		test("creates terminal with PAGER set to cat", () => {
			TerminalRegistry.createTerminal("/test/path")

			assert.ok(mockCreateTerminal.calledWith({
				cwd: "/test/path",
				name: EXTENSION_DISPLAY_NAME as string,
				iconPath: expect.any(Object)) as unknown,
				env: {
					PAGER: "cat",
					PROMPT_COMMAND: "sleep 0.050",
					VTE_VERSION: "0",
				},
