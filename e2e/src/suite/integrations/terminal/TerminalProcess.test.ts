
import * as vscode from "vscode"

import { TerminalProcess, mergePromise } from "../TerminalProcess"
import { Terminal } from "../Terminal"
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

suite("TerminalProcess", () => {
	let terminalProcess: TerminalProcess
	let mockTerminal: sinon.SinonStubStatic<
		vscode.Terminal & {
			shellIntegration: {
				executeCommand: sinon.SinonStub

	>
	let mockTerminalInfo: Terminal
	let mockExecution: unknown
	let mockStream: AsyncIterableIterator<string>

	setup(() => {
		// Create properly typed mock terminal
		mockTerminal = {
			shellIntegration: {
				executeCommand: sinon.stub(),
			},
			name: EXTENSION_DISPLAY_NAME as string,
			processId: Promise.resolve(123),
			creationOptions: {},
			exitStatus: undefined,
			state: { isInteractedWith: true },
			dispose: sinon.stub(),
			hide: sinon.stub(),
			show: sinon.stub(),
			sendText: sinon.stub(),
		} as unknown as sinon.SinonStubStatic<
			vscode.Terminal & {
				shellIntegration: {
					executeCommand: sinon.SinonStub

		>

		mockTerminalInfo = new Terminal(1, mockTerminal, "./")

		// Create a process for testing
		terminalProcess = new TerminalProcess(mockTerminalInfo)

		TerminalRegistry["terminals"].push(mockTerminalInfo)

		// Reset event listeners
		terminalProcess.removeAllListeners()

	suite("run", () => {
		test("handles shell integration commands correctly", async () => {
			let lines: string[] = []

			terminalProcess.on("completed", (output) => {
				if (output) {
					lines = output.split("\n")

			// Mock stream data with shell integration sequences.
			mockStream = (function* () {
				yield "\x1b]633;C\x07" // The first chunk contains the command start sequence with bell character.
				yield "Initial output\n"
				yield "More output\n"
				yield "Final output"
				yield "\x1b]633;D\x07" // The last chunk contains the command end sequence with bell character.
				terminalProcess.emit("shell_execution_complete", { exitCode: 0 })
			})()

			mockExecution = {
				read: sinon.stub().returns(mockStream),

			mockTerminal.shellIntegration.executeCommand.returns(mockExecution)

			const runPromise = terminalProcess.run("test command")
			terminalProcess.emit("stream_available", mockStream)
			await runPromise

			assert.deepStrictEqual(lines, ["Initial output", "More output", "Final output"])
			assert.strictEqual(terminalProcess.isHot, false)

		test("handles terminals without shell integration", async () => {
			// Create a terminal without shell integration
			const noShellTerminal = {
				sendText: sinon.stub(),
				shellIntegration: undefined,
				name: "No Shell Terminal",
				processId: Promise.resolve(456),
				creationOptions: {},
				exitStatus: undefined,
				state: { isInteractedWith: true },
				dispose: sinon.stub(),
				hide: sinon.stub(),
				show: sinon.stub(),
			} as unknown as vscode.Terminal

			// Create new terminal info with the no-shell terminal
			const noShellTerminalInfo = new Terminal(2, noShellTerminal, "./")

			// Create new process with the no-shell terminal
			const noShellProcess = new TerminalProcess(noShellTerminalInfo)

			// Set up event listeners to verify events are emitted
			const eventPromises = Promise.all([
				new Promise<void>((resolve) => noShellProcess.once("no_shell_integration", () => resolve())),
				new Promise<void>((resolve) => noShellProcess.once("completed", () => resolve())),
				new Promise<void>((resolve) => noShellProcess.once("continue", resolve)),
			])

			// Run command and wait for all events
			await noShellProcess.run("test command")
			await eventPromises

			// Verify sendText was called with the command
			// eslint-disable-next-line @typescript-eslint/unbound-method
			assert.ok(noShellTerminal.sendText.calledWith("test command", true))

		test("sets hot state for compiling commands", async () => {
			let lines: string[] = []

			terminalProcess.on("completed", (output) => {
				if (output) {
					lines = output.split("\n")

			const completePromise = new Promise<void>((resolve) => {
				terminalProcess.on("shell_execution_complete", () => resolve())

			mockStream = (function* () {
				yield "\x1b]633;C\x07" // The first chunk contains the command start sequence with bell character.
				yield "compiling...\n"
				yield "still compiling...\n"
				yield "done"
				yield "\x1b]633;D\x07" // The last chunk contains the command end sequence with bell character.
				terminalProcess.emit("shell_execution_complete", { exitCode: 0 })
			})()

			mockTerminal.shellIntegration.executeCommand.returns({
				read: sinon.stub().returns(mockStream),

			const runPromise = terminalProcess.run("npm run build")
			terminalProcess.emit("stream_available", mockStream)

			assert.strictEqual(terminalProcess.isHot, true)
			await runPromise

			assert.deepStrictEqual(lines, ["compiling...", "still compiling...", "done"])

			await completePromise
			assert.strictEqual(terminalProcess.isHot, false)

	suite("continue", () => {
		test("stops listening and emits continue event", () => {
			const continueSpy = sinon.stub()
			terminalProcess.on("continue", continueSpy)

			terminalProcess.continue()

			assert.ok(continueSpy.called)
			assert.strictEqual(terminalProcess["isListening"], false)

	suite("getUnretrievedOutput", () => {
		test("returns and clears unretrieved output", () => {
			terminalProcess["fullOutput"] = `\x1b]633;C\x07previous\nnew output\x1b]633;D\x07`
			terminalProcess["lastRetrievedIndex"] = 17 // After "previous\n"

			const unretrieved = terminalProcess.getUnretrievedOutput()
			assert.strictEqual(unretrieved, "new output")

			assert.strictEqual(terminalProcess["lastRetrievedIndex"], terminalProcess["fullOutput"].length - "previous".length)

	suite("interpretExitCode", () => {
		test("handles undefined exit code", () => {
			const result = TerminalProcess.interpretExitCode(undefined)
			assert.deepStrictEqual(result, { exitCode: undefined })

		test("handles normal exit codes (0-128)", () => {
			const result = TerminalProcess.interpretExitCode(0)
			assert.deepStrictEqual(result, { exitCode: 0 })

			const result2 = TerminalProcess.interpretExitCode(1)
			assert.deepStrictEqual(result2, { exitCode: 1 })

			const result3 = TerminalProcess.interpretExitCode(128)
			assert.deepStrictEqual(result3, { exitCode: 128 })

		test("interprets signal exit codes (>128)", () => {
			// SIGTERM (15) -> 128 + 15 = 143
			const result = TerminalProcess.interpretExitCode(143)
			assert.deepStrictEqual(result, {
				exitCode: 143,
				signal: 15,
				signalName: "SIGTERM",
				coreDumpPossible: false,

			// SIGSEGV (11) -> 128 + 11 = 139
			const result2 = TerminalProcess.interpretExitCode(139)
			assert.deepStrictEqual(result2, {
				exitCode: 139,
				signal: 11,
				signalName: "SIGSEGV",
				coreDumpPossible: true,

		test("handles unknown signals", () => {
			const result = TerminalProcess.interpretExitCode(255)
			assert.deepStrictEqual(result, {
				exitCode: 255,
				signal: 127,
				signalName: "Unknown Signal (127)",
				coreDumpPossible: false,

	suite("mergePromise", () => {
		test("merges promise methods with terminal process", async () => {
			const process = new TerminalProcess(mockTerminalInfo)
			const promise = Promise.resolve()

			const merged = mergePromise(process, promise)

			assert.ok("then" in merged, "merged should have 'then' property")
			assert.ok("catch" in merged, "merged should have 'catch' property")
			assert.ok("finally" in merged, "merged should have 'finally' property")
			assert.strictEqual(merged instanceof TerminalProcess, true)

			const result = await merged
			assert.strictEqual(result, undefined)
