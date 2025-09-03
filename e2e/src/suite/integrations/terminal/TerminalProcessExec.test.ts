
import * as vscode from "vscode"
import { execSync } from "child_process"
import { TerminalProcess, ExitCodeDetails } from "../TerminalProcess"
import { Terminal } from "../Terminal"
import { TerminalRegistry } from "../TerminalRegistry"
// Mock the vscode module
// TODO: Mock setup needs manual migration
import * as assert from 'assert'
import * as sinon from 'sinon'
import * as vscode from 'vscode'
// TODO: Fix mock - needs proxyquire
/*
=> {
	// Store event handlers so we can trigger them in tests
	const eventHandlers = {
		startTerminalShellExecution: null as ((e: unknown) => void) | null,
		endTerminalShellExecution: null as ((e: unknown) => void) | null,

	return {
		workspace: {
			getConfiguration: sinon.stub().returns({
				get: sinon.stub().returns(null),
			}),
		},
		window: {
			createTerminal: sinon.stub(),
			onDidStartTerminalShellExecution: sinon.stub().callsFake((handler) => {
				eventHandlers.startTerminalShellExecution = handler as (e: unknown) => void
				return { dispose: sinon.stub() }
			}),
			onDidEndTerminalShellExecution: sinon.stub().callsFake((handler) => {
				eventHandlers.endTerminalShellExecution = handler as (e: unknown) => void
				return { dispose: sinon.stub() }
			}),
		},
		ThemeIcon: class ThemeIcon {
			constructor(id: string) {
				this.id = id

			id: string
		},
		Uri: {
			file: (path: string) => ({ fsPath: path }),
		},
		// Expose event handlers for testing
		__eventHandlers: eventHandlers,

})*/

// Create a mock stream that uses real command output with realistic chunking
function createRealCommandStream(command: string): { stream: AsyncIterable<string>; exitCode: number } {
	let realOutput: string
	let exitCode: number

	try {
		// Execute the command and get the real output, redirecting stderr to /dev/null
		realOutput = execSync(command + " 2>/dev/null", {
			encoding: "utf8",
			maxBuffer: 100 * 1024 * 1024, // Increase buffer size to 100MB
		exitCode = 0 // Command succeeded
} catch (error) {
			assert.fail('Unexpected error: ' + error.message)
		} catch (err: unknown) {
		// Command failed - get output and exit code from error
		const error = err as { stdout?: Buffer | string; signal?: string; status?: number }
		realOutput = typeof error.stdout === "string" ? error.stdout : error.stdout?.toString() || ""

		// Handle signal termination
		if (error.signal) {
			// Convert signal name to number using Node's constants
			const signals: Record<string, number> = {
				SIGTERM: 15,
				SIGSEGV: 11,
				// Add other signals as needed

			const signalNum = signals[error.signal]
			if (signalNum !== undefined) {
				exitCode = 128 + signalNum // Signal exit codes are 128 + signal number
} else {
				// Log error and default to 1 if signal not recognized
				console.log(`[DEBUG] Unrecognized signal '${error.signal}' from command '${command}'`)
				exitCode = 1
} else {
			exitCode = error.status || 1 // Use status if available, default to 1

	// Create an async iterator that yields the command output with proper markers
	// and realistic chunking (not guaranteed to split on newlines)
	const stream = {
		async *[Symbol.asyncIterator]() {
			await Promise.resolve()
			// First yield the command start marker
			yield "\x1b]633;C\x07"

			// Yield the real output in potentially arbitrary chunks
			// This simulates how terminal data might be received in practice
			if (realOutput.length > 0) {
				// For a simple test like "echo a", we'll just yield the whole output
				// For more complex outputs, we could implement random chunking here
				yield realOutput

			// Last yield the command end marker
			yield "\x1b]633;D\x07"
		},

	return { stream, exitCode }

/**
 * Generalized function to test terminal command execution
 * @param command The command to execute
 * @param expectedOutput The expected output after processing
 * @returns A promise that resolves when the test is complete
 */
async function testTerminalCommand(
	command: string,
	expectedOutput: string,
): Promise<{ executionTimeUs: number; capturedOutput: string; exitDetails: ExitCodeDetails }> {
	let startTime: bigint = BigInt(0)
	let endTime: bigint = BigInt(0)
	let timeRecorded = false
	// Create a mock terminal with shell integration
	const mockTerminal = {
		shellIntegration: {
			executeCommand: sinon.stub(),
			cwd: vscode.Uri.file("/test/path"),
		},
		name: "Thea Code",
		processId: Promise.resolve(123),
		creationOptions: {},
		exitStatus: undefined,
		state: { isInteractedWith: true },
		dispose: sinon.stub(),
		hide: sinon.stub(),
		show: sinon.stub(),
		sendText: sinon.stub(),

	// Create terminal info with running state
	const mockTerminalInfo = new Terminal(1, mockTerminal, "/test/path")
	mockTerminalInfo.running = true

	// Add the terminal to the registry
	TerminalRegistry["terminals"] = [mockTerminalInfo]

	// Create a new terminal process for testing
	startTime = process.hrtime.bigint() // Start timing from terminal process creation
	const terminalProcess = new TerminalProcess(mockTerminalInfo)

	try {
		// Set up the mock stream with real command output and exit code
		const { stream, exitCode } catch (error) {
			assert.fail('Unexpected error: ' + error.message)
		}= createRealCommandStream(command)

		// Configure the mock terminal to return our stream
		mockTerminal.shellIntegration.executeCommand.callsFake(() => {
			return {
				read: sinon.stub().returns(stream),

		// Set up event listeners to capture output
		let capturedOutput = ""
		terminalProcess.on("completed", (output) => {
			if (!timeRecorded) {
				endTime = process.hrtime.bigint() // End timing when completed event is received with output
				timeRecorded = true

			if (output) {
				capturedOutput = output

		// Create a promise that resolves when the command completes
		const completedPromise = new Promise<void>((resolve) => {
			terminalProcess.once("completed", () => {
				resolve()

		// Set the process on the terminal
		mockTerminalInfo.process = terminalProcess

		// Run the command (now handled by constructor)
		// We've already created the process, so we'll trigger the events manually

		// Get the event handlers from the mock
		const eventHandlers = (
			vscode as unknown as {
				__eventHandlers: {
					startTerminalShellExecution: ((e: unknown) => void) | null
					endTerminalShellExecution: ((e: unknown) => void) | null

		).__eventHandlers

		// Execute the command first to set up the process
		await terminalProcess.run(command)

		// Trigger the start terminal shell execution event through VSCode mock
		if (eventHandlers.startTerminalShellExecution) {
			eventHandlers.startTerminalShellExecution({
				terminal: mockTerminal,
				execution: {
					commandLine: { value: command },
					read: () => stream,
				},

		// Wait for some output to be processed
		await new Promise<void>((resolve) => {
			terminalProcess.once("line", () => resolve())

		// Then trigger the end event
		if (eventHandlers.endTerminalShellExecution) {
			eventHandlers.endTerminalShellExecution({
				terminal: mockTerminal,
				exitCode: exitCode,

		// Store exit details for return
		const exitDetails = TerminalProcess.interpretExitCode(exitCode)

		// Set a timeout to avoid hanging tests
		const timeoutPromise = new Promise<void>((_, reject) => {
			setTimeout(() => {
				reject(new Error("Test timed out after 1000ms"))
			}, 1000)

		// Wait for the command to complete or timeout
		await Promise.race([completedPromise, timeoutPromise])
		// Calculate execution time in microseconds
		// If endTime wasn't set (unlikely but possible), set it now
		if (!timeRecorded) {
			endTime = process.hrtime.bigint()

		const executionTimeUs = Number((endTime - startTime) / BigInt(1000))

		// Verify the output matches the expected output
		assert.strictEqual(capturedOutput, expectedOutput)

		return { executionTimeUs, capturedOutput, exitDetails }
	} finally {
		// Clean up
		terminalProcess.removeAllListeners()
		TerminalRegistry["terminals"] = []

suite("TerminalProcess with Real Command Output", () => {
	suiteSetup(() => {
		// Initialize TerminalRegistry event handlers once globally
		TerminalRegistry.initialize()

	setup(() => {
		// Reset the terminals array before each test
		TerminalRegistry["terminals"] = []
		sinon.restore()

	test("should execute 'echo a' and return exactly 'a\\n' with execution time", async () => {
		await testTerminalCommand("echo a", "a\n")

	test("should execute 'echo -n a' and return exactly 'a'", async () => {
		const { executionTimeUs } = await testTerminalCommand("/bin/echo -n a", "a")
		console.log(
			`'echo -n a' execution time: ${executionTimeUs} microseconds (${executionTimeUs / 1000} milliseconds)`,

	test("should execute 'printf \"a\\nb\\n\"' and return 'a\\nb\\n'", async () => {
		const { executionTimeUs } = await testTerminalCommand('printf "a\\nb\\n"', "a\nb\n")
		console.log(
			`'printf "a\\nb\\n"' execution time: ${executionTimeUs} microseconds (${executionTimeUs / 1000} milliseconds)`,

	test("should properly handle terminal shell execution events", async () => {
		// This test is implicitly testing the event handlers since all tests now use them
		const { executionTimeUs } = await testTerminalCommand("echo test", "test\n")
		console.log(
			`'echo test' execution time: ${executionTimeUs} microseconds (${executionTimeUs / 1000} milliseconds)`,

	const TEST_LINES = 1_000_000

	test(`should execute 'yes AAA... | head -n ${TEST_LINES}' and verify ${TEST_LINES} lines of 'A's`, async () => {
		const expectedOutput = Array(TEST_LINES).fill("A".repeat(76)).join("\n") + "\n"

		// This command will generate 1M lines with 76 'A's each.
		const { executionTimeUs, capturedOutput } = await testTerminalCommand(
			`yes "${"A".repeat(76)}" | head -n ${TEST_LINES}`,
			expectedOutput,

		console.log(
			`'yes "${"A".repeat(76)}" | head -n ${TEST_LINES}' execution time: ${executionTimeUs} microseconds (${executionTimeUs / 1000} milliseconds)`,

		// Display a truncated output sample (first 3 lines and last 3 lines)
		const lines = capturedOutput.split("\n")
		const truncatedOutput =
			lines.slice(0, 3).join("\n") +
			`\n... (truncated ${lines.length - 6} lines) ...\n` +
			lines.slice(Math.max(0, lines.length - 3), lines.length).join("\n")

		console.log("Output sample (first 3 lines):\n", truncatedOutput)

		// Verify the output.
		// Check if we have TEST_LINES lines (may have an empty line at the end).
		assert.ok(lines.length >= TEST_LINES)

		// Sample some lines to verify they contain 76 'A' characters.
		// Sample indices at beginning, 1%, 10%, 50%, and end of the output.
		const sampleIndices = [
			0,
			Math.floor(TEST_LINES * 0.01),
			Math.floor(TEST_LINES * 0.1),
			Math.floor(TEST_LINES * 0.5),
			TEST_LINES - 1,
		].filter((i) => i < lines.length)

		for (const index of sampleIndices) {
			assert.strictEqual(lines[index], "A".repeat(76))

	}, 30000) // 30 second timeout for processing 1M lines

	suite("exit code interpretation", () => {
		test("should handle exit 2", async () => {
			const { exitDetails } = await testTerminalCommand("exit 2", "")
			assert.deepStrictEqual(exitDetails, { exitCode: 2 })
		}, 10000) // 10 second timeout

		test("should handle normal exit codes", async () => {
			// Test successful command
			const { exitDetails } = await testTerminalCommand("true", "")
			assert.deepStrictEqual(exitDetails, { exitCode: 0 })

			// Test failed command
			const { exitDetails: exitDetails2 } = await testTerminalCommand("false", "")
			assert.deepStrictEqual(exitDetails2, { exitCode: 1 })
		}, 10000) // 10 second timeout

		test("should interpret SIGTERM exit code", async () => {
			// Run kill in subshell to ensure signal affects the command
			const { exitDetails } = await testTerminalCommand("bash -c 'kill $$'", "")
			assert.deepStrictEqual(exitDetails, {
				exitCode: 143, // 128 + 15 (SIGTERM)
				signal: 15,
				signalName: "SIGTERM",
				coreDumpPossible: false,
		}, 10000) // 10 second timeout

		test("should interpret SIGSEGV exit code", async () => {
			// Run kill in subshell to ensure signal affects the command
			const { exitDetails } = await testTerminalCommand("bash -c 'kill -SIGSEGV $$'", "")
			assert.deepStrictEqual(exitDetails, {
				exitCode: 139, // 128 + 11 (SIGSEGV)
				signal: 11,
				signalName: "SIGSEGV",
				coreDumpPossible: true,
		}, 10000) // 10 second timeout

		test("should handle command not found", async () => {
			// Test a non-existent command
			const { exitDetails } = await testTerminalCommand("nonexistentcommand", "")
			assert.strictEqual(exitDetails?.exitCode, 127) // Command not found
		}, 10000) // 10 second timeout
