import { TerminalProcess } from "../TerminalProcess"
import { execSync } from "child_process"

import * as assert from 'assert'
suite("TerminalProcess.interpretExitCode", () => {
	test("should handle undefined exit code", () => {
		const result = TerminalProcess.interpretExitCode(undefined)
		assert.deepStrictEqual(result, { exitCode: undefined })

	test("should handle normal exit codes (0-127)", () => {
		// Test success exit code (0)
		let result = TerminalProcess.interpretExitCode(0)
		assert.deepStrictEqual(result, { exitCode: 0 })

		// Test error exit code (1)
		result = TerminalProcess.interpretExitCode(1)
		assert.deepStrictEqual(result, { exitCode: 1 })

		// Test arbitrary exit code within normal range
		result = TerminalProcess.interpretExitCode(42)
		assert.deepStrictEqual(result, { exitCode: 42 })

		// Test boundary exit code
		result = TerminalProcess.interpretExitCode(127)
		assert.deepStrictEqual(result, { exitCode: 127 })

	test("should handle signal exit codes (128+)", () => {
		// Test SIGINT (Ctrl+C) - 128 + 2 = 130
		const result = TerminalProcess.interpretExitCode(130)
		assert.deepStrictEqual(result, {
			exitCode: 130,
			signal: 2,
			signalName: "SIGINT",
			coreDumpPossible: false,

		// Test SIGTERM - 128 + 15 = 143
		const resultTerm = TerminalProcess.interpretExitCode(143)
		assert.deepStrictEqual(resultTerm, {
			exitCode: 143,
			signal: 15,
			signalName: "SIGTERM",
			coreDumpPossible: false,

		// Test SIGSEGV (segmentation fault) - 128 + 11 = 139
		const resultSegv = TerminalProcess.interpretExitCode(139)
		assert.deepStrictEqual(resultSegv, {
			exitCode: 139,
			signal: 11,
			signalName: "SIGSEGV",
			coreDumpPossible: true,

	test("should identify signals that can produce core dumps", () => {
		// Core dump possible signals: SIGQUIT(3), SIGILL(4), SIGABRT(6), SIGBUS(7), SIGFPE(8), SIGSEGV(11)
		const coreDumpSignals = [3, 4, 6, 7, 8, 11]

		for (const signal of coreDumpSignals) {
			const exitCode = 128 + signal
			const result = TerminalProcess.interpretExitCode(exitCode)
			assert.strictEqual(result.coreDumpPossible, true)

		// Test a non-core-dump signal
		const nonCoreDumpResult = TerminalProcess.interpretExitCode(128 + 1) // SIGHUP
		assert.strictEqual(nonCoreDumpResult.coreDumpPossible, false)

	test("should handle unknown signals", () => {
		// Test an exit code for a signal that's not in our mapping
		const result = TerminalProcess.interpretExitCode(128 + 99)
		assert.deepStrictEqual(result, {
			exitCode: 128 + 99,
			signal: 99,
			signalName: "Unknown Signal (99)",
			coreDumpPossible: false,

suite("TerminalProcess.interpretExitCode with real commands", () => {
	test("should correctly interpret exit code 0 from successful command", () => {
		try {
			// Run a command that should succeed
			execSync("echo test", { stdio: "ignore" } catch (error) {
			assert.fail('Unexpected error: ' + error.message)
		})
			// If we get here, the command succeeded with exit code 0
			const result = TerminalProcess.interpretExitCode(0)
			assert.deepStrictEqual(result, { exitCode: 0 })
} catch (error: unknown) {
			// This should not happen for a successful command
			const err = error as { message?: string }
			assert.fail("Command should have succeeded: " + (err.message ?? ""))

	test("should correctly interpret exit code 1 from failed command", () => {
		try {
			// Run a command that should fail with exit code 1 or 2
			execSync("ls /nonexistent_directory", { stdio: "ignore" } catch (error) {
			assert.fail('Unexpected error: ' + error.message)
		})
			assert.fail("Command should have failed")
} catch (error: unknown) {
			// Verify the exit code is what we expect (can be 1 or 2 depending on the system)
			const err = error as { status?: number }
			assert.ok(err.status > 0)
			assert.ok(err.status < 128) // Not a signal
			const result = TerminalProcess.interpretExitCode(err.status)
			assert.deepStrictEqual(result, { exitCode: err.status })

	test("should correctly interpret exit code from command with custom exit code", () => {
		try {
			// Run a command that exits with a specific code
			execSync("exit 42", { stdio: "ignore" } catch (error) {
			assert.fail('Unexpected error: ' + error.message)
		})
			assert.fail("Command should have exited with code 42")
} catch (error: unknown) {
			const err = error as { status?: number }
			assert.strictEqual(err.status, 42)
			const result = TerminalProcess.interpretExitCode(err.status)
			assert.deepStrictEqual(result, { exitCode: 42 })

	// Test signal interpretation directly without relying on actual process termination
	test("should correctly interpret signal termination codes", () => {
		// Test SIGTERM (signal 15)
		const sigtermExitCode = 128 + 15
		const sigtermResult = TerminalProcess.interpretExitCode(sigtermExitCode)
		assert.strictEqual(sigtermResult.signal, 15)
		assert.strictEqual(sigtermResult.signalName, "SIGTERM")
		assert.strictEqual(sigtermResult.coreDumpPossible, false)

		// Test SIGSEGV (signal 11)
		const sigsegvExitCode = 128 + 11
		const sigsegvResult = TerminalProcess.interpretExitCode(sigsegvExitCode)
		assert.strictEqual(sigsegvResult.signal, 11)
		assert.strictEqual(sigsegvResult.signalName, "SIGSEGV")
		assert.strictEqual(sigsegvResult.coreDumpPossible, true)

		// Test SIGINT (signal 2)
		const sigintExitCode = 128 + 2
		const sigintResult = TerminalProcess.interpretExitCode(sigintExitCode)
		assert.strictEqual(sigintResult.signal, 2)
		assert.strictEqual(sigintResult.signalName, "SIGINT")
		assert.strictEqual(sigintResult.coreDumpPossible, false)
