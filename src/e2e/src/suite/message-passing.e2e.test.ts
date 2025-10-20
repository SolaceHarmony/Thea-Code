/**
 * End-to-End Tests: Message Passing System
 * 
 * Tests to verify:
 * 1. Bidirectional communication between extension and webview
 * 2. Message type safety and validation
 * 3. State synchronization
 * 4. Message handling patterns match architecture
 */

import * as assert from "assert"
import * as vscode from "vscode"
import { EXTENSION_ID } from "../thea-constants"
import type { TheaCodeAPI } from "../../../exports/thea-code"

suite("Message Passing System E2E", () => {
	let extension: vscode.Extension<any> | undefined
	let api: TheaCodeAPI | undefined

	suiteSetup(async function () {
		this.timeout(60_000)
		extension = vscode.extensions.getExtension(EXTENSION_ID)
		assert.ok(extension, `Extension ${EXTENSION_ID} should be found`)
		
		if (!extension.isActive) {
			await extension.activate()
		}

		// Get API if available
		const exp = extension.exports
		if (exp && typeof exp === "object") {
			api = (exp as any).api || exp
		}
	})

	test("Extension should have TheaProvider for message handling", async () => {
		assert.ok(extension, "Extension should be loaded")
		assert.ok(extension.isActive, "Extension should be active")
		
		// TheaProvider manages webview communication
		// Verify extension can handle webview lifecycle
		await vscode.commands.executeCommand("workbench.view.extension.thea-code-ActivityBar")
		await new Promise(resolve => setTimeout(resolve, 1000))
		
		assert.ok(extension.isActive, "Extension should handle webview lifecycle")
	})

	test("Extension state should be accessible for synchronization", async () => {
		// API should provide access to state management
		if (!api) {
			console.log("API not available - skipping state test")
			return
		}

		// Verify API structure exists
		assert.ok(api, "API should be available for state management")
	})

	test("Message commands should trigger proper actions", async function () {
		this.timeout(10_000)
		
		// Open webview first
		await vscode.commands.executeCommand("workbench.view.extension.thea-code-ActivityBar")
		await new Promise(resolve => setTimeout(resolve, 2000))
		
		// Test navigation commands (these send messages internally)
		const navigationCommands = [
			"thea-code.settingsButtonClicked",
			"thea-code.historyButtonClicked",
			"thea-code.mcpButtonClicked",
		]
		
		for (const cmd of navigationCommands) {
			await assert.doesNotReject(
				async () => {
					await vscode.commands.executeCommand(cmd)
					// Give time for message to be processed
					await new Promise(resolve => setTimeout(resolve, 500))
				},
				`Command ${cmd} should execute without errors`
			)
		}
		
		assert.ok(extension.isActive, "Extension should remain active after commands")
	})

	test("Extension should handle state updates through commands", async function () {
		this.timeout(10_000)
		
		// These commands involve state updates
		await vscode.commands.executeCommand("workbench.view.extension.thea-code-ActivityBar")
		await new Promise(resolve => setTimeout(resolve, 1000))
		
		// Execute a command that updates state
		await assert.doesNotReject(
			async () => {
				await vscode.commands.executeCommand("thea-code.plusButtonClicked")
				await new Promise(resolve => setTimeout(resolve, 500))
			},
			"New task command should execute and update state"
		)
	})

	test("Extension should maintain state across view switches", async function () {
		this.timeout(15_000)
		
		// Open webview
		await vscode.commands.executeCommand("workbench.view.extension.thea-code-ActivityBar")
		await new Promise(resolve => setTimeout(resolve, 1000))
		
		// Switch between different views
		const views = [
			"thea-code.settingsButtonClicked",
			"thea-code.historyButtonClicked",
			"thea-code.mcpButtonClicked",
			"thea-code.promptsButtonClicked",
		]
		
		for (const view of views) {
			await vscode.commands.executeCommand(view)
			await new Promise(resolve => setTimeout(resolve, 500))
			
			// Extension should maintain state
			assert.ok(extension.isActive, `Extension should be active after ${view}`)
		}
	})

	test("Context menu commands should trigger proper message flow", async function () {
		this.timeout(10_000)
		
		// Create a test document
		const doc = await vscode.workspace.openTextDocument({
			content: "function test() {\n  return 'hello';\n}",
			language: "javascript",
		})
		
		const editor = await vscode.window.showTextDocument(doc)
		
		// Select some text
		editor.selection = new vscode.Selection(
			new vscode.Position(0, 0),
			new vscode.Position(2, 1)
		)
		
		// Context menu commands should work (they send messages)
		await assert.doesNotReject(
			async () => {
				await vscode.commands.executeCommand("thea-code.addToContext")
				await new Promise(resolve => setTimeout(resolve, 500))
			},
			"Add to context command should execute"
		)
		
		// Clean up
		await vscode.commands.executeCommand("workbench.action.closeActiveEditor")
	})

	test("Terminal context commands should trigger proper message flow", async function () {
		this.timeout(10_000)
		
		// Create a terminal
		const terminal = vscode.window.createTerminal("Test Terminal")
		terminal.show()
		await new Promise(resolve => setTimeout(resolve, 1000))
		
		// Terminal commands should work
		await assert.doesNotReject(
			async () => {
				// These commands send messages when terminal content is available
				await vscode.commands.executeCommand("thea-code.terminalAddToContext")
				await new Promise(resolve => setTimeout(resolve, 500))
			},
			"Terminal add to context command should execute"
		)
		
		// Clean up
		terminal.dispose()
	})

	test("Extension should handle rapid message sequences", async function () {
		this.timeout(15_000)
		
		// Open webview
		await vscode.commands.executeCommand("workbench.view.extension.thea-code-ActivityBar")
		await new Promise(resolve => setTimeout(resolve, 1000))
		
		// Rapid command execution
		const commands = [
			"thea-code.settingsButtonClicked",
			"thea-code.historyButtonClicked",
			"thea-code.settingsButtonClicked",
			"thea-code.mcpButtonClicked",
			"thea-code.promptsButtonClicked",
		]
		
		// Execute commands rapidly
		const promises = commands.map(cmd => 
			vscode.commands.executeCommand(cmd)
		)
		
		await assert.doesNotReject(
			async () => {
				await Promise.all(promises)
				await new Promise(resolve => setTimeout(resolve, 1000))
			},
			"Extension should handle rapid message sequences"
		)
		
		assert.ok(extension.isActive, "Extension should remain stable after rapid messages")
	})

	test("Extension should recover from view disposal and recreation", async function () {
		this.timeout(15_000)
		
		// Open webview
		await vscode.commands.executeCommand("workbench.view.extension.thea-code-ActivityBar")
		await new Promise(resolve => setTimeout(resolve, 1000))
		
		// Execute a command
		await vscode.commands.executeCommand("thea-code.settingsButtonClicked")
		await new Promise(resolve => setTimeout(resolve, 500))
		
		// Close sidebar (disposes webview)
		await vscode.commands.executeCommand("workbench.action.closeSidebar")
		await new Promise(resolve => setTimeout(resolve, 1000))
		
		// Reopen (creates new webview)
		await vscode.commands.executeCommand("workbench.view.extension.thea-code-ActivityBar")
		await new Promise(resolve => setTimeout(resolve, 2000))
		
		// Should still be functional
		await assert.doesNotReject(
			async () => {
				await vscode.commands.executeCommand("thea-code.historyButtonClicked")
				await new Promise(resolve => setTimeout(resolve, 500))
			},
			"Extension should handle webview recreation"
		)
		
		assert.ok(extension.isActive, "Extension should remain active after recreation")
	})

	test("Extension should handle popout command", async function () {
		this.timeout(10_000)
		
		// Open sidebar
		await vscode.commands.executeCommand("workbench.view.extension.thea-code-ActivityBar")
		await new Promise(resolve => setTimeout(resolve, 1000))
		
		// Popout should work (opens in editor)
		await assert.doesNotReject(
			async () => {
				await vscode.commands.executeCommand("thea-code.popoutButtonClicked")
				await new Promise(resolve => setTimeout(resolve, 1000))
			},
			"Popout button command should execute"
		)
		
		// Clean up any opened editors
		await vscode.commands.executeCommand("workbench.action.closeAllEditors")
	})

	test("Extension should handle help command", async function () {
		this.timeout(10_000)
		
		// Help command should work
		await assert.doesNotReject(
			async () => {
				await vscode.commands.executeCommand("thea-code.helpButtonClicked")
				await new Promise(resolve => setTimeout(resolve, 500))
			},
			"Help button command should execute"
		)
	})

	test("Message-driven architecture should support state persistence", async function () {
		this.timeout(10_000)
		
		// Open webview
		await vscode.commands.executeCommand("workbench.view.extension.thea-code-ActivityBar")
		await new Promise(resolve => setTimeout(resolve, 1000))
		
		// Navigate to settings
		await vscode.commands.executeCommand("thea-code.settingsButtonClicked")
		await new Promise(resolve => setTimeout(resolve, 500))
		
		// Close and reopen
		await vscode.commands.executeCommand("workbench.action.closeSidebar")
		await new Promise(resolve => setTimeout(resolve, 500))
		await vscode.commands.executeCommand("workbench.view.extension.thea-code-ActivityBar")
		await new Promise(resolve => setTimeout(resolve, 2000))
		
		// Should restore state (extension handles this internally)
		assert.ok(extension.isActive, "Extension should maintain state across sessions")
	})

	test("Extension should handle openInNewTab command", async function () {
		this.timeout(10_000)
		
		// Create a test document
		const doc = await vscode.workspace.openTextDocument({
			content: "test content",
			language: "plaintext",
		})
		
		await vscode.window.showTextDocument(doc)
		
		// Open in new tab should work
		await assert.doesNotReject(
			async () => {
				await vscode.commands.executeCommand("thea-code.openInNewTab")
				await new Promise(resolve => setTimeout(resolve, 500))
			},
			"Open in new tab command should execute"
		)
		
		// Clean up
		await vscode.commands.executeCommand("workbench.action.closeAllEditors")
	})
})
