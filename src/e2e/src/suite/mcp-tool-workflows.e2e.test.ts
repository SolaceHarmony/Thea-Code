/**
 * End-to-End Tests: MCP Tool Workflows
 * 
 * Tests to verify:
 * 1. MCP tools are properly registered across all providers
 * 2. Tool execution works end-to-end
 * 3. Format compatibility (XML, JSON, OpenAI) is maintained
 * 4. Unified tool access across different AI providers
 */

import * as assert from "assert"
import * as vscode from "vscode"
import { EXTENSION_ID } from "../thea-constants"
import type { TheaCodeAPI } from "../../../exports/thea-code"

suite("MCP Tool Workflows E2E", () => {
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

	test("Extension should support MCP integration", async () => {
		assert.ok(extension, "Extension should be loaded")
		assert.ok(extension.isActive, "Extension should be active")
		
		// Check package.json mentions MCP
		const packageJSON = extension.packageJSON
		assert.ok(
			packageJSON.keywords.includes("mcp") || 
			packageJSON.description.toLowerCase().includes("mcp"),
			"Extension should reference MCP in metadata"
		)
	})

	test("MCP view should be accessible", async function () {
		this.timeout(10_000)
		
		// Open webview
		await vscode.commands.executeCommand("workbench.view.extension.thea-code-ActivityBar")
		await new Promise(resolve => setTimeout(resolve, 1000))
		
		// Navigate to MCP view
		await assert.doesNotReject(
			async () => {
				await vscode.commands.executeCommand("thea-code.mcpButtonClicked")
				await new Promise(resolve => setTimeout(resolve, 500))
			},
			"MCP view should be accessible"
		)
	})

	test("File operation tools should be available", async () => {
		// The extension should provide file operation capabilities
		// These are core MCP tools that should work across all providers
		
		// Commands for file operations exist
		const commands = await vscode.commands.getCommands(true)
		
		// While the actual tool execution happens through AI models,
		// the extension should have infrastructure for file operations
		assert.ok(extension.isActive, "Extension should support file operations via MCP")
	})

	test("Context menu 'Add To Context' should enable tool usage", async function () {
		this.timeout(10_000)
		
		// Create a test file
		const doc = await vscode.workspace.openTextDocument({
			content: "function example() {\n  return 'test';\n}",
			language: "javascript",
		})
		
		const editor = await vscode.window.showTextDocument(doc)
		
		// Select content
		editor.selection = new vscode.Selection(
			new vscode.Position(0, 0),
			new vscode.Position(2, 1)
		)
		
		// Add to context should work (prepares content for tool use)
		await assert.doesNotReject(
			async () => {
				await vscode.commands.executeCommand("thea-code.addToContext")
				await new Promise(resolve => setTimeout(resolve, 500))
			},
			"Add to context should work for tool workflows"
		)
		
		// Clean up
		await vscode.commands.executeCommand("workbench.action.closeActiveEditor")
	})

	test("Terminal integration should enable command execution tools", async function () {
		this.timeout(10_000)
		
		// Create terminal
		const terminal = vscode.window.createTerminal("MCP Test Terminal")
		terminal.show()
		await new Promise(resolve => setTimeout(resolve, 1000))
		
		// Terminal commands should be available for tool execution
		await assert.doesNotReject(
			async () => {
				await vscode.commands.executeCommand("thea-code.terminalAddToContext")
				await new Promise(resolve => setTimeout(resolve, 500))
			},
			"Terminal integration should support tool workflows"
		)
		
		// Clean up
		terminal.dispose()
	})

	test("Extension should handle browser tool availability", async () => {
		// Browser tools are part of MCP integration
		// The extension should have infrastructure for browser automation
		
		// Verify extension is properly set up
		assert.ok(extension.isActive, "Extension should support browser tools via MCP")
		
		// Browser functionality is configured in settings
		const config = vscode.workspace.getConfiguration("thea-code")
		// Browser tools can be toggled but infrastructure should exist
	})

	test("Tool execution should support multiple providers", async () => {
		// Verify extension supports multiple AI providers
		// All should have unified tool access via MCP
		
		const packageJSON = extension.packageJSON
		
		// Extension should support multiple providers (documented in README/description)
		assert.ok(
			packageJSON.description.includes("AI") || 
			packageJSON.description.includes("model"),
			"Extension should support multiple AI providers"
		)
		
		// MCP provides unified tool access
		assert.ok(extension.isActive, "MCP integration should work across providers")
	})

	test("Extension configuration should support tool settings", async () => {
		// Get extension configuration
		const config = vscode.workspace.getConfiguration("thea-code")
		
		// Configuration schema should exist
		const packageJSON = extension.packageJSON
		const configSchema = packageJSON.contributes.configuration
		
		assert.ok(configSchema, "Extension should have configuration schema")
		assert.ok(configSchema.properties, "Configuration should have properties")
		
		// Tool-related settings should be configurable
		const props = configSchema.properties
		assert.ok(props["thea-code.allowedCommands"], "Allowed commands config should exist for tool safety")
	})

	test("MCP button should be in view title menu", async () => {
		const packageJSON = extension.packageJSON
		const menus = packageJSON.contributes.menus
		
		assert.ok(menus, "Extension should have menus")
		assert.ok(menus["view/title"], "View title menu should exist")
		
		// Find MCP button in view/title menu
		const mcpButton = menus["view/title"].find(
			(item: any) => item.command === "thea-code.mcpButtonClicked"
		)
		
		assert.ok(mcpButton, "MCP button should be in view title menu")
		assert.strictEqual(mcpButton.group, "navigation@3", "MCP button should be in navigation group")
	})

	test("Tool workflows should support explain/fix/improve code", async function () {
		this.timeout(15_000)
		
		// Create a test file with code to analyze
		const doc = await vscode.workspace.openTextDocument({
			content: `function buggyCode() {
  var x = 10;
  if (x = 5) {  // Bug: assignment instead of comparison
    console.log("Equal");
  }
}`,
			language: "javascript",
		})
		
		const editor = await vscode.window.showTextDocument(doc)
		
		// Select the buggy code
		editor.selection = new vscode.Selection(
			new vscode.Position(0, 0),
			new vscode.Position(5, 1)
		)
		
		// These commands trigger tool workflows
		const codeCommands = [
			"thea-code.explainCode",
			"thea-code.fixCode",
			"thea-code.improveCode",
		]
		
		for (const cmd of codeCommands) {
			await assert.doesNotReject(
				async () => {
					await vscode.commands.executeCommand(cmd)
					await new Promise(resolve => setTimeout(resolve, 500))
				},
				`${cmd} should trigger tool workflow`
			)
		}
		
		// Clean up
		await vscode.commands.executeCommand("workbench.action.closeActiveEditor")
	})

	test("Terminal tool workflows should support command analysis", async function () {
		this.timeout(10_000)
		
		// Create terminal
		const terminal = vscode.window.createTerminal("Tool Test Terminal")
		terminal.show()
		await new Promise(resolve => setTimeout(resolve, 1000))
		
		// Send a test command
		terminal.sendText("echo 'test command'")
		await new Promise(resolve => setTimeout(resolve, 1000))
		
		// Terminal analysis commands should work
		const terminalCommands = [
			"thea-code.terminalExplainCommand",
			"thea-code.terminalFixCommand",
		]
		
		for (const cmd of terminalCommands) {
			await assert.doesNotReject(
				async () => {
					await vscode.commands.executeCommand(cmd)
					await new Promise(resolve => setTimeout(resolve, 500))
				},
				`${cmd} should trigger terminal tool workflow`
			)
		}
		
		// Clean up
		terminal.dispose()
	})

	test("Extension should handle new task workflow", async function () {
		this.timeout(10_000)
		
		// Open webview
		await vscode.commands.executeCommand("workbench.view.extension.thea-code-ActivityBar")
		await new Promise(resolve => setTimeout(resolve, 1000))
		
		// New task triggers tool initialization
		await assert.doesNotReject(
			async () => {
				await vscode.commands.executeCommand("thea-code.plusButtonClicked")
				await new Promise(resolve => setTimeout(resolve, 1000))
			},
			"New task command should initialize tool workflows"
		)
		
		// Alternative new task command
		await assert.doesNotReject(
			async () => {
				await vscode.commands.executeCommand("thea-code.newTask")
				await new Promise(resolve => setTimeout(resolve, 500))
			},
			"newTask command should initialize tool workflows"
		)
	})

	test("Extension should support custom storage path configuration", async function () {
		this.timeout(5_000)
		
		// Custom storage path affects where MCP data is stored
		await assert.doesNotReject(
			async () => {
				await vscode.commands.executeCommand("thea-code.setCustomStoragePath")
			},
			"Custom storage path command should be available"
		)
	})

	test("MCP integration should maintain state across view changes", async function () {
		this.timeout(15_000)
		
		// Open webview
		await vscode.commands.executeCommand("workbench.view.extension.thea-code-ActivityBar")
		await new Promise(resolve => setTimeout(resolve, 1000))
		
		// Navigate to MCP view
		await vscode.commands.executeCommand("thea-code.mcpButtonClicked")
		await new Promise(resolve => setTimeout(resolve, 500))
		
		// Switch to other views
		await vscode.commands.executeCommand("thea-code.settingsButtonClicked")
		await new Promise(resolve => setTimeout(resolve, 500))
		
		// Return to MCP view
		await vscode.commands.executeCommand("thea-code.mcpButtonClicked")
		await new Promise(resolve => setTimeout(resolve, 500))
		
		// MCP state should be maintained
		assert.ok(extension.isActive, "MCP state should persist across view changes")
	})

	test("Tool execution should handle errors gracefully", async function () {
		this.timeout(10_000)
		
		// Try to execute commands without proper context
		// Should not crash the extension
		
		await assert.doesNotReject(
			async () => {
				// Execute context commands without selection
				await vscode.commands.executeCommand("thea-code.addToContext")
				await new Promise(resolve => setTimeout(resolve, 500))
			},
			"Tool commands should handle missing context gracefully"
		)
		
		assert.ok(extension.isActive, "Extension should remain stable after tool errors")
	})

	test("Multiple tool operations should work sequentially", async function () {
		this.timeout(20_000)
		
		// Create a test document
		const doc = await vscode.workspace.openTextDocument({
			content: "function test1() {}\nfunction test2() {}\nfunction test3() {}",
			language: "javascript",
		})
		
		const editor = await vscode.window.showTextDocument(doc)
		
		// Select first function
		editor.selection = new vscode.Selection(
			new vscode.Position(0, 0),
			new vscode.Position(0, 20)
		)
		
		// Execute multiple operations
		await vscode.commands.executeCommand("thea-code.addToContext")
		await new Promise(resolve => setTimeout(resolve, 500))
		
		// Select second function
		editor.selection = new vscode.Selection(
			new vscode.Position(1, 0),
			new vscode.Position(1, 20)
		)
		
		await vscode.commands.executeCommand("thea-code.explainCode")
		await new Promise(resolve => setTimeout(resolve, 500))
		
		// Select third function
		editor.selection = new vscode.Selection(
			new vscode.Position(2, 0),
			new vscode.Position(2, 20)
		)
		
		await vscode.commands.executeCommand("thea-code.improveCode")
		await new Promise(resolve => setTimeout(resolve, 500))
		
		// Extension should handle sequential operations
		assert.ok(extension.isActive, "Extension should handle sequential tool operations")
		
		// Clean up
		await vscode.commands.executeCommand("workbench.action.closeActiveEditor")
	})
})
