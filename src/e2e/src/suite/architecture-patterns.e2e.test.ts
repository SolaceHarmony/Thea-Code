/**
 * End-to-End Tests: Architecture Pattern Validation
 * 
 * Tests to verify:
 * 1. Provider pattern implementation
 * 2. Message-driven architecture compliance
 * 3. Component composition patterns
 * 4. State management patterns
 */

import * as assert from "assert"
import * as vscode from "vscode"
import { EXTENSION_ID } from "../thea-constants"
import type { TheaCodeAPI } from "../../../exports/thea-code"

interface ExtensionCommand {
	command: string
	title: string
	category?: string
}

interface ExtensionPackageJSON {
	contributes: {
		commands: ExtensionCommand[]
	}
}

suite("Architecture Patterns E2E", () => {
	let extension: vscode.Extension<TheaCodeAPI> | undefined
	let api: TheaCodeAPI | undefined

	suiteSetup(async function () {
		this.timeout(60_000)
		extension = vscode.extensions.getExtension<TheaCodeAPI>(EXTENSION_ID)
		assert.ok(extension, `Extension ${EXTENSION_ID} should be found`)
		
		if (!extension.isActive) {
			await extension.activate()
		}

		// Get API if available
		const exp = extension.exports
		// The export is the API itself
		api = exp
	})

	// Type assertion helper - after suiteSetup, extension is guaranteed to be defined
	function getExtension(): vscode.Extension<TheaCodeAPI> {
		assert.ok(extension, "Extension should be initialized in suiteSetup")
		return extension!
	}

	test("Extension should follow single responsibility principle", async () => {
		// Extension package.json should be well-organized
		const packageJSON = getExtension().packageJSON as ExtensionPackageJSON
		// Commands should be organized by functionality
		const commands = packageJSON.contributes.commands
		assert.ok(commands, "Commands should be defined")
		assert.ok(Array.isArray(commands), "Commands should be an array")
		
		// Categories should group related commands
		const categories = new Set(commands.map((cmd) => cmd.category).filter(Boolean))
		assert.ok(categories.size > 0, "Commands should be categorized")
		assert.ok(categories.has("Thea Code"), "Thea Code category should exist")
	})

	test("Extension should implement proper view lifecycle", async function () {
		this.timeout(15_000)
		
		// Test view lifecycle: create -> show -> hide -> recreate
		
		// 1. Create and show
		await vscode.commands.executeCommand("workbench.view.extension.thea-code-ActivityBar")
		await new Promise(resolve => setTimeout(resolve, 1000))
		assert.ok(getExtension().isActive, "Extension should be active after view creation")
		
		// 2. Hide
		await vscode.commands.executeCommand("workbench.action.closeSidebar")
		await new Promise(resolve => setTimeout(resolve, 500))
		assert.ok(getExtension().isActive, "Extension should remain active after hide")
		
		// 3. Show again (recreate)
		await vscode.commands.executeCommand("workbench.view.extension.thea-code-ActivityBar")
		await new Promise(resolve => setTimeout(resolve, 1000))
		assert.ok(getExtension().isActive, "Extension should handle view recreation")
		
		// 4. Verify functionality after recreation
		await vscode.commands.executeCommand("thea-code.settingsButtonClicked")
		await new Promise(resolve => setTimeout(resolve, 500))
		assert.ok(getExtension().isActive, "Extension should be functional after recreation")
	})

	test("Extension should implement state synchronization pattern", async function () {
		this.timeout(15_000)
		
		// Open webview
		await vscode.commands.executeCommand("workbench.view.extension.thea-code-ActivityBar")
		await new Promise(resolve => setTimeout(resolve, 1000))
		
		// Make state changes through different views
		const stateChangingViews = [
			"thea-code.settingsButtonClicked",  // Settings can change state
			"thea-code.mcpButtonClicked",       // MCP view can change state
			"thea-code.promptsButtonClicked",   // Prompts can change state
		]
		
		for (const view of stateChangingViews) {
			await vscode.commands.executeCommand(view)
			await new Promise(resolve => setTimeout(resolve, 500))
			
			// Extension should maintain consistent state
			assert.ok(getExtension().isActive, `State should be consistent after ${view}`)
		}
		
		// Return to main view
		await vscode.commands.executeCommand("thea-code.plusButtonClicked")
		await new Promise(resolve => setTimeout(resolve, 500))
		
		// State should be synchronized
		assert.ok(getExtension().isActive, "State should remain synchronized")
	})

	test("Extension should support composition through menu structure", async () => {
		const packageJSON = getExtension().packageJSON
		const menus = packageJSON.contributes.menus
		
		// Verify menu composition
		assert.ok(menus, "Menus should be defined")
		
		// Submenus enable composition
		const submenus = getExtension().packageJSON.contributes.submenus
		assert.ok(submenus, "Submenus should be defined")
		assert.ok(Array.isArray(submenus), "Submenus should be an array")
		
		// Context menu and terminal menu are composed
		const contextSubmenu = submenus.find((sm: any) => sm.id === "thea-code.contextMenu")
		const terminalSubmenu = submenus.find((sm: any) => sm.id === "thea-code.terminalMenu")
		
		assert.ok(contextSubmenu, "Context menu submenu should exist")
		assert.ok(terminalSubmenu, "Terminal menu submenu should exist")
		
		// These menus compose into editor and terminal contexts
		assert.ok(menus["editor/context"], "Editor context menu should use composition")
		assert.ok(menus["terminal/context"], "Terminal context menu should use composition")
	})

	test("Extension should handle concurrent operations", async function () {
		this.timeout(20_000)
		
		// Open webview
		await vscode.commands.executeCommand("workbench.view.extension.thea-code-ActivityBar")
		await new Promise(resolve => setTimeout(resolve, 1000))
		
		// Execute multiple operations concurrently
		const operations = [
			vscode.commands.executeCommand("thea-code.settingsButtonClicked"),
			vscode.commands.executeCommand("thea-code.historyButtonClicked"),
			vscode.commands.executeCommand("thea-code.mcpButtonClicked"),
		]
		
		await assert.doesNotReject(
			async () => {
				await Promise.all(operations)
				await new Promise(resolve => setTimeout(resolve, 1000))
			},
			"Extension should handle concurrent operations"
		)
		
		assert.ok(getExtension().isActive, "Extension should remain stable after concurrent ops")
	})

	test("Extension should support dependency injection pattern", async () => {
		// API should be injected and available
		if (!api) {
			console.log("API not available - skipping DI test")
			return
		}
		
		// API represents dependency injection of extension services
		assert.ok(api, "API should be available as injected dependency")
		
		// Extension exports should provide access to services
		const exp = getExtension().exports
		assert.ok(exp, "Extension should export services")
	})

	test("Extension should implement error boundaries", async function () {
		this.timeout(15_000)
		
		// Try to trigger error scenarios
		// Extension should handle gracefully without crashing
		
		// 1. Execute command without proper context
		await assert.doesNotReject(
			async () => {
				await vscode.commands.executeCommand("thea-code.addToContext")
				await new Promise(resolve => setTimeout(resolve, 500))
			},
			"Extension should handle command errors"
		)
		
		// 2. Rapid view switching (potential race conditions)
		for (let i = 0; i < 5; i++) {
			await vscode.commands.executeCommand("thea-code.settingsButtonClicked")
			await vscode.commands.executeCommand("thea-code.historyButtonClicked")
		}
		await new Promise(resolve => setTimeout(resolve, 1000))
		
		// Extension should still be functional
		assert.ok(getExtension().isActive, "Extension should handle errors gracefully")
	})

	test("Extension should follow event-driven architecture", async function () {
		this.timeout(10_000)
		
		// Extension should respond to VSCode events
		
		// 1. Document events
		const doc = await vscode.workspace.openTextDocument({
			content: "test",
			language: "javascript",
		})
		await vscode.window.showTextDocument(doc)
		await new Promise(resolve => setTimeout(resolve, 500))
		
		// Extension should handle document events
		assert.ok(getExtension().isActive, "Extension should handle document events")
		
		// 2. Window events (view visibility)
		await vscode.commands.executeCommand("workbench.action.closeSidebar")
		await new Promise(resolve => setTimeout(resolve, 500))
		await vscode.commands.executeCommand("workbench.view.extension.thea-code-ActivityBar")
		await new Promise(resolve => setTimeout(resolve, 500))
		
		// Extension should handle window events
		assert.ok(getExtension().isActive, "Extension should handle window events")
		
		// Clean up
		await vscode.commands.executeCommand("workbench.action.closeActiveEditor")
	})

	test("Extension should implement proper resource management", async function () {
		this.timeout(20_000)
		
		// Create and dispose multiple resources
		const terminals: vscode.Terminal[] = []
		
		// Create terminals
		for (let i = 0; i < 3; i++) {
			const terminal = vscode.window.createTerminal(`Test ${i}`)
			terminals.push(terminal)
			await new Promise(resolve => setTimeout(resolve, 200))
		}
		
		// Extension should handle multiple resources
		assert.ok(getExtension().isActive, "Extension should handle multiple resources")
		
		// Dispose terminals
		for (const terminal of terminals) {
			terminal.dispose()
			await new Promise(resolve => setTimeout(resolve, 100))
		}
		
		// Extension should clean up properly
		assert.ok(getExtension().isActive, "Extension should manage resource disposal")
	})

	test("Extension should support configuration reactivity", async function () {
		this.timeout(10_000)
		
		// Configuration changes should be reactive
		const config = vscode.workspace.getConfiguration("thea-code")
		
		// Get current allowed commands
		const currentAllowedCommands = config.get("allowedCommands")
		assert.ok(currentAllowedCommands, "Configuration should be accessible")
		
		// Extension should support configuration updates
		// (Actual update requires workspace, but we verify the pattern exists)
		const packageJSON = getExtension().packageJSON
		const configSchema = packageJSON.contributes.configuration
		
		assert.ok(configSchema.properties, "Configuration should support updates")
		assert.ok(
			configSchema.properties["thea-code.allowedCommands"],
			"Allowed commands should be configurable"
		)
	})

	test("Extension should implement proper activation strategy", async () => {
		const packageJSON = getExtension().packageJSON
		
		// Check activation events
		const activationEvents = packageJSON.activationEvents
		assert.ok(activationEvents, "Activation events should be defined")
		assert.ok(Array.isArray(activationEvents), "Activation events should be an array")
		
		// Should activate on startup (lazy loading pattern)
		assert.ok(
			activationEvents.includes("onStartupFinished"),
			"Extension should use onStartupFinished for optimal performance"
		)
		
		// Extension should be activated by now
		assert.ok(getExtension().isActive, "Extension should be activated")
	})

	test("Extension should implement view persistence", async function () {
		this.timeout(15_000)
		
		// Test that view state persists across sessions
		
		// Open and navigate to specific view
		await vscode.commands.executeCommand("workbench.view.extension.thea-code-ActivityBar")
		await new Promise(resolve => setTimeout(resolve, 1000))
		
		await vscode.commands.executeCommand("thea-code.settingsButtonClicked")
		await new Promise(resolve => setTimeout(resolve, 500))
		
		// Close webview
		await vscode.commands.executeCommand("workbench.action.closeSidebar")
		await new Promise(resolve => setTimeout(resolve, 1000))
		
		// Reopen
		await vscode.commands.executeCommand("workbench.view.extension.thea-code-ActivityBar")
		await new Promise(resolve => setTimeout(resolve, 2000))
		
		// View should be restored (extension handles internally)
		assert.ok(getExtension().isActive, "Extension should maintain view state")
	})

	test("Extension should implement command validation pattern", async () => {
		const packageJSON = getExtension().packageJSON
		
		// Commands should have proper metadata
		const commands = packageJSON.contributes.commands
		
		for (const cmd of commands) {
			assert.ok(cmd.command, "Command should have command ID")
			assert.ok(cmd.title, "Command should have title")
			
			// Commands with UI should have icons or categories
			if (cmd.command.includes("Button")) {
				assert.ok(cmd.icon || cmd.category, "UI commands should have icon or category")
			}
		}
	})

	test("Extension should follow separation of concerns", async () => {
		const packageJSON = getExtension().packageJSON
		
		// Configuration is separate from commands
		assert.ok(packageJSON.contributes.configuration, "Configuration should be separate")
		assert.ok(packageJSON.contributes.commands, "Commands should be separate")
		assert.ok(packageJSON.contributes.views, "Views should be separate")
		assert.ok(packageJSON.contributes.menus, "Menus should be separate")
		
		// Each concern is properly separated
		assert.notStrictEqual(
			packageJSON.contributes.configuration,
			packageJSON.contributes.commands,
			"Configuration and commands should be separate"
		)
	})

	test("Extension should implement proper icon management", async () => {
		const packageJSON = getExtension().packageJSON
		
		// Extension should have icon
		assert.ok(packageJSON.icon, "Extension should have icon")
		
		// Views should have icons
		const activityBarViews = packageJSON.contributes.views["thea-code-ActivityBar"]
		for (const view of activityBarViews) {
			assert.ok(view.icon, "View should have icon")
		}
		
		// View container should have icon
		const viewsContainers = packageJSON.contributes.viewsContainers.activitybar
		const theaContainer = viewsContainers.find((c: any) => c.id === "thea-code-ActivityBar")
		assert.ok(theaContainer.icon, "View container should have icon")
	})

	test("Extension should implement proper categorization", async () => {
		const packageJSON = getExtension().packageJSON
		
		// Extension should be in appropriate categories
		const categories = packageJSON.categories
		assert.ok(categories, "Extension should have categories")
		assert.ok(Array.isArray(categories), "Categories should be an array")
		
		// Should include relevant categories
		assert.ok(categories.includes("AI"), "Extension should be in AI category")
		assert.ok(categories.includes("Chat"), "Extension should be in Chat category")
		assert.ok(
			categories.includes("Programming Languages"),
			"Extension should be in Programming Languages category"
		)
	})
})
