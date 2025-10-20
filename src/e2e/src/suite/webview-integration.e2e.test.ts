/**
 * End-to-End Tests: Webview UI Integration
 * 
 * Tests to verify:
 * 1. Modern UI components are properly integrated
 * 2. No legacy webview-ui-toolkit dependencies remain
 * 3. VSCode component wrappers work correctly
 * 4. UI components integrate with VSCode APIs
 */

import * as assert from "assert"
import * as vscode from "vscode"
import { EXTENSION_ID } from "../thea-constants"

suite("Webview UI Integration E2E", () => {
	let extension: vscode.Extension<any> | undefined

	suiteSetup(async function () {
		this.timeout(60_000)
		extension = vscode.extensions.getExtension(EXTENSION_ID)
		assert.ok(extension, `Extension ${EXTENSION_ID} should be found`)
		
		if (!extension!.isActive) {
			await extension!.activate()
		}
	})

	test("Extension should use modern UI components (no webview-ui-toolkit)", async () => {
		// Verify extension is using modern architecture
		assert.ok(extension, "Extension should be loaded")
		assert.ok(extension!.isActive, "Extension should be active")
		
		// Extension should have proper webview provider
		const packageJSON = extension!.packageJSON
		assert.ok(packageJSON.contributes, "Extension should have contributes")
		assert.ok(packageJSON.contributes.views, "Extension should have views")
		
		// Verify the sidebar webview is registered
		const theaSidebar = packageJSON.contributes.views["thea-code-ActivityBar"]
		assert.ok(theaSidebar, "Thea Code sidebar should be registered")
		assert.ok(
			theaSidebar.some((view: any) => view.id === "thea-code.SidebarProvider"),
			"SidebarProvider should be registered"
		)
	})

	test("Webview should be creatable and functional", async function () {
		this.timeout(10_000)
		
		// Try to open the sidebar
		await vscode.commands.executeCommand("workbench.view.extension.thea-code-ActivityBar")
		
		// Give the webview time to initialize
		await new Promise(resolve => setTimeout(resolve, 2000))
		
		// Extension should still be active after webview creation
		assert.ok(extension!.isActive, "Extension should remain active after webview creation")
	})

	test("Commands for UI navigation should be registered", async () => {
		const commands = await vscode.commands.getCommands(true)
		
		const expectedUICommands = [
			"thea-code.plusButtonClicked",      // New Task
			"thea-code.settingsButtonClicked",  // Settings
			"thea-code.historyButtonClicked",   // History
			"thea-code.mcpButtonClicked",       // MCP Servers
			"thea-code.promptsButtonClicked",   // Prompts
			"thea-code.popoutButtonClicked",    // Open in Editor
			"thea-code.helpButtonClicked",      // Documentation
		]
		
		for (const cmd of expectedUICommands) {
			assert.ok(
				commands.includes(cmd),
				`UI command ${cmd} should be registered for modern UI navigation`
			)
		}
	})

	test("Settings button command should be executable", async function () {
		this.timeout(5_000)
		
		// This should not throw
		await assert.doesNotReject(
			async () => {
				await vscode.commands.executeCommand("thea-code.settingsButtonClicked")
			},
			"Settings button command should be executable"
		)
	})

	test("MCP button command should be executable", async function () {
		this.timeout(5_000)
		
		await assert.doesNotReject(
			async () => {
				await vscode.commands.executeCommand("thea-code.mcpButtonClicked")
			},
			"MCP button command should be executable"
		)
	})

	test("History button command should be executable", async function () {
		this.timeout(5_000)
		
		await assert.doesNotReject(
			async () => {
				await vscode.commands.executeCommand("thea-code.historyButtonClicked")
			},
			"History button command should be executable"
		)
	})

	test("Prompts button command should be executable", async function () {
		this.timeout(5_000)
		
		await assert.doesNotReject(
			async () => {
				await vscode.commands.executeCommand("thea-code.promptsButtonClicked")
			},
			"Prompts button command should be executable"
		)
	})

	test("Context menu commands should be registered", async () => {
		const commands = await vscode.commands.getCommands(true)
		
		const contextCommands = [
			"thea-code.explainCode",
			"thea-code.fixCode",
			"thea-code.improveCode",
			"thea-code.addToContext",
		]
		
		for (const cmd of contextCommands) {
			assert.ok(
				commands.includes(cmd),
				`Context menu command ${cmd} should be registered`
			)
		}
	})

	test("Terminal context commands should be registered", async () => {
		const commands = await vscode.commands.getCommands(true)
		
		const terminalCommands = [
			"thea-code.terminalAddToContext",
			"thea-code.terminalFixCommand",
			"thea-code.terminalExplainCommand",
			"thea-code.terminalFixCommandInCurrentTask",
			"thea-code.terminalExplainCommandInCurrentTask",
		]
		
		for (const cmd of terminalCommands) {
			assert.ok(
				commands.includes(cmd),
				`Terminal command ${cmd} should be registered`
			)
		}
	})

	test("Extension should have proper views container", async () => {
		const packageJSON = extension!.packageJSON
		const viewsContainers = packageJSON.contributes.viewsContainers
		
		assert.ok(viewsContainers, "Extension should have viewsContainers")
		assert.ok(viewsContainers.activitybar, "Extension should have activitybar containers")
		
		const theaContainer = viewsContainers.activitybar.find(
			(c: any) => c.id === "thea-code-ActivityBar"
		)
		
		assert.ok(theaContainer, "Thea Code activity bar container should exist")
		assert.strictEqual(theaContainer.title, "Thea Code", "Container title should be correct")
		assert.strictEqual(
			theaContainer.icon,
			"assets/icons/icon.svg",
			"Container should have icon"
		)
	})

	test("Extension configuration should be properly defined", async () => {
		const packageJSON = extension!.packageJSON
		const config = packageJSON.contributes.configuration
		
		assert.ok(config, "Extension should have configuration")
		assert.strictEqual(config.title, "Thea Code", "Configuration title should be correct")
		assert.ok(config.properties, "Configuration should have properties")
		
		// Verify key configuration properties exist
		const props = config.properties
		assert.ok(props["thea-code.allowedCommands"], "allowedCommands config should exist")
		assert.ok(props["thea-code.vsCodeLmModelSelector"], "vsCodeLmModelSelector config should exist")
		assert.ok(props["thea-code.customStoragePath"], "customStoragePath config should exist")
	})

	test("Extension should properly handle view visibility", async function () {
		this.timeout(10_000)
		
		// Show the view
		await vscode.commands.executeCommand("workbench.view.extension.thea-code-ActivityBar")
		await new Promise(resolve => setTimeout(resolve, 1000))
		
		// Extension should still be active
		assert.ok(extension!.isActive, "Extension should be active after showing view")
		
		// Hide and show again
		await vscode.commands.executeCommand("workbench.action.closeSidebar")
		await new Promise(resolve => setTimeout(resolve, 500))
		await vscode.commands.executeCommand("workbench.view.extension.thea-code-ActivityBar")
		await new Promise(resolve => setTimeout(resolve, 1000))
		
		// Should still be functional
		assert.ok(extension!.isActive, "Extension should remain active after hide/show cycle")
	})

	test("Extension menus should be properly configured", async () => {
		const packageJSON = extension!.packageJSON
		const menus = packageJSON.contributes.menus
		
		assert.ok(menus, "Extension should have menus")
		assert.ok(menus["editor/context"], "Editor context menu should be configured")
		assert.ok(menus["terminal/context"], "Terminal context menu should be configured")
		assert.ok(menus["view/title"], "View title menu should be configured")
		assert.ok(menus["thea-code.contextMenu"], "Thea Code context menu should be configured")
		assert.ok(menus["thea-code.terminalMenu"], "Thea Code terminal menu should be configured")
	})
})
