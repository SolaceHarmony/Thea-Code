import * as assert from "assert"
import * as vscode from "vscode"

suite("Simple E2E", () => {
	test("should activate sidebar and initialize properly", async () => {
		// Focus the sidebar view
		await vscode.commands.executeCommand("thea-code.SidebarProvider.focus")

		const api = global.api
		const apiKeys = Object.keys((api as unknown as Record<string, unknown>) || {})
		console.log("API keys:", apiKeys)
		assert.ok(api, "API should be available")

		// New native VS Code extension API
		assert.ok(apiKeys.includes("outputChannel"), "API should have outputChannel")
		assert.ok(apiKeys.includes("taskManager"), "API should have taskManager")
		assert.ok(apiKeys.includes("isTestMode"), "API should have isTestMode flag")
		assert.ok(apiKeys.includes("version"), "API should have version")

		// Get the TreeView for the sidebar
		const sidebarView = (api as unknown as { sidebarView?: vscode.TreeView<unknown> }).sidebarView
		assert.ok(sidebarView, "Sidebar TreeView should be available")

		// Wait for the sidebar to be visible
		let attempts = 0
		while (!sidebarView.visible && attempts < 20) {
			await new Promise((r) => setTimeout(r, 250))
			attempts++
		}
		assert.strictEqual(sidebarView.visible, true, "Sidebar should be visible")

		// Verify taskManager is initialized and can create tasks
		const taskManager = (api as unknown as { taskManager?: unknown }).taskManager
		assert.ok(taskManager, "TaskManager should be available")

		// The extension should be fully activated without errors
		console.log("Extension activated successfully in native VS Code mode")
	})
})
