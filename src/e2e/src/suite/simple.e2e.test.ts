import * as assert from "assert"
import * as vscode from "vscode"

suite("Simple E2E", () => {
	test("should activate sidebar and resolve webview", async () => {
		// Focus the sidebar view
		await vscode.commands.executeCommand("thea-code.SidebarProvider.focus")

		const api = global.api
		console.log("API keys:", Object.keys((api as unknown as Record<string, unknown>) || {}))
		assert.ok(api, "API should be available")

		// The extension returns an API instance; the underlying webview provider is held on `api.provider`.
		const provider = (api as unknown as { provider?: unknown }).provider as
			| {
					view?: vscode.WebviewView | vscode.WebviewPanel
					isViewLaunched?: boolean
			  }
			| undefined
		assert.ok(provider, "Provider should be available via api.provider")

		// Wait for the webview to resolve
		let attempts = 0
		while (!provider.view && attempts < 40) {
			await new Promise((r) => setTimeout(r, 250))
			attempts++
		}
		assert.ok(provider.view, "WebviewView should be set")
		assert.strictEqual(provider.view.visible, true, "Webview should be visible")

		// Check if HTML is set
		const html = provider.view.webview.html
		assert.ok(html.length > 0, "Webview HTML should not be empty")
		assert.ok(html.includes("<!DOCTYPE html>"), "Webview HTML should be valid")

		// Wait for webview to launch (React app to start and send webviewDidLaunch)
		// This verifies that the React app didn't crash on startup
		attempts = 0
		while (!provider.isViewLaunched && attempts < 40) {
			await new Promise((r) => setTimeout(r, 250))
			attempts++
		}
		assert.strictEqual(provider.isViewLaunched, true, "Webview should have launched (received webviewDidLaunch)")
	})
})
