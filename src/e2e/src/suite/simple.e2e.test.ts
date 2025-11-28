import * as assert from "assert"
import * as vscode from "vscode"

suite("Simple E2E", () => {
    test("should activate sidebar and resolve webview", async () => {
        // Focus the sidebar view
        await vscode.commands.executeCommand("thea-code.SidebarProvider.focus")

        // Wait for webview to resolve
        await new Promise(r => setTimeout(r, 2000))

        // Get the provider from the global API
        const api = (global as any).api
        console.log("API keys:", Object.keys(api || {}))
        assert.ok(api, "API should be available")
        assert.ok(api.sidebarProvider, "SidebarProvider should be available")

        const provider = api.sidebarProvider
        assert.ok(provider.view, "WebviewView should be set")
        assert.strictEqual(provider.view.visible, true, "Webview should be visible")

        // Check if HTML is set
        assert.ok(provider.view.webview.html.length > 0, "Webview HTML should not be empty")
        assert.ok(provider.view.webview.html.includes("<!DOCTYPE html>"), "Webview HTML should be valid")
    })
})
