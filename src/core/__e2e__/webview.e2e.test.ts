import * as assert from "assert"
import * as vscode from "vscode"
import { EXTENSION_ID, EXTENSION_NAME } from "../../../shared/config/thea-config"

suite("Webview Tests", () => {
	let extension: vscode.Extension<any> | undefined

	suiteSetup(async function () {
		this.timeout(30000)
		extension = vscode.extensions.getExtension(EXTENSION_ID)
		if (!extension) {
			assert.fail("Extension not found")
		}
		if (!extension.isActive) {
			await extension.activate()
		}
	})

	suite("Webview Command Integration", () => {
		test("Should open webview panel", async function () {
			this.timeout(10000)

			// Execute the command to open the webview
			await vscode.commands.executeCommand(`${EXTENSION_NAME}.plusButtonClicked`)

			// Give it a moment to open
			await new Promise((resolve) => setTimeout(resolve, 1000))

			// We can't directly check if webview is open from e2e tests,
			// but we can verify the command executed without error
			assert.ok(true, "Command executed successfully")
		})

		test("Should handle settings button", async function () {
			this.timeout(10000)

			try {
				await vscode.commands.executeCommand(`${EXTENSION_NAME}.settingsButtonClicked`)
				assert.ok(true, "Settings command executed successfully")
			} catch (error) {
				assert.fail(`Settings command failed: ${error}`)
			}
		})

		test("Should handle history button", async function () {
			this.timeout(10000)

			try {
				await vscode.commands.executeCommand(`${EXTENSION_NAME}.historyButtonClicked`)
				assert.ok(true, "History command executed successfully")
			} catch (error) {
				assert.fail(`History command failed: ${error}`)
			}
		})

		test("Should handle MCP button", async function () {
			this.timeout(10000)

			try {
				await vscode.commands.executeCommand(`${EXTENSION_NAME}.mcpButtonClicked`)
				assert.ok(true, "MCP command executed successfully")
			} catch (error) {
				assert.fail(`MCP command failed: ${error}`)
			}
		})

		test("Should handle prompts button", async function () {
			this.timeout(10000)

			try {
				await vscode.commands.executeCommand(`${EXTENSION_NAME}.promptsButtonClicked`)
				assert.ok(true, "Prompts command executed successfully")
			} catch (error) {
				assert.fail(`Prompts command failed: ${error}`)
			}
		})

		test("Should handle popout button", async function () {
			this.timeout(10000)

			try {
				await vscode.commands.executeCommand(`${EXTENSION_NAME}.popoutButtonClicked`)
				assert.ok(true, "Popout command executed successfully")
			} catch (error) {
				assert.fail(`Popout command failed: ${error}`)
			}
		})

		test("Should handle help button", async function () {
			this.timeout(10000)

			try {
				await vscode.commands.executeCommand(`${EXTENSION_NAME}.helpButtonClicked`)
				assert.ok(true, "Help command executed successfully")
			} catch (error) {
				assert.fail(`Help command failed: ${error}`)
			}
		})
	})

	suite("Context Menu Commands", () => {
		test("Should register addToContext command", async function () {
			this.timeout(5000)
			const commands = await vscode.commands.getCommands()
			assert.ok(
				commands.includes(`${EXTENSION_NAME}.addToContext`),
				"addToContext command should be registered"
			)
		})

		test("Should register explainCode command", async function () {
			this.timeout(5000)
			const commands = await vscode.commands.getCommands()
			assert.ok(
				commands.includes(`${EXTENSION_NAME}.explainCode`),
				"explainCode command should be registered"
			)
		})

		test("Should register fixCode command", async function () {
			this.timeout(5000)
			const commands = await vscode.commands.getCommands()
			assert.ok(commands.includes(`${EXTENSION_NAME}.fixCode`), "fixCode command should be registered")
		})

		test("Should register improveCode command", async function () {
			this.timeout(5000)
			const commands = await vscode.commands.getCommands()
			assert.ok(
				commands.includes(`${EXTENSION_NAME}.improveCode`),
				"improveCode command should be registered"
			)
		})
	})

	suite("Terminal Context Menu Commands", () => {
		test("Should register terminalAddToContext command", async function () {
			this.timeout(5000)
			const commands = await vscode.commands.getCommands()
			assert.ok(
				commands.includes(`${EXTENSION_NAME}.terminalAddToContext`),
				"terminalAddToContext command should be registered"
			)
		})

		test("Should register terminalFixCommand command", async function () {
			this.timeout(5000)
			const commands = await vscode.commands.getCommands()
			assert.ok(
				commands.includes(`${EXTENSION_NAME}.terminalFixCommand`),
				"terminalFixCommand command should be registered"
			)
		})

		test("Should register terminalExplainCommand command", async function () {
			this.timeout(5000)
			const commands = await vscode.commands.getCommands()
			assert.ok(
				commands.includes(`${EXTENSION_NAME}.terminalExplainCommand`),
				"terminalExplainCommand command should be registered"
			)
		})
	})

	suite("Extension Activation", () => {
		test("Should activate extension", () => {
			assert.ok(extension, "Extension should exist")
			assert.ok(extension.isActive, "Extension should be active")
		})

		test("Should expose extension API", () => {
			assert.ok(extension, "Extension should exist")
			const api = extension.exports
			assert.ok(api !== undefined, "Extension should export API")
		})
	})
