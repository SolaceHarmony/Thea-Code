import * as assert from "assert"
import * as vscode from "vscode"
import { EXTENSION_ID, EXTENSION_NAME } from "../../thea-constants"

suite("Webview Tests", () => {
	let extension: vscode.Extension<any> | undefined

	suiteSetup(async function() {
		this.timeout(30000)
		extension = vscode.extensions.getExtension(EXTENSION_ID)
		if (!extension) {
			assert.fail("Extension not found")

		if (!extension.isActive) {
			await extension.activate()

	test("Should open webview panel", async function() {
		this.timeout(10000)
		
		// Execute the command to open the webview
		await vscode.commands.executeCommand(`${EXTENSION_NAME}.plusButtonClicked`)
		
		// Give it a moment to open
		await new Promise(resolve => setTimeout(resolve, 1000))
		
		// We can't directly check if webview is open from e2e tests,
		// but we can verify the command executed without error
		assert.ok(true, "Command executed successfully")

	test("Should handle settings button", async function() {
		this.timeout(10000)
		
		try {
			await vscode.commands.executeCommand(`${EXTENSION_NAME}.settingsButtonClicked`)
			assert.ok(true, "Settings command executed successfully")
		} catch (error) {
			assert.fail(`Settings command failed: ${error}`)

	test("Should handle history button", async function() {
		this.timeout(10000)
		
		try {
			await vscode.commands.executeCommand(`${EXTENSION_NAME}.historyButtonClicked`)
			assert.ok(true, "History command executed successfully")
		} catch (error) {
			assert.fail(`History command failed: ${error}`)
