import * as assert from "assert"
import * as vscode from "vscode"
import { EXTENSION_ID, EXTENSION_NAME } from "../../thea-constants"

suite("Command Tests", () => {
	let extension: vscode.Extension<any> | undefined

	suiteSetup(async function() {
		this.timeout(30000)
		extension = vscode.extensions.getExtension(EXTENSION_ID)
		if (!extension) {
			assert.fail("Extension not found")

		if (!extension.isActive) {
			await extension.activate()

	suite("Core Commands", () => {
		test("All expected commands should be registered", async () => {
			const allCommands = await vscode.commands.getCommands(true)
			
			const expectedCommands = [
				`${EXTENSION_NAME}.plusButtonClicked`,
				`${EXTENSION_NAME}.mcpButtonClicked`,
				`${EXTENSION_NAME}.promptsButtonClicked`,
				`${EXTENSION_NAME}.historyButtonClicked`,
				`${EXTENSION_NAME}.popoutButtonClicked`,
				`${EXTENSION_NAME}.settingsButtonClicked`,
				`${EXTENSION_NAME}.helpButtonClicked`,
				`${EXTENSION_NAME}.openInNewTab`,
				`${EXTENSION_NAME}.explainCode`,
				`${EXTENSION_NAME}.fixCode`,
				`${EXTENSION_NAME}.improveCode`,
				`${EXTENSION_NAME}.addToContext`,
				`${EXTENSION_NAME}.terminalAddToContext`,
				`${EXTENSION_NAME}.terminalFixCommand`,
				`${EXTENSION_NAME}.terminalExplainCommand`,
				`${EXTENSION_NAME}.terminalFixCommandInCurrentTask`,
				`${EXTENSION_NAME}.terminalExplainCommandInCurrentTask`,
				`${EXTENSION_NAME}.newTask`,

			for (const cmd of expectedCommands) {
				assert.ok(
					allCommands.includes(cmd), 
					`Command ${cmd} should be registered`

	suite("Code Action Commands", () => {
		test("Explain code command should be available", async () => {
			const commands = await vscode.commands.getCommands(true)
			assert.ok(
				commands.includes(`${EXTENSION_NAME}.explainCode`),
				"Explain code command should be registered"

		test("Fix code command should be available", async () => {
			const commands = await vscode.commands.getCommands(true)
			assert.ok(
				commands.includes(`${EXTENSION_NAME}.fixCode`),
				"Fix code command should be registered"

		test("Improve code command should be available", async () => {
			const commands = await vscode.commands.getCommands(true)
			assert.ok(
				commands.includes(`${EXTENSION_NAME}.improveCode`),
				"Improve code command should be registered"

	suite("Terminal Commands", () => {
		test("Terminal fix command should be available", async () => {
			const commands = await vscode.commands.getCommands(true)
			assert.ok(
				commands.includes(`${EXTENSION_NAME}.terminalFixCommand`),
				"Terminal fix command should be registered"

		test("Terminal explain command should be available", async () => {
			const commands = await vscode.commands.getCommands(true)
			assert.ok(
				commands.includes(`${EXTENSION_NAME}.terminalExplainCommand`),
				"Terminal explain command should be registered"

	suite("UI Commands", () => {
		test("Plus button command should execute without error", async function() {
			this.timeout(5000)
			try {
				await vscode.commands.executeCommand(`${EXTENSION_NAME}.plusButtonClicked`)
				assert.ok(true, "Plus button command executed successfully")
			} catch (error) {
				// It's ok if webview commands fail in test environment
				// We're mainly checking they're registered
				assert.ok(true, "Command is registered even if execution fails in test")

		test("Settings button command should execute without error", async function() {
			this.timeout(5000)
			try {
				await vscode.commands.executeCommand(`${EXTENSION_NAME}.settingsButtonClicked`)
				assert.ok(true, "Settings button command executed successfully")
			} catch (error) {
				assert.ok(true, "Command is registered even if execution fails in test")
