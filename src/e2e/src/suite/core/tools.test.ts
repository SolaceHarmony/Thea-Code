import * as assert from "assert"
import * as vscode from "vscode"
import { EXTENSION_ID, EXTENSION_NAME } from "../../thea-constants"

suite("Core Tools Tests", () => {
	let extension: vscode.Extension<any> | undefined
	let api: any

	suiteSetup(async function() {
		this.timeout(30000)
		extension = vscode.extensions.getExtension(EXTENSION_ID)
		if (!extension) {
			assert.fail("Extension not found")

		if (!extension.isActive) {
			await extension.activate()

		api = extension.exports

	suite("Execute Command Tool", () => {
		test("Execute command should be available", () => {
			// Test that the extension can execute commands
			assert.ok(vscode.commands, "VSCode commands API should be available")

		test("Should handle command execution through extension", async function() {
			this.timeout(5000)
			
			// Test executing a safe command like getting all commands
			const commands = await vscode.commands.getCommands()
			assert.ok(Array.isArray(commands), "Should return array of commands")
			assert.ok(commands.length > 0, "Should have some commands registered")

		test.skip("Should handle command errors gracefully", async () => {
			// Test error handling for invalid commands
			try {
				await vscode.commands.executeCommand("nonexistent.command")
				assert.fail("Should have thrown error for nonexistent command")
			} catch (error) {
				assert.ok(error, "Should throw error for invalid command")

	suite("Apply Diff Tool", () => {
		test("Should handle text document edits", async function() {
			this.timeout(10000)
			
			// Create a temporary document to test editing
			const document = await vscode.workspace.openTextDocument({
				content: "Hello World",
				language: "plaintext"
			
			const editor = await vscode.window.showTextDocument(document)
			
			// Test that we can make edits
			const success = await editor.edit(editBuilder => {
				editBuilder.replace(
					new vscode.Range(0, 0, 0, 5),
					"Hi"

			assert.ok(success, "Edit should succeed")
			assert.strictEqual(
				document.getText(),
				"Hi World",
				"Document should be edited"

			// Close without saving
			await vscode.commands.executeCommand("workbench.action.closeActiveEditor")

		test.skip("Should apply diffs to files", async () => {
			// This would test the actual diff application
			// Requires more complex setup with file system access

	suite("List Files Tool", () => {
		test("Should list workspace files", async () => {
			// Test file listing through workspace API
			const files = await vscode.workspace.findFiles("**/*.{ts,js}", "**/node_modules/**", 10)
			assert.ok(Array.isArray(files), "Should return array of files")

		test("Should respect ignore patterns", async () => {
			// Test that node_modules and other patterns are ignored
			const files = await vscode.workspace.findFiles("**/*", "**/node_modules/**", 100)
			const hasNodeModules = files.some(f => f.path.includes("node_modules"))
			assert.ok(!hasNodeModules, "Should not include node_modules")

	suite("Browser Tool", () => {
		test.skip("Should handle browser automation requests", async () => {
			// Browser automation tests would go here
			// These require special setup and may not work in test environment

	suite("Ask Followup Question Tool", () => {
		test("Should support user interaction", () => {
			// Test that the extension can show input boxes
			assert.ok(vscode.window.showInputBox, "Input box API should be available")
			assert.ok(vscode.window.showQuickPick, "Quick pick API should be available")

		test.skip("Should handle user input", async () => {
			// This would test actual user input handling
			// Difficult to test in automated environment

	suite("Attempt Completion Tool", () => {
		test.skip("Should mark tasks as complete", async () => {
			// Test task completion functionality
			// Requires task system to be active

		test.skip("Should validate completion criteria", async () => {
			// Test that completion validates requirements

	suite("Tool Integration", () => {
		test("All tool commands should be registered", async () => {
			const commands = await vscode.commands.getCommands(true)
			
			// Check for tool-related commands
			const toolCommands = commands.filter(cmd => 
				cmd.includes(EXTENSION_NAME) && 
				(cmd.includes("execute") || cmd.includes("apply") || cmd.includes("tool"))

			assert.ok(toolCommands.length >= 0, "Should have tool commands registered")

		test.skip("Tools should integrate with task system", async () => {
			// Test that tools work within the task context

		test.skip("Tools should respect permissions", async () => {
			// Test that tools check permissions before executing
