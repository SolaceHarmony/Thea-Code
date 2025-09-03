import * as assert from "assert"
import * as vscode from "vscode"
import { EXTENSION_NAME, EXTENSION_DISPLAY_NAME } from "../thea-constants"

suite(`${EXTENSION_DISPLAY_NAME} Extension`, () => {
	test("OPENROUTER_API_KEY environment variable is set (or skipped)", function() {
		if (!process.env.OPENROUTER_API_KEY) {
			console.log("Warning: OPENROUTER_API_KEY not set - skipping API tests")
			this.skip()

	test("Commands should be registered", async () => {
		const expectedCommands = [
			`${EXTENSION_NAME}.plusButtonClicked`,
			`${EXTENSION_NAME}.mcpButtonClicked`,
			`${EXTENSION_NAME}.historyButtonClicked`,
			`${EXTENSION_NAME}.popoutButtonClicked`,
			`${EXTENSION_NAME}.settingsButtonClicked`,
			`${EXTENSION_NAME}.openInNewTab`,
			`${EXTENSION_NAME}.explainCode`,
			`${EXTENSION_NAME}.fixCode`,
			`${EXTENSION_NAME}.improveCode`,

		const commands = await vscode.commands.getCommands(true)

		for (const cmd of expectedCommands) {
			assert.ok(commands.includes(cmd), `Command ${cmd} should be registered`)
