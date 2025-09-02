import * as assert from "assert"
import * as vscode from "vscode"
import { EXTENSION_ID, EXTENSION_DISPLAY_NAME } from "../thea-constants"

suite(`${EXTENSION_DISPLAY_NAME} Basic Tests`, () => {
	test("Extension should be present", () => {
		const extension = vscode.extensions.getExtension(EXTENSION_ID)
		assert.ok(extension, "Extension should be installed")

	test("Extension should activate", async function() {
		this.timeout(30000) // 30 second timeout for activation
		const extension = vscode.extensions.getExtension(EXTENSION_ID)
		if (!extension) {
			assert.fail("Extension not found")

		try {
			await extension.activate()
			assert.ok(extension.isActive, "Extension should be active")
		} catch (error) {
			console.error("Extension activation failed:", error)
			assert.fail(`Extension failed to activate: ${error}`)

	test("VSCode version check", () => {
		const version = vscode.version
		console.log("VSCode version:", version)
		assert.ok(version, "VSCode version should be available")
