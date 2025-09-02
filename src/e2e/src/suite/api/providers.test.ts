import * as assert from "assert"
import * as vscode from "vscode"
import { EXTENSION_ID } from "../../thea-constants"

suite("API Provider Tests", () => {
	let extension: vscode.Extension<any> | undefined

	suiteSetup(async function() {
		this.timeout(30000)
		extension = vscode.extensions.getExtension(EXTENSION_ID)
		if (!extension) {
			assert.fail("Extension not found")

		if (!extension.isActive) {
			await extension.activate()

	test("Extension exports API", () => {
		assert.ok(extension?.exports, "Extension should export an API")

	// Placeholder for provider-specific tests
	suite("OpenAI Provider", () => {
		test.skip("should handle streaming responses", () => {
			// To be implemented when we migrate OpenAI tests

	suite("Anthropic Provider", () => {
		test.skip("should handle Claude models", () => {
			// To be implemented when we migrate Anthropic tests

	suite("Model Registry", () => {
		test.skip("should list available models", () => {
			// To be implemented when we migrate model registry tests
