import GenericProviderMock, { PROVIDER_CONFIGS } from "../../../../test/generic-provider-mock/server"
import { createTestHelper, testProviderBehavior } from "../../../../test/generic-provider-mock/test-helpers"

import * as assert from 'assert'
import * as sinon from 'sinon'
suite("Generic Provider Mock", () => {
	let mock: GenericProviderMock
	let port: number

	teardown(async () => {
		if (mock) {
			await mock.stop()

	suite("Server Lifecycle", () => {
		test("should start and stop server", async () => {
			mock = new GenericProviderMock(PROVIDER_CONFIGS.generic)
			port = await mock.start()
			
			assert.ok(port > 0)
			expect(mock.getPort()).toBe(port)
			
			await mock.stop()
			expect(mock.getPort()).toBeNull()

		test("should handle multiple start calls", async () => {
			mock = new GenericProviderMock(PROVIDER_CONFIGS.generic)
			const port1 = await mock.start()
			const port2 = await mock.start() // Should return same port
			
			assert.strictEqual(port1, port2)

	suite("OpenAI Format Support", () => {
		setup(async () => {
			mock = new GenericProviderMock(PROVIDER_CONFIGS.openai)
			port = await mock.start()
			// Verify server started
			assert.ok(port > 0)
			expect(mock.getPort()).toBe(port)

		test("should list models", async () => {
			const response = await fetch(`http://127.0.0.1:${port}/v1/models`)
			assert.strictEqual(response.status, 200)
			
			const data = await response.json()
			assert.strictEqual(data.object, "list")
			expect(Array.isArray(data.data)).toBe(true)
			assert.ok(data.data.length > 0)
			expect(data.data[0]).toHaveProperty("id")

		test("should handle chat completion", async () => {
			const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model: "gpt-4",
					messages: [{ role: "user", content: "Test" }],
					stream: false,
				}),
			
			assert.strictEqual(response.status, 200)
			const data = await response.json()
			expect(data).toHaveProperty("choices")
			expect(data.choices[0].message).toHaveProperty("content")

		test("should handle streaming", async () => {
			const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model: "gpt-4",
					messages: [{ role: "user", content: "Test" }],
					stream: true,
				}),
			
			assert.strictEqual(response.status, 200)
			expect(response.headers.get("content-type")).toBe("text/event-stream")
			
			const text = await response.text()
			assert.ok(text.includes("data:"))
			assert.ok(text.includes("[DONE]"))

		test("should handle tool calls", async () => {
			const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model: "gpt-4",
					messages: [{ role: "user", content: "Use tool" }],
					tools: [{
						type: "function",
						function: {
							name: "test_func",
							description: "Test",
							parameters: {},
						},
					}],
				}),
			
			assert.strictEqual(response.status, 200)
			const data = await response.json()
			assert.ok(data.choices[0].message.tool_calls !== undefined)
			assert.strictEqual(data.choices[0].message.tool_calls[0].function.name, "test_func")

	suite("Anthropic Format Support", () => {
		setup(async () => {
			mock = new GenericProviderMock(PROVIDER_CONFIGS.anthropic)
			port = await mock.start()

		test("should handle messages endpoint", async () => {
			const response = await fetch(`http://127.0.0.1:${port}/v1/messages`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model: "claude-3-sonnet-20240229",
					messages: [{ role: "user", content: "Test" }],
				}),
			
			assert.strictEqual(response.status, 200)
			const data = await response.json()
			assert.strictEqual(data.type, "message")
			assert.strictEqual(data.content[0].type, "text")

		test("should handle token counting", async () => {
			const response = await fetch(`http://127.0.0.1:${port}/v1/messages/count_tokens`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					messages: [{ role: "user", content: "Count these tokens" }],
				}),
			
			assert.strictEqual(response.status, 200)
			const data = await response.json()
			expect(data).toHaveProperty("input_tokens")
			assert.ok(data.input_tokens > 0)

		test("should include thinking for thinking models", async () => {
			const response = await fetch(`http://127.0.0.1:${port}/v1/messages`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model: "claude-3-5-sonnet-thinking",
					messages: [{ role: "user", content: "Think" }],
				}),
			
			const data = await response.json()
			assert.ok(data.content[0].text.includes("<think>"))

	suite("Ollama Format Support", () => {
		setup(async () => {
			mock = new GenericProviderMock(PROVIDER_CONFIGS.ollama)
			port = await mock.start()

		test("should handle generate endpoint", async () => {
			const response = await fetch(`http://127.0.0.1:${port}/api/generate`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model: "llama2",
					prompt: "Test",
					stream: false,
				}),
			
			assert.strictEqual(response.status, 200)
			const data = await response.json()
			expect(data).toHaveProperty("response")
			assert.strictEqual(data.done, true)

		test("should handle chat endpoint", async () => {
			const response = await fetch(`http://127.0.0.1:${port}/api/chat`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model: "llama2",
					messages: [{ role: "user", content: "Test" }],
					stream: false,
				}),
			
			assert.strictEqual(response.status, 200)
			const data = await response.json()
			assert.strictEqual(data.message.role, "assistant")
			expect(data.message).toHaveProperty("content")

	suite("Response Overrides", () => {
		setup(async () => {
			mock = new GenericProviderMock(PROVIDER_CONFIGS.generic)
			port = await mock.start()

		test("should override responses", async () => {
			mock.setResponseOverride("chat_model-1", {
				id: "override",
				choices: [{
					message: { content: "Overridden" },
				}],
			
			const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model: "model-1",
					messages: [{ role: "user", content: "Test" }],
				}),
			
			const data = await response.json()
			assert.strictEqual(data.id, "override")
			assert.strictEqual(data.choices[0].message.content, "Overridden")

	suite("Request Logging", () => {
		setup(async () => {
			mock = new GenericProviderMock(PROVIDER_CONFIGS.generic)
			port = await mock.start()

		test("should log requests", async () => {
			await fetch(`http://127.0.0.1:${port}/v1/models`)
			await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model: "test",
					messages: [{ role: "user", content: "Log me" }],
				}),
			
			const log = mock.getRequestLog()
			assert.strictEqual(log.length, 2)
			assert.strictEqual(log[0].path, "/v1/models")
			assert.strictEqual(log[1].body.messages[0].content, "Log me")

		test("should clear request log", () => {
			mock.clearRequestLog()
			expect(mock.getRequestLog()).toHaveLength(0)

	suite("Test Helper Integration", () => {
		test("should work with test helper", async () => {
			const helper = createTestHelper("openai")
			
			try {
				const url = await helper.start()
				assert.ok(url.match(/http:\/\/127\.0\.0\.1:\d+/))
				
				const response = await fetch(`${url}/v1/models`)
				assert.strictEqual(response.status, 200)
			} finally {
				await helper.stop()

		test("should run test scenarios", async () => {
			const results = await testProviderBehavior("openai", ["basic_chat"])
			
			assert.strictEqual(results.failed, 0)
			assert.ok(results.passed > 0)

	suite("Configuration Updates", () => {
		test("should update configuration", async () => {
			mock = new GenericProviderMock(PROVIDER_CONFIGS.generic)
			port = await mock.start()
			
			// Update config
			mock.updateConfig({
				responsePatterns: {
					simple: "Updated response",
				},
			
			const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model: "model-1",
					messages: [{ role: "user", content: "Test" }],
				}),
			
			const data = await response.json()
			assert.strictEqual(data.choices[0].message.content, "Updated response")
