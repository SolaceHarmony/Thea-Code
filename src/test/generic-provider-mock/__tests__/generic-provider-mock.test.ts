import GenericProviderMock, { PROVIDER_CONFIGS } from "../../../../test/generic-provider-mock/server"
import { createTestHelper, testProviderBehavior } from "../../../../test/generic-provider-mock/test-helpers"

describe("Generic Provider Mock", () => {
	let mock: GenericProviderMock
	let port: number

	afterEach(async () => {
		if (mock) {
			await mock.stop()
		}
	})

	describe("Server Lifecycle", () => {
		test("should start and stop server", async () => {
			mock = new GenericProviderMock(PROVIDER_CONFIGS.generic)
			port = await mock.start()
			
			expect(port).toBeGreaterThan(0)
			expect(mock.getPort()).toBe(port)
			
			await mock.stop()
			expect(mock.getPort()).toBeNull()
		})

		test("should handle multiple start calls", async () => {
			mock = new GenericProviderMock(PROVIDER_CONFIGS.generic)
			const port1 = await mock.start()
			const port2 = await mock.start() // Should return same port
			
			expect(port1).toBe(port2)
		})
	})

	describe("OpenAI Format Support", () => {
		beforeEach(async () => {
			mock = new GenericProviderMock(PROVIDER_CONFIGS.openai)
			port = await mock.start()
			// Verify server started
			expect(port).toBeGreaterThan(0)
			expect(mock.getPort()).toBe(port)
		})

		test("should list models", async () => {
			const response = await fetch(`http://127.0.0.1:${port}/v1/models`)
			expect(response.status).toBe(200)
			
			const data = await response.json()
			expect(data.object).toBe("list")
			expect(Array.isArray(data.data)).toBe(true)
			expect(data.data.length).toBeGreaterThan(0)
			expect(data.data[0]).toHaveProperty("id")
		})

		test("should handle chat completion", async () => {
			const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model: "gpt-4",
					messages: [{ role: "user", content: "Test" }],
					stream: false,
				}),
			})
			
			expect(response.status).toBe(200)
			const data = await response.json()
			expect(data).toHaveProperty("choices")
			expect(data.choices[0].message).toHaveProperty("content")
		})

		test("should handle streaming", async () => {
			const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model: "gpt-4",
					messages: [{ role: "user", content: "Test" }],
					stream: true,
				}),
			})
			
			expect(response.status).toBe(200)
			expect(response.headers.get("content-type")).toBe("text/event-stream")
			
			const text = await response.text()
			expect(text).toContain("data:")
			expect(text).toContain("[DONE]")
		})

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
			})
			
			expect(response.status).toBe(200)
			const data = await response.json()
			expect(data.choices[0].message.tool_calls).toBeDefined()
			expect(data.choices[0].message.tool_calls[0].function.name).toBe("test_func")
		})
	})

	describe("Anthropic Format Support", () => {
		beforeEach(async () => {
			mock = new GenericProviderMock(PROVIDER_CONFIGS.anthropic)
			port = await mock.start()
		})

		test("should handle messages endpoint", async () => {
			const response = await fetch(`http://127.0.0.1:${port}/v1/messages`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model: "claude-3-sonnet-20240229",
					messages: [{ role: "user", content: "Test" }],
				}),
			})
			
			expect(response.status).toBe(200)
			const data = await response.json()
			expect(data.type).toBe("message")
			expect(data.content[0].type).toBe("text")
		})

		test("should handle token counting", async () => {
			const response = await fetch(`http://127.0.0.1:${port}/v1/messages/count_tokens`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					messages: [{ role: "user", content: "Count these tokens" }],
				}),
			})
			
			expect(response.status).toBe(200)
			const data = await response.json()
			expect(data).toHaveProperty("input_tokens")
			expect(data.input_tokens).toBeGreaterThan(0)
		})

		test("should include thinking for thinking models", async () => {
			const response = await fetch(`http://127.0.0.1:${port}/v1/messages`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model: "claude-3-5-sonnet-thinking",
					messages: [{ role: "user", content: "Think" }],
				}),
			})
			
			const data = await response.json()
			expect(data.content[0].text).toContain("<think>")
		})
	})

	describe("Ollama Format Support", () => {
		beforeEach(async () => {
			mock = new GenericProviderMock(PROVIDER_CONFIGS.ollama)
			port = await mock.start()
		})

		test("should handle generate endpoint", async () => {
			const response = await fetch(`http://127.0.0.1:${port}/api/generate`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model: "llama2",
					prompt: "Test",
					stream: false,
				}),
			})
			
			expect(response.status).toBe(200)
			const data = await response.json()
			expect(data).toHaveProperty("response")
			expect(data.done).toBe(true)
		})

		test("should handle chat endpoint", async () => {
			const response = await fetch(`http://127.0.0.1:${port}/api/chat`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model: "llama2",
					messages: [{ role: "user", content: "Test" }],
					stream: false,
				}),
			})
			
			expect(response.status).toBe(200)
			const data = await response.json()
			expect(data.message.role).toBe("assistant")
			expect(data.message).toHaveProperty("content")
		})
	})

	describe("Response Overrides", () => {
		beforeEach(async () => {
			mock = new GenericProviderMock(PROVIDER_CONFIGS.generic)
			port = await mock.start()
		})

		test("should override responses", async () => {
			mock.setResponseOverride("chat_model-1", {
				id: "override",
				choices: [{
					message: { content: "Overridden" },
				}],
			})
			
			const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model: "model-1",
					messages: [{ role: "user", content: "Test" }],
				}),
			})
			
			const data = await response.json()
			expect(data.id).toBe("override")
			expect(data.choices[0].message.content).toBe("Overridden")
		})
	})

	describe("Request Logging", () => {
		beforeEach(async () => {
			mock = new GenericProviderMock(PROVIDER_CONFIGS.generic)
			port = await mock.start()
		})

		test("should log requests", async () => {
			await fetch(`http://127.0.0.1:${port}/v1/models`)
			await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model: "test",
					messages: [{ role: "user", content: "Log me" }],
				}),
			})
			
			const log = mock.getRequestLog()
			expect(log).toHaveLength(2)
			expect(log[0].path).toBe("/v1/models")
			expect(log[1].body.messages[0].content).toBe("Log me")
		})

		test("should clear request log", () => {
			mock.clearRequestLog()
			expect(mock.getRequestLog()).toHaveLength(0)
		})
	})

	describe("Test Helper Integration", () => {
		test("should work with test helper", async () => {
			const helper = createTestHelper("openai")
			
			try {
				const url = await helper.start()
				expect(url).toMatch(/http:\/\/127\.0\.0\.1:\d+/)
				
				const response = await fetch(`${url}/v1/models`)
				expect(response.status).toBe(200)
			} finally {
				await helper.stop()
			}
		})

		test("should run test scenarios", async () => {
			const results = await testProviderBehavior("openai", ["basic_chat"])
			
			expect(results.failed).toBe(0)
			expect(results.passed).toBeGreaterThan(0)
		})
	})

	describe("Configuration Updates", () => {
		test("should update configuration", async () => {
			mock = new GenericProviderMock(PROVIDER_CONFIGS.generic)
			port = await mock.start()
			
			// Update config
			mock.updateConfig({
				responsePatterns: {
					simple: "Updated response",
				},
			})
			
			const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model: "model-1",
					messages: [{ role: "user", content: "Test" }],
				}),
			})
			
			const data = await response.json()
			expect(data.choices[0].message.content).toBe("Updated response")
		})
	})
})