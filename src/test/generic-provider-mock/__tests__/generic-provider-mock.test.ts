import { expect } from "chai"

/* eslint-disable @typescript-eslint/no-unused-expressions, @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-unsafe-return, @typescript-eslint/no-explicit-any */
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
		it("should start and stop server", async () => {
			mock = new GenericProviderMock(PROVIDER_CONFIGS.generic)
			port = await mock.start()
			
			expect(port).to.be.greaterThan(0)
			expect(mock.getPort()).to.equal(port)
			
			await mock.stop()
			expect(mock.getPort()).to.be.null
		})

		it("should handle multiple start calls", async () => {
			mock = new GenericProviderMock(PROVIDER_CONFIGS.generic)
			const port1 = await mock.start()
			const port2 = await mock.start() // Should return same port
			
			expect(port1).to.equal(port2)
		})
	})

	describe("OpenAI Format Support", () => {
		beforeEach(async () => {
			mock = new GenericProviderMock(PROVIDER_CONFIGS.openai)
			port = await mock.start()
			// Verify server started
			expect(port).to.be.greaterThan(0)
			expect(mock.getPort()).to.equal(port)
		})

		it("should list models", async () => {
			const response = await fetch(`http://127.0.0.1:${port}/v1/models`)
			expect(response.status).to.equal(200)
			
			const data = await response.json()
			expect(data.object).to.equal("list")
			expect(Array.isArray(data.data)).to.be.true
			expect(data.data.length).to.be.greaterThan(0)
			expect(data.data[0]).to.have.property("id")
		})

		it("should handle chat completion", async () => {
			const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model: "gpt-4",
					messages: [{ role: "user", content: "Test" }],
					stream: false,
				}),
			})
			
			expect(response.status).to.equal(200)
			const data = await response.json()
			expect(data).to.have.property("choices")
			expect(data.choices[0].message).to.have.property("content")
		})

		it("should handle streaming", async () => {
			const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model: "gpt-4",
					messages: [{ role: "user", content: "Test" }],
					stream: true,
				}),
			})
			
			expect(response.status).to.equal(200)
			expect(response.headers.get("content-type")).to.equal("text/event-stream")
			
			const text = await response.text()
			expect(text).to.contain("data:")
			expect(text).to.contain("[DONE]")
		})

		it("should handle tool calls", async () => {
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
			
			expect(response.status).to.equal(200)
			const data = await response.json()
			expect(data.choices[0].message.tool_calls).to.not.be.undefined
			expect(data.choices[0].message.tool_calls[0].function.name).to.equal("test_func")
		})
	})

	describe("Anthropic Format Support", () => {
		beforeEach(async () => {
			mock = new GenericProviderMock(PROVIDER_CONFIGS.anthropic)
			port = await mock.start()
		})

		it("should handle messages endpoint", async () => {
			const response = await fetch(`http://127.0.0.1:${port}/v1/messages`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model: "claude-3-sonnet-20240229",
					messages: [{ role: "user", content: "Test" }],
				}),
			})
			
			expect(response.status).to.equal(200)
			const data = await response.json()
			expect(data.type).to.equal("message")
			expect(data.content[0].type).to.equal("text")
		})

		it("should handle token counting", async () => {
			const response = await fetch(`http://127.0.0.1:${port}/v1/messages/count_tokens`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					messages: [{ role: "user", content: "Count these tokens" }],
				}),
			})
			
			expect(response.status).to.equal(200)
			const data = await response.json()
			expect(data).to.have.property("input_tokens")
			expect(data.input_tokens).to.be.greaterThan(0)
		})

		it("should include thinking for thinking models", async () => {
			const response = await fetch(`http://127.0.0.1:${port}/v1/messages`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model: "claude-3-5-sonnet-thinking",
					messages: [{ role: "user", content: "Think" }],
				}),
			})
			
			const data = await response.json()
			expect(data.content[0].text).to.contain("<think>")
		})
	})

	describe("Ollama Format Support", () => {
		beforeEach(async () => {
			mock = new GenericProviderMock(PROVIDER_CONFIGS.ollama)
			port = await mock.start()
		})

		it("should handle generate endpoint", async () => {
			const response = await fetch(`http://127.0.0.1:${port}/api/generate`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model: "llama2",
					prompt: "Test",
					stream: false,
				}),
			})
			
			expect(response.status).to.equal(200)
			const data = await response.json()
			expect(data).to.have.property("response")
			expect(data.done).to.be.true
		})

		it("should handle chat endpoint", async () => {
			const response = await fetch(`http://127.0.0.1:${port}/api/chat`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model: "llama2",
					messages: [{ role: "user", content: "Test" }],
					stream: false,
				}),
			})
			
			expect(response.status).to.equal(200)
			const data = await response.json()
			expect(data.message.role).to.equal("assistant")
			expect(data.message).to.have.property("content")
		})
	})

	describe("Response Overrides", () => {
		beforeEach(async () => {
			mock = new GenericProviderMock(PROVIDER_CONFIGS.generic)
			port = await mock.start()
		})

		it("should override responses", async () => {
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
			expect(data.id).to.equal("override")
			expect(data.choices[0].message.content).to.equal("Overridden")
		})
	})

	describe("Request Logging", () => {
		beforeEach(async () => {
			mock = new GenericProviderMock(PROVIDER_CONFIGS.generic)
			port = await mock.start()
		})

		it("should log requests", async () => {
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
			expect(log).to.have.lengthOf(2)
			expect(log[0].path).to.equal("/v1/models")
			expect(log[1].body.messages[0].content).to.equal("Log me")
		})

		it("should clear request log", () => {
			mock.clearRequestLog()
			expect(mock.getRequestLog()).to.have.lengthOf(0)
		})
	})

	describe("Test Helper Integration", () => {
		it("should work with test helper", async () => {
			const helper = createTestHelper("openai")
			
			try {
				const url = await helper.start()
				expect(url).to.match(/http:\/\/127\.0\.0\.1:\d+/)
				
				const response = await fetch(`${url}/v1/models`)
				expect(response.status).to.equal(200)
			} finally {
				await helper.stop()
			}
		})

		it("should run test scenarios", async () => {
			const results = await testProviderBehavior("openai", ["basic_chat"])
			
			expect(results.failed).to.equal(0)
			expect(results.passed).to.be.greaterThan(0)
		})
	})

	describe("Configuration Updates", () => {
		it("should update configuration", async () => {
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
			expect(data.choices[0].message.content).to.equal("Updated response")
		})
	})
})
