import { expect } from "chai"

/* eslint-disable @typescript-eslint/no-unused-expressions, @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-unsafe-return, @typescript-eslint/no-explicit-any */

/**
 * Test suite for all dynamic model provider mock servers
 * Tests that our enhanced mock server can simulate all provider APIs
 */

describe("All Providers Runtime Test", () => {
	let providerPorts: Record<string, number>

	before(() => {
		// Get the ports stored by our global setup
		providerPorts = (globalThis as any).__PROVIDER_PORTS__ || {}
		console.log("📋 Available provider ports:", providerPorts)
	})

	describe("Mock Server Availability", () => {
		it("should have all 8 provider mock servers running", () => {
			expect(providerPorts).to.not.be.undefined
			
			const expectedProviders = [
				"openai", 
				"anthropic", 
				"bedrock", 
				"gemini", 
				"vertex", 
				"mistral", 
				"deepseek", 
				"ollama"
			]
			
			expectedProviders.forEach(provider => {
				expect(providerPorts[provider]).to.not.be.undefined
				expect(typeof providerPorts[provider]).to.equal("number")
				expect(providerPorts[provider]).to.be.greaterThan(10000)
			})
			
			console.log(`✅ All ${expectedProviders.length} provider mock servers are running`)
		})
	})

	describe("OpenAI-Compatible Providers", () => {
		const openaiCompatible = ["openai", "bedrock", "gemini", "mistral", "deepseek", "ollama"]

		openaiCompatible.forEach(provider => {
			it(`${provider} should respond to /v1/models endpoint`, async () => {
				const port = providerPorts[provider]
				expect(port).to.not.be.undefined
				
				const response = await fetch(`http://127.0.0.1:${port}/v1/models`)
				expect(response.ok).to.be.true
				
				const data = await response.json()
				expect(data).to.have.property("data")
				expect(Array.isArray(data.data)).to.be.true
				expect(data.data.length).to.be.greaterThan(0)
				
				console.log(`✅ ${provider}: Found ${data.data.length} models`)
				console.log(`   Sample: ${data.data[0].id}`)
			})

			it(`${provider} should handle chat completions`, async () => {
				const port = providerPorts[provider]
				const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({
						model: "test-model",
						messages: [{ role: "user", content: "Hello" }]
					})
				})
				
				expect(response.ok).to.be.true
				const data = await response.json()
				expect(data).to.have.property("choices")
				expect(data.choices[0]).to.have.property("message")
				
				console.log(`✅ ${provider}: Chat completion response received`)
			})
		})
	})

	describe("Anthropic-Compatible Providers", () => {
		const anthropicCompatible = ["anthropic", "vertex"]

		anthropicCompatible.forEach(provider => {
			it(`${provider} should handle /v1/messages endpoint`, async () => {
				const port = providerPorts[provider]
				const response = await fetch(`http://127.0.0.1:${port}/v1/messages`, {
					method: "POST",
					headers: { 
						"Content-Type": "application/json",
						"x-api-key": "test-key",
						"anthropic-version": "2023-06-01"
					},
					body: JSON.stringify({
						model: "claude-3-sonnet-20240229",
						max_tokens: 100,
						messages: [{ role: "user", content: "Hello" }]
					})
				})
				
				expect(response.ok).to.be.true
				const data = await response.json()
				expect(data).to.have.property("content")
				expect(data).to.have.property("role", "assistant")
				
				console.log(`✅ ${provider}: Anthropic messages response received`)
			})
		})
	})

	describe("AWS Bedrock Provider", () => {
		it("should handle ListFoundationModels operation", async () => {
			const port = providerPorts.bedrock
			const response = await fetch(`http://127.0.0.1:${port}/`, {
				method: "POST",
				headers: { 
					"Content-Type": "application/x-amz-json-1.1",
					"X-Amz-Target": "AmazonBedrockControlPlaneService.ListFoundationModels"
				},
				body: JSON.stringify({})
			})
			
			expect(response.ok).to.be.true
			const data = await response.json()
			expect(data).to.have.property("modelSummaries")
			expect(Array.isArray(data.modelSummaries)).to.be.true
			
			console.log(`✅ Bedrock: Found ${data.modelSummaries.length} foundation models`)
		})
	})

	describe("Google Vertex AI Provider", () => {
		it("should handle publisher models endpoint", async () => {
			const port = providerPorts.vertex
			const response = await fetch(`http://127.0.0.1:${port}/v1/projects/test-project/locations/us-central1/publishers/anthropic/models`)
			
			expect(response.ok).to.be.true
			const data = await response.json()
			expect(data).to.have.property("models")
			expect(Array.isArray(data.models)).to.be.true
			
			console.log(`✅ Vertex: Found ${data.models.length} publisher models`)
		})

		it("should handle foundation models endpoint", async () => {
			const port = providerPorts.vertex
			const response = await fetch(`http://127.0.0.1:${port}/v1/projects/test-project/locations/us-central1/models`)
			
			expect(response.ok).to.be.true
			const data = await response.json()
			expect(data).to.have.property("models")
			expect(Array.isArray(data.models)).to.be.true
			
			console.log(`✅ Vertex: Found ${data.models.length} foundation models`)
		})
	})

	describe("Ollama Provider", () => {
		it("should handle /api/tags endpoint", async () => {
			const port = providerPorts.ollama
			const response = await fetch(`http://127.0.0.1:${port}/api/tags`)
			
			expect(response.ok).to.be.true
			const data = await response.json()
			expect(data).to.have.property("models")
			expect(Array.isArray(data.models)).to.be.true
			
			console.log(`✅ Ollama: Found ${data.models.length} local models`)
		})

		it("should handle /api/chat endpoint", async () => {
			const port = providerPorts.ollama
			const response = await fetch(`http://127.0.0.1:${port}/api/chat`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model: "llama2",
					messages: [{ role: "user", content: "Hello" }],
					stream: false
				})
			})
			
			expect(response.ok).to.be.true
			const data = await response.json()
			expect(data).to.have.property("message")
			expect(data.message).to.have.property("role", "assistant")
			
			console.log(`✅ Ollama: Chat response received`)
		})
	})

	describe("Performance Test", () => {
		it("should handle concurrent requests to all providers", async () => {
			const startTime = Date.now()
			
			const providers = Object.keys(providerPorts)
			const promises = providers.map(async provider => {
				try {
					const port = providerPorts[provider]
					const response = await fetch(`http://127.0.0.1:${port}/v1/models`)
					return {
						provider,
						success: response.ok,
						status: response.status,
						modelCount: response.ok ? (await response.json()).data?.length || 0 : 0
					}
				} catch (error) {
					return {
						provider,
						success: false,
						error: error.message
					}
				}
			})
			
			const results = await Promise.all(promises)
			const endTime = Date.now()
			const totalTime = endTime - startTime
			
			console.log(`\n🚀 Concurrent Test Results (${totalTime}ms):`)
			results.forEach(result => {
				if (result.success) {
					console.log(`   ✅ ${result.provider}: ${result.modelCount} models`)
				} else {
					console.log(`   ❌ ${result.provider}: ${result.error || "Failed"}`)
				}
			})
			
			// At least 6 providers should respond successfully
			const successCount = results.filter(r => r.success).length
			expect(successCount).to.be.at.least(6)
			
			// Should complete within 10 seconds
		expect(totalTime).to.be.lessThan(10000)
		})
	})
})
