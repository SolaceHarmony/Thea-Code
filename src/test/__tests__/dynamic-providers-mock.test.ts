/**
 * Test suite for all dynamic model provider mock servers
 * Tests that our enhanced mock server can simulate all 14 provider APIs
 */

describe("Dynamic Provider Mock Servers", () => {
	let providerPorts: Record<string, number>

	beforeAll(() => {
		// Get the ports stored by our global setup
		providerPorts = (globalThis as any).__PROVIDER_PORTS__ || {}
		console.log("📋 Available provider ports:", providerPorts)
	})

	describe("Mock Server Availability", () => {
		test("should have all 8 provider mock servers running", () => {
			expect(providerPorts).toBeDefined()
			
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
				expect(providerPorts[provider]).toBeDefined()
				expect(typeof providerPorts[provider]).toBe("number")
				expect(providerPorts[provider]).toBeGreaterThan(10000)
			})
			
			console.log(`✅ All ${expectedProviders.length} provider mock servers are running`)
		})
	})

	describe("OpenAI-Compatible Providers", () => {
		const openaiCompatible = ["openai", "bedrock", "gemini", "mistral", "deepseek", "ollama"]

		openaiCompatible.forEach(provider => {
			test(`${provider} should respond to /v1/models endpoint`, async () => {
				const port = providerPorts[provider]
				expect(port).toBeDefined()
				
				const response = await fetch(`http://127.0.0.1:${port}/v1/models`)
				expect(response.ok).toBe(true)
				
				const data = await response.json()
				expect(data).toHaveProperty("data")
				expect(Array.isArray(data.data)).toBe(true)
				expect(data.data.length).toBeGreaterThan(0)
				
				console.log(`✅ ${provider}: Found ${data.data.length} models`)
				console.log(`   Sample: ${data.data[0].id}`)
			})

			test(`${provider} should handle chat completions`, async () => {
				const port = providerPorts[provider]
				const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({
						model: "test-model",
						messages: [{ role: "user", content: "Hello" }]
					})
				})
				
				expect(response.ok).toBe(true)
				const data = await response.json()
				expect(data).toHaveProperty("choices")
				expect(data.choices[0]).toHaveProperty("message")
				
				console.log(`✅ ${provider}: Chat completion response received`)
			})
		})
	})

	describe("Anthropic-Compatible Providers", () => {
		const anthropicCompatible = ["anthropic", "vertex"]

		anthropicCompatible.forEach(provider => {
			test(`${provider} should handle /v1/messages endpoint`, async () => {
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
				
				expect(response.ok).toBe(true)
				const data = await response.json()
				expect(data).toHaveProperty("content")
				expect(data).toHaveProperty("role", "assistant")
				
				console.log(`✅ ${provider}: Anthropic messages response received`)
			})
		})
	})

	describe("AWS Bedrock Provider", () => {
		test("should handle ListFoundationModels operation", async () => {
			const port = providerPorts.bedrock
			const response = await fetch(`http://127.0.0.1:${port}/`, {
				method: "POST",
				headers: { 
					"Content-Type": "application/x-amz-json-1.1",
					"X-Amz-Target": "AmazonBedrockControlPlaneService.ListFoundationModels"
				},
				body: JSON.stringify({})
			})
			
			expect(response.ok).toBe(true)
			const data = await response.json()
			expect(data).toHaveProperty("modelSummaries")
			expect(Array.isArray(data.modelSummaries)).toBe(true)
			
			console.log(`✅ Bedrock: Found ${data.modelSummaries.length} foundation models`)
		})
	})

	describe("Google Vertex AI Provider", () => {
		test("should handle publisher models endpoint", async () => {
			const port = providerPorts.vertex
			const response = await fetch(`http://127.0.0.1:${port}/v1/projects/test-project/locations/us-central1/publishers/anthropic/models`)
			
			expect(response.ok).toBe(true)
			const data = await response.json()
			expect(data).toHaveProperty("models")
			expect(Array.isArray(data.models)).toBe(true)
			
			console.log(`✅ Vertex: Found ${data.models.length} publisher models`)
		})

		test("should handle foundation models endpoint", async () => {
			const port = providerPorts.vertex
			const response = await fetch(`http://127.0.0.1:${port}/v1/projects/test-project/locations/us-central1/models`)
			
			expect(response.ok).toBe(true)
			const data = await response.json()
			expect(data).toHaveProperty("models")
			expect(Array.isArray(data.models)).toBe(true)
			
			console.log(`✅ Vertex: Found ${data.models.length} foundation models`)
		})
	})

	describe("Ollama Provider", () => {
		test("should handle /api/tags endpoint", async () => {
			const port = providerPorts.ollama
			const response = await fetch(`http://127.0.0.1:${port}/api/tags`)
			
			expect(response.ok).toBe(true)
			const data = await response.json()
			expect(data).toHaveProperty("models")
			expect(Array.isArray(data.models)).toBe(true)
			
			console.log(`✅ Ollama: Found ${data.models.length} local models`)
		})

		test("should handle /api/chat endpoint", async () => {
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
			
			expect(response.ok).toBe(true)
			const data = await response.json()
			expect(data).toHaveProperty("message")
			expect(data.message).toHaveProperty("role", "assistant")
			
			console.log(`✅ Ollama: Chat response received`)
		})
	})

	describe("Streaming Support", () => {
		test("OpenAI streaming should work", async () => {
			const port = providerPorts.openai
			const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model: "gpt-4",
					messages: [{ role: "user", content: "Hello" }],
					stream: true
				})
			})
			
			expect(response.ok).toBe(true)
			expect(response.headers.get("content-type")).toContain("text/event-stream")
			
			const text = await response.text()
			expect(text).toContain("data:")
			expect(text).toContain("[DONE]")
			
			console.log("✅ OpenAI streaming works")
		})

		test("Anthropic streaming should work", async () => {
			const port = providerPorts.anthropic
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
					messages: [{ role: "user", content: "Hello" }],
					stream: true
				})
			})
			
			expect(response.ok).toBe(true)
			expect(response.headers.get("content-type")).toContain("text/event-stream")
			
			const text = await response.text()
			expect(text).toContain("data:")
			expect(text).toContain("[DONE]")
			
			console.log("✅ Anthropic streaming works")
		})
	})

	describe("Performance Test", () => {
		test("should handle concurrent requests to all providers", async () => {
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
			
			// All providers should respond successfully
			const successCount = results.filter(r => r.success).length
			expect(successCount).toBe(providers.length)
			
			// Should complete within 5 seconds
			expect(totalTime).toBeLessThan(5000)
		})
	})
})