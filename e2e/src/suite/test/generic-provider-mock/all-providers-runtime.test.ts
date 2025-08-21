import * as assert from 'assert'
import * as sinon from 'sinon'

/**
 * Test suite for all dynamic model provider mock servers
 * Tests that our enhanced mock server can simulate all provider APIs
 */

suite("All Providers Runtime Test", () => {
	let providerPorts: Record<string, number>

	suiteSetup(() => {
		// Get the ports stored by our global setup
		providerPorts = (globalThis as any).__PROVIDER_PORTS__ || {}
		console.log("📋 Available provider ports:", providerPorts)

	suite("Mock Server Availability", () => {
		test("should have all 8 provider mock servers running", () => {
			assert.ok(providerPorts !== undefined)
			
			const expectedProviders = [
				"openai", 
				"anthropic", 
				"bedrock", 
				"gemini", 
				"vertex", 
				"mistral", 
				"deepseek", 
				"ollama"

			expectedProviders.forEach(provider => {
				assert.ok(providerPorts[provider] !== undefined)
				assert.strictEqual(typeof providerPorts[provider], "number")
				assert.ok(providerPorts[provider] > 10000)
			
			console.log(`✅ All ${expectedProviders.length} provider mock servers are running`)

	suite("OpenAI-Compatible Providers", () => {
		const openaiCompatible = ["openai", "bedrock", "gemini", "mistral", "deepseek", "ollama"]

		openaiCompatible.forEach(provider => {
			test(`${provider} should respond to /v1/models endpoint`, async () => {
				const port = providerPorts[provider]
				assert.ok(port !== undefined)
				
				const response = await fetch(`http://127.0.0.1:${port}/v1/models`)
				assert.strictEqual(response.ok, true)
				
				const data = await response.json()
				expect(data).toHaveProperty("data")
				expect(Array.isArray(data.data)).toBe(true)
				assert.ok(data.data.length > 0)
				
				console.log(`✅ ${provider}: Found ${data.data.length} models`)
				console.log(`   Sample: ${data.data[0].id}`)

			test(`${provider} should handle chat completions`, async () => {
				const port = providerPorts[provider]
				const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({
						model: "test-model",
						messages: [{ role: "user", content: "Hello" }]
				
				assert.strictEqual(response.ok, true)
				const data = await response.json()
				expect(data).toHaveProperty("choices")
				expect(data.choices[0]).toHaveProperty("message")
				
				console.log(`✅ ${provider}: Chat completion response received`)

	suite("Anthropic-Compatible Providers", () => {
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
				
				assert.strictEqual(response.ok, true)
				const data = await response.json()
				expect(data).toHaveProperty("content")
				expect(data).toHaveProperty("role", "assistant")
				
				console.log(`✅ ${provider}: Anthropic messages response received`)

	suite("AWS Bedrock Provider", () => {
		test("should handle ListFoundationModels operation", async () => {
			const port = providerPorts.bedrock
			const response = await fetch(`http://127.0.0.1:${port}/`, {
				method: "POST",
				headers: { 
					"Content-Type": "application/x-amz-json-1.1",
					"X-Amz-Target": "AmazonBedrockControlPlaneService.ListFoundationModels"
				},
				body: JSON.stringify({})
			
			assert.strictEqual(response.ok, true)
			const data = await response.json()
			expect(data).toHaveProperty("modelSummaries")
			expect(Array.isArray(data.modelSummaries)).toBe(true)
			
			console.log(`✅ Bedrock: Found ${data.modelSummaries.length} foundation models`)

	suite("Google Vertex AI Provider", () => {
		test("should handle publisher models endpoint", async () => {
			const port = providerPorts.vertex
			const response = await fetch(`http://127.0.0.1:${port}/v1/projects/test-project/locations/us-central1/publishers/anthropic/models`)
			
			assert.strictEqual(response.ok, true)
			const data = await response.json()
			expect(data).toHaveProperty("models")
			expect(Array.isArray(data.models)).toBe(true)
			
			console.log(`✅ Vertex: Found ${data.models.length} publisher models`)

		test("should handle foundation models endpoint", async () => {
			const port = providerPorts.vertex
			const response = await fetch(`http://127.0.0.1:${port}/v1/projects/test-project/locations/us-central1/models`)
			
			assert.strictEqual(response.ok, true)
			const data = await response.json()
			expect(data).toHaveProperty("models")
			expect(Array.isArray(data.models)).toBe(true)
			
			console.log(`✅ Vertex: Found ${data.models.length} foundation models`)

	suite("Ollama Provider", () => {
		test("should handle /api/tags endpoint", async () => {
			const port = providerPorts.ollama
			const response = await fetch(`http://127.0.0.1:${port}/api/tags`)
			
			assert.strictEqual(response.ok, true)
			const data = await response.json()
			expect(data).toHaveProperty("models")
			expect(Array.isArray(data.models)).toBe(true)
			
			console.log(`✅ Ollama: Found ${data.models.length} local models`)

		test("should handle /api/chat endpoint", async () => {
			const port = providerPorts.ollama
			const response = await fetch(`http://127.0.0.1:${port}/api/chat`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					model: "llama2",
					messages: [{ role: "user", content: "Hello" }],
					stream: false
			
			assert.strictEqual(response.ok, true)
			const data = await response.json()
			expect(data).toHaveProperty("message")
			expect(data.message).toHaveProperty("role", "assistant")
			
			console.log(`✅ Ollama: Chat response received`)

	suite("Performance Test", () => {
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

				} catch (error) {
					return {
						provider,
						success: false,
						error: error.message

			const results = await Promise.all(promises)
			const endTime = Date.now()
			const totalTime = endTime - startTime
			
			console.log(`\n🚀 Concurrent Test Results (${totalTime}ms):`)
			results.forEach(result => {
				if (result.success) {
					console.log(`   ✅ ${result.provider}: ${result.modelCount} models`)
				} else {
					console.log(`   ❌ ${result.provider}: ${result.error || "Failed"}`)

			// At least 6 providers should respond successfully
			const successCount = results.filter(r => r.success).length
			assert.ok(successCount >= 6)
			
			// Should complete within 10 seconds
			assert.ok(totalTime < 10000)
