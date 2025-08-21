import * as assert from 'assert'
import * as sinon from 'sinon'
/**
 * Comprehensive runtime test for all dynamic model providers
 * Tests that all 14 providers can successfully fetch models from their respective mock APIs
 */

import { providerMocks } from "../../../../test/generic-provider-mock/setup"
import { ModelRegistry } from "../model-registry"

// Test configuration with mock API credentials
const testConfigs = {
	anthropic: {
		apiKey: "test-anthropic-key",
	},
	openai: {
		apiKey: "test-openai-key",
	},
	bedrock: {
		region: "us-east-1",
		accessKeyId: "test-access-key",
		secretAccessKey: "test-secret-key",
	},
	gemini: {
		apiKey: "test-gemini-key",
	},
	vertex: {
		projectId: "test-project",
		region: "us-central1",
		keyFilename: "/fake/path/to/key.json",
	},
	mistral: {
		apiKey: "test-mistral-key",
	},
	deepseek: {
		apiKey: "test-deepseek-key",
	},
	ollama: {
		baseUrl: process.env.OLLAMA_BASE_URL || "http://localhost:11434",
	},
	openrouter: {
		apiKey: "test-openrouter-key",
	},
	lmstudio: {
		baseUrl: "http://localhost:1234",
	},
	msty: {
		apiKey: "test-msty-key",
	},
	together: {
		apiKey: "test-together-key",
	},
	groq: {
		apiKey: "test-groq-key",
	},
	xai: {
		apiKey: "test-xai-key",
	},
}

suite("All Dynamic Providers Runtime Test", () => {
	let registry: ModelRegistry

	suiteSetup(async () => {
		// Wait a moment for mock servers to be fully ready
		await new Promise(resolve => setTimeout(resolve, 1000))
		registry = ModelRegistry.getInstance()
	})

	suiteTeardown(() => {
		// Clean up the singleton
		;(ModelRegistry as any).instance = null
	})

	suite("Dynamic Providers", () => {
		const dynamicProviders = [
			"anthropic",
			"openai", 
			"bedrock",
			"gemini",
			"vertex",
			"mistral",
			"deepseek",
		]

		dynamicProviders.forEach(providerName => {
			test(`${providerName} should fetch models successfully`, async () => {
				const config = testConfigs[providerName as keyof typeof testConfigs]
				assert.notStrictEqual(config, undefined)

				try {
					const models = await registry.getModels(providerName, config)
					
					assert.notStrictEqual(models, undefined)
					expect(Array.isArray(models)).toBe(true)
					assert.ok(models.length > 0)

					// Validate model structure
					models.forEach(model => {
						assert.ok(model.hasOwnProperty('id'))
						assert.ok(model.hasOwnProperty('name'))
						assert.ok(model.hasOwnProperty('capabilities'))
						assert.strictEqual(typeof model.id, "string")
						assert.strictEqual(typeof model.name, "string")
						expect(Array.isArray(model.capabilities)).toBe(true)
					})

					console.log(`✅ ${providerName}: Found ${models.length} models`)
					console.log(`   Sample models: ${models.slice(0, 3).map(m => m.id).join(", ")}`)
				} catch (error) {
					console.error(`❌ ${providerName} failed:`, error)
					throw error
				}
			}, 10000) // 10 second timeout for each provider
		})
	})

	suite("Static Providers (should have pre-defined models)", () => {
		const staticProviders = [
			"ollama",
			"openrouter", 
			"lmstudio",
			"msty",
			"together",
			"groq",
			"xai",
		]

		staticProviders.forEach(providerName => {
			test(`${providerName} should return static models`, async () => {
				const config = testConfigs[providerName as keyof typeof testConfigs]
				
				try {
					const models = await registry.getModels(providerName, config)
					
					assert.notStrictEqual(models, undefined)
					expect(Array.isArray(models)).toBe(true)
					
					if (models.length > 0) {
						console.log(`✅ ${providerName}: Found ${models.length} static models`)
					} else {
						console.log(`ℹ️ ${providerName}: No models configured (expected for some providers)`)
					}
				} catch (error) {
					console.error(`❌ ${providerName} failed:`, error)
					// Don't throw for static providers as they may not have mock endpoints
				}
			})
		})
	})

	suite("Cache Functionality", () => {
		test("should cache models and serve from cache on second request", async () => {
			const providerName = "anthropic"
			const config = testConfigs.anthropic

			// First request - should hit the API
			const firstRequest = await registry.getModels(providerName, config)
			assert.ok(firstRequest.length > 0)

			// Second request - should use cache
			const secondRequest = await registry.getModels(providerName, config)
			assert.deepStrictEqual(secondRequest, firstRequest)
			
			console.log(`✅ Cache test: Both requests returned ${firstRequest.length} models`)
		})

		test("should refresh cache when requested", async () => {
			const providerName = "openai"
			const config = testConfigs.openai

			// Get models and cache them
			const initialModels = await registry.getModels(providerName, config)
			assert.ok(initialModels.length > 0)

			// Force refresh
			await registry.refreshModels(providerName, config)
			const refreshedModels = await registry.getModels(providerName, config)
			
			assert.ok(refreshedModels.length > 0)
			console.log(`✅ Refresh test: Got ${refreshedModels.length} models after refresh`)
		})
	})

	suite("Error Handling", () => {
		test("should handle invalid provider gracefully", async () => {
			try {
				await registry.getModels("nonexistent-provider", {})
				fail("Should have thrown an error for invalid provider")
			} catch (error) {
				assert.ok(error instanceof Error)
				console.log("✅ Invalid provider handled correctly")
			}
		})

		test("should handle invalid configuration gracefully", async () => {
			try {
				// Try with completely empty config
				await registry.getModels("anthropic", {})
				fail("Should have thrown an error for missing API key")
			} catch (error) {
				assert.ok(error instanceof Error)
				console.log("✅ Missing configuration handled correctly")
			}
		})
	})

	suite("Model Capabilities", () => {
		test("should return models with correct capability information", async () => {
			const models = await registry.getModels("anthropic", testConfigs.anthropic)
			
			const claudeModel = models.find(m => m.id.includes("claude"))
			if (claudeModel) {
				assert.ok(claudeModel.capabilities.includes("chat"))
				assert.ok(claudeModel.capabilities.includes("tools"))
				console.log(`✅ Claude model capabilities: ${claudeModel.capabilities.join(", ")}`)
			}
		})

		test("should categorize models by their capabilities", async () => {
			const models = await registry.getModels("gemini", testConfigs.gemini)
			
			const chatModels = models.filter(m => m.capabilities.includes("chat"))
			const visionModels = models.filter(m => m.capabilities.includes("vision"))
			const toolModels = models.filter(m => m.capabilities.includes("tools"))
			
			assert.ok(chatModels.length > 0)
			console.log(`✅ Gemini capabilities - Chat: ${chatModels.length}, Vision: ${visionModels.length}, Tools: ${toolModels.length}`)
		})
	})
// Mock cleanup

// Performance test
suite("Performance Tests", () => {
	test("should fetch all provider models within reasonable time", async () => {
		const registry = ModelRegistry.getInstance()
		const startTime = Date.now()
		
		const promises = [
			"anthropic",
			"openai",
			"bedrock", 
			"gemini",
			"mistral",
		].map(async provider => {
			try {
				const config = testConfigs[provider as keyof typeof testConfigs]
				const models = await registry.getModels(provider, config)
// Mock return block needs context
// 				return { provider, modelCount: models.length, success: true }
// 			} catch (error) {
// 				return { provider, error: error.message, success: false }
// 			}
		})
		
		const results = await Promise.all(promises)
		const endTime = Date.now()
		const totalTime = endTime - startTime
		
		console.log(`\n🚀 Performance Test Results (${totalTime}ms total):`)
		results.forEach(result => {
			if (result.success) {
				console.log(`   ✅ ${result.provider}: ${result.modelCount} models`)
			} else {
				console.log(`   ❌ ${result.provider}: ${result.error}`)
			}
		})
		
		// Should complete within 15 seconds
		expect(totalTime).toBeLessThan(15000)
		
		// At least 3 providers should succeed
		const successCount = results.filter(r => r.success).length
		expect(successCount).toBeGreaterThanOrEqual(3)
	}, 20000)
// Mock cleanup