import * as assert from 'assert'
import { expect } from 'chai'
import * as sinon from 'sinon'
/**
 * Model Registry Tests
 * Tests for the dynamic model information extraction system
 */

import { 
	ModelRegistry, 
	ModelProvider, 
	ModelListing,
	StaticModelProvider,
	migrateStaticModels
} from "../model-registry"
import { ModelInfo } from "../../../schemas"

suite("ModelRegistry", () => {
	let registry: ModelRegistry
	let mockProvider: sinon.SinonStubbedInstance<ModelProvider>
	
	setup(() => {
		// Reset registry
		ModelRegistry.reset()
		registry = ModelRegistry.getInstance()
		
		// Create mock provider
		mockProvider = {
			listModels: sinon.stub(),
			getModelInfo: sinon.stub(),
			getDefaultModelId: sinon.stub()
		}
	})
	
	teardown(() => {
		sinon.restore()
		ModelRegistry.reset()
	})
	
	suite("Singleton Pattern", () => {
		test("should return the same instance", () => {
			const instance1 = ModelRegistry.getInstance()
			const instance2 = ModelRegistry.getInstance()
			assert.strictEqual(instance1, instance2)
		})
		
		test("should create new instance after reset", () => {
			const instance1 = ModelRegistry.getInstance()
			ModelRegistry.reset()
			const instance2 = ModelRegistry.getInstance()
			assert.notStrictEqual(instance1, instance2)
		})
	})
	
	suite("Provider Registration", () => {
		test("should register a provider", () => {
			registry.registerProvider("test", mockProvider)
			expect(registry.hasProvider("test")).to.be.true
		})
		
		test("should unregister a provider", () => {
			registry.registerProvider("test", mockProvider)
			registry.unregisterProvider("test")
			expect(registry.hasProvider("test")).to.be.false
		})
		
		test("should list registered providers", () => {
			registry.registerProvider("provider1", mockProvider)
			registry.registerProvider("provider2", mockProvider)
			
			const providers = registry.getProviderNames()
			assert.ok(providers.includes("provider1"))
			assert.ok(providers.includes("provider2"))
		})
	})
	
	suite("Model Listing", () => {
		const mockModels: ModelListing[] = [
			{
				id: "model-1",
				info: {
					maxTokens: 4096,
					contextWindow: 8192,
					supportsImages: true,
					inputPrice: 1,
					outputPrice: 2
				} as ModelInfo,
				displayName: "Model 1"
			},
			{
				id: "model-2",
				info: {
					maxTokens: 8192,
					contextWindow: 16384,
					supportsImages: false,
					inputPrice: 2,
					outputPrice: 4
				} as ModelInfo,
				displayName: "Model 2"
			}
		]
		
		setup(() => {
			mockProvider.listModels.resolves(mockModels)
			registry.registerProvider("test", mockProvider)
		})
		
		test("should fetch models from provider", async () => {
			const models = await registry.getModels("test")
			
			assert.deepStrictEqual(models, mockModels)
			assert.strictEqual(mockProvider.listModels.callCount, 1)
		})
		
		test("should cache models", async () => {
			// First call
			await registry.getModels("test")
			// Second call should use cache
			await registry.getModels("test")
			
			assert.strictEqual(mockProvider.listModels.callCount, 1)
		})
		
		test("should force refresh when requested", async () => {
			// First call
			await registry.getModels("test")
			// Force refresh
			await registry.getModels("test", true)
			
			assert.strictEqual(mockProvider.listModels.callCount, 2)
		})
		
		test("should return empty array for unregistered provider", async () => {
			const models = await registry.getModels("nonexistent")
			assert.deepStrictEqual(models, [])
		})
		
		test("should use expired cache on provider error", async () => {
			// First successful call
			await registry.getModels("test")
			
			// Make provider fail
			mockProvider.listModels.rejects(new Error("API error"))
			
			// Force refresh should return cached data
			const models = await registry.getModels("test", true)
			assert.deepStrictEqual(models, mockModels)
		})
	})
	
	suite("Model Info", () => {
		const mockModels: ModelListing[] = [
			{
				id: "model-1",
				info: {
					maxTokens: 4096,
					contextWindow: 8192
				} as ModelInfo
			}
		]
		
		setup(() => {
			mockProvider.listModels.resolves(mockModels)
			mockProvider.getModelInfo.resolves({
				maxTokens: 2048,
				contextWindow: 4096
			} as ModelInfo)
			registry.registerProvider("test", mockProvider)
		})
		
		test("should get model info from cached list", async () => {
			const info = await registry.getModelInfo("test", "model-1")
			
			assert.deepStrictEqual(info, mockModels[0].info)
			assert.ok(!mockProvider.getModelInfo.called)
		})
		
		test("should query provider for unknown model", async () => {
			const info = await registry.getModelInfo("test", "unknown-model")
			
			assert.deepStrictEqual(info, {
				maxTokens: 2048,
				contextWindow: 4096
			})
			assert.ok(mockProvider.getModelInfo.calledWith("unknown-model"))
		})
		
		test("should return null for nonexistent model", async () => {
			mockProvider.getModelInfo.resolves(null)
			
			const info = await registry.getModelInfo("test", "nonexistent")
			assert.strictEqual(info, null)
		})
	})
	
	suite("Default Model", () => {
		setup(() => {
			mockProvider.getDefaultModelId.returns("default-model")
			mockProvider.listModels.resolves([
				{ id: "model-1", info: {} as ModelInfo },
				{ id: "model-2", info: {} as ModelInfo }
			])
			registry.registerProvider("test", mockProvider)
		})
		
		test("should get default model from provider", async () => {
			const modelId = await registry.getDefaultModelId("test")
			assert.strictEqual(modelId, "default-model")
		})
		
		test("should fallback to first model on error", async () => {
			mockProvider.getDefaultModelId.callsFake(() => {
				throw new Error("Error")
			})
			
			const modelId = await registry.getDefaultModelId("test")
			assert.strictEqual(modelId, "model-1")
		})
		
		test("should return empty string for unregistered provider", async () => {
			const modelId = await registry.getDefaultModelId("nonexistent")
			assert.strictEqual(modelId, "")
		})
	})
	
	suite("Cache Management", () => {
		setup(() => {
			mockProvider.listModels.resolves([])
			registry.registerProvider("test1", mockProvider)
			registry.registerProvider("test2", mockProvider)
		})
		
		test("should clear cache for specific provider", async () => {
			await registry.getModels("test1")
			await registry.getModels("test2")
			
			registry.clearCache("test1")
			
			// test1 should refetch, test2 should use cache
			await registry.getModels("test1")
			await registry.getModels("test2")
			
			// test1: 2 calls (initial + after clear)
			// test2: 1 call (initial only)
			assert.strictEqual(mockProvider.listModels.callCount, 3)
		})
		
		test("should clear all caches", async () => {
			await registry.getModels("test1")
			await registry.getModels("test2")
			
			registry.clearCache()
			
			await registry.getModels("test1")
			await registry.getModels("test2")
			
			// Each provider: 2 calls (initial + after clear)
			assert.strictEqual(mockProvider.listModels.callCount, 4)
		})
		
		test("should respect custom TTL", async () => {
			// Set very short TTL
			await registry.getModels("test1")
			registry.setCacheTTL("test1", 1) // 1ms TTL
			
			// Wait for TTL to expire
			await new Promise(resolve => setTimeout(resolve, 10))
			
			// Should refetch
			await registry.getModels("test1")
			assert.strictEqual(mockProvider.listModels.callCount, 2)
		})
	})
	
	suite("StaticModelProvider", () => {
		test("should wrap static models", async () => {
			const staticModels = {
				"static-1": {
					maxTokens: 1000,
					contextWindow: 2000,
					supportsImages: false
				} as ModelInfo,
				"static-2": {
					maxTokens: 2000,
					contextWindow: 4000,
					supportsImages: true
				} as ModelInfo
			}
			
			const provider = new StaticModelProvider(staticModels, "static-1")
			
			const models = await provider.listModels()
			assert.strictEqual(models.length, 2)
			assert.strictEqual(models[0].id, "static-1")
			
			const info = await provider.getModelInfo("static-2")
			assert.strictEqual(info?.maxTokens, 2000)
			
			const defaultId = provider.getDefaultModelId()
			assert.strictEqual(defaultId, "static-1")
		})
	})
	
	suite("migrateStaticModels", () => {
		test("should convert static models to listings", () => {
			const staticModels = {
				"model-a": {
					maxTokens: 1000,
					contextWindow: 2000
				} as ModelInfo,
				"model-b": {
					maxTokens: 2000,
					contextWindow: 4000
				} as ModelInfo
			}
			
			const listings = migrateStaticModels(staticModels)
			
			assert.strictEqual(listings.length, 2)
			assert.deepStrictEqual(listings[0], {
				id: "model-a",
				info: staticModels["model-a"],
				displayName: "model-a",
				deprecated: false,
				releaseDate: undefined
			})
		})
	})
// Mock cleanup
