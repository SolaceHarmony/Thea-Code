/**
 * Model Registry Tests
 * Tests for the dynamic model information extraction system
 */

import { describe, expect, it, jest, beforeEach, afterEach } from "@jest/globals"
import { 
	ModelRegistry, 
	ModelProvider, 
	ModelListing,
	StaticModelProvider,
	migrateStaticModels
} from "../model-registry"
import { ModelInfo } from "../../../schemas"

describe("ModelRegistry", () => {
	let registry: ModelRegistry
	let mockProvider: jest.Mocked<ModelProvider>
	
	beforeEach(() => {
		// Reset registry
		ModelRegistry.reset()
		registry = ModelRegistry.getInstance()
		
		// Create mock provider
		mockProvider = {
			listModels: jest.fn(),
			getModelInfo: jest.fn(),
			getDefaultModelId: jest.fn()
		}
	})
	
	afterEach(() => {
		jest.clearAllMocks()
		ModelRegistry.reset()
	})
	
	describe("Singleton Pattern", () => {
		it("should return the same instance", () => {
			const instance1 = ModelRegistry.getInstance()
			const instance2 = ModelRegistry.getInstance()
			expect(instance1).toBe(instance2)
		})
		
		it("should create new instance after reset", () => {
			const instance1 = ModelRegistry.getInstance()
			ModelRegistry.reset()
			const instance2 = ModelRegistry.getInstance()
			expect(instance1).not.toBe(instance2)
		})
	})
	
	describe("Provider Registration", () => {
		it("should register a provider", () => {
			registry.registerProvider("test", mockProvider)
			expect(registry.hasProvider("test")).toBe(true)
		})
		
		it("should unregister a provider", () => {
			registry.registerProvider("test", mockProvider)
			registry.unregisterProvider("test")
			expect(registry.hasProvider("test")).toBe(false)
		})
		
		it("should list registered providers", () => {
			registry.registerProvider("provider1", mockProvider)
			registry.registerProvider("provider2", mockProvider)
			
			const providers = registry.getProviderNames()
			expect(providers).toContain("provider1")
			expect(providers).toContain("provider2")
		})
	})
	
	describe("Model Listing", () => {
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
		
		beforeEach(() => {
			mockProvider.listModels.mockResolvedValue(mockModels)
			registry.registerProvider("test", mockProvider)
		})
		
		it("should fetch models from provider", async () => {
			const models = await registry.getModels("test")
			
			expect(models).toEqual(mockModels)
			expect(mockProvider.listModels).toHaveBeenCalledTimes(1)
		})
		
		it("should cache models", async () => {
			// First call
			await registry.getModels("test")
			// Second call should use cache
			await registry.getModels("test")
			
			expect(mockProvider.listModels).toHaveBeenCalledTimes(1)
		})
		
		it("should force refresh when requested", async () => {
			// First call
			await registry.getModels("test")
			// Force refresh
			await registry.getModels("test", true)
			
			expect(mockProvider.listModels).toHaveBeenCalledTimes(2)
		})
		
		it("should return empty array for unregistered provider", async () => {
			const models = await registry.getModels("nonexistent")
			expect(models).toEqual([])
		})
		
		it("should use expired cache on provider error", async () => {
			// First successful call
			await registry.getModels("test")
			
			// Make provider fail
			mockProvider.listModels.mockRejectedValue(new Error("API error"))
			
			// Force refresh should return cached data
			const models = await registry.getModels("test", true)
			expect(models).toEqual(mockModels)
		})
	})
	
	describe("Model Info", () => {
		const mockModels: ModelListing[] = [
			{
				id: "model-1",
				info: {
					maxTokens: 4096,
					contextWindow: 8192
				} as ModelInfo
			}
		]
		
		beforeEach(() => {
			mockProvider.listModels.mockResolvedValue(mockModels)
			mockProvider.getModelInfo.mockResolvedValue({
				maxTokens: 2048,
				contextWindow: 4096
			} as ModelInfo)
			registry.registerProvider("test", mockProvider)
		})
		
		it("should get model info from cached list", async () => {
			const info = await registry.getModelInfo("test", "model-1")
			
			expect(info).toEqual(mockModels[0].info)
			expect(mockProvider.getModelInfo).not.toHaveBeenCalled()
		})
		
		it("should query provider for unknown model", async () => {
			const info = await registry.getModelInfo("test", "unknown-model")
			
			expect(info).toEqual({
				maxTokens: 2048,
				contextWindow: 4096
			})
			expect(mockProvider.getModelInfo).toHaveBeenCalledWith("unknown-model")
		})
		
		it("should return null for nonexistent model", async () => {
			mockProvider.getModelInfo.mockResolvedValue(null)
			
			const info = await registry.getModelInfo("test", "nonexistent")
			expect(info).toBeNull()
		})
	})
	
	describe("Default Model", () => {
		beforeEach(() => {
			mockProvider.getDefaultModelId.mockReturnValue("default-model")
			mockProvider.listModels.mockResolvedValue([
				{ id: "model-1", info: {} as ModelInfo },
				{ id: "model-2", info: {} as ModelInfo }
			])
			registry.registerProvider("test", mockProvider)
		})
		
		it("should get default model from provider", async () => {
			const modelId = await registry.getDefaultModelId("test")
			expect(modelId).toBe("default-model")
		})
		
		it("should fallback to first model on error", async () => {
			mockProvider.getDefaultModelId.mockImplementation(() => {
				throw new Error("Error")
			})
			
			const modelId = await registry.getDefaultModelId("test")
			expect(modelId).toBe("model-1")
		})
		
		it("should return empty string for unregistered provider", async () => {
			const modelId = await registry.getDefaultModelId("nonexistent")
			expect(modelId).toBe("")
		})
	})
	
	describe("Cache Management", () => {
		beforeEach(() => {
			mockProvider.listModels.mockResolvedValue([])
			registry.registerProvider("test1", mockProvider)
			registry.registerProvider("test2", mockProvider)
		})
		
		it("should clear cache for specific provider", async () => {
			await registry.getModels("test1")
			await registry.getModels("test2")
			
			registry.clearCache("test1")
			
			// test1 should refetch, test2 should use cache
			await registry.getModels("test1")
			await registry.getModels("test2")
			
			// test1: 2 calls (initial + after clear)
			// test2: 1 call (initial only)
			expect(mockProvider.listModels).toHaveBeenCalledTimes(3)
		})
		
		it("should clear all caches", async () => {
			await registry.getModels("test1")
			await registry.getModels("test2")
			
			registry.clearCache()
			
			await registry.getModels("test1")
			await registry.getModels("test2")
			
			// Each provider: 2 calls (initial + after clear)
			expect(mockProvider.listModels).toHaveBeenCalledTimes(4)
		})
		
		it("should respect custom TTL", async () => {
			// Set very short TTL
			await registry.getModels("test1")
			registry.setCacheTTL("test1", 1) // 1ms TTL
			
			// Wait for TTL to expire
			await new Promise(resolve => setTimeout(resolve, 10))
			
			// Should refetch
			await registry.getModels("test1")
			expect(mockProvider.listModels).toHaveBeenCalledTimes(2)
		})
	})
	
	describe("StaticModelProvider", () => {
		it("should wrap static models", async () => {
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
			expect(models).toHaveLength(2)
			expect(models[0].id).toBe("static-1")
			
			const info = await provider.getModelInfo("static-2")
			expect(info?.maxTokens).toBe(2000)
			
			const defaultId = provider.getDefaultModelId()
			expect(defaultId).toBe("static-1")
		})
	})
	
	describe("migrateStaticModels", () => {
		it("should convert static models to listings", () => {
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
			
			expect(listings).toHaveLength(2)
			expect(listings[0]).toEqual({
				id: "model-a",
				info: staticModels["model-a"],
				displayName: "model-a",
				deprecated: false,
				releaseDate: undefined
			})
		})
	})
})