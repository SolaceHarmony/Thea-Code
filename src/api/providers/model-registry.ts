/**
 * Dynamic Model Registry System
 * 
 * This module provides a dynamic model information extraction system that
 * queries providers for their available models rather than using hardcoded
 * definitions. This makes the system more flexible and maintainable.
 */

import { ModelInfo } from "../../schemas"
import { ApiHandlerOptions } from "../../shared/api"
import { AnthropicModelProvider } from "./anthropic-model-provider"
import { OpenAIModelProvider } from "./openai-model-provider"
import { 
	BedrockModelProvider,
	GeminiModelProvider,
	VertexModelProvider,
	MistralModelProvider,
	DeepSeekModelProvider,
} from "./model-providers"

/**
 * Interface for providers to implement dynamic model listing
 */
export interface ModelProvider {
	/**
	 * List all available models from the provider
	 * @returns Promise resolving to array of model IDs and their info
	 */
	listModels(): Promise<ModelListing[]>
	
	/**
	 * Get detailed information about a specific model
	 * @param modelId The model ID to get info for
	 * @returns Promise resolving to model info or null if not found
	 */
	getModelInfo(modelId: string): Promise<ModelInfo | null>
	
	/**
	 * Get the default model ID for this provider
	 * @returns The default model ID
	 */
	getDefaultModelId(): string | Promise<string>

	/** Optional: configure provider with API options */
	configure?(options: ApiHandlerOptions): void
}

/**
 * Model listing entry returned by providers
 */
export interface ModelListing {
	id: string
	info: ModelInfo
	displayName?: string
	deprecated?: boolean
	releaseDate?: string
}

/**
 * Cache entry for model information
 */
interface CacheEntry {
	models: ModelListing[]
	timestamp: number
	ttl: number // Time to live in milliseconds
}

/**
 * Dynamic model registry that manages model information from all providers
 */
export class ModelRegistry {
	private static instance: ModelRegistry | null = null
	private cache = new Map<string, CacheEntry>()
	private providers = new Map<string, ModelProvider>()
	
	// Default cache TTL: 1 hour
	private static readonly DEFAULT_TTL = 60 * 60 * 1000
	
	private constructor() {
		// Initialize with default providers
		this.initializeProviders()
	}
	
	/**
	 * Initialize default model providers
	 */
	private initializeProviders(): void {
		// Register all dynamic providers
		this.registerProvider("anthropic", new AnthropicModelProvider())
		this.registerProvider("openai", new OpenAIModelProvider())
		this.registerProvider("bedrock", new BedrockModelProvider())
		this.registerProvider("gemini", new GeminiModelProvider())
		this.registerProvider("vertex", new VertexModelProvider())
		this.registerProvider("mistral", new MistralModelProvider())
		this.registerProvider("deepseek", new DeepSeekModelProvider())
	}
	
	/**
	 * Get the singleton instance of ModelRegistry
	 */
	static getInstance(): ModelRegistry {
		if (!ModelRegistry.instance) {
			ModelRegistry.instance = new ModelRegistry()
		}
		return ModelRegistry.instance
	}
	
	/**
	 * Configure a provider with API options (keys, endpoints, etc.)
	 * @param providerName Name of the provider to configure
	 * @param options Configuration options for the provider
	 */
	configureProvider(providerName: string, options: ApiHandlerOptions): void {
		const provider = this.providers.get(providerName)
		provider?.configure?.(options)
	}
	
	/**
	 * Register a model provider
	 * @param name Provider name (e.g., "anthropic", "openai")
	 * @param provider Provider instance implementing ModelProvider interface
	 */
	registerProvider(name: string, provider: ModelProvider): void {
		this.providers.set(name, provider)
	}
	
	/**
	 * Get a registered provider by name
	 * @param name Provider name
	 * @returns ModelProvider instance or undefined if not found
	 */
	getProvider(name: string): ModelProvider | undefined {
		return this.providers.get(name)
	}
	
	/**
	 * Get all available models from a provider
	 * @param providerName Name of the provider
	 * @param forceRefresh Force refresh bypassing cache
	 * @returns Promise resolving to array of model listings
	 */
	async getModels(providerName: string, forceRefresh = false): Promise<ModelListing[]> {
		// Check cache first
		if (!forceRefresh) {
			const cached = this.cache.get(providerName)
			if (cached && Date.now() - cached.timestamp < cached.ttl) {
				return cached.models
			}
		}
		
		// Get provider
		const provider = this.providers.get(providerName)
		if (!provider) {
			console.warn(`Provider ${providerName} not registered`)
			return []
		}
		
		try {
			// Fetch models from provider
			const models = await provider.listModels()
			
			// Update cache
			this.cache.set(providerName, {
				models,
				timestamp: Date.now(),
				ttl: ModelRegistry.DEFAULT_TTL
			})
			
			return models
		} catch (error) {
			console.error(`Failed to fetch models from ${providerName}:`, error)
			
			// Return cached data if available, even if expired
			const cached = this.cache.get(providerName)
			if (cached) {
				console.warn(`Using expired cache for ${providerName}`)
				return cached.models
			}
			
			return []
		}
	}
	
	/**
	 * Get information about a specific model
	 * @param providerName Name of the provider
	 * @param modelId Model ID
	 * @returns Promise resolving to model info or null
	 */
	async getModelInfo(providerName: string, modelId: string): Promise<ModelInfo | null> {
		// First check if model is in cached list
		const models = await this.getModels(providerName)
		const model = models.find(m => m.id === modelId)
		if (model) {
			return model.info
		}
		
		// If not in list, try direct query (for dynamic or unlisted models)
		const provider = this.providers.get(providerName)
		if (provider) {
			try {
				return await provider.getModelInfo(modelId)
			} catch (error) {
				console.error(`Failed to get model info for ${modelId} from ${providerName}:`, error)
			}
		}
		
		return null
	}
	
	/**
	 * Get the default model for a provider
	 * @param providerName Name of the provider
	 * @returns Default model ID or empty string
	 */
	async getDefaultModelId(providerName: string): Promise<string> {
		const provider = this.providers.get(providerName)
		if (!provider) {
			return ""
		}
		
		try {
			return provider.getDefaultModelId()
		} catch (error) {
			console.error(`Failed to get default model for ${providerName}:`, error)
			
			// Fallback to first available model
			const models = await this.getModels(providerName)
			return models.length > 0 ? models[0].id : ""
		}
	}
	
	/**
	 * Clear cache for a specific provider or all providers
	 * @param providerName Optional provider name, clears all if not specified
	 */
	clearCache(providerName?: string): void {
		if (providerName) {
			this.cache.delete(providerName)
		} else {
			this.cache.clear()
		}
	}
	
	/**
	 * Set cache TTL for a provider
	 * @param providerName Provider name
	 * @param ttl Time to live in milliseconds
	 */
	setCacheTTL(providerName: string, ttl: number): void {
		const cached = this.cache.get(providerName)
		if (cached) {
			cached.ttl = ttl
		}
	}
	
	/**
	 * Get all registered provider names
	 * @returns Array of provider names
	 */
	getProviderNames(): string[] {
		return Array.from(this.providers.keys())
	}
	
	/**
	 * Check if a provider is registered
	 * @param providerName Provider name
	 * @returns True if provider is registered
	 */
	hasProvider(providerName: string): boolean {
		return this.providers.has(providerName)
	}
	
	/**
	 * Unregister a provider
	 * @param providerName Provider name
	 */
	unregisterProvider(providerName: string): void {
		this.providers.delete(providerName)
		this.cache.delete(providerName)
	}
	
	/**
	 * Reset the registry (for testing)
	 */
	static reset(): void {
		if (ModelRegistry.instance) {
			ModelRegistry.instance.providers.clear()
			ModelRegistry.instance.cache.clear()
		}
		ModelRegistry.instance = null
	}
}

/**
 * Helper function to migrate from static model maps to dynamic registry
 * @param staticModels Static model map
 * @returns Array of model listings
 */
export function migrateStaticModels(
	staticModels: Record<string, ModelInfo>
): ModelListing[] {
	return Object.entries(staticModels).map(([id, info]) => ({
		id,
		info,
		displayName: id, // Can be enhanced with better display names
		deprecated: false,
		releaseDate: undefined
	}))
}

/**
 * Create a fallback provider that uses static model definitions
 * This is useful during migration from static to dynamic models
 */
export class StaticModelProvider implements ModelProvider {
	constructor(
		private models: Record<string, ModelInfo>,
		private defaultModelId: string
	) {}
	
	listModels(): Promise<ModelListing[]> {
		return Promise.resolve(migrateStaticModels(this.models))
	}
	
	getModelInfo(modelId: string): Promise<ModelInfo | null> {
		return Promise.resolve(this.models[modelId] || null)
	}
	
	getDefaultModelId(): string {
		return this.defaultModelId
	}
}

/**
 * Export singleton instance for convenience
 */
export const modelRegistry = ModelRegistry.getInstance()
