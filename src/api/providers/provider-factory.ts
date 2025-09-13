/**
 * Provider Factory with Dynamic Model Support
 * 
 * Creates provider instances with dynamic model information extraction
 * instead of relying on hardcoded model definitions.
 */

import { ApiHandlerOptions, ProviderName } from "../../shared/api"
import { SingleCompletionHandler } from "../index"
import { modelRegistry } from "./model-registry"

// Dynamic providers
import { DynamicAnthropicHandler } from "./anthropic-dynamic"
import { AnthropicModelProvider } from "./anthropic-model-provider"
import { OpenAIModelProvider } from "./openai-model-provider"

// Legacy providers (to be migrated)
import { OpenAiHandler } from "./openai"
import { VertexHandler } from "./vertex"
import { BedrockHandler } from "./bedrock"
import { OllamaHandler } from "./ollama"

/**
 * Provider factory that creates handlers with dynamic model support
 */
export class ProviderFactory {
	private static instance: ProviderFactory | null = null
	private initialized = false
	
	/**
	 * Get singleton instance
	 */
	static getInstance(): ProviderFactory {
		if (!ProviderFactory.instance) {
			ProviderFactory.instance = new ProviderFactory()
		}
		return ProviderFactory.instance
	}
	
	/**
	 * Initialize the factory and register model providers
	 */
	initialize(options?: {
		anthropicApiKey?: string
		openAiApiKey?: string
		anthropicBaseUrl?: string
		openAiBaseUrl?: string
	}): void {
		if (this.initialized) {
			return
		}
		
		// Register Anthropic model provider
		if (!modelRegistry.hasProvider("anthropic")) {
			const anthropicProvider = new AnthropicModelProvider(
				options?.anthropicApiKey,
				options?.anthropicBaseUrl
			)
			modelRegistry.registerProvider("anthropic", anthropicProvider)
		}
		
		// Register OpenAI model provider
		if (!modelRegistry.hasProvider("openai")) {
			const openAiProvider = new OpenAIModelProvider(
				options?.openAiApiKey,
				options?.openAiBaseUrl
			)
			modelRegistry.registerProvider("openai", openAiProvider)
		}
		
		// TODO: Register other providers as they are implemented
		// - Vertex
		// - Bedrock
		// - Ollama
		// - Mistral
		// - Glama
		
		this.initialized = true
	}
	
	/**
	 * Create a provider handler
	 */
createHandler(
    providerName: ProviderName,
    options: ApiHandlerOptions
): SingleCompletionHandler {
		// Ensure factory is initialized
		this.initialize({
			anthropicApiKey: options.apiKey,
			openAiApiKey: options.openAiApiKey,
			anthropicBaseUrl: options.anthropicBaseUrl,
			openAiBaseUrl: options.openAiBaseUrl
		})
		
		switch (providerName) {
			case "anthropic":
				// Use dynamic handler for Anthropic
				return new DynamicAnthropicHandler(options)
			
			case "openai":
            // TODO: Create DynamicOpenAIHandler; for now use legacy handler
            const OpenAiCtor = OpenAiHandler as unknown as new (o: ApiHandlerOptions) => SingleCompletionHandler
            return new OpenAiCtor(options)
			
			case "vertex":
            // TODO: Create DynamicVertexHandler
            const VertexCtor = VertexHandler as unknown as new (o: ApiHandlerOptions) => SingleCompletionHandler
            return new VertexCtor(options)
			
			case "bedrock":
            // TODO: Create DynamicBedrockHandler
            const BedrockCtor = BedrockHandler as unknown as new (o: ApiHandlerOptions) => SingleCompletionHandler
            return new BedrockCtor(options)
			
			case "ollama":
            // TODO: Create DynamicOllamaHandler
            const OllamaCtor = OllamaHandler as unknown as new (o: ApiHandlerOptions) => SingleCompletionHandler
            return new OllamaCtor(options)
			
			// Add other providers as needed
			default:
				throw new Error(`Unsupported provider: ${providerName}`)
		}
	}
	
	/**
	 * Get available models for a provider
	 */
	async getAvailableModels(providerName: string) {
		this.initialize()
		return modelRegistry.getModels(providerName)
	}
	
	/**
	 * Get model info for a specific model
	 */
	async getModelInfo(providerName: string, modelId: string) {
		this.initialize()
		return modelRegistry.getModelInfo(providerName, modelId)
	}
	
	/**
	 * Get default model for a provider
	 */
	async getDefaultModel(providerName: string) {
		this.initialize()
		return modelRegistry.getDefaultModelId(providerName)
	}
	
	/**
	 * Refresh model list for a provider
	 */
	async refreshModels(providerName: string) {
		this.initialize()
		return modelRegistry.getModels(providerName, true)
	}
	
	/**
	 * Clear all caches
	 */
	clearCaches() {
		modelRegistry.clearCache()
	}
	
	/**
	 * Reset factory (for testing)
	 */
	static reset() {
		ProviderFactory.instance = null
		modelRegistry.clearCache()
	}
}

/**
 * Export singleton instance
 */
export const providerFactory = ProviderFactory.getInstance()

/**
 * Helper function to get model info across all providers
 */
export async function findModelAcrossProviders(modelId: string): Promise<{
	provider: string
	info: import("../../schemas").ModelInfo
} | null> {
	const factory = ProviderFactory.getInstance()
	factory.initialize()
	
        const providers = modelRegistry.getProviderNames()

        const infos = await Promise.all(
                providers.map(async (provider) => ({
                        provider,
                        info: await modelRegistry.getModelInfo(provider, modelId),
                })),
        )

        return infos.find(({ info }) => info) || null
}

/**
 * Helper to migrate from static model selection
 */
export async function selectModel(
	providerName: string,
	preferredModelId?: string
): Promise<string> {
	const factory = ProviderFactory.getInstance()
	factory.initialize()
	
	if (preferredModelId) {
		// Check if the model exists
		const info = await modelRegistry.getModelInfo(providerName, preferredModelId)
		if (info) {
			return preferredModelId
		}
		
		console.warn(`Model ${preferredModelId} not found for ${providerName}, using default`)
	}
	
	// Return default model
	return modelRegistry.getDefaultModelId(providerName)
}
