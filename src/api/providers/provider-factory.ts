/**
 * Provider Factory with Dynamic Model Support
 * 
 * Creates provider instances with dynamic model information extraction
 * instead of relying on hardcoded model definitions.
 */

import { ApiHandlerOptions } from "../../shared/api"
import type { ProviderName } from "../../schemas"
import { SingleCompletionHandler } from "../index"
import { modelRegistry } from "./model-registry"

// Dynamic providers
import { DynamicAnthropicHandler } from "./anthropic-dynamic"
import { DynamicVertexHandler } from "./vertex-dynamic"
import { DynamicBedrockHandler } from "./bedrock-dynamic"
import { DynamicOllamaHandler } from "./ollama-dynamic"
import { DynamicGeminiHandler } from "./gemini-dynamic"
import { DynamicMistralHandler } from "./mistral-dynamic"
import { DynamicDeepSeekHandler } from "./deepseek-dynamic"
import { DynamicGlamaHandler } from "./glama-dynamic"
import { AnthropicModelProvider } from "./anthropic-model-provider"
import { OpenAIModelProvider } from "./openai-model-provider"
import { DynamicOpenAIHandler } from "./openai-dynamic"



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
		
		// Note: Other providers register themselves when their dynamic handler is instantiated
		
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
				// Use dynamic handler for OpenAI
				return new DynamicOpenAIHandler(options)
			
			case "vertex":
				// Use dynamic handler for Vertex
				return new DynamicVertexHandler(options)
			
			case "bedrock":
				// Use dynamic handler for Bedrock
				return new DynamicBedrockHandler(options)
			
			case "ollama":
				// Use dynamic handler for Ollama
				return new DynamicOllamaHandler(options)
			
			case "gemini":
				// Use dynamic handler for Gemini
				return new DynamicGeminiHandler(options)
			
			case "mistral":
				// Use dynamic handler for Mistral
				return new DynamicMistralHandler(options)
			
			case "deepseek":
				// Use dynamic handler for DeepSeek
				return new DynamicDeepSeekHandler(options)
			
			case "glama":
				// Use dynamic handler for Glama
				return new DynamicGlamaHandler(options)
			
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
export async function findModelAcrossProviders(modelId: string): Promise<{ provider: string; info: import("../../schemas").ModelInfo } | null> {
	const factory = ProviderFactory.getInstance()
	factory.initialize()
	for (const provider of modelRegistry.getProviderNames()) {
		const info = await modelRegistry.getModelInfo(provider, modelId)
		if (info) return { provider, info }
	}
	return null
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

