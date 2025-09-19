/**
 * Anthropic Dynamic Model Provider
 * 
 * Fetches available models from Anthropic's API dynamically
 * rather than using hardcoded definitions.
 */

import { ModelInfo } from "../../schemas"
import { ModelProvider, ModelListing } from "./model-registry"
import { NeutralAnthropicClient } from "../../services/anthropic"

/**
 * Anthropic model provider that fetches models dynamically
 */
export class AnthropicModelProvider implements ModelProvider {
	private client: NeutralAnthropicClient
	private cachedModels: ModelListing[] | null = null
	
	constructor(apiKey?: string, baseURL?: string) {
		this.client = new NeutralAnthropicClient({
			apiKey: apiKey || "",
			baseURL: baseURL
		})
	}
	
	async listModels(): Promise<ModelListing[]> {
		// If we have cached models, return them
		if (this.cachedModels) {
			return this.cachedModels
		}
		
		try {
			// Attempt to fetch models from API
			// Note: Anthropic doesn't currently have a public models endpoint,
			// so we'll use a combination of known models and capability detection
			const models = await this.fetchModelsFromAPI()
			
			if (models.length > 0) {
				this.cachedModels = models
				return models
			}
		} catch (error) {
			console.warn("Failed to fetch Anthropic models from API, using defaults:", error)
		}
		
		// Fallback to default known models with dynamic capability detection
		return this.getDefaultModels()
	}
	
	async getModelInfo(modelId: string): Promise<ModelInfo | null> {
		// First check if it's in our list
		const models = await this.listModels()
		const model = models.find(m => m.id === modelId)
		if (model) {
			return model.info
		}
		
		// Try to detect capabilities for unknown model
		return Promise.resolve(this.detectModelCapabilities(modelId))
	}
	
	getDefaultModelId(): string {
		// Use the latest stable Sonnet model as default
		return "claude-3-7-sonnet-20250219"
	}
	
	/**
	 * Attempt to fetch models from Anthropic API
	 * This would use a models endpoint if/when available
	 */
	private fetchModelsFromAPI(): Promise<ModelListing[]> {
		// Anthropic doesn't currently expose a models list API
		// When available, this would look something like:
		// const response = await this.client.listModels()
		// return response.models.map(m => this.convertToModelListing(m))
		
		// For now, return empty to trigger fallback
		return Promise.resolve([])
	}
	
	/**
	 * Detect model capabilities based on model ID patterns
	 */
	private detectModelCapabilities(modelId: string): ModelInfo | null {
		// Parse model ID to extract information
		const patterns = {
			// Claude 3.7 models
			claude37: /claude-3-7-(sonnet|opus|haiku)(?:-(\d{8}))?(?::thinking)?/,
			// Claude 3.5 models
			claude35: /claude-3-5-(sonnet|opus|haiku)(?:-(\d{8}))?/,
			// Claude 3 models
			claude3: /claude-3-(opus|sonnet|haiku)(?:-(\d{8}))?/,
			// Legacy Claude models
			legacy: /claude-(?:instant-)?(\d+(?:\.\d+)?)/
		}
		
		let baseInfo: Partial<ModelInfo> = {
			contextWindow: 200_000, // Default for Claude 3+
			supportsImages: true,
			supportsPromptCache: true,
			supportsComputerUse: false
		}
		
		// Detect model family and tier
		if (patterns.claude37.test(modelId)) {
			const match = modelId.match(patterns.claude37)
			const tier = match?.[1] || "sonnet"
			const hasThinking = modelId.includes(":thinking")
			
			baseInfo = {
				...baseInfo,
				maxTokens: hasThinking ? 128_000 : 8192,
				supportsComputerUse: true,
				thinking: hasThinking,
				...this.getTierPricing(tier, "3.7")
			}
		} else if (patterns.claude35.test(modelId)) {
			const match = modelId.match(patterns.claude35)
			const tier = match?.[1] || "sonnet"
			
			baseInfo = {
				...baseInfo,
				maxTokens: 8192,
				supportsComputerUse: tier === "sonnet",
				...this.getTierPricing(tier, "3.5")
			}
		} else if (patterns.claude3.test(modelId)) {
			const match = modelId.match(patterns.claude3)
			const tier = match?.[1] || "sonnet"
			
			baseInfo = {
				...baseInfo,
				maxTokens: 4096,
				...this.getTierPricing(tier, "3")
			}
		} else if (patterns.legacy.test(modelId)) {
			// Legacy models have limited capabilities
			baseInfo = {
				maxTokens: 4096,
				contextWindow: 100_000,
				supportsImages: false,
				supportsPromptCache: false,
				supportsComputerUse: false,
				inputPrice: 8,
				outputPrice: 24
			}
		} else {
			// Unknown model pattern
			return null
		}
		
		return baseInfo as ModelInfo
	}
	
	/**
	 * Get pricing based on model tier and version
	 */
	private getTierPricing(tier: string, version: string): Partial<ModelInfo> {
		const pricing: Record<string, Record<string, Partial<ModelInfo>>> = {
			"3.7": {
				sonnet: {
					inputPrice: 3.0,
					outputPrice: 15.0,
					cacheWritesPrice: 3.75,
					cacheReadsPrice: 0.3
				},
				opus: {
					inputPrice: 15.0,
					outputPrice: 75.0,
					cacheWritesPrice: 18.75,
					cacheReadsPrice: 1.5
				},
				haiku: {
					inputPrice: 1.0,
					outputPrice: 5.0,
					cacheWritesPrice: 1.25,
					cacheReadsPrice: 0.1
				}
			},
			"3.5": {
				sonnet: {
					inputPrice: 3.0,
					outputPrice: 15.0,
					cacheWritesPrice: 3.75,
					cacheReadsPrice: 0.3
				},
				haiku: {
					inputPrice: 1.0,
					outputPrice: 5.0,
					cacheWritesPrice: 1.25,
					cacheReadsPrice: 0.1,
					supportsImages: false // 3.5 Haiku doesn't support images
				}
			},
			"3": {
				opus: {
					inputPrice: 15.0,
					outputPrice: 75.0,
					cacheWritesPrice: 18.75,
					cacheReadsPrice: 1.5
				},
				sonnet: {
					inputPrice: 3.0,
					outputPrice: 15.0,
					cacheWritesPrice: 3.75,
					cacheReadsPrice: 0.3
				},
				haiku: {
					inputPrice: 0.25,
					outputPrice: 1.25,
					cacheWritesPrice: 0.3,
					cacheReadsPrice: 0.03
				}
			}
		}
		
		return pricing[version]?.[tier] || {}
	}
	
	/**
	 * Get default models when API is unavailable
	 */
	private getDefaultModels(): ModelListing[] {
		// These are the currently known models as of the last update
		// This list should be periodically updated or fetched from a CDN
		return [
			{
				id: "claude-3-7-sonnet-20250219:thinking",
				displayName: "Claude 3.7 Sonnet (Thinking)",
				info: {
					maxTokens: 128_000,
					contextWindow: 200_000,
					supportsImages: true,
					supportsComputerUse: true,
					supportsPromptCache: true,
					inputPrice: 3.0,
					outputPrice: 15.0,
					cacheWritesPrice: 3.75,
					cacheReadsPrice: 0.3,
					thinking: true,
					description: "Claude 3.7 Sonnet with extended reasoning mode for complex tasks"
				},
				releaseDate: "2025-02-19"
			},
			{
				id: "claude-3-7-sonnet-20250219",
				displayName: "Claude 3.7 Sonnet",
				info: {
					maxTokens: 8192,
					contextWindow: 200_000,
					supportsImages: true,
					supportsComputerUse: true,
					supportsPromptCache: true,
					inputPrice: 3.0,
					outputPrice: 15.0,
					cacheWritesPrice: 3.75,
					cacheReadsPrice: 0.3,
					thinking: false,
					description: "Claude 3.7 Sonnet - Advanced model with improved reasoning and coding"
				},
				releaseDate: "2025-02-19"
			},
			{
				id: "claude-3-5-sonnet-20241022",
				displayName: "Claude 3.5 Sonnet",
				info: {
					maxTokens: 8192,
					contextWindow: 200_000,
					supportsImages: true,
					supportsComputerUse: true,
					supportsPromptCache: true,
					inputPrice: 3.0,
					outputPrice: 15.0,
					cacheWritesPrice: 3.75,
					cacheReadsPrice: 0.3,
					description: "Claude 3.5 Sonnet - Balanced performance and capability"
				},
				releaseDate: "2024-10-22"
			},
			{
				id: "claude-3-5-haiku-20241022",
				displayName: "Claude 3.5 Haiku",
				info: {
					maxTokens: 8192,
					contextWindow: 200_000,
					supportsImages: false,
					supportsPromptCache: true,
					inputPrice: 1.0,
					outputPrice: 5.0,
					cacheWritesPrice: 1.25,
					cacheReadsPrice: 0.1,
					description: "Claude 3.5 Haiku - Fast and efficient for simple tasks"
				},
				releaseDate: "2024-10-22"
			},
			{
				id: "claude-3-opus-20240229",
				displayName: "Claude 3 Opus",
				info: {
					maxTokens: 4096,
					contextWindow: 200_000,
					supportsImages: true,
					supportsPromptCache: true,
					inputPrice: 15.0,
					outputPrice: 75.0,
					cacheWritesPrice: 18.75,
					cacheReadsPrice: 1.5,
					description: "Claude 3 Opus - Most capable Claude 3 model"
				},
				releaseDate: "2024-02-29"
			},
			{
				id: "claude-3-haiku-20240307",
				displayName: "Claude 3 Haiku",
				info: {
					maxTokens: 4096,
					contextWindow: 200_000,
					supportsImages: true,
					supportsPromptCache: true,
					inputPrice: 0.25,
					outputPrice: 1.25,
					cacheWritesPrice: 0.3,
					cacheReadsPrice: 0.03,
					description: "Claude 3 Haiku - Fastest Claude 3 model"
				},
				releaseDate: "2024-03-07"
			}
		]
	}
	
	/**
	 * Clear cached models
	 */
	clearCache(): void {
		this.cachedModels = null
	}
}
