/**
 * OpenAI Dynamic Model Provider
 * 
 * Fetches available models from OpenAI's API dynamically
 */

import OpenAI from "openai"
import { ModelInfo } from "../../schemas"
import { ModelProvider, ModelListing } from "./model-registry"

/**
 * OpenAI model provider that fetches models dynamically from the API
 */
export class OpenAIModelProvider implements ModelProvider {
	private client: OpenAI
	private cachedModels: ModelListing[] | null = null
	
	constructor(apiKey?: string, baseURL?: string) {
		this.client = new OpenAI({
			apiKey: apiKey || "not-provided",
			baseURL: baseURL || "https://api.openai.com/v1"
		})
	}
	
	async listModels(): Promise<ModelListing[]> {
		// If we have cached models, return them
		if (this.cachedModels) {
			return this.cachedModels
		}
		
		try {
			// OpenAI provides a models endpoint
			const response = await this.client.models.list()
			const models: ModelListing[] = []
			
			for (const model of response.data) {
				// Only include chat models
				if (this.isChatModel(model.id)) {
					const info = await this.getModelCapabilities(model.id)
					if (info) {
						models.push({
							id: model.id,
							displayName: this.getDisplayName(model.id),
							info,
							deprecated: this.isDeprecated(model.id),
							releaseDate: this.getReleaseDate(model.id)
						})
					}
				}
			}
			
			// Sort models by capability and recency
			models.sort((a, b) => {
				// Prioritize non-deprecated models
				if (a.deprecated !== b.deprecated) {
					return a.deprecated ? 1 : -1
				}
				// Then by release date (newer first)
				if (a.releaseDate && b.releaseDate) {
					return b.releaseDate.localeCompare(a.releaseDate)
				}
				// Then alphabetically
				return a.id.localeCompare(b.id)
			})
			
			this.cachedModels = models
			return models
		} catch (error) {
			console.warn("Failed to fetch OpenAI models from API:", error)
			return this.getDefaultModels()
		}
	}
	
	async getModelInfo(modelId: string): Promise<ModelInfo | null> {
		// First check if it's in our list
		const models = await this.listModels()
		const model = models.find(m => m.id === modelId)
		if (model) {
			return model.info
		}
		
		// Try to get capabilities for unknown model
		return this.getModelCapabilities(modelId)
	}
	
	getDefaultModelId(): string {
		return "gpt-4o"
	}
	
	/**
	 * Check if a model ID represents a chat model
	 */
	private isChatModel(modelId: string): boolean {
		const chatPrefixes = ["gpt-4", "gpt-3.5", "o1", "o3", "chatgpt"]
		return chatPrefixes.some(prefix => modelId.startsWith(prefix))
	}
	
	/**
	 * Get model capabilities based on model ID
	 */
	private async getModelCapabilities(modelId: string): Promise<ModelInfo | null> {
		// Try to get actual capabilities from API if possible
		// OpenAI doesn't expose detailed capabilities, so we use patterns
		
		const baseInfo: Partial<ModelInfo> = {
			supportsImages: false,
			supportsPromptCache: false,
			supportsComputerUse: false,
			supportsAssistantTool: true
		}
		
		// GPT-4 models
		if (modelId.startsWith("gpt-4o")) {
			return {
				...baseInfo,
				maxTokens: modelId.includes("mini") ? 16_384 : 128_000,
				contextWindow: 128_000,
				supportsImages: true,
				inputPrice: modelId.includes("mini") ? 0.15 : 2.5,
				outputPrice: modelId.includes("mini") ? 0.6 : 10,
				reasoningEffort: "low",
				description: modelId.includes("mini") ? "Fast and affordable GPT-4o model" : "Most capable GPT-4o model"
			} as ModelInfo
		} else if (modelId.startsWith("gpt-4-turbo")) {
			return {
				...baseInfo,
				maxTokens: 4096,
				contextWindow: 128_000,
				supportsImages: true,
				inputPrice: 10,
				outputPrice: 30,
				reasoningEffort: "low",
				description: "GPT-4 Turbo with vision capabilities"
			} as ModelInfo
		} else if (modelId.startsWith("gpt-4")) {
			return {
				...baseInfo,
				maxTokens: 8192,
				contextWindow: modelId.includes("32k") ? 32_768 : 8192,
				inputPrice: modelId.includes("32k") ? 60 : 30,
				outputPrice: modelId.includes("32k") ? 120 : 60,
				reasoningEffort: "low",
				description: "Original GPT-4 model"
			} as ModelInfo
		}
		
		// O1 reasoning models
		else if (modelId.startsWith("o1")) {
			const isPreview = modelId.includes("preview")
			const isMini = modelId.includes("mini")
			
			return {
				...baseInfo,
				maxTokens: isPreview ? 32_768 : (isMini ? 65_536 : 100_000),
				contextWindow: 128_000,
				supportsImages: !isPreview,
				supportsTemperature: false, // O1 models don't support temperature
				inputPrice: isMini ? 3 : 15,
				outputPrice: isMini ? 12 : 60,
				reasoningEffort: "high",
				reasoningTokens: true,
				description: `O1 ${isMini ? "mini" : "full"} reasoning model`
			} as ModelInfo
		}
		
		// O3 models
		else if (modelId.startsWith("o3")) {
			const isMini = modelId.includes("mini")
			
			return {
				...baseInfo,
				maxTokens: isMini ? 65_536 : 100_000,
				contextWindow: 128_000,
				supportsImages: true,
				supportsTemperature: false,
				inputPrice: isMini ? 1.1 : 15, // Estimated
				outputPrice: isMini ? 4.4 : 60, // Estimated
				reasoningEffort: isMini ? "medium" : "extreme",
				reasoningTokens: true,
				description: `O3 ${isMini ? "mini" : "full"} advanced reasoning model`
			} as ModelInfo
		}
		
		// GPT-3.5 models
		else if (modelId.startsWith("gpt-3.5")) {
			return {
				...baseInfo,
				maxTokens: 4096,
				contextWindow: modelId.includes("16k") ? 16_384 : 4096,
				inputPrice: 0.5,
				outputPrice: 1.5,
				reasoningEffort: "low",
				description: "Fast and affordable GPT-3.5 model"
			} as ModelInfo
		}
		
		// ChatGPT models
		else if (modelId.startsWith("chatgpt")) {
			return {
				...baseInfo,
				maxTokens: 16_384,
				contextWindow: 128_000,
				supportsImages: true,
				inputPrice: 5,
				outputPrice: 15,
				reasoningEffort: "low",
				description: "ChatGPT model optimized for conversation"
			} as ModelInfo
		}
		
		return null
	}
	
	/**
	 * Get display name for a model
	 */
	private getDisplayName(modelId: string): string {
		const displayNames: Record<string, string> = {
			"gpt-4o": "GPT-4o",
			"gpt-4o-mini": "GPT-4o Mini",
			"gpt-4-turbo": "GPT-4 Turbo",
			"gpt-4": "GPT-4",
			"gpt-3.5-turbo": "GPT-3.5 Turbo",
			"o1-preview": "O1 Preview",
			"o1-mini": "O1 Mini",
			"o1": "O1",
			"o3-mini": "O3 Mini",
			"o3": "O3",
			"chatgpt-4o-latest": "ChatGPT-4o Latest"
		}
		
		// Check for exact match
		if (displayNames[modelId]) {
			return displayNames[modelId]
		}
		
		// Generate display name from ID
		return modelId
			.split("-")
			.map(part => part.charAt(0).toUpperCase() + part.slice(1))
			.join(" ")
	}
	
	/**
	 * Check if a model is deprecated
	 */
	private isDeprecated(modelId: string): boolean {
		const deprecatedPatterns = [
			/gpt-4-\d{4}/, // Date-versioned GPT-4 models
			/gpt-3\.5-turbo-\d{4}/, // Date-versioned GPT-3.5 models
			/text-/, // Text completion models
			/davinci/, // Legacy models
			/curie/,
			/babbage/,
			/ada/
		]
		
		return deprecatedPatterns.some(pattern => pattern.test(modelId))
	}
	
	/**
	 * Get release date for a model
	 */
	private getReleaseDate(modelId: string): string | undefined {
		// Extract date from model ID if present
		const dateMatch = modelId.match(/(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?/)
		if (dateMatch) {
			const [, year, month = "01", day = "01"] = dateMatch
			return `${year}-${month}-${day}`
		}
		
		// Known release dates
		const releaseDates: Record<string, string> = {
			"gpt-4o": "2024-05-13",
			"gpt-4o-mini": "2024-07-18",
			"gpt-4-turbo": "2024-04-09",
			"gpt-4": "2023-03-14",
			"gpt-3.5-turbo": "2022-11-30",
			"o1-preview": "2024-09-12",
			"o1-mini": "2024-09-12",
			"o1": "2024-12-17",
			"o3-mini": "2025-01-31",
			"o3": "2025-01-31"
		}
		
		return releaseDates[modelId]
	}
	
	/**
	 * Get default models when API is unavailable
	 */
	private getDefaultModels(): ModelListing[] {
		return [
			{
				id: "gpt-4o",
				displayName: "GPT-4o",
				info: {
					maxTokens: 128_000,
					contextWindow: 128_000,
					supportsImages: true,
					supportsPromptCache: false,
					supportsComputerUse: false,
					supportsAssistantTool: true,
					inputPrice: 2.5,
					outputPrice: 10,
					reasoningEffort: "low",
					description: "Most capable GPT-4o model with vision"
				},
				releaseDate: "2024-05-13"
			},
			{
				id: "gpt-4o-mini",
				displayName: "GPT-4o Mini",
				info: {
					maxTokens: 16_384,
					contextWindow: 128_000,
					supportsImages: true,
					supportsPromptCache: false,
					supportsComputerUse: false,
					supportsAssistantTool: true,
					inputPrice: 0.15,
					outputPrice: 0.6,
					reasoningEffort: "low",
					description: "Affordable and fast GPT-4o model"
				},
				releaseDate: "2024-07-18"
			},
			{
				id: "o1",
				displayName: "O1",
				info: {
					maxTokens: 100_000,
					contextWindow: 128_000,
					supportsImages: true,
					supportsPromptCache: false,
					supportsComputerUse: false,
					supportsAssistantTool: true,
					supportsTemperature: false,
					inputPrice: 15,
					outputPrice: 60,
					reasoningEffort: "high",
					reasoningTokens: true,
					description: "Advanced reasoning model"
				},
				releaseDate: "2024-12-17"
			},
			{
				id: "o1-mini",
				displayName: "O1 Mini",
				info: {
					maxTokens: 65_536,
					contextWindow: 128_000,
					supportsImages: true,
					supportsPromptCache: false,
					supportsComputerUse: false,
					supportsAssistantTool: true,
					supportsTemperature: false,
					inputPrice: 3,
					outputPrice: 12,
					reasoningEffort: "high",
					reasoningTokens: true,
					description: "Efficient reasoning model"
				},
				releaseDate: "2024-09-12"
			},
			{
				id: "o3-mini",
				displayName: "O3 Mini",
				info: {
					maxTokens: 65_536,
					contextWindow: 128_000,
					supportsImages: true,
					supportsPromptCache: false,
					supportsComputerUse: false,
					supportsAssistantTool: true,
					supportsTemperature: false,
					inputPrice: 1.1,
					outputPrice: 4.4,
					reasoningEffort: "medium",
					reasoningTokens: true,
					description: "Latest efficient reasoning model"
				},
				releaseDate: "2025-01-31"
			},
			{
				id: "gpt-3.5-turbo",
				displayName: "GPT-3.5 Turbo",
				info: {
					maxTokens: 4096,
					contextWindow: 4096,
					supportsImages: false,
					supportsPromptCache: false,
					supportsComputerUse: false,
					supportsAssistantTool: true,
					inputPrice: 0.5,
					outputPrice: 1.5,
					reasoningEffort: "low",
					description: "Fast and affordable model"
				},
				releaseDate: "2022-11-30"
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