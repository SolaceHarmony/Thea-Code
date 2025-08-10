import axios from "axios"
import { ModelProvider, ModelListing } from "../model-registry"
import { ModelInfo } from "../../../shared/api"

interface OpenAICompatibleModel {
  id: string
  object: string
  created?: number
  owned_by?: string
  permission?: any[]
}

interface OpenAICompatibleResponse {
  object: string
  data: OpenAICompatibleModel[]
}

interface CacheEntry {
  models: ModelListing[]
  timestamp: number
}

/**
 * Model provider for DeepSeek that dynamically fetches available models
 * Uses OpenAI-compatible API format
 */
export class DeepSeekModelProvider implements ModelProvider {
  private apiKey: string = ""
  private cache: Map<string, CacheEntry> = new Map()
  private cacheTTL = 3600000 // 1 hour
  private baseUrl = "https://api.deepseek.com/v1"

  /**
   * Known model capabilities and pricing for DeepSeek models
   * Prices are per million tokens
   */
  private modelCapabilities: Record<string, Partial<ModelInfo>> = {
    "deepseek-chat": {
      maxTokens: 4096,
      contextWindow: 32768,
      supportsImages: false,
      supportsPromptCache: false,
      inputPrice: 0.14,  // $0.14 per 1M input tokens
      outputPrice: 0.28, // $0.28 per 1M output tokens
      description: "DeepSeek's flagship conversational AI model",
    },
    "deepseek-coder": {
      maxTokens: 4096,
      contextWindow: 16384,
      supportsImages: false,
      supportsPromptCache: false,
      inputPrice: 0.14,
      outputPrice: 0.28,
      description: "DeepSeek's specialized coding model",
    },
    "deepseek-r1": {
      maxTokens: 8192,
      contextWindow: 32768,
      supportsImages: false,
      supportsPromptCache: false,
      thinking: true,
      inputPrice: 0.55,  // Higher price for reasoning model
      outputPrice: 2.19,
      description: "DeepSeek's advanced reasoning model with thinking capabilities",
    },
    "deepseek-r1-lite-preview": {
      maxTokens: 8192,
      contextWindow: 32768,
      supportsImages: false,
      supportsPromptCache: false,
      thinking: true,
      inputPrice: 0.14,  // Preview pricing
      outputPrice: 0.28,
      description: "Lite version of DeepSeek R1 reasoning model (preview)",
    },
  }

  configure(options: any): void {
    this.apiKey = options.deepSeekApiKey || ""
  }

  async getModels(forceRefresh = false): Promise<ModelListing[]> {
    const cacheKey = "deepseek_models"
    
    // Check cache first
    if (!forceRefresh) {
      const cached = this.cache.get(cacheKey)
      if (cached && Date.now() - cached.timestamp < this.cacheTTL) {
        return cached.models
      }
    }

    try {
      // Fetch fresh models from API
      const models = await this.fetchModels()
      
      // Update cache
      this.cache.set(cacheKey, {
        models,
        timestamp: Date.now(),
      })
      
      return models
    } catch (error) {
      console.error("Failed to fetch DeepSeek models:", error)
      
      // Return cached models if available
      const cached = this.cache.get(cacheKey)
      if (cached) {
        console.log("Using cached DeepSeek models due to API error")
        return cached.models
      }
      
      // Return fallback models
      return this.getFallbackModels()
    }
  }

  private async fetchModels(): Promise<ModelListing[]> {
    if (!this.apiKey) {
      throw new Error("DeepSeek API key not configured")
    }

    const response = await axios.get(`${this.baseUrl}/models`, {
      headers: {
        "Authorization": `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
    })
    
    const data = response.data as OpenAICompatibleResponse
    const models: ModelListing[] = []
    
    for (const model of data.data || []) {
      const modelId = model.id
      
      // Get capabilities from known models or derive them
      const capabilities = this.getModelCapabilities(modelId)
      
      models.push({
        modelId,
        info: capabilities,
      })
    }
    
    return models
  }

  private getModelCapabilities(modelId: string): ModelInfo {
    // Check for exact match first
    if (this.modelCapabilities[modelId]) {
      return this.modelCapabilities[modelId] as ModelInfo
    }
    
    // Check for partial matches
    for (const [key, capabilities] of Object.entries(this.modelCapabilities)) {
      if (modelId.includes(key)) {
        return capabilities as ModelInfo
      }
    }
    
    // Default capabilities for unknown models
    const isR1Model = modelId.includes("r1") || modelId.includes("reasoning")
    const isCoder = modelId.includes("coder") || modelId.includes("code")
    
    return {
      maxTokens: isR1Model ? 8192 : 4096,
      contextWindow: 32768,
      supportsImages: false,
      supportsPromptCache: false,
      thinking: isR1Model,
      inputPrice: isR1Model ? 0.55 : 0.14,
      outputPrice: isR1Model ? 2.19 : 0.28,
      description: isCoder ? "DeepSeek coding model" : isR1Model ? "DeepSeek reasoning model" : "DeepSeek chat model",
      supportsTemperature: true,
      supportsTopP: true,
    }
  }

  async getModelInfo(modelId: string): Promise<ModelInfo | null> {
    const models = await this.getModels()
    const model = models.find((m) => m.modelId === modelId)
    return model?.info || null
  }

  async getDefaultModelId(): Promise<string> {
    // Default to deepseek-chat
    const models = await this.getModels()
    
    // Try to find deepseek-chat
    const chatModel = models.find((m) => m.modelId === "deepseek-chat")
    if (chatModel) return chatModel.modelId
    
    // Fallback to any non-R1 model (R1 models are expensive)
    const regularModel = models.find((m) => !m.modelId.includes("r1"))
    if (regularModel) return regularModel.modelId
    
    // Return first available model
    return models[0]?.modelId || "deepseek-chat"
  }

  private getFallbackModels(): ModelListing[] {
    // Return known DeepSeek models as fallback
    return Object.entries(this.modelCapabilities).map(([modelId, info]) => ({
      modelId,
      info: info as ModelInfo,
    }))
  }

  /**
   * Check if a model is a reasoning model (R1 family)
   */
  static isReasoningModel(modelId: string): boolean {
    return modelId.includes("r1") || modelId.includes("reasoning")
  }

  /**
   * Check if a model is a coding specialist model
   */
  static isCodingModel(modelId: string): boolean {
    return modelId.includes("coder") || modelId.includes("code")
  }
}