import axios from "axios"
import { ModelProvider, ModelListing } from "../model-registry"
import { ModelInfo, ApiHandlerOptions } from "../../../shared/api"

interface OpenAICompatibleModel {
  id: string
  object: string
  created?: number
  owned_by?: string
  permission?: unknown[]
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
 * Model provider for Mistral AI that dynamically fetches available models
 * Uses OpenAI-compatible API format
 */
export class MistralModelProvider implements ModelProvider {
  private apiKey: string = ""
  private cache: Map<string, CacheEntry> = new Map()
  private cacheTTL = 3600000 // 1 hour
  private baseUrl = "https://api.mistral.ai/v1"

  /**
   * Known model capabilities and pricing for Mistral models
   * Prices are per million tokens
   */
  private modelCapabilities: Record<string, Partial<ModelInfo>> = {
    "mistral-large-latest": {
      maxTokens: 8192,
      contextWindow: 128000,
      supportsImages: false,
      supportsPromptCache: false,
      inputPrice: 2.0,   // $2 per 1M input tokens
      outputPrice: 6.0,  // $6 per 1M output tokens
      description: "Mistral's flagship model for complex reasoning tasks",
      supportsTemperature: true,
      supportsTopP: true,
    },
    "mistral-large-2407": {
      maxTokens: 8192,
      contextWindow: 128000,
      supportsImages: false,
      supportsPromptCache: false,
      inputPrice: 2.0,
      outputPrice: 6.0,
      description: "Mistral Large model from July 2024",
    },
    "mistral-medium-latest": {
      maxTokens: 8192,
      contextWindow: 32000,
      supportsImages: false,
      supportsPromptCache: false,
      inputPrice: 2.7,   // Per 1M tokens
      outputPrice: 8.1,
      description: "Balanced performance and efficiency",
    },
    "mistral-small-latest": {
      maxTokens: 8192,
      contextWindow: 32000,
      supportsImages: false,
      supportsPromptCache: false,
      inputPrice: 0.2,
      outputPrice: 0.6,
      description: "Fast and efficient for simpler tasks",
    },
    "mistral-tiny": {
      maxTokens: 4096,
      contextWindow: 32000,
      supportsImages: false,
      supportsPromptCache: false,
      inputPrice: 0.14,
      outputPrice: 0.42,
      description: "Smallest and fastest Mistral model",
    },
    "codestral-latest": {
      maxTokens: 8192,
      contextWindow: 32000,
      supportsImages: false,
      supportsPromptCache: false,
      inputPrice: 0.2,
      outputPrice: 0.6,
      description: "Specialized model for code generation and completion",
    },
    "codestral-mamba-latest": {
      maxTokens: 8192,
      contextWindow: 256000, // Much larger context for code
      supportsImages: false,
      supportsPromptCache: false,
      inputPrice: 0.25,
      outputPrice: 0.25, // Same price for input/output
      description: "Mamba-based model optimized for code with large context",
    },
    "open-mistral-7b": {
      maxTokens: 8192,
      contextWindow: 32000,
      supportsImages: false,
      supportsPromptCache: false,
      inputPrice: 0.25,
      outputPrice: 0.25,
      description: "Open-source 7B parameter model",
    },
    "open-mixtral-8x7b": {
      maxTokens: 8192,
      contextWindow: 32000,
      supportsImages: false,
      supportsPromptCache: false,
      inputPrice: 0.7,
      outputPrice: 0.7,
      description: "Open-source mixture of experts model",
    },
    "open-mixtral-8x22b": {
      maxTokens: 8192,
      contextWindow: 64000,
      supportsImages: false,
      supportsPromptCache: false,
      inputPrice: 2.0,
      outputPrice: 6.0,
      description: "Large open-source mixture of experts model",
    },
  }

  configure(options: ApiHandlerOptions): void {
    this.apiKey = options.mistralApiKey || ""
  }

  async listModels(forceRefresh = false): Promise<ModelListing[]> { return this.getModels(forceRefresh) }
  async getModels(forceRefresh = false): Promise<ModelListing[]> {
    const cacheKey = "mistral_models"
    
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
      console.error("Failed to fetch Mistral models:", error)
      
      // Return cached models if available
      const cached = this.cache.get(cacheKey)
      if (cached) {
        console.log("Using cached Mistral models due to API error")
        return cached.models
      }
      
      // Return fallback models
      return this.getFallbackModels()
    }
  }

  private async fetchModels(): Promise<ModelListing[]> {
    if (!this.apiKey) {
      throw new Error("Mistral API key not configured")
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
        id: modelId,
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
    
    // Check for partial matches (handle versioned models)
    for (const [key, capabilities] of Object.entries(this.modelCapabilities)) {
      if (modelId.includes(key.replace("-latest", ""))) {
        return capabilities as ModelInfo
      }
    }
    
    // Derive capabilities from model name
    const isLarge = modelId.includes("large")
    const isMedium = modelId.includes("medium") 
    const isSmall = modelId.includes("small") || modelId.includes("tiny")
    const isCodestral = modelId.includes("codestral")
    const isMamba = modelId.includes("mamba")
    const isOpen = modelId.includes("open")
    const isMixtral = modelId.includes("mixtral")
    
    // Default capabilities based on model tier
    let maxTokens = 8192
    let contextWindow = 32000
    let inputPrice = 0.7
    let outputPrice = 2.1
    
    if (isLarge) {
      inputPrice = 2.0
      outputPrice = 6.0
      contextWindow = 128000
    } else if (isMedium) {
      inputPrice = 2.7
      outputPrice = 8.1
    } else if (isSmall) {
      inputPrice = 0.2
      outputPrice = 0.6
    }
    
    if (isMamba) {
      contextWindow = 256000 // Mamba has large context
      inputPrice = 0.25
      outputPrice = 0.25
    }
    
    if (isMixtral && modelId.includes("8x22b")) {
      contextWindow = 64000
      inputPrice = 2.0
      outputPrice = 6.0
    }
    
    return {
      maxTokens,
      contextWindow,
      supportsImages: false,
      supportsPromptCache: false,
      inputPrice,
      outputPrice,
      description: isCodestral ? "Code generation model" : 
                   isOpen ? "Open-source model" :
                   isMixtral ? "Mixture of experts model" :
                   isLarge ? "High-performance model for complex tasks" :
                   isMedium ? "Balanced performance model" :
                   "Efficient model for simpler tasks",
      supportsTemperature: true,
      supportsTopP: true,
    }
  }

  async getModelInfo(modelId: string): Promise<ModelInfo | null> {
    const models = await this.getModels()
    const model = models.find((m) => m.id === modelId)
    return model?.info || null
  }

  async getDefaultModelId(): Promise<string> {
    // Default to mistral-small (good balance of cost and performance)
    const models = await this.getModels()
    
    // Try to find mistral-small-latest
    const smallModel = models.find((m) => m.id === "mistral-small-latest")
    if (smallModel) return smallModel.id
    
    // Fallback to any small model
    const anySmall = models.find((m) => m.id.includes("small"))
    if (anySmall) return anySmall.id
    
    // Fallback to any model that's not "large" (to avoid high costs)
    const affordable = models.find((m) => !m.id.includes("large"))
    if (affordable) return affordable.id
    
    // Return first available model
    return models[0]?.id || "mistral-small-latest"
  }

  private getFallbackModels(): ModelListing[] {
    // Return known Mistral models as fallback
    return Object.entries(this.modelCapabilities).map(([id, info]) => ({
      id,
      info: info as ModelInfo,
    }))
  }

  /**
   * Check if a model is a coding specialist (Codestral)
   */
  static isCodingModel(modelId: string): boolean {
    return modelId.includes("codestral")
  }

  /**
   * Check if a model is open-source
   */
  static isOpenSourceModel(modelId: string): boolean {
    return modelId.includes("open-")
  }

  /**
   * Check if a model is a mixture of experts (Mixtral)
   */
  static isMixtralModel(modelId: string): boolean {
    return modelId.includes("mixtral")
  }

  /**
   * Get model tier (large, medium, small, tiny)
   */
  static getModelTier(modelId: string): string {
    if (modelId.includes("large")) return "large"
    if (modelId.includes("medium")) return "medium" 
    if (modelId.includes("small")) return "small"
    if (modelId.includes("tiny")) return "tiny"
    return "unknown"
  }
}
