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
 * Model provider for Glama that dynamically fetches available models
 * Uses OpenAI-compatible API format
 */
export class GlamaModelProvider implements ModelProvider {
  private apiKey: string = ""
  private cache: Map<string, CacheEntry> = new Map()
  private cacheTTL = 3600000 // 1 hour
  private baseUrl = "https://glama.ai/api/gateway/openai/v1"

  /**
   * Known model capabilities and pricing for Glama models
   * Prices are per million tokens
   */
  private modelCapabilities: Record<string, Partial<ModelInfo>> = {
    "anthropic/claude-3-5-sonnet": {
      maxTokens: 8192,
      contextWindow: 200000,
      supportsImages: true,
      supportsPromptCache: true,
      inputPrice: 3.0,
      outputPrice: 15.0,
    },
    "anthropic/claude-3-opus": {
      maxTokens: 4096,
      contextWindow: 200000,
      supportsImages: true,
      supportsPromptCache: true,
      inputPrice: 15.0,
      outputPrice: 75.0,
    },
    "openai/gpt-4o": {
      maxTokens: 4096,
      contextWindow: 128000,
      supportsImages: true,
      supportsPromptCache: false,
      inputPrice: 5.0,
      outputPrice: 15.0,
    },
  }

  configure(options: ApiHandlerOptions): void {
    this.apiKey = options.glamaApiKey || ""
  }

  async listModels(): Promise<ModelListing[]> {
    return this.getModels()
  }

  async getModels(forceRefresh = false): Promise<ModelListing[]> {
    const cacheKey = "models"

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
      console.error("Failed to fetch Glama models:", error)

      // Return cached models if available
      const cached = this.cache.get(cacheKey)
      if (cached) {
        console.log("Using cached Glama models due to API error")
        return cached.models
      }

      // Return fallback models
      return this.getFallbackModels()
    }
  }

  private async fetchModels(): Promise<ModelListing[]> {
    if (!this.apiKey) {
      return this.getFallbackModels()
    }

    const response = await axios.get<OpenAICompatibleResponse>(`${this.baseUrl}/models`, {
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
      },
    })

    const models: ModelListing[] = []

    for (const model of response.data.data) {
      const modelId = model.id
      
      // Get known capabilities or derive them
      const capabilities = this.modelCapabilities[modelId] || this.deriveCapabilities(modelId)

      models.push({
        id: modelId,
        info: {
          maxTokens: capabilities.maxTokens || 4096,
          contextWindow: capabilities.contextWindow || 128000,
          supportsImages: capabilities.supportsImages || false,
          supportsPromptCache: capabilities.supportsPromptCache || false,
          inputPrice: capabilities.inputPrice,
          outputPrice: capabilities.outputPrice,
          description: `Glama model: ${modelId}`,
        },
      })
    }

    return models
  }

  private deriveCapabilities(modelId: string): Partial<ModelInfo> {
    const capabilities: Partial<ModelInfo> = {}
    const lowerId = modelId.toLowerCase()

    // Derive from model name
    if (lowerId.includes("claude") || lowerId.includes("anthropic")) {
      capabilities.supportsImages = true
      capabilities.supportsPromptCache = true
      capabilities.contextWindow = 200000
    } else if (lowerId.includes("gpt-4") || lowerId.includes("openai")) {
      capabilities.supportsImages = true
      capabilities.contextWindow = 128000
    } else if (lowerId.includes("gemini")) {
      capabilities.supportsImages = true
      capabilities.contextWindow = 1000000 // Gemini usually has large context
    } else if (lowerId.includes("llama")) {
      capabilities.contextWindow = 8192
    } else if (lowerId.includes("mistral")) {
      capabilities.contextWindow = 32000
    }

    return capabilities
  }

  async getModelInfo(modelId: string): Promise<ModelInfo | null> {
    const models = await this.getModels()
    const model = models.find((m) => m.id === modelId)
    return model?.info || null
  }

  getDefaultModelId(): string {
    return "anthropic/claude-3-5-sonnet"
  }

  private getFallbackModels(): ModelListing[] {
    // Return a basic set of known models as fallback
    return Object.entries(this.modelCapabilities).map(([id, info]) => ({
      id,
      info: info as ModelInfo,
    }))
  }
}
