import axios from "axios"
import { ModelProvider, ModelListing } from "../model-registry"
import { ModelInfo, ApiHandlerOptions } from "../../../shared/api"

interface GeminiModel {
  name: string
  version?: string
  displayName?: string
  description?: string
  inputTokenLimit?: number
  outputTokenLimit?: number
  supportedGenerationMethods?: string[]
  temperature?: number
  topP?: number
  topK?: number
}

interface GeminiModelsResponse {
  models: GeminiModel[]
}

interface CacheEntry {
  models: ModelListing[]
  timestamp: number
}

/**
 * Model provider for Google Gemini (AI Studio) that dynamically fetches available models
 */
export class GeminiModelProvider implements ModelProvider {
  private apiKey: string = ""
  private cache: Map<string, CacheEntry> = new Map()
  private cacheTTL = 3600000 // 1 hour
  private baseUrl = "https://generativelanguage.googleapis.com/v1"

  /**
   * Known model pricing (Gemini API doesn't return pricing info)
   * Prices are per million tokens
   */
  private modelPricing: Record<string, { input: number; output: number }> = {
    "gemini-1.5-pro": {
      input: 3.5,    // $3.50 per million input tokens
      output: 10.5,  // $10.50 per million output tokens
    },
    "gemini-1.5-pro-latest": {
      input: 3.5,
      output: 10.5,
    },
    "gemini-1.5-flash": {
      input: 0.075,  // $0.075 per million input tokens
      output: 0.3,   // $0.30 per million output tokens
    },
    "gemini-1.5-flash-latest": {
      input: 0.075,
      output: 0.3,
    },
    "gemini-1.0-pro": {
      input: 0.5,    // $0.50 per million input tokens
      output: 1.5,   // $1.50 per million output tokens
    },
    "gemini-2.0-flash-exp": {
      input: 0,      // Free during experimental phase
      output: 0,
    },
  }

  configure(options: ApiHandlerOptions): void {
    this.apiKey = options.geminiApiKey || ""
  }

  async getModels(forceRefresh = false): Promise<ModelListing[]> {
    const cacheKey = "gemini_models"
    
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
      console.error("Failed to fetch Gemini models:", error)
      
      // Return cached models if available
      const cached = this.cache.get(cacheKey)
      if (cached) {
        console.log("Using cached Gemini models due to API error")
        return cached.models
      }
      
      // Return fallback models
      return this.getFallbackModels()
    }
  }

  private async fetchModels(): Promise<ModelListing[]> {
    if (!this.apiKey) {
      throw new Error("Gemini API key not configured")
    }

    const response = await axios.get(`${this.baseUrl}/models`, {
      params: {
        key: this.apiKey,
      },
    })
    
    const data = response.data as GeminiModelsResponse
    const models: ModelListing[] = []
    
    for (const model of data.models || []) {
      // Extract model ID from the full name (e.g., "models/gemini-1.5-pro" -> "gemini-1.5-pro")
      const modelId = model.name.replace("models/", "")
      
      // Skip non-generative models
      if (!model.supportedGenerationMethods?.includes("generateContent")) {
        continue
      }
      
      // Get pricing info
      const pricing = this.getPricing(modelId)
      
      // Determine capabilities based on model name
      const supportsImages = this.supportsImages(modelId)
      const supportsSystemInstructions = this.supportsSystemInstructions(modelId)
      const thinking = modelId.includes("thinking")
      
      models.push({
        modelId,
        info: {
          maxTokens: model.outputTokenLimit || 8192,
          contextWindow: model.inputTokenLimit || 32768,
          supportsImages,
          supportsPromptCache: false, // Gemini doesn't support prompt caching yet
          supportsSystemInstructions,
          inputPrice: pricing.input,
          outputPrice: pricing.output,
          description: model.description || model.displayName,
          thinking,
          temperature: model.temperature,
          topP: model.topP,
          topK: model.topK,
        },
      })
    }
    
    return models
  }

  private getPricing(modelId: string): { input: number; output: number } {
    // Find the best match for pricing
    for (const [key, pricing] of Object.entries(this.modelPricing)) {
      if (modelId.includes(key)) {
        return pricing
      }
    }
    
    // Default pricing if not found
    return { input: 0, output: 0 }
  }

  private supportsImages(modelId: string): boolean {
    // Most Gemini models support multimodal input
    const imageModels = [
      "gemini-1.5-pro",
      "gemini-1.5-flash",
      "gemini-1.0-pro-vision",
      "gemini-2.0-flash",
    ]
    
    return imageModels.some(m => modelId.includes(m))
  }

  private supportsSystemInstructions(modelId: string): boolean {
    // Gemini 1.5+ models support system instructions
    return modelId.includes("gemini-1.5") || modelId.includes("gemini-2")
  }

  async getModelInfo(modelId: string): Promise<ModelInfo | null> {
    const models = await this.getModels()
    const model = models.find((m) => m.modelId === modelId)
    return model?.info || null
  }

  async getDefaultModelId(): Promise<string> {
    // Default to Gemini 1.5 Flash (good balance of speed and capability)
    const models = await this.getModels()
    
    // Try to find Gemini 1.5 Flash
    const flash = models.find((m) => m.modelId === "gemini-1.5-flash-latest")
    if (flash) return flash.modelId
    
    // Fallback to any 1.5 model
    const any15 = models.find((m) => m.modelId.includes("gemini-1.5"))
    if (any15) return any15.modelId
    
    // Return first available model
    return models[0]?.modelId || "gemini-1.5-flash-latest"
  }

  private getFallbackModels(): ModelListing[] {
    // Return a basic set of known Gemini models as fallback
    return [
      {
        modelId: "gemini-1.5-flash-latest",
        info: {
          maxTokens: 8192,
          contextWindow: 1048576, // 1M tokens
          supportsImages: true,
          supportsPromptCache: false,
          supportsSystemInstructions: true,
          inputPrice: 0.075,
          outputPrice: 0.3,
          description: "Fast, versatile multimodal model for scaling",
        },
      },
      {
        modelId: "gemini-1.5-pro-latest",
        info: {
          maxTokens: 8192,
          contextWindow: 2097152, // 2M tokens
          supportsImages: true,
          supportsPromptCache: false,
          supportsSystemInstructions: true,
          inputPrice: 3.5,
          outputPrice: 10.5,
          description: "Advanced multimodal model with extended context",
        },
      },
      {
        modelId: "gemini-1.0-pro",
        info: {
          maxTokens: 2048,
          contextWindow: 32768,
          supportsImages: false,
          supportsPromptCache: false,
          supportsSystemInstructions: true,
          inputPrice: 0.5,
          outputPrice: 1.5,
          description: "Versatile text model for various tasks",
        },
      },
      {
        modelId: "gemini-2.0-flash-exp",
        info: {
          maxTokens: 8192,
          contextWindow: 1048576,
          supportsImages: true,
          supportsPromptCache: false,
          supportsSystemInstructions: true,
          inputPrice: 0,
          outputPrice: 0,
          description: "Experimental next-generation model (free during preview)",
          thinking: false,
        },
      },
    ]
  }

  /**
   * Get free tier limits for Gemini models
   * Returns requests per minute for free tier
   */
  static getFreeTierLimits(modelId: string): number {
    if (modelId.includes("flash")) {
      return 15 // 15 requests per minute for Flash models
    } else if (modelId.includes("pro")) {
      return 2 // 2 requests per minute for Pro models
    }
    return 10 // Default
  }
}