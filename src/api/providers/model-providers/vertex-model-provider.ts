import { GoogleAuth } from "google-auth-library"
import axios from "axios"
import { ModelProvider, ModelListing } from "../model-registry"
import { ModelInfo, ApiHandlerOptions } from "../../../shared/api"

// Model interfaces are omitted; responses are typed inline per request

interface CacheEntry {
  models: ModelListing[]
  timestamp: number
}

/**
 * Model provider for Google Vertex AI that dynamically fetches available models
 * Supports both Claude (Anthropic on Vertex) and Gemini models
 */
export class VertexModelProvider implements ModelProvider {
  private projectId: string = ""
  private region: string = "us-east5"
  private credentials: Record<string, unknown> | null = null
  private keyFile: string = ""
  private cache: Map<string, CacheEntry> = new Map()
  private cacheTTL = 3600000 // 1 hour
  private auth: GoogleAuth | null = null

  /**
   * Known model capabilities for Vertex AI models
   * Combines Claude and Gemini models available on Vertex
   */
  private modelCapabilities: Record<string, Partial<ModelInfo>> = {
    // Claude models on Vertex AI
    "claude-3-opus": {
      maxTokens: 4096,
      contextWindow: 200000,
      supportsImages: true,
      supportsPromptCache: true,
      inputPrice: 15.0,
      outputPrice: 75.0,
      description: "Claude 3 Opus via Vertex AI",
    },
    "claude-3-sonnet": {
      maxTokens: 4096,
      contextWindow: 200000,
      supportsImages: true,
      supportsPromptCache: true,
      inputPrice: 3.0,
      outputPrice: 15.0,
      description: "Claude 3 Sonnet via Vertex AI",
    },
    "claude-3-haiku": {
      maxTokens: 4096,
      contextWindow: 200000,
      supportsImages: true,
      supportsPromptCache: true,
      inputPrice: 0.25,
      outputPrice: 1.25,
      description: "Claude 3 Haiku via Vertex AI",
    },
    "claude-3-5-sonnet": {
      maxTokens: 8192,
      contextWindow: 200000,
      supportsImages: true,
      supportsPromptCache: true,
      inputPrice: 3.0,
      outputPrice: 15.0,
      description: "Claude 3.5 Sonnet via Vertex AI",
    },
    // Gemini models on Vertex AI
    "gemini-1.5-pro": {
      maxTokens: 8192,
      contextWindow: 2097152, // 2M tokens
      supportsImages: true,
      supportsPromptCache: false,
      inputPrice: 1.25,  // Vertex pricing different from AI Studio
      outputPrice: 5.0,
      description: "Gemini 1.5 Pro via Vertex AI",
    },
    "gemini-1.5-flash": {
      maxTokens: 8192,
      contextWindow: 1048576, // 1M tokens
      supportsImages: true,
      supportsPromptCache: false,
      inputPrice: 0.075,
      outputPrice: 0.3,
      description: "Gemini 1.5 Flash via Vertex AI",
    },
    "gemini-1.0-pro": {
      maxTokens: 2048,
      contextWindow: 32768,
      supportsImages: false,
      supportsPromptCache: false,
      inputPrice: 0.5,
      outputPrice: 1.5,
      description: "Gemini 1.0 Pro via Vertex AI",
    },
    "gemini-1.0-pro-vision": {
      maxTokens: 2048,
      contextWindow: 16384,
      supportsImages: true,
      supportsPromptCache: false,
      inputPrice: 0.25,
      outputPrice: 0.5,
      description: "Gemini 1.0 Pro Vision via Vertex AI",
    },
  }

  configure(options: ApiHandlerOptions): void {
    this.projectId = options.vertexProjectId || ""
    this.region = options.vertexRegion || "us-east5"
    this.keyFile = options.vertexKeyFile || ""
    
    if (options.vertexJsonCredentials) {
      try {
        this.credentials = JSON.parse(options.vertexJsonCredentials) as Record<string, unknown>
      } catch (error) {
        console.error("Failed to parse Vertex credentials:", error)
      }
    }
    
    // Initialize Google Auth
    this.initializeAuth()
  }

  private initializeAuth(): void {
    try {
      const authOptions: Record<string, unknown> = {
        scopes: ["https://www.googleapis.com/auth/cloud-platform"],
      }
      
      if (this.credentials) {
        authOptions.credentials = this.credentials
      } else if (this.keyFile) {
        authOptions.keyFile = this.keyFile
      }
      // If neither is provided, will use Application Default Credentials
      
      this.auth = new GoogleAuth(authOptions)
    } catch (error) {
      console.error("Failed to initialize Vertex AI authentication:", error)
      this.auth = null
    }
  }

  async listModels(forceRefresh = false): Promise<ModelListing[]> { return this.getModels(forceRefresh) }

  async getModels(forceRefresh = false): Promise<ModelListing[]> {
    const cacheKey = `vertex_models_${this.projectId}_${this.region}`
    
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
      console.error("Failed to fetch Vertex AI models:", error)
      
      // Return cached models if available
      const cached = this.cache.get(cacheKey)
      if (cached) {
        console.log("Using cached Vertex AI models due to API error")
        return cached.models
      }
      
      // Return fallback models
      return this.getFallbackModels()
    }
  }

  private async fetchModels(): Promise<ModelListing[]> {
    if (!this.projectId || !this.auth) {
      throw new Error("Vertex AI project ID or authentication not configured")
    }

    // Get access token
    const client = await this.auth.getClient()
    const accessToken = await client.getAccessToken()
    
    if (!accessToken.token) {
      throw new Error("Failed to get Vertex AI access token")
    }

    // Fetch both publisher models and foundation models
    const [publisherModels, foundationModels] = await Promise.all([
      this.fetchPublisherModels(accessToken.token),
      this.fetchFoundationModels(accessToken.token),
    ])

    return [...publisherModels, ...foundationModels]
  }

  private async fetchPublisherModels(accessToken: string): Promise<ModelListing[]> {
    // Fetch Google's publisher models (Gemini)
    const url = `https://${this.region}-aiplatform.googleapis.com/v1/projects/${this.projectId}/locations/${this.region}/publishers/google/models`
    
    try {
      const response = await axios.get<{ models?: Array<{ name?: string }> }>(url, {
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
      })
      
      const models: ModelListing[] = []
      
      for (const model of response.data.models || []) {
        const modelName = model.name || ""
        const modelId = this.extractModelId(modelName)
        
        if (this.isGeminiModel(modelId)) {
          const capabilities = this.getModelCapabilities(modelId)
          models.push({ id: modelId, info: capabilities })
        }
      }
      
      return models
    } catch (error) {
      console.warn("Failed to fetch publisher models:", error)
      return []
    }
  }

  private async fetchFoundationModels(accessToken: string): Promise<ModelListing[]> {
    // Fetch Anthropic models (Claude) via partner models
    const url = `https://${this.region}-aiplatform.googleapis.com/v1/projects/${this.projectId}/locations/${this.region}/models`
    
    try {
      const response = await axios.get<{ models?: Array<{ name?: string }> }>(url, {
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        params: {
          filter: 'publisher="anthropic"',
        },
      })
      
      const models: ModelListing[] = []
      
      for (const model of response.data.models || []) {
        const modelName = model.name || ""
        const modelId = this.extractModelId(modelName)
        
        if (this.isClaudeModel(modelId)) {
          const capabilities = this.getModelCapabilities(modelId)
          models.push({ id: modelId, info: capabilities })
        }
      }
      
      return models
    } catch (error) {
      console.warn("Failed to fetch foundation models:", error)
      return []
    }
  }

  private extractModelId(modelName: string): string {
    // Extract model ID from full resource name
    // e.g., "projects/PROJECT/locations/REGION/publishers/google/models/gemini-1.5-pro"
    const parts = modelName.split("/")
    return parts[parts.length - 1] || modelName
  }

  private isClaudeModel(modelId: string): boolean {
    return modelId.includes("claude")
  }

  private isGeminiModel(modelId: string): boolean {
    return modelId.includes("gemini")
  }

  private getModelCapabilities(modelId: string): ModelInfo {
    // Check for exact match first
    if (this.modelCapabilities[modelId]) {
      return this.modelCapabilities[modelId] as ModelInfo
    }
    
    // Check for partial matches
    for (const [key, capabilities] of Object.entries(this.modelCapabilities)) {
      if (modelId.includes(key.replace(/[-@].*/, ""))) {
        return capabilities as ModelInfo
      }
    }
    
    // Derive capabilities from model name
    const isClaude = this.isClaudeModel(modelId)
    const isGemini = this.isGeminiModel(modelId)
    
    if (isClaude) {
      const isOpus = modelId.includes("opus")
      const isSonnet = modelId.includes("sonnet")
      
      return {
        maxTokens: isSonnet && modelId.includes("3-5") ? 8192 : 4096,
        contextWindow: 200000,
        supportsImages: true,
        supportsPromptCache: true,
        inputPrice: isOpus ? 15.0 : isSonnet ? 3.0 : 0.25,
        outputPrice: isOpus ? 75.0 : isSonnet ? 15.0 : 1.25,
        description: `Claude model via Vertex AI`,
      }
    } else if (isGemini) {
      const isPro = modelId.includes("pro")
      const isFlash = modelId.includes("flash")
      const hasVision = modelId.includes("vision")
      const version = modelId.includes("1.5") ? "1.5" : "1.0"
      
      return {
        maxTokens: version === "1.5" ? 8192 : 2048,
        contextWindow: version === "1.5" && isPro ? 2097152 : 
                       version === "1.5" && isFlash ? 1048576 :
                       hasVision ? 16384 : 32768,
        supportsImages: isPro || isFlash || hasVision,
        supportsPromptCache: false,
        inputPrice: isPro && version === "1.5" ? 1.25 : 
                    isFlash ? 0.075 : 
                    hasVision ? 0.25 : 0.5,
        outputPrice: isPro && version === "1.5" ? 5.0 :
                     isFlash ? 0.3 :
                     hasVision ? 0.5 : 1.5,
        description: `Gemini model via Vertex AI`,
      }
    }
    
    // Default fallback
    return {
      maxTokens: 4096,
      contextWindow: 32768,
      supportsImages: false,
      supportsPromptCache: false,
      inputPrice: 1.0,
      outputPrice: 3.0,
      description: "Model via Vertex AI",
    }
  }

  async getModelInfo(modelId: string): Promise<ModelInfo | null> {
    const models = await this.getModels()
    const model = models.find((m) => m.id === modelId)
    return model?.info || null
  }

  async getDefaultModelId(): Promise<string> {
    // Default to Claude 3.5 Sonnet if available
    const models = await this.getModels()
    
    // Try to find Claude 3.5 Sonnet
    const claude35 = models.find((m) => m.id.includes("claude-3-5-sonnet"))
    if (claude35) return claude35.id
    
    // Fallback to Gemini 1.5 Flash (good performance/cost ratio)
    const geminiFlash = models.find((m) => m.id.includes("gemini-1.5-flash"))
    if (geminiFlash) return geminiFlash.id
    
    // Fallback to any Claude model
    const anyClaude = models.find((m) => m.id.includes("claude"))
    if (anyClaude) return anyClaude.id
    
    // Return first available model
    return models[0]?.id || "claude-3-5-sonnet@20241022"
  }

  private getFallbackModels(): ModelListing[] {
    // Return known Vertex AI models as fallback
    return Object.entries(this.modelCapabilities).map(([id, info]) => ({
      id,
      info: info as ModelInfo,
    }))
  }

  /**
   * Check if the provider is properly configured
   */
  isConfigured(): boolean {
    return !!(this.projectId && this.auth)
  }

  /**
   * Get available regions for Vertex AI
   */
  static getAvailableRegions(): string[] {
    return [
      "us-central1",
      "us-east1", 
      "us-east4",
      "us-west1",
      "us-west4",
      "europe-west1",
      "europe-west4",
      "asia-east1",
      "asia-northeast1",
      "asia-southeast1",
    ]
  }

  /**
   * Validate project ID format
   */
  static validateProjectId(projectId: string): { isValid: boolean; errorMessage?: string } {
    if (!projectId) {
      return { isValid: false, errorMessage: "Project ID is required" }
    }
    
    // GCP project ID rules
    if (!/^[a-z][\w-]+[a-z0-9]$/.test(projectId) || projectId.length > 30) {
      return { 
        isValid: false, 
        errorMessage: "Invalid project ID format. Must be 6-30 characters, lowercase letters, digits, and hyphens only." 
      }
    }
    
    return { isValid: true }
  }
}
