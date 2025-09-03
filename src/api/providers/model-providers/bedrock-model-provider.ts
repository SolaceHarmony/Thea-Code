import { BedrockClient, ListFoundationModelsCommand } from "@aws-sdk/client-bedrock"
import { ModelProvider, ModelListing } from "../model-registry"
import { ModelInfo, ApiHandlerOptions } from "../../../shared/api"

interface BedrockModelSummary {
  modelArn?: string
  modelId?: string
  modelName?: string
  providerName?: string
  inputModalities?: string[]
  outputModalities?: string[]
  responseStreamingSupported?: boolean
  customizationsSupported?: string[]
  inferenceTypesSupported?: string[]
  modelLifecycle?: {
    status?: string
  }
}

interface CacheEntry {
  models: ModelListing[]
  timestamp: number
}

/**
 * Model provider for AWS Bedrock that dynamically fetches available foundation models
 */
export class BedrockModelProvider implements ModelProvider {
  private client: BedrockClient | null = null
  private cache: Map<string, CacheEntry> = new Map()
  private cacheTTL = 3600000 // 1 hour
  private region: string = "us-east-1"
  private credentials: { accessKeyId: string; secretAccessKey: string; sessionToken?: string } | null = null

  /**
   * Known model capabilities based on model ID patterns
   * This provides fallback information when API doesn't return complete details
   */
  private modelCapabilities: Record<string, Partial<ModelInfo>> = {
    // Anthropic Claude models on Bedrock
    "anthropic.claude-3-opus": {
      maxTokens: 4096,
      contextWindow: 200000,
      supportsImages: true,
      supportsPromptCache: true,
      inputPrice: 15.0,
      outputPrice: 75.0,
    },
    "anthropic.claude-3-sonnet": {
      maxTokens: 4096,
      contextWindow: 200000,
      supportsImages: true,
      supportsPromptCache: true,
      inputPrice: 3.0,
      outputPrice: 15.0,
    },
    "anthropic.claude-3-haiku": {
      maxTokens: 4096,
      contextWindow: 200000,
      supportsImages: true,
      supportsPromptCache: true,
      inputPrice: 0.25,
      outputPrice: 1.25,
    },
    "anthropic.claude-3-5-sonnet": {
      maxTokens: 8192,
      contextWindow: 200000,
      supportsImages: true,
      supportsPromptCache: true,
      inputPrice: 3.0,
      outputPrice: 15.0,
    },
    // Amazon Titan models
    "amazon.titan-text-express": {
      maxTokens: 8192,
      contextWindow: 8192,
      supportsImages: false,
      inputPrice: 0.2,
      outputPrice: 0.6,
    },
    "amazon.titan-text-lite": {
      maxTokens: 4096,
      contextWindow: 4096,
      supportsImages: false,
      inputPrice: 0.15,
      outputPrice: 0.45,
    },
    // Meta Llama models
    "meta.llama3-8b-instruct": {
      maxTokens: 2048,
      contextWindow: 8192,
      supportsImages: false,
      inputPrice: 0.3,
      outputPrice: 0.6,
    },
    "meta.llama3-70b-instruct": {
      maxTokens: 2048,
      contextWindow: 8192,
      supportsImages: false,
      inputPrice: 2.65,
      outputPrice: 3.5,
    },
    // Cohere models
    "cohere.command-r-plus": {
      maxTokens: 4000,
      contextWindow: 128000,
      supportsImages: false,
      inputPrice: 3.0,
      outputPrice: 15.0,
    },
    // Mistral models
    "mistral.mistral-large": {
      maxTokens: 8192,
      contextWindow: 32000,
      supportsImages: false,
      inputPrice: 8.0,
      outputPrice: 24.0,
    },
  }

  configure(options: ApiHandlerOptions): void {
    this.region = options.awsRegion || "us-east-1"
    this.credentials = options.awsAccessKey && options.awsSecretKey
      ? { accessKeyId: options.awsAccessKey, secretAccessKey: options.awsSecretKey, sessionToken: options.awsSessionToken }
      : null

    // Initialize Bedrock client with configuration
    this.client = new BedrockClient({
      region: this.region,
      credentials: this.credentials ?? undefined,
    })
  }

  async getModels(forceRefresh = false): Promise<ModelListing[]> {
    const cacheKey = `models_${this.region}`
    
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
      console.error("Failed to fetch Bedrock models:", error)
      
      // Return cached models if available
      const cached = this.cache.get(cacheKey)
      if (cached) {
        console.log("Using cached Bedrock models due to API error")
        return cached.models
      }
      
      // Return fallback models
      return this.getFallbackModels()
    }
  }

  private async fetchModels(): Promise<ModelListing[]> {
    if (!this.client) {
      throw new Error("Bedrock client not configured")
    }

    const command = new ListFoundationModelsCommand({})
    const response = await this.client.send(command)
    
    const models: ModelListing[] = []
    
    for (const model of response.modelSummaries || []) {
      const modelId = model.modelId || ""
      const baseId = this.getBaseModelId(modelId)
      
      // Get known capabilities or derive them
      const capabilities = this.modelCapabilities[baseId] || this.deriveCapabilities(model)
      
      models.push({
        id: modelId,
        info: {
          maxTokens: capabilities.maxTokens || 4096,
          contextWindow: capabilities.contextWindow || 4096,
          supportsImages: model.inputModalities?.includes("IMAGE") || capabilities.supportsImages || false,
          supportsPromptCache: capabilities.supportsPromptCache || false,
          inputPrice: capabilities.inputPrice,
          outputPrice: capabilities.outputPrice,
          supportsStreaming: model.responseStreamingSupported || false,
          provider: model.providerName,
          description: `${model.providerName} ${model.modelName}`,
        },
      })
    }
    
    return models
  }

  private getBaseModelId(modelId: string): string {
    // Remove version numbers and date stamps
    // e.g., "anthropic.claude-3-opus-20240229-v1:0" -> "anthropic.claude-3-opus"
    return modelId.split("-").slice(0, 3).join("-").split(":")[0]
  }

  private deriveCapabilities(model: BedrockModelSummary): Partial<ModelInfo> {
    const capabilities: Partial<ModelInfo> = {}
    
    // Derive from provider name
    const provider = model.providerName?.toLowerCase()
    
    if (provider?.includes("anthropic")) {
      capabilities.supportsPromptCache = true
      capabilities.contextWindow = 200000
    } else if (provider?.includes("meta")) {
      capabilities.contextWindow = 8192
    } else if (provider?.includes("amazon")) {
      capabilities.contextWindow = 8192
    }
    
    // Check modalities
    if (model.inputModalities?.includes("IMAGE")) {
      capabilities.supportsImages = true
    }
    
    return capabilities
  }

  async getModelInfo(modelId: string): Promise<ModelInfo | null> {
    const models = await this.getModels()
    const model = models.find((m) => m.modelId === modelId)
    return model?.info || null
  }

  async getDefaultModelId(): Promise<string> {
    // Default to Claude 3.5 Sonnet if available
    const models = await this.getModels()
    
    // Try to find Claude 3.5 Sonnet
    const claude35 = models.find((m) => m.id.includes("claude-3-5-sonnet"))
    if (claude35) return claude35.id
    
    // Fallback to any Claude model
    const anyClaud = models.find((m) => m.id.includes("claude"))
    if (anyClaud) return anyClaud.id
    
    // Return first available model
    return models[0]?.id || "anthropic.claude-3-5-sonnet-20241022-v2:0"
  }

  private getFallbackModels(): ModelListing[] {
    // Return a basic set of known models as fallback
    return [
      {
        id: "anthropic.claude-3-5-sonnet-20241022-v2:0",
        info: this.modelCapabilities["anthropic.claude-3-5-sonnet"] as ModelInfo,
      },
      {
        id: "anthropic.claude-3-opus-20240229-v1:0",
        info: this.modelCapabilities["anthropic.claude-3-opus"] as ModelInfo,
      },
      {
        id: "anthropic.claude-3-haiku-20240307-v1:0",
        info: this.modelCapabilities["anthropic.claude-3-haiku"] as ModelInfo,
      },
    ]
  }

  /**
   * Check if custom ARN is a valid Bedrock model ARN
   */
  static validateArn(arn: string): { isValid: boolean; errorMessage?: string } {
    const arnRegex = /^arn:aws:bedrock:[^:]+:\d+:(foundation-model|provisioned-model|inference-profile)\/[^:]+$/
    
    if (!arnRegex.test(arn)) {
      return {
        isValid: false,
        errorMessage: "Invalid ARN format. Expected: arn:aws:bedrock:region:account:resource-type/resource-id",
      }
    }
    
    return { isValid: true }
  }
}
