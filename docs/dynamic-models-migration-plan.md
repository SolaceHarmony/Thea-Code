# Dynamic Models Migration Plan

## Executive Summary
Migrate 5 remaining static model providers (Vertex, Bedrock, Gemini, DeepSeek, Mistral) to dynamically fetch models from their respective APIs, following the pattern established with AnthropicModelProvider.

## Provider API Research

### 1. AWS Bedrock
**API Endpoint**: `ListFoundationModels`
```typescript
// AWS SDK method
const command = new ListFoundationModelsCommand({});
const response = await bedrockClient.send(command);
```
**Response includes**:
- Model ID/ARN
- Model name
- Provider name
- Input/output modalities
- Supported use cases
- Model lifecycle status

### 2. Google Vertex AI
**API Endpoint**: `projects.locations.models.list`
```typescript
// REST API
GET https://REGION-aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/REGION/models
```
**Alternative**: Use publisher models endpoint
```typescript
GET https://REGION-aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/REGION/publishers/google/models
```

### 3. Google Gemini (AI Studio)
**API Endpoint**: `models.list`
```typescript
// REST API
GET https://generativelanguage.googleapis.com/v1/models
```
**Response includes**:
- Model name
- Supported generation methods
- Input/output token limits
- Temperature ranges

### 4. Mistral AI
**API Endpoint**: `/v1/models`
```typescript
// REST API
GET https://api.mistral.ai/v1/models
```
**Response**: OpenAI-compatible format with model list

### 5. DeepSeek
**API Endpoint**: `/v1/models` (OpenAI-compatible)
```typescript
// REST API
GET https://api.deepseek.com/v1/models
```
**Response**: OpenAI-compatible format

## Implementation Architecture

### Phase 1: Backend Model Providers (Week 1)

#### 1.1 Create Base Infrastructure
```typescript
// src/api/providers/model-providers/base-cloud-provider.ts
export abstract class BaseCloudModelProvider implements ModelProvider {
  protected cache: Map<string, CacheEntry> = new Map()
  protected cacheTTL: number = 3600000 // 1 hour
  
  abstract fetchModels(): Promise<ModelListing[]>
  abstract getDefaultModelId(): string
  
  async getModels(forceRefresh = false): Promise<ModelListing[]> {
    // Implement caching logic
    // Handle API failures with fallback
    // Transform API response to ModelListing
  }
}
```

#### 1.2 Implement BedrockModelProvider
```typescript
// src/api/providers/model-providers/bedrock-model-provider.ts
export class BedrockModelProvider extends BaseCloudModelProvider {
  private client: BedrockClient
  
  async fetchModels(): Promise<ModelListing[]> {
    const command = new ListFoundationModelsCommand({})
    const response = await this.client.send(command)
    
    return response.modelSummaries.map(model => ({
      modelId: model.modelId,
      info: {
        maxTokens: this.getMaxTokens(model),
        contextWindow: this.getContextWindow(model),
        supportsImages: model.inputModalities?.includes('IMAGE'),
        supportsPromptCache: this.supportsCache(model),
        inputPrice: this.getInputPrice(model),
        outputPrice: this.getOutputPrice(model),
      }
    }))
  }
}
```

#### 1.3 Implement VertexModelProvider
```typescript
// src/api/providers/model-providers/vertex-model-provider.ts
export class VertexModelProvider extends BaseCloudModelProvider {
  async fetchModels(): Promise<ModelListing[]> {
    // Fetch both Claude and Gemini models available on Vertex
    const [claudeModels, geminiModels] = await Promise.all([
      this.fetchClaudeModels(),
      this.fetchGeminiModels()
    ])
    
    return [...claudeModels, ...geminiModels]
  }
  
  private async fetchClaudeModels() {
    // Use Vertex AI API to list Claude models
  }
  
  private async fetchGeminiModels() {
    // Use Vertex AI API to list Gemini models
  }
}
```

#### 1.4 Implement GeminiModelProvider
```typescript
// src/api/providers/model-providers/gemini-model-provider.ts
export class GeminiModelProvider extends BaseCloudModelProvider {
  async fetchModels(): Promise<ModelListing[]> {
    const response = await fetch('https://generativelanguage.googleapis.com/v1/models', {
      headers: { 'x-goog-api-key': this.apiKey }
    })
    
    const data = await response.json()
    return data.models.map(this.transformModel)
  }
}
```

#### 1.5 Implement MistralModelProvider
```typescript
// src/api/providers/model-providers/mistral-model-provider.ts
export class MistralModelProvider extends BaseCloudModelProvider {
  async fetchModels(): Promise<ModelListing[]> {
    const response = await fetch('https://api.mistral.ai/v1/models', {
      headers: { 'Authorization': `Bearer ${this.apiKey}` }
    })
    
    const data = await response.json()
    return data.data.map(this.transformModel)
  }
}
```

#### 1.6 Implement DeepSeekModelProvider
```typescript
// src/api/providers/model-providers/deepseek-model-provider.ts
export class DeepSeekModelProvider extends BaseCloudModelProvider {
  async fetchModels(): Promise<ModelListing[]> {
    const response = await fetch('https://api.deepseek.com/v1/models', {
      headers: { 'Authorization': `Bearer ${this.apiKey}` }
    })
    
    const data = await response.json()
    return data.data.map(this.transformModel)
  }
}
```

### Phase 2: Model Registry Integration (Day 3-4)

#### 2.1 Register Providers
```typescript
// src/api/providers/model-registry.ts
export class ModelRegistry {
  constructor() {
    // Register cloud providers
    this.registerProvider('bedrock', new BedrockModelProvider())
    this.registerProvider('vertex', new VertexModelProvider())
    this.registerProvider('gemini', new GeminiModelProvider())
    this.registerProvider('mistral', new MistralModelProvider())
    this.registerProvider('deepseek', new DeepSeekModelProvider())
  }
}
```

#### 2.2 Add Helper Functions
```typescript
// src/api/providers/provider-helpers.ts
export async function getBedrockModels(options: ApiHandlerOptions) {
  const registry = ModelRegistry.getInstance()
  const provider = registry.getProvider('bedrock')
  provider.configure(options)
  return await provider.getModels()
}

// Similar for other providers...
```

### Phase 3: Backend Message Handlers (Day 4-5)

#### 3.1 Update webviewMessageHandler.ts
```typescript
// Add new message handlers
case "refreshBedrockModels": {
  const modelRegistry = ModelRegistry.getInstance()
  const bedrockModels = await modelRegistry.getModels("bedrock", true)
  
  if (bedrockModels.length > 0) {
    const modelsRecord = transformToRecord(bedrockModels)
    await cacheModels("bedrock", modelsRecord)
    await provider.postMessageToWebview({ 
      type: "bedrockModels", 
      bedrockModels: modelsRecord 
    })
  }
  break
}

// Similar for other providers...
```

#### 3.2 Add Initialization Code
```typescript
// On webview launch, load all dynamic models
const modelProviders = ['bedrock', 'vertex', 'gemini', 'mistral', 'deepseek']

for (const providerName of modelProviders) {
  // Load cached models
  void provider.readModelsFromCache(`${providerName}_models.json`).then(models => {
    if (models) {
      void provider.postMessageToWebview({ 
        type: `${providerName}Models`, 
        [`${providerName}Models`]: models 
      })
    }
  })
  
  // Fetch fresh models
  void modelRegistry.getModels(providerName).then(models => {
    // Cache and send to webview
  })
}
```

### Phase 4: Frontend Updates (Day 5-6)

#### 4.1 Update ApiOptions.tsx
```typescript
// Add state for each provider
const [bedrockModels, setBedrockModels] = useState<Record<string, ModelInfo> | null>(null)
const [vertexModels, setVertexModels] = useState<Record<string, ModelInfo> | null>(null)
const [geminiModels, setGeminiModels] = useState<Record<string, ModelInfo> | null>(null)
const [mistralModels, setMistralModels] = useState<Record<string, ModelInfo> | null>(null)
const [deepseekModels, setDeepseekModels] = useState<Record<string, ModelInfo> | null>(null)

// Update refresh logic
useDebounce(() => {
  switch(selectedProvider) {
    case "bedrock":
      vscode.postMessage({ type: "refreshBedrockModels" })
      break
    case "vertex":
      vscode.postMessage({ type: "refreshVertexModels" })
      break
    // ... other cases
  }
}, 250, [selectedProvider])

// Update message handler
case "bedrockModels": {
  const updatedModels = message.bedrockModels ?? {}
  setBedrockModels(updatedModels)
  break
}
// ... other cases
```

#### 4.2 Update Model Selection Logic
```typescript
const selectedProviderModelOptions = useMemo(() => {
  // Use dynamic models for all providers
  const dynamicProviders = {
    anthropic: anthropicModels,
    bedrock: bedrockModels,
    vertex: vertexModels,
    gemini: geminiModels,
    mistral: mistralModels,
    deepseek: deepseekModels,
  }
  
  if (selectedProvider in dynamicProviders) {
    const models = dynamicProviders[selectedProvider]
    if (models) {
      return Object.keys(models).map(modelId => ({
        value: modelId,
        label: modelId,
      }))
    }
  }
  
  // Fallback to static models (temporarily)
  return MODELS_BY_PROVIDER[selectedProvider] ? 
    Object.keys(MODELS_BY_PROVIDER[selectedProvider]).map(modelId => ({
      value: modelId,
      label: modelId,
    })) : []
}, [selectedProvider, anthropicModels, bedrockModels, vertexModels, geminiModels, mistralModels, deepseekModels])
```

#### 4.3 Update normalizeApiConfiguration
```typescript
export function normalizeApiConfiguration(
  apiConfiguration?: ApiConfiguration,
  dynamicModels?: {
    anthropicModels?: Record<string, ModelInfo> | null
    bedrockModels?: Record<string, ModelInfo> | null
    vertexModels?: Record<string, ModelInfo> | null
    geminiModels?: Record<string, ModelInfo> | null
    mistralModels?: Record<string, ModelInfo> | null
    deepseekModels?: Record<string, ModelInfo> | null
  }
) {
  // Use dynamic models when available
  switch (provider) {
    case "bedrock":
      const models = dynamicModels?.bedrockModels || bedrockModels
      return getProviderData(models, bedrockDefaultModelId)
    // ... other cases
  }
}
```

### Phase 5: Testing & Verification (Day 7)

#### 5.1 Unit Tests
```typescript
// Test each model provider
describe('BedrockModelProvider', () => {
  it('should fetch models from AWS', async () => {
    const provider = new BedrockModelProvider()
    const models = await provider.fetchModels()
    expect(models).toHaveLength(greaterThan(0))
    expect(models[0]).toHaveProperty('modelId')
    expect(models[0]).toHaveProperty('info')
  })
  
  it('should handle API failures gracefully', async () => {
    // Mock API failure
    const provider = new BedrockModelProvider()
    const models = await provider.getModels()
    expect(models).toEqual(fallbackModels)
  })
})
```

#### 5.2 Integration Tests
- Test model refresh on provider switch
- Test caching behavior
- Test fallback to static models
- Test API authentication
- Test model selection UI updates

#### 5.3 Manual Testing Checklist
- [ ] Each provider loads models on webview launch
- [ ] Models refresh when switching providers
- [ ] Model selection dropdown shows dynamic models
- [ ] Cached models load offline
- [ ] API failures show appropriate fallbacks
- [ ] Model info (pricing, capabilities) displays correctly

### Phase 6: Cleanup (Day 8)

#### 6.1 Remove Static Definitions
```typescript
// src/shared/api.ts
// Remove:
// - bedrockModels
// - vertexModels
// - geminiModels
// - mistralModels
// - deepSeekModels
```

#### 6.2 Update Documentation
- Update provider setup guides
- Document new environment variables
- Add troubleshooting section
- Update API reference

## Implementation Schedule

### Week 1
- **Day 1-2**: Implement model providers (backend)
- **Day 3**: Model Registry integration
- **Day 4**: Message handlers
- **Day 5**: Frontend updates
- **Day 6**: Testing
- **Day 7**: Cleanup & documentation

## Risk Mitigation

### 1. API Rate Limits
- **Risk**: Hitting provider API rate limits
- **Mitigation**: 
  - Implement exponential backoff
  - Cache models for 1 hour minimum
  - Add manual refresh cooldown

### 2. API Changes
- **Risk**: Provider APIs change format
- **Mitigation**:
  - Implement response validation
  - Maintain fallback to known models
  - Version API responses

### 3. Authentication Issues
- **Risk**: Complex auth for cloud providers
- **Mitigation**:
  - Support multiple auth methods
  - Clear error messages
  - Fallback to static models

### 4. Network Failures
- **Risk**: API calls fail due to network
- **Mitigation**:
  - Use cached models when available
  - Implement retry logic
  - Show offline indicators

## Success Criteria

1. **All 5 providers fetch models dynamically**
2. **No regression in existing functionality**
3. **Models update without code changes**
4. **Consistent UX across all providers**
5. **Robust error handling and fallbacks**
6. **Performance unchanged or improved**

## Configuration Requirements

### Environment Variables
```env
# AWS Bedrock
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx

# Google Vertex
VERTEX_PROJECT_ID=my-project
VERTEX_REGION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/path/to/creds.json

# Google Gemini
GEMINI_API_KEY=xxx

# Mistral
MISTRAL_API_KEY=xxx

# DeepSeek
DEEPSEEK_API_KEY=xxx
```

## Benefits When Complete

1. **Zero maintenance** for model updates
2. **Real-time pricing** information
3. **Accurate capabilities** per model
4. **Automatic new model** availability
5. **Consistent implementation** across providers
6. **Reduced bundle size** (no static lists)
7. **Better user experience** with live data

## Rollback Plan

If issues arise:
1. Keep static models as fallback
2. Feature flag for dynamic models
3. Gradual rollout per provider
4. Monitor error rates
5. Quick revert capability