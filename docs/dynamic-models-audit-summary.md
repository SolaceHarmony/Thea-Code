# Dynamic Models Audit Summary

## Current Implementation Status

### ✅ Providers with Full Dynamic Support
These providers already fetch models dynamically and display them in the UI:

1. **OpenRouter** 
   - Backend: `getOpenRouterModels()` fetches from API
   - Frontend: Uses `ModelPicker` component
   - Refresh: `refreshOpenRouterModels` message
   - Status: **Fully functional**

2. **Glama**
   - Backend: `getGlamaModels()` fetches from API
   - Frontend: Uses `ModelPicker` component  
   - Refresh: `refreshGlamaModels` message
   - Status: **Fully functional**

3. **Unbound**
   - Backend: `getUnboundModels()` (currently returns static data)
   - Frontend: Uses `ModelPicker` component
   - Refresh: `refreshUnboundModels` message
   - Status: **Semi-functional** (structured but not fetching from API)

4. **Requesty**
   - Backend: `getRequestyModels()` fetches from API
   - Frontend: Uses `ModelPicker` component
   - Refresh: `refreshRequestyModels` message
   - Status: **Fully functional**

5. **OpenAI**
   - Backend: `getOpenAiModels()` fetches from API
   - Frontend: Uses dynamic state (`openAiModels`)
   - Refresh: `refreshOpenAiModels` message
   - Status: **Fully functional**

6. **Ollama**
   - Backend: `getOllamaModels()` fetches from local instance
   - Frontend: Uses dynamic state (`ollamaModels`)
   - Refresh: `requestOllamaModels` message
   - Status: **Fully functional**

7. **LM Studio**
   - Backend: `getLmStudioModels()` fetches from local instance
   - Frontend: Uses dynamic state (`lmStudioModels`)
   - Refresh: `requestLmStudioModels` message
   - Status: **Fully functional**

8. **VS Code LM**
   - Backend: `getVsCodeLmModels()` fetches from VS Code API
   - Frontend: Uses dynamic state (`vsCodeLmModels`)
   - Refresh: `requestVsCodeLmModels` message
   - Status: **Fully functional**

9. **Anthropic** (NEW)
   - Backend: Uses `ModelRegistry` with `AnthropicModelProvider`
   - Frontend: Uses dynamic state (`anthropicModels`)
   - Refresh: `refreshAnthropicModels` message
   - Status: **Fully functional**

### ❌ Providers Still Using Static Models

These providers use `MODELS_BY_PROVIDER` and static imports from `shared/api.ts`:

1. **Vertex**
   - Currently: Uses `vertexModels` static import
   - Frontend: Uses `selectedProviderModelOptions` from `MODELS_BY_PROVIDER`
   - Needs: Dynamic model provider implementation

2. **Bedrock**
   - Currently: Uses `bedrockModels` static import
   - Frontend: Uses `selectedProviderModelOptions` from `MODELS_BY_PROVIDER`
   - Special: Has custom ARN support
   - Needs: Dynamic model provider implementation

3. **Gemini**
   - Currently: Uses `geminiModels` static import
   - Frontend: Uses `selectedProviderModelOptions` from `MODELS_BY_PROVIDER`
   - Needs: Dynamic model provider implementation

4. **DeepSeek**
   - Currently: Uses `deepSeekModels` static import
   - Frontend: Uses `selectedProviderModelOptions` from `MODELS_BY_PROVIDER`
   - Needs: Dynamic model provider implementation

5. **Mistral**
   - Currently: Uses `mistralModels` static import
   - Frontend: Uses `selectedProviderModelOptions` from `MODELS_BY_PROVIDER`
   - Needs: Dynamic model provider implementation

## Frontend Model Selection Patterns

### Pattern 1: ModelPicker Component
Used by: OpenRouter, Glama, Unbound, Requesty
```tsx
<ModelPicker
  models={dynamicModels}
  modelIdKey="providerModelId"
  modelInfoKey="providerModelInfo"
  ...
/>
```

### Pattern 2: Direct Select Component
Used by: Vertex, Bedrock, Gemini, DeepSeek, Mistral
```tsx
<Select value={selectedModelId}>
  {selectedProviderModelOptions.map(option => 
    <SelectItem value={option.value}>{option.label}</SelectItem>
  )}
</Select>
```

### Pattern 3: Custom Implementation
Used by: OpenAI, Ollama, LM Studio, VS Code LM
- Each has unique UI requirements
- Uses dynamic state directly

## Required Updates

### Backend Tasks
1. Create model providers for static providers:
   - `VertexModelProvider`
   - `BedrockModelProvider`
   - `GeminiModelProvider`
   - `DeepSeekModelProvider`
   - `MistralModelProvider`

2. Register providers with `ModelRegistry`

3. Add refresh handlers in `webviewMessageHandler.ts`

### Frontend Tasks
1. Add dynamic model states for each provider
2. Update `selectedProviderModelOptions` logic to use dynamic models
3. Add refresh triggers for each provider
4. Update `normalizeApiConfiguration` to handle all dynamic models

## Key Observations

1. **ModelPicker vs Select**: Providers using `ModelPicker` already have better support for dynamic models
2. **Caching**: Most dynamic providers cache models locally for offline use
3. **Refresh Pattern**: All dynamic providers follow similar refresh message patterns
4. **Initialization**: Models are loaded on webview launch and refreshed on provider selection

## Migration Strategy

1. **Phase 1**: Implement backend model providers (Already done for Anthropic)
2. **Phase 2**: Update frontend to use dynamic models (In progress)
3. **Phase 3**: Remove static model definitions
4. **Phase 4**: Unify model selection UI components

## Benefits When Complete

- No more manual model list updates
- Real-time pricing and capability information
- Automatic support for new models
- Consistent model selection experience
- Reduced maintenance burden