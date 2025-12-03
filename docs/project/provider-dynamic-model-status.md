# Provider Dynamic Model Support Status

## Summary
Overview of which providers support dynamic model fetching vs static model definitions.

## Provider Status

### ✅ Fully Dynamic (Already Implemented)
These providers fetch models dynamically from their APIs:

1. **OpenRouter** (`getOpenRouterModels`)
   - Fetches from: `https://openrouter.ai/api/v1/models`
   - Returns complete model info with pricing, capabilities
   - Used in webview via `refreshOpenRouterModels`

2. **Glama** (`getGlamaModels`)
   - Fetches from: `https://glama.ai/api/gateway/v1/models`
   - Returns model capabilities and pricing
   - Used in webview via `refreshGlamaModels`

3. **Requesty** (`getRequestyModels`)
   - Fetches from: `https://router.requesty.ai/v1/models`
   - Returns model info with pricing
   - Used in webview via `refreshRequestyModels`

4. **Ollama** (`getOllamaModels`)
   - Fetches from local Ollama instance
   - Returns available local models
   - Used in webview via `requestOllamaModels`

5. **OpenAI** (`getOpenAiModels`)
   - Fetches from OpenAI API or compatible endpoints
   - Returns model list
   - Used in webview via `refreshOpenAiModels`

6. **Anthropic** (NEW - just implemented)
   - Uses ModelRegistry with AnthropicModelProvider
   - Pattern-based capability detection
   - Used in webview via `refreshAnthropicModels`

### ⚠️ Semi-Dynamic (Structured but Static)
These providers have functions that return models but currently use static data:

7. **Unbound** (`getUnboundModels`)
   - Returns static model list
   - Could easily be converted to fetch from API

### ❌ Static Models (Need Dynamic Implementation)
These providers still use hardcoded model definitions:

8. **Vertex** 
   - Uses: `vertexModels` from `shared/api.ts`
   - Supports both Claude and Gemini models on GCP
   - Needs: Dynamic fetching from Vertex AI API

9. **Bedrock**
   - Uses: `bedrockModels` from `shared/api.ts`
   - Supports various models on AWS
   - Needs: Dynamic fetching from Bedrock API

10. **Gemini** (Google AI Studio)
    - Uses: `geminiModels` from `shared/api.ts`
    - Direct Google Gemini API
    - Needs: Dynamic fetching from Gemini API

11. **DeepSeek**
    - Uses: `deepSeekModels` from `shared/api.ts`
    - Needs: Dynamic fetching from DeepSeek API

12. **Mistral**
    - Uses: `mistralModels` from `shared/api.ts`
    - Needs: Dynamic fetching from Mistral API

### 🔧 Other Providers

13. **LM Studio** (`getLmStudioModels`)
    - Already dynamic, fetches from local instance
    - Used in webview via `requestLmStudioModels`

14. **VS Code LM** (`getVsCodeLmModels`)
    - Already dynamic, fetches from VS Code API
    - Used in webview via `requestVsCodeLmModels`

15. **Human Relay**
    - Special provider, doesn't need model selection

## Implementation Priority

### High Priority (Popular providers)
1. **Vertex** - Major cloud provider (GCP)
2. **Bedrock** - Major cloud provider (AWS)
3. **Gemini** - Direct Google AI access

### Medium Priority
4. **Mistral** - Popular European AI provider
5. **DeepSeek** - Growing in popularity

### Low Priority
6. **Unbound** - Already semi-dynamic, easy to convert when needed

## Next Steps

1. **Create model providers for static providers:**
   - VertexModelProvider
   - BedrockModelProvider
   - GeminiModelProvider
   - MistralModelProvider
   - DeepSeekModelProvider

2. **Add to ModelRegistry:**
   - Register each new provider
   - Implement caching and fallback

3. **Update webview message handlers:**
   - Add refresh handlers for each provider
   - Update initialization code

4. **Update frontend:**
   - Add dynamic model state for each provider
   - Update model selection logic

5. **Remove static definitions:**
   - Once all providers are dynamic
   - Clean up `shared/api.ts`

## Benefits of Dynamic Models

- **Automatic Updates**: New models available without code changes
- **Accurate Pricing**: Real-time pricing information
- **Capability Detection**: Accurate feature support info
- **Reduced Maintenance**: No manual model list updates
- **Better UX**: Users see only available models