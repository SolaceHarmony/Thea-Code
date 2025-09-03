/**
 * Helper functions to get models from dynamic providers
 * These functions provide a simple interface for the webview message handler
 */

import { ModelRegistry } from "./model-registry"
import { ApiHandlerOptions } from "../../shared/api"

/**
 * Get Bedrock models using the dynamic provider
 */
export async function getBedrockModels(options?: ApiHandlerOptions) {
  const registry = ModelRegistry.getInstance()
  
  if (options) {
    registry.configureProvider("bedrock", options)
  }
  
  const models = await registry.getModels("bedrock")
  
  // Transform to Record<string, ModelInfo> format expected by frontend
  const modelsRecord: Record<string, import("../../schemas").ModelInfo> = {}
  for (const model of models) {
    modelsRecord[model.id] = model.info
  }
  
  return modelsRecord
}

/**
 * Get Gemini models using the dynamic provider
 */
export async function getGeminiModels(options?: ApiHandlerOptions) {
  const registry = ModelRegistry.getInstance()
  
  if (options) {
    registry.configureProvider("gemini", options)
  }
  
  const models = await registry.getModels("gemini")
  
  // Transform to Record<string, ModelInfo> format expected by frontend
  const modelsRecord: Record<string, import("../../schemas").ModelInfo> = {}
  for (const model of models) {
    modelsRecord[model.id] = model.info
  }
  
  return modelsRecord
}

/**
 * Get Vertex models using the dynamic provider
 */
export async function getVertexModels(options?: ApiHandlerOptions) {
  const registry = ModelRegistry.getInstance()
  
  if (options) {
    registry.configureProvider("vertex", options)
  }
  
  const models = await registry.getModels("vertex")
  
  // Transform to Record<string, ModelInfo> format expected by frontend
  const modelsRecord: Record<string, import("../../schemas").ModelInfo> = {}
  for (const model of models) {
    modelsRecord[model.id] = model.info
  }
  
  return modelsRecord
}

/**
 * Get Mistral models using the dynamic provider
 */
export async function getMistralModels(options?: ApiHandlerOptions) {
  const registry = ModelRegistry.getInstance()
  
  if (options) {
    registry.configureProvider("mistral", options)
  }
  
  const models = await registry.getModels("mistral")
  
  // Transform to Record<string, ModelInfo> format expected by frontend
  const modelsRecord: Record<string, import("../../schemas").ModelInfo> = {}
  for (const model of models) {
    modelsRecord[model.id] = model.info
  }
  
  return modelsRecord
}

/**
 * Get DeepSeek models using the dynamic provider
 */
export async function getDeepSeekModels(options?: ApiHandlerOptions) {
  const registry = ModelRegistry.getInstance()
  
  if (options) {
    registry.configureProvider("deepseek", options)
  }
  
  const models = await registry.getModels("deepseek")
  
  // Transform to Record<string, ModelInfo> format expected by frontend
  const modelsRecord: Record<string, import("../../schemas").ModelInfo> = {}
  for (const model of models) {
    modelsRecord[model.id] = model.info
  }
  
  return modelsRecord
}

/**
 * Get Anthropic models using the dynamic provider (alternative to direct registry access)
 */
export async function getAnthropicModels(options?: ApiHandlerOptions) {
  const registry = ModelRegistry.getInstance()
  
  if (options) {
    registry.configureProvider("anthropic", options)
  }
  
  const models = await registry.getModels("anthropic")
  
  // Transform to Record<string, ModelInfo> format expected by frontend
  const modelsRecord: Record<string, import("../../schemas").ModelInfo> = {}
  for (const model of models) {
    modelsRecord[model.id] = model.info
  }
  
  return modelsRecord
}
