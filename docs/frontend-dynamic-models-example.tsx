/*
  Documentation example: Dynamic model fetching and selection across providers.
  This file is a self-contained TSX example and not part of the extension build.
*/

import React, { useCallback, useEffect, useMemo, useState } from 'react'

// Minimal type definitions used by the example
export type ProviderId =
  | 'anthropic'
  | 'bedrock'
  | 'vertex'
  | 'gemini'
  | 'mistral'
  | 'deepseek'
  | 'openrouter'
  | 'glama'

export interface ModelInfo {
  id?: string
  label?: string
  maxTokens?: number
  contextWindow?: number
  supportsPromptCache?: boolean
  supportsImages?: boolean
  // Any extra provider-specific metadata
  [key: string]: unknown
}

export interface ApiConfiguration {
  apiProvider?: ProviderId
  apiModelId?: string
}

// Messages from the extension to the webview UI
export type ExtensionMessage =
  | { type: 'anthropicModels'; anthropicModels: Record<string, ModelInfo> }
  | { type: 'bedrockModels'; bedrockModels: Record<string, ModelInfo> }
  | { type: 'vertexModels'; vertexModels: Record<string, ModelInfo> }
  | { type: 'geminiModels'; geminiModels: Record<string, ModelInfo> }
  | { type: 'mistralModels'; mistralModels: Record<string, ModelInfo> }
  | { type: 'deepseekModels'; deepseekModels: Record<string, ModelInfo> }

// Messages from the webview UI to the extension
export type UiRefreshMessage =
  | { type: 'refreshAnthropicModels' }
  | { type: 'refreshBedrockModels' }
  | { type: 'refreshVertexModels' }
  | { type: 'refreshGeminiModels' }
  | { type: 'refreshMistralModels' }
  | { type: 'refreshDeepSeekModels' }
  | { type: 'refreshOpenRouterModels' }
  | { type: 'refreshGlamaModels' }

// Safe shim for VSCode webview API in doc environment
const vscode: { postMessage: (message: UiRefreshMessage) => void } =
  typeof window !== 'undefined' && (window as any).acquireVsCodeApi
    ? (window as any).acquireVsCodeApi()
    : { postMessage: () => {} }

// Lightweight debounce hook for the example
function useDebounce(callback: () => void, delay: number, deps: React.DependencyList) {
  useEffect(() => {
    const handle = setTimeout(callback, delay)
    return () => clearTimeout(handle)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
}

// Fallback static models map (if a provider has not yet returned dynamic models)
const MODELS_BY_PROVIDER: Partial<Record<ProviderId, Record<string, ModelInfo>>> = {
  anthropic: {
    'claude-3-7-sonnet-latest': { contextWindow: 200_000, supportsImages: true },
  },
}

// Default model ids used as fallbacks per provider
const DEFAULT_MODEL_BY_PROVIDER: Record<ProviderId, string> = {
  anthropic: 'claude-3-7-sonnet-latest',
  bedrock: 'anthropic.claude-3-7-sonnet-2025-02-19-v1:0',
  vertex: 'claude-3-7-sonnet@20250219',
  gemini: 'gemini-1.5-pro',
  mistral: 'mistral-large-latest',
  deepseek: 'deepseek-chat',
  openrouter: 'openrouter/anthropic/claude-3.7-sonnet',
  glama: 'glama/sonnet',
}

// Normalize API configuration and select proper model metadata
export function normalizeApiConfiguration(
  apiConfiguration?: ApiConfiguration,
  dynamicModels?: Partial<Record<`${ProviderId}Models`, Record<string, ModelInfo> | null>>,
): {
  selectedProvider: ProviderId
  selectedModelId: string
  selectedModelInfo: ModelInfo
} {
  const provider: ProviderId = apiConfiguration?.apiProvider || 'anthropic'
  const requestedModelId = apiConfiguration?.apiModelId

  const pick = (
    models: Record<string, ModelInfo> | null | undefined,
    defaultId: string,
  ): { selectedModelId: string; selectedModelInfo: ModelInfo } => {
    const available = models && Object.keys(models).length > 0 ? models : MODELS_BY_PROVIDER[provider]
    const modelId = requestedModelId || Object.keys(available || {})[0] || defaultId
    const info: ModelInfo = (available?.[modelId] as ModelInfo) || { }
    return { selectedModelId: modelId, selectedModelInfo: info }
  }

  const key = `${provider}Models` as keyof typeof dynamicModels
  const modelsForProvider = dynamicModels?.[key]
  const { selectedModelId, selectedModelInfo } = pick(modelsForProvider || null, DEFAULT_MODEL_BY_PROVIDER[provider])

  return { selectedProvider: provider, selectedModelId, selectedModelInfo }
}

export default function DynamicModelsExample() {
  // 1) State for dynamic model providers
  const [anthropicModels, setAnthropicModels] = useState<Record<string, ModelInfo> | null>(null)
  const [bedrockModels, setBedrockModels] = useState<Record<string, ModelInfo> | null>(null)
  const [vertexModels, setVertexModels] = useState<Record<string, ModelInfo> | null>(null)
  const [geminiModels, setGeminiModels] = useState<Record<string, ModelInfo> | null>(null)
  const [mistralModels, setMistralModels] = useState<Record<string, ModelInfo> | null>(null)
  const [deepseekModels, setDeepseekModels] = useState<Record<string, ModelInfo> | null>(null)

  // Example API configuration (would normally come from settings/state)
  const [apiConfiguration, setApiConfiguration] = useState<ApiConfiguration>({ apiProvider: 'anthropic' })
  const selectedProvider = apiConfiguration.apiProvider ?? 'anthropic'

  // 2) Debounced refresh message for the selected provider
  useDebounce(
    () => {
      switch (selectedProvider) {
        case 'anthropic':
          vscode.postMessage({ type: 'refreshAnthropicModels' })
          break
        case 'bedrock':
          vscode.postMessage({ type: 'refreshBedrockModels' })
          break
        case 'vertex':
          vscode.postMessage({ type: 'refreshVertexModels' })
          break
        case 'gemini':
          vscode.postMessage({ type: 'refreshGeminiModels' })
          break
        case 'mistral':
          vscode.postMessage({ type: 'refreshMistralModels' })
          break
        case 'deepseek':
          vscode.postMessage({ type: 'refreshDeepSeekModels' })
          break
        case 'openrouter':
          vscode.postMessage({ type: 'refreshOpenRouterModels' })
          break
        case 'glama':
          vscode.postMessage({ type: 'refreshGlamaModels' })
          break
        default:
          break
      }
    },
    250,
    [selectedProvider],
  )

  // 3) Message handler: update model caches per provider
  const onMessage = useCallback((event: MessageEvent) => {
    const message = event.data as ExtensionMessage
    switch (message.type) {
      case 'anthropicModels':
        setAnthropicModels(message.anthropicModels ?? {})
        break
      case 'bedrockModels':
        setBedrockModels(message.bedrockModels ?? {})
        break
      case 'vertexModels':
        setVertexModels(message.vertexModels ?? {})
        break
      case 'geminiModels':
        setGeminiModels(message.geminiModels ?? {})
        break
      case 'mistralModels':
        setMistralModels(message.mistralModels ?? {})
        break
      case 'deepseekModels':
        setDeepseekModels(message.deepseekModels ?? {})
        break
      default:
        break
    }
  }, [])

  useEffect(() => {
    window.addEventListener('message', onMessage)
    return () => window.removeEventListener('message', onMessage)
  }, [onMessage])

  // 4) Model selection options for the current provider
  const selectedProviderModelOptions = useMemo(() => {
    const dynamicProviders: Partial<Record<ProviderId, Record<string, ModelInfo> | null>> = {
      anthropic: anthropicModels,
      bedrock: bedrockModels,
      vertex: vertexModels,
      gemini: geminiModels,
      mistral: mistralModels,
      deepseek: deepseekModels,
    }

    const models = dynamicProviders[selectedProvider]
    if (models && Object.keys(models).length > 0) {
      return Object.keys(models).map((modelId) => ({ value: modelId, label: modelId }))
    }

    const fallback = MODELS_BY_PROVIDER[selectedProvider]
    return fallback ? Object.keys(fallback).map((modelId) => ({ value: modelId, label: modelId })) : []
  }, [
    selectedProvider,
    anthropicModels,
    bedrockModels,
    vertexModels,
    geminiModels,
    mistralModels,
    deepseekModels,
  ])

  // 5) Use normalizeApiConfiguration to compute selected model
  const { selectedModelId, selectedModelInfo } = useMemo(
    () =>
      normalizeApiConfiguration(apiConfiguration, {
        anthropicModels,
        bedrockModels,
        vertexModels,
        geminiModels,
        mistralModels,
        deepseekModels,
      }),
    [apiConfiguration, anthropicModels, bedrockModels, vertexModels, geminiModels, mistralModels, deepseekModels],
  )

  // Render a minimal UI purely for demonstration
  return (
    <div style={{ padding: 12, fontFamily: 'var(--vscode-font-family, system-ui)' }}>
      <h3>Dynamic Models Example</h3>

      <label>
        Provider:
        <select
          value={selectedProvider}
          onChange={(e) => setApiConfiguration((prev) => ({ ...prev, apiProvider: e.target.value as ProviderId }))}
        >
          {(['anthropic', 'bedrock', 'vertex', 'gemini', 'mistral', 'deepseek'] as ProviderId[]).map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
      </label>

      <div style={{ marginTop: 8 }}>
        <label>
          Model:
          <select
            value={selectedModelId}
            onChange={(e) => setApiConfiguration((prev) => ({ ...prev, apiModelId: e.target.value }))}
          >
            {selectedProviderModelOptions.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </label>
      </div>

      <pre style={{ marginTop: 12, background: '#1115', padding: 8 }}>
        {JSON.stringify({ selectedProvider, selectedModelId, selectedModelInfo }, null, 2)}
      </pre>
    </div>
  )
}