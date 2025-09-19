/**
 * Dynamic Anthropic Handler
 * 
 * An updated version of AnthropicHandler that uses the dynamic model registry
 * instead of hardcoded model definitions.
 */

import { NeutralAnthropicClient } from "../../services/anthropic"
import { ApiHandlerOptions, ModelInfo } from "../../shared/api"
import type { NeutralConversationHistory, NeutralMessageContent } from "../../shared/neutral-history"
import { ApiStream } from "../transform/stream"
import { BaseProvider } from "./base-provider"
import { ANTHROPIC_DEFAULT_MAX_TOKENS } from "./constants"
import { SingleCompletionHandler, getModelParams } from "../index"
import { supportsThinking } from "../../utils/model-capabilities"
import { getBaseModelId, isThinkingModel } from "../../utils/model-pattern-detection"
import { modelRegistry } from "./model-registry"
import { AnthropicModelProvider } from "./anthropic-model-provider"

export class DynamicAnthropicHandler extends BaseProvider implements SingleCompletionHandler {
	private options: ApiHandlerOptions
	private client: NeutralAnthropicClient
	private modelProvider: AnthropicModelProvider

	constructor(options: ApiHandlerOptions) {
		super()
		this.options = options

		// Create client
		this.client = new NeutralAnthropicClient({
			apiKey: this.options.apiKey || "",
			baseURL: this.options.anthropicBaseUrl,
		})

		// Create model provider
		this.modelProvider = new AnthropicModelProvider(this.options.apiKey, this.options.anthropicBaseUrl)

		// Register with the model registry if not already registered
		if (!modelRegistry.hasProvider("anthropic")) {
			modelRegistry.registerProvider("anthropic", this.modelProvider)
		}
	}

	override async *createMessage(systemPrompt: string, messages: NeutralConversationHistory): ApiStream {
		const model = await this.getModelAsync()

		const stream = this.client.createMessage({
			model: model.id,
			systemPrompt,
			messages,
			maxTokens: model.maxTokens ?? ANTHROPIC_DEFAULT_MAX_TOKENS,
			temperature: model.temperature,
			thinking: model.thinking,
		})

		for await (const chunk of stream) {
			if (chunk.type === "tool_use") {
				const toolResult = await this.processToolUse({
					id: chunk.id,
					name: chunk.name,
					input: chunk.input,
				})

				const toolResultString = typeof toolResult === "string" ? toolResult : JSON.stringify(toolResult)

				yield {
					type: "tool_result",
					id: chunk.id,
					content: toolResultString,
				}
			} else {
				yield chunk
			}
		}
	}

	/**
	 * Get model information asynchronously from the registry
	 */
	async getModelAsync() {
		const modelId = this.options.apiModelId || (await modelRegistry.getDefaultModelId("anthropic"))

		// Get model info from registry
		let info = await modelRegistry.getModelInfo("anthropic", modelId)

		// If model not found, try to detect capabilities
		if (!info) {
			info = await this.modelProvider.getModelInfo(modelId)

			// If still not found, use a default
			if (!info) {
				const defaultId = await modelRegistry.getDefaultModelId("anthropic")
				info = await modelRegistry.getModelInfo("anthropic", defaultId)

				if (!info) {
					// Ultimate fallback
					info = {
						maxTokens: ANTHROPIC_DEFAULT_MAX_TOKENS,
						contextWindow: 200_000,
						supportsImages: true,
						supportsPromptCache: true,
						inputPrice: 3,
						outputPrice: 15,
					} as ModelInfo
				}
			}
		}

		// Track the original model ID for special variant handling
		const virtualId = modelId
		let actualId = modelId

		// Special handling for thinking variants
		if (isThinkingModel(modelId)) {
			actualId = getBaseModelId(modelId)
		}

		// Get base model parameters
		const baseParams = getModelParams({
			options: this.options,
			model: info,
			defaultMaxTokens: ANTHROPIC_DEFAULT_MAX_TOKENS,
		})

		// Model-specific thinking adjustments
		if (supportsThinking(info)) {
			const customMaxTokens = this.options.modelMaxTokens
			const customMaxThinkingTokens = this.options.modelMaxThinkingTokens

			if (isThinkingModel(virtualId) || supportsThinking(info)) {
				// Clamp the thinking budget
				const effectiveMaxTokens = customMaxTokens ?? baseParams.maxTokens ?? ANTHROPIC_DEFAULT_MAX_TOKENS
				const maxBudgetTokens = Math.floor(effectiveMaxTokens * 0.8)
				const budgetTokens = Math.max(
					Math.min(customMaxThinkingTokens ?? maxBudgetTokens, maxBudgetTokens),
					1024,
				)

				baseParams.thinking = {
					type: "enabled",
					budget_tokens: budgetTokens,
				}
			}
		}

		return {
			id: actualId,
			info,
			virtualId,
			...baseParams,
		}
	}

	/**
	 * Synchronous getModel for compatibility
	 * Note: This uses cached model info and may not be as accurate as getModelAsync
	 */
	getModel() {
		const modelId = this.options.apiModelId || "claude-3-7-sonnet-20250219"

		// Use a simplified synchronous approach for compatibility
		// In production, we should migrate all callers to use getModelAsync
		const info: ModelInfo = {
			maxTokens: ANTHROPIC_DEFAULT_MAX_TOKENS,
			contextWindow: 200_000,
			supportsImages: true,
			supportsPromptCache: true,
			inputPrice: 3,
			outputPrice: 15,
			thinking: modelId.includes("thinking"),
		} as ModelInfo

		const virtualId = modelId
		let actualId = modelId

		if (isThinkingModel(modelId)) {
			actualId = getBaseModelId(modelId)
		}

		const baseParams = getModelParams({
			options: this.options,
			model: info,
			defaultMaxTokens: ANTHROPIC_DEFAULT_MAX_TOKENS,
		})

		if (supportsThinking(info) && (isThinkingModel(virtualId) || supportsThinking(info))) {
			const customMaxTokens = this.options.modelMaxTokens
			const customMaxThinkingTokens = this.options.modelMaxThinkingTokens
			const effectiveMaxTokens = customMaxTokens ?? baseParams.maxTokens ?? ANTHROPIC_DEFAULT_MAX_TOKENS
			const maxBudgetTokens = Math.floor(effectiveMaxTokens * 0.8)
			const budgetTokens = Math.max(Math.min(customMaxThinkingTokens ?? maxBudgetTokens, maxBudgetTokens), 1024)

			baseParams.thinking = {
				type: "enabled",
				budget_tokens: budgetTokens,
			}
		}

		return {
			id: actualId,
			info,
			virtualId,
			...baseParams,
		}
	}

	async completePrompt(prompt: string) {
		const model = await this.getModelAsync()

		let text = ""
		const stream = this.client.createMessage({
			model: model.id,
			systemPrompt: "",
			messages: [{ role: "user", content: prompt }],
			maxTokens: ANTHROPIC_DEFAULT_MAX_TOKENS,
			temperature: model.temperature,
		})

		for await (const chunk of stream) {
			if (chunk.type === "text") {
				text += chunk.text
			}
		}

		return text
	}

	override async countTokens(content: string | NeutralMessageContent): Promise<number> {
		try {
			const model = await this.getModelAsync()
			return await this.client.countTokens(model.id, content)
		} catch (error) {
			console.warn("Anthropic token counting failed, using fallback", error)
			return super.countTokens(content)
		}
	}

	/**
	 * Refresh model list from provider
	 */
	async refreshModels(): Promise<void> {
		await modelRegistry.getModels("anthropic", true)
	}

	/**
	 * Get available models
	 */
	async getAvailableModels() {
		return modelRegistry.getModels("anthropic")
	}
}