/**
 * Dynamic OpenAI Handler
 * 
 * An updated version of OpenAiHandler that uses the dynamic model registry
 * instead of hardcoded model definitions.
 */

import { OpenAiHandler } from "./openai"
import { ApiHandlerOptions, ModelInfo } from "../../shared/api"
import { modelRegistry } from "./model-registry"
import { OpenAIModelProvider } from "./openai-model-provider"
import { SingleCompletionHandler } from "../index"
import type { NeutralConversationHistory } from "../../shared/neutral-history"
import { ApiStream } from "../transform/stream"

export class DynamicOpenAIHandler extends OpenAiHandler implements SingleCompletionHandler {
	private modelProvider: OpenAIModelProvider

	constructor(options: ApiHandlerOptions) {
		super(options)

		// Create model provider
		// Note: OpenAIModelProvider constructor signature is (apiKey, baseURL)
		this.modelProvider = new OpenAIModelProvider(
			options.openAiApiKey,
			options.openAiBaseUrl
		)

		// Register with the model registry if not already registered
		if (!modelRegistry.hasProvider("openai")) {
			modelRegistry.registerProvider("openai", this.modelProvider)
		}
	}

	override getModel(): { id: string; info: ModelInfo } {
		return this.currentModelConfig || super.getModel()
	}

	private currentModelConfig: { id: string; info: ModelInfo } | null = null

	override async *createMessage(systemPrompt: string, messages: NeutralConversationHistory): ApiStream {
		// Fetch model info dynamically before processing
		const modelId = this.options.apiModelId || await modelRegistry.getDefaultModelId("openai")
		const info = await modelRegistry.getModelInfo("openai", modelId)
		
		if (info) {
			this.currentModelConfig = {
				id: modelId,
				info
			}
		} else {
			// Fallback to base behavior if not found
			this.currentModelConfig = null
		}

		// Call base implementation
		yield* super.createMessage(systemPrompt, messages)
	}
}
