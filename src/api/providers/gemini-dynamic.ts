/**
 * Dynamic Gemini Handler
 * 
 * An updated version of GeminiHandler that uses the dynamic model registry
 * instead of hardcoded model definitions.
 */

import { GeminiHandler } from "./gemini"
import { ApiHandlerOptions, ModelInfo } from "../../shared/api"
import { modelRegistry } from "./model-registry"
import { GeminiModelProvider } from "./model-providers/gemini-model-provider"
import { SingleCompletionHandler } from "../index"
import type { NeutralConversationHistory } from "../../shared/neutral-history"
import { ApiStream } from "../transform/stream"

export class DynamicGeminiHandler extends GeminiHandler implements SingleCompletionHandler {
	private modelProvider: GeminiModelProvider

	constructor(options: ApiHandlerOptions) {
		super(options)

		// Create model provider
		this.modelProvider = new GeminiModelProvider()
		this.modelProvider.configure(options)

		// Register with the model registry if not already registered
		if (!modelRegistry.hasProvider("gemini")) {
			modelRegistry.registerProvider("gemini", this.modelProvider)
		}
	}

	override getModel(): { id: string; info: ModelInfo } {
		return this.currentModelConfig || super.getModel()
	}

	private currentModelConfig: { id: string; info: ModelInfo } | null = null

	override async *createMessage(systemPrompt: string, messages: NeutralConversationHistory): ApiStream {
		// Fetch model info dynamically before processing
		const modelId = this.options.apiModelId || await modelRegistry.getDefaultModelId("gemini")
		const info = await modelRegistry.getModelInfo("gemini", modelId)
		
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
