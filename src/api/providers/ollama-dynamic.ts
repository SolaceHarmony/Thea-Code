/**
 * Dynamic Ollama Handler
 * 
 * An updated version of OllamaHandler that uses the dynamic model registry
 * instead of hardcoded model definitions.
 */

import { OllamaHandler } from "./ollama"
import { ApiHandlerOptions, ModelInfo } from "../../shared/api"
import { modelRegistry } from "./model-registry"
import { OllamaModelProvider } from "./model-providers/ollama-model-provider"
import { SingleCompletionHandler } from "../index"
import type { NeutralConversationHistory } from "../../shared/neutral-history"
import { ApiStream } from "../transform/stream"

export class DynamicOllamaHandler extends OllamaHandler implements SingleCompletionHandler {
	private modelProvider: OllamaModelProvider

	constructor(options: ApiHandlerOptions) {
		super(options)

		// Create model provider
		this.modelProvider = new OllamaModelProvider()
		this.modelProvider.configure(options)

		// Register with the model registry if not already registered
		if (!modelRegistry.hasProvider("ollama")) {
			modelRegistry.registerProvider("ollama", this.modelProvider)
		}
	}

	override getModel(): { id: string; info: ModelInfo } {
		return this.currentModelConfig || super.getModel()
	}

	private currentModelConfig: { id: string; info: ModelInfo } | null = null

	override async *createMessage(systemPrompt: string, messages: NeutralConversationHistory): ApiStream {
		// Fetch model info dynamically before processing
		const modelId = this.options.apiModelId || await modelRegistry.getDefaultModelId("ollama")
		const info = await modelRegistry.getModelInfo("ollama", modelId)
		
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
