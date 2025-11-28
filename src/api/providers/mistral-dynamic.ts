/**
 * Dynamic Mistral Handler
 * 
 * An updated version of MistralHandler that uses the dynamic model registry
 * instead of hardcoded model definitions.
 */

import { MistralHandler } from "./mistral"
import { ApiHandlerOptions, ModelInfo } from "../../shared/api"
import { modelRegistry } from "./model-registry"
import { MistralModelProvider } from "./model-providers/mistral-model-provider"
import { SingleCompletionHandler } from "../index"
import type { NeutralConversationHistory } from "../../shared/neutral-history"
import { ApiStream } from "../transform/stream"

export class DynamicMistralHandler extends MistralHandler implements SingleCompletionHandler {
	private modelProvider: MistralModelProvider

	constructor(options: ApiHandlerOptions) {
		super(options)

		// Create model provider
		this.modelProvider = new MistralModelProvider()
		this.modelProvider.configure(options)

		// Register with the model registry if not already registered
		if (!modelRegistry.hasProvider("mistral")) {
			modelRegistry.registerProvider("mistral", this.modelProvider)
		}
	}

	override getModel(): { id: string; info: ModelInfo } {
		return this.currentModelConfig || super.getModel()
	}

	private currentModelConfig: { id: string; info: ModelInfo } | null = null

	override async *createMessage(systemPrompt: string, messages: NeutralConversationHistory): ApiStream {
		// Fetch model info dynamically before processing
		const modelId = this.options.apiModelId || await modelRegistry.getDefaultModelId("mistral")
		const info = await modelRegistry.getModelInfo("mistral", modelId)
		
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
