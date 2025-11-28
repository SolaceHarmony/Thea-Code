/**
 * Dynamic Glama Handler
 * 
 * An updated version of GlamaHandler that uses the dynamic model registry
 * instead of hardcoded model definitions.
 */

import { GlamaHandler } from "./glama"
import { ApiHandlerOptions, ModelInfo } from "../../shared/api"
import { modelRegistry } from "./model-registry"
import { GlamaModelProvider } from "./model-providers/glama-model-provider"
import { SingleCompletionHandler } from "../index"
import type { NeutralConversationHistory } from "../../shared/neutral-history"
import { ApiStream } from "../transform/stream"

export class DynamicGlamaHandler extends GlamaHandler implements SingleCompletionHandler {
	private modelProvider: GlamaModelProvider

	constructor(options: ApiHandlerOptions) {
		super(options)

		// Create model provider
		this.modelProvider = new GlamaModelProvider()
		this.modelProvider.configure(options)

		// Register with the model registry if not already registered
		if (!modelRegistry.hasProvider("glama")) {
			modelRegistry.registerProvider("glama", this.modelProvider)
		}
	}

	override getModel(): { id: string; info: ModelInfo } {
		return this.currentModelConfig || super.getModel()
	}

	private currentModelConfig: { id: string; info: ModelInfo } | null = null

	override async *createMessage(systemPrompt: string, messages: NeutralConversationHistory): ApiStream {
		// Fetch model info dynamically before processing
		const modelId = this.options.apiModelId || await modelRegistry.getDefaultModelId("glama")
		const info = await modelRegistry.getModelInfo("glama", modelId)
		
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
