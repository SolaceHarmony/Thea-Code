/**
 * Dynamic Bedrock Handler
 * 
 * An updated version of AwsBedrockHandler that uses the dynamic model registry
 * instead of hardcoded model definitions.
 */

import { AwsBedrockHandler } from "./bedrock"
import { ApiHandlerOptions, ModelInfo } from "../../shared/api"
import { modelRegistry } from "./model-registry"
import { BedrockModelProvider } from "./model-providers/bedrock-model-provider"
import { SingleCompletionHandler } from "../index"
import type { NeutralConversationHistory } from "../../shared/neutral-history"
import { ApiStream } from "../transform/stream"

export class DynamicBedrockHandler extends AwsBedrockHandler implements SingleCompletionHandler {
	private modelProvider: BedrockModelProvider

	constructor(options: ApiHandlerOptions) {
		super(options)

		// Create model provider
		this.modelProvider = new BedrockModelProvider()
		this.modelProvider.configure(options)

		// Register with the model registry if not already registered
		if (!modelRegistry.hasProvider("bedrock")) {
			modelRegistry.registerProvider("bedrock", this.modelProvider)
		}
	}

	override getModel(): { id: string; info: ModelInfo } {
		return this.currentModelConfig || super.getModel()
	}

	private currentModelConfig: { id: string; info: ModelInfo } | null = null

	override async *createMessage(systemPrompt: string, messages: NeutralConversationHistory): ApiStream {
		// Fetch model info dynamically before processing
		const modelId = this.options.apiModelId || await modelRegistry.getDefaultModelId("bedrock")
		const info = await modelRegistry.getModelInfo("bedrock", modelId)
		
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
