/**
 * Dynamic Vertex Handler
 * 
 * An updated version of VertexHandler that uses the dynamic model registry
 * instead of hardcoded model definitions.
 */

import { NeutralVertexClient } from "../../services/vertex"
import { ModelInfo, ApiHandlerOptions } from "../../shared/api"
import type { NeutralConversationHistory } from "../../shared/neutral-history"
import { ApiStream } from "../transform/stream"
import { BaseProvider } from "./base-provider"
import { ANTHROPIC_DEFAULT_MAX_TOKENS } from "./constants"
import { SingleCompletionHandler } from "../index"
import { modelRegistry } from "./model-registry"
import { VertexModelProvider } from "./model-providers/vertex-model-provider"

export class DynamicVertexHandler extends BaseProvider implements SingleCompletionHandler {
	private options: ApiHandlerOptions
	private client: NeutralVertexClient
	private modelProvider: VertexModelProvider

	constructor(options: ApiHandlerOptions) {
		super()
		this.options = options

		// Parse credentials if provided
		let credentials: Record<string, unknown> | undefined
		if (this.options.vertexJsonCredentials) {
			try {
				credentials = JSON.parse(this.options.vertexJsonCredentials) as Record<string, unknown>
			} catch (error) {
				console.error("Failed to parse Vertex credentials:", error)
			}
		}

		// Create client
		this.client = new NeutralVertexClient({
			projectId: this.options.vertexProjectId || "",
			region: this.options.vertexRegion || "us-east5",
			credentials,
			keyFile: this.options.vertexKeyFile,
		})

		// Create model provider
		this.modelProvider = new VertexModelProvider()
		this.modelProvider.configure(this.options)

		// Register with the model registry if not already registered
		if (!modelRegistry.hasProvider("vertex")) {
			modelRegistry.registerProvider("vertex", this.modelProvider)
		}
	}

	override async *createMessage(systemPrompt: string, messages: NeutralConversationHistory): ApiStream {
		const model = await this.getModelAsync()
		let stream: ApiStream

		if (model.id.startsWith("claude")) {
			stream = this.client.createClaudeMessage({
				model: model.id,
				systemPrompt,
				messages,
				maxTokens: model.maxTokens ?? ANTHROPIC_DEFAULT_MAX_TOKENS,
				temperature: model.temperature,
			})
		} else if (model.id.startsWith("gemini")) {
			stream = this.client.createGeminiMessage({
				model: model.id,
				systemPrompt,
				messages,
				maxTokens: model.maxTokens ?? ANTHROPIC_DEFAULT_MAX_TOKENS,
				temperature: model.temperature,
			})
		} else {
			throw new Error(`Unknown model ID: ${model.id}`)
		}

		for await (const chunk of stream) {
			// Vertex client already handles tool use internally or returns tool calls
			// If NeutralVertexClient returns tool calls in a specific format, we might need to handle them here
			// But based on anthropic-dynamic.ts, it seems we just yield them or process them if needed.
			// Checking anthropic-dynamic.ts again, it processes tool_use chunks.
			
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
		const modelId = this.options.apiModelId || (await modelRegistry.getDefaultModelId("vertex"))

		// Get model info from registry
		let info = await modelRegistry.getModelInfo("vertex", modelId)

		// If model not found, try to detect capabilities
		if (!info) {
			info = await this.modelProvider.getModelInfo(modelId)

			// If still not found, use a default
			if (!info) {
				const defaultId = await modelRegistry.getDefaultModelId("vertex")
				info = await modelRegistry.getModelInfo("vertex", defaultId)

				if (!info) {
					// Ultimate fallback
					info = {
						maxTokens: ANTHROPIC_DEFAULT_MAX_TOKENS,
						contextWindow: 200_000,
						supportsImages: true,
						supportsPromptCache: true,
						inputPrice: 0,
						outputPrice: 0,
						description: "Unknown Vertex Model",
					}
				}
			}
		}

		return {
			id: modelId,
			...info,
		}
	}

	getModel(): { id: string; info: ModelInfo } {
		const modelId = this.options.apiModelId || "claude-3-5-sonnet-v2@20241022"
		return {
			id: modelId,
			info: {
				maxTokens: ANTHROPIC_DEFAULT_MAX_TOKENS,
				contextWindow: 200_000,
				supportsImages: true,
				supportsPromptCache: true,
				inputPrice: 0,
				outputPrice: 0,
				description: "Vertex Model",
			},
		}
	}

	async completePrompt(prompt: string): Promise<string> {
		try {
			const stream = this.createMessage("", [{ role: "user", content: [{ type: "text", text: prompt }] }])

			let completion = ""
			for await (const chunk of stream) {
				if (chunk.type === "text") {
					completion += chunk.text
				}
			}
			return completion
		} catch (error) {
			console.error("Error in completePrompt:", error)
			throw error
		}
	}
}
