import axios from "axios"
import { ModelProvider, ModelListing } from "../model-registry"
import { ModelInfo, ApiHandlerOptions } from "../../../shared/api"

interface OllamaModelResponse {
	models: {
		name: string
		model: string
		modified_at: string
		size: number
		digest: string
		details: {
			parent_model: string
			format: string
			family: string
			families: string[]
			parameter_size: string
			quantization_level: string
		}
	}[]
}

export class OllamaModelProvider implements ModelProvider {
	private baseUrl: string = "http://localhost:11434"

	configure(options: ApiHandlerOptions): void {
		this.baseUrl = options.ollamaBaseUrl || "http://localhost:11434"
	}

	async getModels(): Promise<ModelListing[]> {
		try {
			const response = await axios.get<OllamaModelResponse>(`${this.baseUrl}/api/tags`)
			
			return response.data.models.map(model => {
				const info = this.deriveModelInfo(model)
				return {
					id: model.name,
					info
				}
			})
		} catch (error) {
			console.error("Failed to fetch Ollama models:", error)
			return []
		}
	}

	async listModels(): Promise<ModelListing[]> {
		return this.getModels()
	}

	async getModelInfo(modelId: string): Promise<ModelInfo | null> {
		const models = await this.getModels()
		const model = models.find(m => m.id === modelId)
		return model?.info || null
	}

	getDefaultModelId(): string {
		return "llama3"
	}

	private deriveModelInfo(model: OllamaModelResponse['models'][0]): ModelInfo {
		// Default capabilities
		const info: ModelInfo = {
			maxTokens: 4096, // Default context window
			contextWindow: 4096,
			supportsImages: false,
			supportsPromptCache: false,
			inputPrice: 0,
			outputPrice: 0,
			description: `Ollama model: ${model.name} (${model.details.parameter_size})`
		}

		// Detect capabilities based on model family/name
		const name = model.name.toLowerCase()
		const family = model.details.family.toLowerCase()

		if (name.includes("llama3") || family.includes("llama3")) {
			info.contextWindow = 8192
			info.maxTokens = 8192
		} else if (name.includes("mistral") || family.includes("mistral")) {
			info.contextWindow = 32000
			info.maxTokens = 32000
		} else if (name.includes("gemma") || family.includes("gemma")) {
			info.contextWindow = 8192
			info.maxTokens = 8192
		} else if (name.includes("llava") || family.includes("clip")) {
			info.supportsImages = true
		}

		return info
	}
}
