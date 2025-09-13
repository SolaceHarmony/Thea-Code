import OpenAI from "openai"
import axios from "axios"

import { SingleCompletionHandler } from "../"
import { ApiHandlerOptions, ModelInfo, openAiModelInfoSaneDefaults } from "../../shared/api"
import type { NeutralConversationHistory, NeutralMessageContent } from "../../shared/neutral-history"
import { convertToOllamaHistory, convertToOllamaContentBlocks } from "../transform/neutral-ollama-format"
import { ApiStream, type ApiStreamChunk } from "../transform/stream"
import { HybridMatcher } from "../../utils/json-xml-bridge"
import { BaseProvider } from "./base-provider"
import { OpenAiHandler } from "./openai"

export class OllamaHandler extends BaseProvider implements SingleCompletionHandler {
	protected options: ApiHandlerOptions
	private client: OpenAI
	private openAiHandler: OpenAiHandler

	constructor(options: ApiHandlerOptions) {
		super()
		this.options = options
		this.client = new OpenAI({
			baseURL: (this.options.ollamaBaseUrl || "http://localhost:10000") + "/v1",
			apiKey: "ollama", // Ollama uses a dummy key via OpenAI client
		})

		// Create an OpenAI handler for tool use detection and processing
		this.openAiHandler = new OpenAiHandler({
			...options,
			// Override any OpenAI-specific options as needed
			openAiApiKey: "ollama", // Use the same dummy key
			openAiBaseUrl: (this.options.ollamaBaseUrl || "http://localhost:10000") + "/v1",
			openAiModelId: this.options.ollamaModelId || "",
		})
	}

	// Updated to use NeutralConversationHistory
	override async *createMessage(systemPrompt: string, messages: NeutralConversationHistory): ApiStream {
		// Convert neutral history to Ollama format
		const openAiMessages = convertToOllamaHistory(messages)

		// Add system prompt if not already included
		const hasSystemMessage = openAiMessages.some((msg) => msg.role === "system")
		if (systemPrompt && systemPrompt.trim() !== "" && !hasSystemMessage) {
			openAiMessages.unshift({ role: "system", content: systemPrompt })
		}

		const stream = await this.client.chat.completions.create({
			model: this.getModel().id,
			messages: openAiMessages,
			temperature: this.options.modelTemperature ?? 0,
			stream: true,
		})

		// Hybrid matching logic for reasoning/thinking blocks only
		const matcher = new HybridMatcher(
			"think", // XML tag name for reasoning
			"thinking", // JSON type for reasoning
			// Removed transformFn, transformation will happen in the loop
		)

		for await (const chunk of stream) {
			const delta = chunk.choices[0]?.delta ?? {}

			if (delta.content) {
				// Use OpenAI handler for all tool use detection (including XML and JSON patterns)
				const toolCalls = this.openAiHandler.extractToolCalls(delta)

                                if (toolCalls.length > 0) {
                                        const results = await Promise.all(
                                                toolCalls.map(async (toolCall) => {
                                                        if (!toolCall.function) return null
                                                        const toolResult = await this.processToolUse({
                                                                id: toolCall.id,
                                                                name: toolCall.function.name,
                                                                input: JSON.parse(toolCall.function.arguments || "{}"),
                                                        })
                                                        const toolResultString =
                                                                typeof toolResult === "string" ? toolResult : JSON.stringify(toolResult)
                                                        return { toolCall, toolResultString }
                                                }),
                                        )
                                        for (const r of results) {
                                                if (r) {
                                                        yield {
                                                                type: "tool_result",
                                                                id: r.toolCall.id,
                                                                content: r.toolResultString,
                                                        }
                                                }
                                        }
                                } else {
                                        // If no tool use was detected, use the matcher for regular content
                                        for (const matchedChunk of matcher.update(delta.content)) {
                                                yield {
                                                        type: matchedChunk.matched ? "reasoning" : "text",
							text:
								typeof matchedChunk.data === "string"
									? matchedChunk.data
									: JSON.stringify(matchedChunk.data),
						} as ApiStreamChunk // Ensure it conforms to ApiStreamChunk
					}
				}
			}
		}

		for (const finalChunk of matcher.final()) {
			yield {
				type: finalChunk.matched ? "reasoning" : "text",
				text: typeof finalChunk.data === "string" ? finalChunk.data : JSON.stringify(finalChunk.data),
			} as ApiStreamChunk // Ensure it conforms to ApiStreamChunk
		}
	}

	// Implement countTokens method for NeutralMessageContent
	override async countTokens(content: NeutralMessageContent): Promise<number> {
		try {
			// Convert neutral content to Ollama format (string)
			const ollamaContent = convertToOllamaContentBlocks(content)

			// Use the base provider's implementation for token counting
			// This will use tiktoken to count tokens in the string
			return super.countTokens([{ type: "text", text: ollamaContent }])
		} catch (error) {
			console.warn("Ollama token counting error, using fallback", error)
			return super.countTokens(content)
		}
	}

	override getModel(): { id: string; info: ModelInfo } {
		return {
			id: this.options.ollamaModelId || "",
			info: openAiModelInfoSaneDefaults,
		}
	}

	// Update completePrompt to use the same conversion logic
	async completePrompt(prompt: string): Promise<string> {
		try {
			// Create a simple neutral history with a single user message
			const neutralHistory: NeutralConversationHistory = [
				{ role: "user", content: [{ type: "text", text: prompt }] },
			]

			// Convert to Ollama format
			const openAiMessages = convertToOllamaHistory(neutralHistory)

			const response = await this.client.chat.completions.create({
				model: this.getModel().id,
				messages: openAiMessages,
				temperature: this.options.modelTemperature ?? 0,
				stream: false,
			})

			return response.choices[0]?.message.content || ""
		} catch (error) {
			if (error instanceof Error) {
				throw new Error(`Ollama completion error: ${error.message}`)
			}
			throw error
		}
	}
}

// getOllamaModels function remains the same
export async function getOllamaModels(baseUrl = "http://localhost:10000") {
	try {
		if (!URL.canParse(baseUrl)) {
			return []
		}

		const response = await axios.get(`${baseUrl}/api/tags`, { timeout: 2000 })
		const responseData = response.data as { models: { name: string; [key: string]: unknown }[] } | undefined
		const modelsArray = responseData?.models?.map((model: { name: string }) => model.name) || []
		return [...new Set<string>(modelsArray)] // Assert modelsArray is string[] for Set
	} catch {
		// Silently return empty array on error
		return []
	}
}
