/**
 * Anthropic provider edge case tests
 * Tests thinking budget clamping, tool_use to tool_result conversion, countTokens fallback
 */

import { describe, expect, it, jest, beforeEach, afterEach } from "@jest/globals"
import { AnthropicHandler } from "../anthropic"
import { NeutralAnthropicClient } from "../../../services/anthropic"
import { ApiHandlerOptions } from "../../../shared/api"
import type { NeutralConversationHistory, NeutralMessageContent } from "../../../shared/neutral-history"

// Mock constants
jest.mock("../constants", () => ({
	ANTHROPIC_DEFAULT_MAX_TOKENS: 8192
}))

// Mock anthropic models
jest.mock("../../../shared/api", () => ({
	...jest.requireActual("../../../shared/api"),
	anthropicDefaultModelId: "claude-3-5-sonnet-20241022",
	anthropicModels: {
		"claude-3-5-sonnet-20241022": {
			maxTokens: 8192,
			contextWindow: 200000,
			supportsImages: true,
			supportsPromptCache: true,
			inputPrice: 3,
			outputPrice: 15,
			supportsThinking: true
		},
		"claude-3-5-sonnet-20241022-thinking": {
			maxTokens: 8192,
			contextWindow: 200000,
			supportsImages: true,
			supportsPromptCache: true,
			inputPrice: 3,
			outputPrice: 15,
			supportsThinking: true
		},
		"claude-3-5-haiku-20241022": {
			maxTokens: 8192,
			contextWindow: 200000,
			supportsImages: true,
			supportsPromptCache: true,
			inputPrice: 1,
			outputPrice: 5,
			supportsThinking: true
		},
		"claude-3-5-haiku-20241022-thinking": {
			maxTokens: 8192,
			contextWindow: 200000,
			supportsImages: true,
			supportsPromptCache: true,
			inputPrice: 1,
			outputPrice: 5,
			supportsThinking: true
		},
		"claude-2.1": {
			maxTokens: 4096,
			contextWindow: 100000,
			supportsImages: false,
			supportsPromptCache: false,
			inputPrice: 8,
			outputPrice: 24,
			supportsThinking: false
		}
	},
	ANTHROPIC_DEFAULT_MAX_TOKENS: 8192
}))

// Mock the NeutralAnthropicClient
jest.mock("../../../services/anthropic", () => ({
	NeutralAnthropicClient: jest.fn()
}))

// Mock model capabilities
jest.mock("../../../utils/model-capabilities", () => ({
	supportsThinking: jest.fn((model) => {
		// Mock that certain models support thinking
		return model?.supportsThinking === true
	})
}))

// Mock model pattern detection  
jest.mock("../../../utils/model-pattern-detection", () => ({
	isThinkingModel: jest.fn((id) => {
		return typeof id === 'string' && id.includes('-thinking')
	}),
	getBaseModelId: jest.fn((id) => {
		if (typeof id === 'string' && id.includes('-thinking')) {
			return id.replace('-thinking', '')
		}
		return id
	})
}))

// Mock getModelParams
jest.mock("../../index", () => ({
	...jest.requireActual("../../index"),
	getModelParams: jest.fn((options) => {
		const customMaxTokens = options.options?.modelMaxTokens
		const customTemperature = options.options?.modelTemperature
		return {
			maxTokens: customMaxTokens !== undefined ? customMaxTokens : 8192,
			temperature: customTemperature !== undefined ? customTemperature : 0,
			thinking: undefined
		}
	})
}))

// Mock the BaseProvider's processToolUse
jest.mock("../base-provider", () => {
	const actual = jest.requireActual("../base-provider")
	return {
		...actual,
		BaseProvider: class MockBaseProvider {
			mcpIntegration: any
			
			constructor() {
				// Mock MCP integration
				this.mcpIntegration = {
					registerTool: jest.fn(),
					executeTool: jest.fn()
				}
			}
			
			registerTools() {
				// Mock method
			}
			
			async processToolUse(toolUse: any) {
				// Mock tool processing
				return `Tool result for ${toolUse.name}`
			}
			
			async countTokens(content: any) {
				// Mock fallback token counting
				return 100
			}
		}
	}
})

describe("AnthropicHandler - Edge Cases", () => {
	let handler: AnthropicHandler
	let mockClient: jest.Mocked<NeutralAnthropicClient>
	let mockOptions: ApiHandlerOptions
	let consoleWarnSpy: jest.SpyInstance

	beforeEach(() => {
		// Reset mocks
		jest.clearAllMocks()
		
		// Setup console spy
		consoleWarnSpy = jest.spyOn(console, 'warn').mockImplementation()

		// Create mock client
		mockClient = {
			createMessage: jest.fn(),
			countTokens: jest.fn()
		} as any

		// Mock the constructor
		(NeutralAnthropicClient as jest.Mock).mockImplementation(() => mockClient)

		// Default options
		mockOptions = {
			apiKey: "test-key",
			apiModelId: "claude-3-5-sonnet-20241022",
			anthropicBaseUrl: "https://api.anthropic.com"
		}

		handler = new AnthropicHandler(mockOptions)
	})

	afterEach(() => {
		jest.restoreAllMocks()
	})

	describe("Thinking Budget Clamping", () => {
		it("should clamp thinking budget to 80% of max tokens", () => {
			// Set custom max tokens
			mockOptions.modelMaxTokens = 10000
			mockOptions.modelMaxThinkingTokens = 9000 // Too high, should be clamped
			mockOptions.apiModelId = "claude-3-5-sonnet-20241022-thinking"
			
			handler = new AnthropicHandler(mockOptions)
			const model = handler.getModel()
			
			// Should clamp to 80% of 10000 = 8000
			expect(model.thinking).toEqual({
				type: "enabled",
				budget_tokens: 8000
			})
		})

		it("should ensure minimum thinking budget of 1024 tokens", () => {
			// Set very low max tokens
			mockOptions.modelMaxTokens = 1000
			mockOptions.apiModelId = "claude-3-5-sonnet-20241022-thinking"
			
			handler = new AnthropicHandler(mockOptions)
			const model = handler.getModel()
			
			// Should use minimum of 1024 despite 80% being 800
			expect(model.thinking).toEqual({
				type: "enabled",
				budget_tokens: 1024
			})
		})

		it("should handle thinking model variants correctly", () => {
			mockOptions.apiModelId = "claude-3-5-haiku-20241022-thinking"
			
			handler = new AnthropicHandler(mockOptions)
			const model = handler.getModel()
			
			// Should detect thinking variant and enable thinking
			expect(model.virtualId).toBe("claude-3-5-haiku-20241022-thinking")
			expect(model.id).toBe("claude-3-5-haiku-20241022") // Base model ID
			expect(model.thinking?.type).toBe("enabled")
		})

		it("should not enable thinking for non-thinking models", () => {
			mockOptions.apiModelId = "claude-2.1"
			
			handler = new AnthropicHandler(mockOptions)
			const model = handler.getModel()
			
			// Should not have thinking enabled
			expect(model.thinking).toBeUndefined()
		})

		it("should respect custom thinking budget within limits", () => {
			mockOptions.modelMaxTokens = 10000
			mockOptions.modelMaxThinkingTokens = 5000 // Within 80% limit
			mockOptions.apiModelId = "claude-3-5-sonnet-20241022-thinking"
			
			handler = new AnthropicHandler(mockOptions)
			const model = handler.getModel()
			
			// Should use the custom value
			expect(model.thinking).toEqual({
				type: "enabled",
				budget_tokens: 5000
			})
		})
	})

	describe("Tool Use to Tool Result Conversion", () => {
		it("should convert tool_use chunks to tool_result", async () => {
			const mockStream = async function* () {
				yield { type: "text", text: "Let me help you with that." }
				yield { 
					type: "tool_use", 
					id: "tool-123",
					name: "calculator",
					input: { operation: "add", a: 2, b: 3 }
				}
				yield { type: "text", text: "The result is above." }
			}

			mockClient.createMessage.mockReturnValue(mockStream())

			const messages: NeutralConversationHistory = [
				{ role: "user", content: "What is 2 + 3?" }
			]

			const stream = handler.createMessage("You are a helpful assistant", messages)
			const results = []
			
			for await (const chunk of stream) {
				results.push(chunk)
			}

			// Should have converted tool_use to tool_result
			expect(results).toHaveLength(3)
			expect(results[0]).toEqual({ type: "text", text: "Let me help you with that." })
			expect(results[1]).toEqual({
				type: "tool_result",
				id: "tool-123",
				content: "Tool result for calculator"
			})
			expect(results[2]).toEqual({ type: "text", text: "The result is above." })
		})

		it("should handle tool result as JSON string when not a string", async () => {
			// Override processToolUse to return an object
			handler.processToolUse = jest.fn().mockResolvedValue({ 
				answer: 5, 
				explanation: "2 + 3 = 5" 
			})

			const mockStream = async function* () {
				yield { 
					type: "tool_use", 
					id: "tool-456",
					name: "complex_tool",
					input: { data: "test" }
				}
			}

			mockClient.createMessage.mockReturnValue(mockStream())

			const messages: NeutralConversationHistory = []
			const stream = handler.createMessage("", messages)
			const results = []
			
			for await (const chunk of stream) {
				results.push(chunk)
			}

			// Should JSON stringify non-string results
			expect(results[0]).toEqual({
				type: "tool_result",
				id: "tool-456",
				content: JSON.stringify({ answer: 5, explanation: "2 + 3 = 5" })
			})
		})

		it("should pass through non-tool_use chunks unchanged", async () => {
			const mockStream = async function* () {
				yield { type: "text", text: "Regular text" }
				yield { type: "thinking", text: "Internal thoughts" }
				yield { type: "usage", input_tokens: 10, output_tokens: 20 }
			}

			mockClient.createMessage.mockReturnValue(mockStream())

			const messages: NeutralConversationHistory = []
			const stream = handler.createMessage("", messages)
			const results = []
			
			for await (const chunk of stream) {
				results.push(chunk)
			}

			// All chunks should pass through unchanged
			expect(results).toEqual([
				{ type: "text", text: "Regular text" },
				{ type: "thinking", text: "Internal thoughts" },
				{ type: "usage", input_tokens: 10, output_tokens: 20 }
			])
		})
	})

	describe("countTokens Fallback", () => {
		it("should use client countTokens when successful", async () => {
			mockClient.countTokens.mockResolvedValue(42)
			
			const content: NeutralMessageContent = [
				{ type: "text", text: "Test content" }
			]
			
			const count = await handler.countTokens(content)
			
			expect(count).toBe(42)
			expect(mockClient.countTokens).toHaveBeenCalledWith(
				"claude-3-5-sonnet-20241022",
				content
			)
			expect(consoleWarnSpy).not.toHaveBeenCalled()
		})

		it("should fallback to base implementation on error", async () => {
			mockClient.countTokens.mockRejectedValue(new Error("API error"))
			
			const content: NeutralMessageContent = [
				{ type: "text", text: "Test content" }
			]
			
			const count = await handler.countTokens(content)
			
			// Should use fallback value from mocked BaseProvider
			expect(count).toBe(100)
			expect(consoleWarnSpy).toHaveBeenCalledWith(
				"Anthropic token counting failed, using fallback",
				expect.any(Error)
			)
		})

		it("should handle various content types in fallback", async () => {
			mockClient.countTokens.mockRejectedValue(new Error("Network error"))
			
			const content: NeutralMessageContent = [
				{ type: "text", text: "Text content" },
				{ 
					type: "image", 
					source: { 
						type: "base64", 
						media_type: "image/png", 
						data: "base64data" 
					} 
				},
				{
					type: "tool_use",
					id: "tool-1",
					name: "test_tool",
					input: { param: "value" }
				}
			]
			
			const count = await handler.countTokens(content)
			
			// Should still return fallback count
			expect(count).toBe(100)
			expect(consoleWarnSpy).toHaveBeenCalled()
		})
	})

	describe("completePrompt Helper", () => {
		it("should concatenate text chunks from stream", async () => {
			const mockStream = async function* () {
				yield { type: "text", text: "Part 1 " }
				yield { type: "text", text: "Part 2 " }
				yield { type: "thinking", text: "Ignored" }
				yield { type: "text", text: "Part 3" }
			}

			mockClient.createMessage.mockReturnValue(mockStream())
			
			const result = await handler.completePrompt("Test prompt")
			
			// Should only concatenate text type chunks
			expect(result).toBe("Part 1 Part 2 Part 3")
			expect(mockClient.createMessage).toHaveBeenCalledWith({
				model: "claude-3-5-sonnet-20241022",
				systemPrompt: "",
				messages: [{ role: "user", content: "Test prompt" }],
				maxTokens: 8192,
				temperature: 0
			})
		})

		it("should handle empty stream", async () => {
			const mockStream = async function* () {
				// Empty stream
			}

			mockClient.createMessage.mockReturnValue(mockStream())
			
			const result = await handler.completePrompt("Test prompt")
			
			expect(result).toBe("")
		})
	})

	describe("Model Selection Edge Cases", () => {
		it("should fallback to default model for invalid ID", () => {
			mockOptions.apiModelId = "invalid-model-id"
			
			handler = new AnthropicHandler(mockOptions)
			const model = handler.getModel()
			
			// Should use default model
			expect(model.id).toBe("claude-3-5-sonnet-20241022")
		})

		it("should handle missing API key", () => {
			mockOptions.apiKey = undefined
			
			handler = new AnthropicHandler(mockOptions)
			
			// Should create client with empty string
			expect(NeutralAnthropicClient).toHaveBeenCalledWith({
				apiKey: "",
				baseURL: "https://api.anthropic.com"
			})
		})

		it("should pass custom base URL to client", () => {
			mockOptions.anthropicBaseUrl = "https://custom.api.com"
			
			handler = new AnthropicHandler(mockOptions)
			
			expect(NeutralAnthropicClient).toHaveBeenCalledWith({
				apiKey: "test-key",
				baseURL: "https://custom.api.com"
			})
		})
	})

	describe("Temperature and MaxTokens Handling", () => {
		it("should use custom temperature", () => {
			mockOptions.modelTemperature = 0.7
			
			handler = new AnthropicHandler(mockOptions)
			const model = handler.getModel()
			
			expect(model.temperature).toBe(0.7)
		})

		it("should use custom max tokens", () => {
			mockOptions.modelMaxTokens = 4096
			
			handler = new AnthropicHandler(mockOptions)
			const model = handler.getModel()
			
			expect(model.maxTokens).toBe(4096)
		})

		it("should apply default max tokens when not specified", () => {
			handler = new AnthropicHandler(mockOptions)
			const model = handler.getModel()
			
			// Should use ANTHROPIC_DEFAULT_MAX_TOKENS (8192)
			expect(model.maxTokens).toBe(8192)
		})
	})
})