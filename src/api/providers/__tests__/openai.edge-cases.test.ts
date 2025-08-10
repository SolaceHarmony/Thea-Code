/**
 * OpenAI provider edge case tests
 * Tests streaming/non-streaming, XmlMatcher reasoning, tool_calls to MCP conversion
 */

import { describe, expect, it, jest, beforeEach, afterEach } from "@jest/globals"
import { OpenAiHandler } from "../openai"
import OpenAI from "openai"
import { ApiHandlerOptions, ModelInfo } from "../../../shared/api"
import type { NeutralConversationHistory, NeutralMessageContent } from "../../../shared/neutral-history"
import { XmlMatcher } from "../../../utils/xml-matcher"

// Mock OpenAI client
jest.mock("openai", () => {
	const mockClient = {
		chat: {
			completions: {
				create: jest.fn()
			}
		}
	}
	return {
		default: jest.fn(() => mockClient),
		AzureOpenAI: jest.fn(() => mockClient)
	}
})

// Mock axios for any HTTP requests
jest.mock("axios")

// Mock XmlMatcher
jest.mock("../../../utils/xml-matcher", () => ({
	XmlMatcher: jest.fn()
}))

// Mock conversion functions
jest.mock("../../transform/neutral-openai-format", () => ({
	convertToOpenAiHistory: jest.fn((messages) => {
		// Simple mock conversion
		return messages.map((msg: any) => ({
			role: msg.role,
			content: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)
		}))
	}),
	convertToOpenAiContentBlocks: jest.fn((content) => {
		if (typeof content === 'string') return content
		return content.map((block: any) => {
			if (block.type === 'text') return { type: 'text', text: block.text }
			if (block.type === 'image') return { type: 'image_url', image_url: { url: 'data:image/png;base64,test' } }
			return block
		})
	})
}))

// Mock shared tool use functions
jest.mock("../shared/tool-use", () => ({
	hasToolCalls: jest.fn(),
	ToolCallAggregator: jest.fn().mockImplementation(() => ({
		processChunk: jest.fn(),
		getCompletedToolCalls: jest.fn().mockReturnValue([])
	}))
}))

// Mock BaseProvider
jest.mock("../base-provider", () => ({
	BaseProvider: class MockBaseProvider {
		mcpIntegration: any
		
		constructor() {
			this.mcpIntegration = {
				registerTool: jest.fn(),
				executeTool: jest.fn()
			}
		}
		
		registerTools() {}
		
		async processToolUse(toolUse: any) {
			return `Tool result for ${toolUse.name}`
		}
		
		async countTokens(content: any) {
			return 100
		}
	}
}))

// Mock model capabilities
jest.mock("../../../utils/model-capabilities", () => ({
	supportsTemperature: jest.fn((model) => {
		// Mock that most models support temperature
		return model?.reasoningEffort !== 'extreme'
	}),
	hasCapability: jest.fn()
}))

// Mock constants
jest.mock("../constants", () => ({
	ANTHROPIC_DEFAULT_MAX_TOKENS: 8192
}))

// Mock branded constants
jest.mock("../../../../dist/thea-config", () => ({
	API_REFERENCES: {
		HOMEPAGE: "https://example.com",
		APP_TITLE: "Test App"
	}
}))

describe("OpenAiHandler - Edge Cases", () => {
	let handler: OpenAiHandler
	let mockClient: jest.Mocked<OpenAI>
	let mockOptions: ApiHandlerOptions
	let consoleWarnSpy: jest.SpyInstance
	let mockXmlMatcher: jest.Mocked<XmlMatcher>

	beforeEach(() => {
		jest.clearAllMocks()
		
		// Setup console spy
		consoleWarnSpy = jest.spyOn(console, 'warn').mockImplementation()

		// Default options - define first
		mockOptions = {
			openAiApiKey: "test-key",
			openAiModelId: "gpt-4",
			openAiBaseUrl: "https://api.openai.com/v1",
			openAiStreamingEnabled: true
		}

		// Create handler which will create the mock client
		handler = new OpenAiHandler(mockOptions)
		
		// Mock getModel to return reasonable defaults
		handler.getModel = jest.fn().mockReturnValue({
			id: mockOptions.openAiModelId || "gpt-4",
			info: {
				maxTokens: 8192,
				contextWindow: 128000,
				supportsImages: true,
				supportsPromptCache: false,
				inputPrice: 30,
				outputPrice: 60,
				reasoningEffort: "low"
			}
		})
		
		// Get the mock client that was created
		const OpenAIMock = jest.requireMock("openai").default
		mockClient = OpenAIMock.mock.results[0]?.value || OpenAIMock()

		// Create mock XmlMatcher
		mockXmlMatcher = {
			processChunk: jest.fn(),
			final: jest.fn().mockReturnValue([])
		} as any

		// Mock XmlMatcher constructor
		(XmlMatcher as jest.Mock).mockImplementation(() => mockXmlMatcher)
	})

	afterEach(() => {
		jest.restoreAllMocks()
	})

	describe("Streaming vs Non-Streaming", () => {
		it("should handle streaming mode with XmlMatcher for reasoning", async () => {
			const mockStream = {
				async *[Symbol.asyncIterator]() {
					yield {
						choices: [{
							delta: {
								content: "<thinking>Internal reasoning</thinking>Regular text"
							}
						}]
					}
					yield {
						choices: [{
							delta: {
								reasoning_content: "More reasoning"
							}
						}]
					}
					yield {
						usage: {
							prompt_tokens: 10,
							completion_tokens: 20
						}
					}
				}
			}

			mockClient.chat.completions.create.mockReturnValue(mockStream as any)
			
			// Mock XmlMatcher to extract thinking tags
			mockXmlMatcher.processChunk.mockReturnValueOnce([
				{ type: "thinking", text: "Internal reasoning" },
				{ type: "text", text: "Regular text" }
			])

			const messages: NeutralConversationHistory = [
				{ role: "user", content: "Test message" }
			]

			const stream = handler.createMessage("System prompt", messages)
			const results = []
			
			for await (const chunk of stream) {
				results.push(chunk)
			}

			// Should have processed through XmlMatcher
			expect(mockXmlMatcher.processChunk).toHaveBeenCalled()
			
			// Should include reasoning from reasoning_content
			expect(results).toContainEqual({
				type: "reasoning",
				text: "More reasoning"
			})
			
			// Should include usage
			expect(results).toContainEqual({
				type: "usage",
				input_tokens: 10,
				output_tokens: 20
			})
		})

		it("should handle non-streaming mode", async () => {
			mockOptions.openAiStreamingEnabled = false
			handler = new OpenAiHandler(mockOptions)

			const mockResponse = {
				choices: [{
					message: {
						content: "Response text"
					}
				}],
				usage: {
					prompt_tokens: 15,
					completion_tokens: 25
				}
			}

			mockClient.chat.completions.create.mockResolvedValue(mockResponse as any)

			const messages: NeutralConversationHistory = [
				{ role: "user", content: "Test" }
			]

			const stream = handler.createMessage("System", messages)
			const results = []
			
			for await (const chunk of stream) {
				results.push(chunk)
			}

			// Should return text and usage
			expect(results).toEqual([
				{ type: "text", text: "Response text" },
				{ type: "usage", input_tokens: 15, output_tokens: 25 }
			])
		})

		it("should handle streaming with no usage data", async () => {
			const mockStream = {
				async *[Symbol.asyncIterator]() {
					yield {
						choices: [{
							delta: {
								content: "Text without usage"
							}
						}]
					}
				}
			}

			mockClient.chat.completions.create.mockReturnValue(mockStream as any)

			const messages: NeutralConversationHistory = []
			const stream = handler.createMessage("", messages)
			const results = []
			
			for await (const chunk of stream) {
				results.push(chunk)
			}

			// Should not have usage chunk
			expect(results).not.toContainEqual(
				expect.objectContaining({ type: "usage" })
			)
		})
	})

	describe("Tool Calls to MCP Conversion", () => {
		it("should convert tool_calls to MCP tool_result", async () => {
			const mockStream = {
				async *[Symbol.asyncIterator]() {
					yield {
						choices: [{
							delta: {
								tool_calls: [{
									id: "call_123",
									function: {
										name: "calculator",
										arguments: '{"operation": "add", "a": 2, "b": 3}'
									}
								}]
							}
						}]
					}
				}
			}

			mockClient.chat.completions.create.mockReturnValue(mockStream as any)

			// Mock ToolCallAggregator to return completed calls
			const mockAggregator = {
				processChunk: jest.fn(),
				getCompletedToolCalls: jest.fn().mockReturnValue([{
					id: "call_123",
					name: "calculator",
					arguments: { operation: "add", a: 2, b: 3 }
				}])
			}
			
			jest.requireMock("../shared/tool-use").ToolCallAggregator.mockImplementation(() => mockAggregator)

			const messages: NeutralConversationHistory = []
			const stream = handler.createMessage("", messages)
			const results = []
			
			for await (const chunk of stream) {
				results.push(chunk)
			}

			// Should convert to tool_result
			expect(results).toContainEqual({
				type: "tool_result",
				id: "call_123",
				content: "Tool result for calculator"
			})
		})

		it("should handle multiple tool calls in sequence", async () => {
			const mockStream = {
				async *[Symbol.asyncIterator]() {
					yield {
						choices: [{
							delta: {
								tool_calls: [{
									id: "call_1",
									function: { name: "tool1", arguments: '{}' }
								}]
							}
						}]
					}
					yield {
						choices: [{
							delta: {
								tool_calls: [{
									id: "call_2",
									function: { name: "tool2", arguments: '{}' }
								}]
							}
						}]
					}
				}
			}

			mockClient.chat.completions.create.mockReturnValue(mockStream as any)

			const mockAggregator = {
				processChunk: jest.fn(),
				getCompletedToolCalls: jest.fn()
					.mockReturnValueOnce([{ id: "call_1", name: "tool1", arguments: {} }])
					.mockReturnValueOnce([{ id: "call_2", name: "tool2", arguments: {} }])
			}
			
			jest.requireMock("../shared/tool-use").ToolCallAggregator.mockImplementation(() => mockAggregator)

			const messages: NeutralConversationHistory = []
			const stream = handler.createMessage("", messages)
			const results = []
			
			for await (const chunk of stream) {
				results.push(chunk)
			}

			// Should have both tool results
			expect(results.filter(r => r.type === "tool_result")).toHaveLength(2)
		})
	})

	describe("XmlMatcher Reasoning Extraction", () => {
		it("should extract thinking tags from content", async () => {
			const mockStream = {
				async *[Symbol.asyncIterator]() {
					yield {
						choices: [{
							delta: {
								content: "<thinking>Let me analyze this</thinking>The answer is 42"
							}
						}]
					}
				}
			}

			mockClient.chat.completions.create.mockReturnValue(mockStream as any)
			
			mockXmlMatcher.processChunk.mockReturnValue([
				{ type: "thinking", text: "Let me analyze this" },
				{ type: "text", text: "The answer is 42" }
			])

			const messages: NeutralConversationHistory = []
			const stream = handler.createMessage("", messages)
			const results = []
			
			for await (const chunk of stream) {
				results.push(chunk)
			}

			// Should have both thinking and text
			expect(results).toContainEqual({ type: "thinking", text: "Let me analyze this" })
			expect(results).toContainEqual({ type: "text", text: "The answer is 42" })
		})

		it("should call XmlMatcher.final() for remaining content", async () => {
			const mockStream = {
				async *[Symbol.asyncIterator]() {
					yield {
						choices: [{
							delta: { content: "Partial <thinking>incomplete" }
						}]
					}
				}
			}

			mockClient.chat.completions.create.mockReturnValue(mockStream as any)
			
			mockXmlMatcher.processChunk.mockReturnValue([])
			mockXmlMatcher.final.mockReturnValue([
				{ type: "text", text: "Partial " },
				{ type: "thinking", text: "incomplete" }
			])

			const messages: NeutralConversationHistory = []
			const stream = handler.createMessage("", messages)
			const results = []
			
			for await (const chunk of stream) {
				results.push(chunk)
			}

			// Should call final() to get remaining content
			expect(mockXmlMatcher.final).toHaveBeenCalled()
			expect(results).toContainEqual({ type: "thinking", text: "incomplete" })
		})
	})

	describe("countTokens Edge Cases", () => {
		it("should count tokens using OpenAI API", async () => {
			mockClient.chat.completions.create.mockResolvedValue({
				choices: [{ message: { content: "" } }],
				usage: { prompt_tokens: 42, completion_tokens: 0 }
			} as any)

			const content: NeutralMessageContent = [
				{ type: "text", text: "Test content" }
			]

			const count = await handler.countTokens(content)

			expect(count).toBe(42)
			expect(mockClient.chat.completions.create).toHaveBeenCalledWith(
				expect.objectContaining({
					model: "gpt-4",
					stream: false
				})
			)
		})

		it("should fallback on API error", async () => {
			mockClient.chat.completions.create.mockRejectedValue(new Error("API error"))

			const content: NeutralMessageContent = [
				{ type: "text", text: "Test" }
			]

			const count = await handler.countTokens(content)

			// Should use fallback
			expect(count).toBe(100)
			expect(consoleWarnSpy).toHaveBeenCalledWith(
				"OpenAI token counting failed, using fallback",
				expect.any(Error)
			)
		})

		it("should handle mixed content types", async () => {
			mockClient.chat.completions.create.mockResolvedValue({
				choices: [{ message: { content: "" } }],
				usage: { prompt_tokens: 150, completion_tokens: 0 }
			} as any)

			const content: NeutralMessageContent = [
				{ type: "text", text: "Text" },
				{ 
					type: "image", 
					source: { type: "base64", media_type: "image/png", data: "base64" } 
				}
			]

			const count = await handler.countTokens(content)

			expect(count).toBe(150)
		})
	})

	describe("Azure OpenAI Handling", () => {
		it("should detect Azure URLs and use AzureOpenAI client", () => {
			mockOptions.openAiBaseUrl = "https://myinstance.openai.azure.com/v1"
			
			const { AzureOpenAI } = jest.requireMock("openai")
			AzureOpenAI.mockImplementation(() => mockClient)

			handler = new OpenAiHandler(mockOptions)

			expect(AzureOpenAI).toHaveBeenCalledWith(
				expect.objectContaining({
					baseURL: "https://myinstance.openai.azure.com/v1",
					apiKey: "test-key"
				})
			)
		})

		it("should use AzureOpenAI when openAiUseAzure flag is set", () => {
			mockOptions.openAiUseAzure = true
			
			const { AzureOpenAI } = jest.requireMock("openai")
			AzureOpenAI.mockImplementation(() => mockClient)

			handler = new OpenAiHandler(mockOptions)

			expect(AzureOpenAI).toHaveBeenCalled()
		})

		it("should handle invalid base URLs gracefully", () => {
			mockOptions.openAiBaseUrl = "not-a-valid-url"
			
			// Should not throw
			expect(() => new OpenAiHandler(mockOptions)).not.toThrow()
			
			// Should use regular OpenAI client
			expect(OpenAI).toHaveBeenCalled()
		})
	})

	describe("Model-Specific Handling", () => {
		it("should handle models without temperature support", async () => {
			// Mock a model that doesn't support temperature
			handler.getModel = jest.fn().mockReturnValue({
				id: "o3-mini",
				info: {
					reasoningEffort: "extreme",
					maxTokens: 8192
				}
			})

			// Mock the O3 family handler
			handler.handleO3FamilyMessage = jest.fn().mockImplementation(async function* () {
				yield { type: "text", text: "O3 response" }
			})

			const messages: NeutralConversationHistory = []
			const stream = handler.createMessage("", messages)
			const results = []
			
			for await (const chunk of stream) {
				results.push(chunk)
			}

			// Should delegate to O3 handler
			expect(handler.handleO3FamilyMessage).toHaveBeenCalled()
			expect(results).toContainEqual({ type: "text", text: "O3 response" })
		})

		it("should detect reasoning models by capability", async () => {
			// Mock a reasoning model
			handler.getModel = jest.fn().mockReturnValue({
				id: "reasoning-model",
				info: {
					reasoningEffort: "high",
					maxTokens: 8192
				}
			})

			const mockStream = {
				async *[Symbol.asyncIterator]() {
					yield {
						choices: [{
							delta: { reasoning_content: "Reasoning output" }
						}]
					}
				}
			}

			mockClient.chat.completions.create.mockReturnValue(mockStream as any)

			const messages: NeutralConversationHistory = []
			const stream = handler.createMessage("", messages)
			const results = []
			
			for await (const chunk of stream) {
				results.push(chunk)
			}

			// Should handle reasoning content
			expect(results).toContainEqual({
				type: "reasoning",
				text: "Reasoning output"
			})
		})
	})

	describe("Default Values and Fallbacks", () => {
		it("should use default API key when not provided", () => {
			mockOptions.openAiApiKey = undefined
			
			handler = new OpenAiHandler(mockOptions)
			
			expect(OpenAI).toHaveBeenCalledWith(
				expect.objectContaining({
					apiKey: "not-provided"
				})
			)
		})

		it("should use default base URL when not provided", () => {
			mockOptions.openAiBaseUrl = undefined
			
			handler = new OpenAiHandler(mockOptions)
			
			expect(OpenAI).toHaveBeenCalledWith(
				expect.objectContaining({
					baseURL: "https://api.openai.com/v1"
				})
			)
		})

		it("should handle empty model ID", async () => {
			mockOptions.openAiModelId = ""
			handler = new OpenAiHandler(mockOptions)
			
			const mockStream = {
				async *[Symbol.asyncIterator]() {
					yield { choices: [{ delta: { content: "Test" } }] }
				}
			}

			mockClient.chat.completions.create.mockReturnValue(mockStream as any)

			const messages: NeutralConversationHistory = []
			const stream = handler.createMessage("", messages)
			
			// Should not throw
			const results = []
			for await (const chunk of stream) {
				results.push(chunk)
			}
			
			expect(results.length).toBeGreaterThan(0)
		})
	})
})