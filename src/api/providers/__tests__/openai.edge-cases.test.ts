/**
 * OpenAI provider edge case tests
 * Tests streaming/non-streaming, XmlMatcher reasoning, tool_calls to MCP conversion
 */

import { OpenAiHandler } from "../openai"
import OpenAI from "openai"
import { ApiHandlerOptions } from "../../../shared/api"
import type { 
	NeutralConversationHistory, 
	NeutralMessageContent,
	NeutralMessage,
	NeutralContentBlock,
	NeutralTextContentBlock 
} from "../../../shared/neutral-history"
import { XmlMatcher, XmlMatcherResult } from "../../../utils/xml-matcher"

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
		__esModule: true,
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
	convertToOpenAiHistory: jest.fn((messages: NeutralMessage[]) => {
		// Simple mock conversion
		return messages.map((msg: NeutralMessage) => ({
			role: msg.role,
			content: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)
		}))
	}),
	convertToOpenAiContentBlocks: jest.fn((content: string | NeutralMessageContent) => {
		if (typeof content === 'string') return content
		return content.map((block: NeutralContentBlock) => {
			if (block.type === 'text') return { type: 'text', text: (block as NeutralTextContentBlock).text }
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
		mcpIntegration: { registerTool: jest.Mock; executeTool: jest.Mock }
		
		constructor() {
			this.mcpIntegration = {
				registerTool: jest.fn(),
				executeTool: jest.fn()
			}
		}
		
		registerTools() {}
		
		processToolUse(toolUse: { name: string }) {
			return Promise.resolve(`Tool result for ${toolUse.name}`)
		}
		
		countTokens(_content: string | NeutralMessageContent) {
			return Promise.resolve(100)
		}
	}
}))

// Mock model capabilities
jest.mock("../../../utils/model-capabilities", () => ({
	supportsTemperature: jest.fn((model: { reasoningEffort?: string }) => {
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
jest.mock("../../../shared/config/thea-config", () => ({
	API_REFERENCES: {
		HOMEPAGE: "https://example.com",
		APP_TITLE: "Test App"
	}
}))

describe("OpenAiHandler - Edge Cases", () => {
	let handler: OpenAiHandler
	let mockClient: jest.Mocked<OpenAI>
	let mockOptions: ApiHandlerOptions
	let consoleWarnSpy: jest.SpyInstance<void, [message?: unknown, ...optionalParams: unknown[]]>
	let mockXmlMatcher: jest.Mocked<XmlMatcher>

	beforeEach(() => {
		jest.clearAllMocks()
		
		// Setup console spy
		consoleWarnSpy = jest.spyOn(console, 'warn').mockImplementation(() => undefined)

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
		;(handler.getModel as unknown as jest.Mock) = jest.fn().mockReturnValue({
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

		// Create mock XmlMatcher with update/final API
		mockXmlMatcher = {
			update: jest.fn().mockReturnValue([]),
			final: jest.fn().mockReturnValue([]),
			processChunk: jest.fn().mockReturnValue([])
		} as unknown as jest.Mocked<XmlMatcher>

		// Mock XmlMatcher constructor to return our mock
		;(XmlMatcher as unknown as jest.Mock).mockImplementation(() => mockXmlMatcher)
	})

	afterEach(() => {
		jest.restoreAllMocks()
	})

	describe("Streaming vs Non-Streaming", () => {
		it("should handle streaming mode with XmlMatcher for reasoning", async () => {
			const mockStream = {
				*[Symbol.asyncIterator]() {
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

			mockClient.chat.completions.create.mockReturnValue(mockStream as unknown as ReturnType<typeof mockClient.chat.completions.create>)
			
			// Mock XmlMatcher to extract thinking tags
			mockXmlMatcher.update.mockReturnValueOnce([
				{ content: "Internal reasoning", tag: "reasoning" },
				{ content: "Regular text", tag: "text" }
			])

			const messages: NeutralConversationHistory = [
				{ role: "user", content: "Test message" }
			]

			const stream = handler.createMessage("System prompt", messages)
			const results: unknown[] = []
			
			for await (const chunk of stream) {
				results.push(chunk)
			}

			// Should have processed through XmlMatcher
			expect(mockXmlMatcher.update).toHaveBeenCalled()
			
			// Should include reasoning from reasoning_content
			expect(results).toContainEqual({
				type: "reasoning",
				text: "More reasoning"
			})
			
			// Should include usage
			expect(results).toContainEqual({
				type: "usage",
				inputTokens: 10,
				outputTokens: 20
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

			mockClient.chat.completions.create.mockResolvedValue(mockResponse as unknown as Awaited<ReturnType<typeof mockClient.chat.completions.create>>)

			const messages: NeutralConversationHistory = [
				{ role: "user", content: "Test" }
			]

			const stream = handler.createMessage("System", messages)
			const results: unknown[] = []
			
			for await (const chunk of stream) {
				results.push(chunk)
			}

			// Should return text and usage
		    expect(results).toEqual([
			    { type: "text", text: "Response text" },
			    { type: "usage", inputTokens: 15, outputTokens: 25 }
		    ])
		})

		it("should handle streaming with no usage data", async () => {
			const mockStream = {
				*[Symbol.asyncIterator]() {
					yield {
						choices: [{
							delta: {
								content: "Text without usage"
							}
						}]
					}
				}
			}

			mockClient.chat.completions.create.mockReturnValue(mockStream as unknown as ReturnType<typeof mockClient.chat.completions.create>)

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
				*[Symbol.asyncIterator]() {
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

			mockClient.chat.completions.create.mockReturnValue(mockStream as unknown as ReturnType<typeof mockClient.chat.completions.create>)

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
				*[Symbol.asyncIterator]() {
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

			mockClient.chat.completions.create.mockReturnValue(mockStream as unknown as ReturnType<typeof mockClient.chat.completions.create>)

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
				*[Symbol.asyncIterator]() {
					yield {
						choices: [{
							delta: {
								content: "<thinking>Let me analyze this</thinking>The answer is 42"
							}
						}]
					}
				}
			}

			mockClient.chat.completions.create.mockReturnValue(mockStream as unknown as ReturnType<typeof mockClient.chat.completions.create>)
			
			mockXmlMatcher.update.mockReturnValue([
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
				*[Symbol.asyncIterator]() {
					yield {
						choices: [{
							delta: { content: "Partial <thinking>incomplete" }
						}]
					}
				}
			}

			mockClient.chat.completions.create.mockReturnValue(mockStream as unknown as ReturnType<typeof mockClient.chat.completions.create>)
			
			mockXmlMatcher.update.mockReturnValue([])
			mockXmlMatcher.final.mockReturnValue([
				{ matched: true, data: "Partial " },
				{ matched: true, data: "incomplete" }
			] as XmlMatcherResult[])

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
			} as Awaited<ReturnType<typeof mockClient.chat.completions.create>>)

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
			} as Awaited<ReturnType<typeof mockClient.chat.completions.create>>)

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
			;(handler.getModel as unknown as jest.Mock) = jest.fn().mockReturnValue({
				id: "o3-mini",
				info: {
					reasoningEffort: "extreme",
					maxTokens: 8192
				}
			})

			// Mock the O3 family handler
			const handlerWithO3 = handler as unknown as { handleO3FamilyMessage: jest.Mock }
			handlerWithO3.handleO3FamilyMessage = jest.fn().mockImplementation(async function* () {
				yield { type: "text", text: "O3 response" }
			})

			const messages: NeutralConversationHistory = []
			const stream = handler.createMessage("", messages)
			const results = []
			
			for await (const chunk of stream) {
				results.push(chunk)
			}

			// Should delegate to O3 handler
			expect(handlerWithO3.handleO3FamilyMessage).toHaveBeenCalled()
			expect(results).toContainEqual({ type: "text", text: "O3 response" })
		})

		it("should detect reasoning models by capability", async () => {
			// Mock a reasoning model
			;(handler.getModel as unknown as jest.Mock) = jest.fn().mockReturnValue({
				id: "reasoning-model",
				info: {
					reasoningEffort: "high",
					maxTokens: 8192
				}
			})

			const mockStream = {
				*[Symbol.asyncIterator]() {
					yield {
						choices: [{
							delta: { reasoning_content: "Reasoning output" }
						}]
					}
				}
			}

			mockClient.chat.completions.create.mockReturnValue(mockStream as unknown as ReturnType<typeof mockClient.chat.completions.create>)

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
				*[Symbol.asyncIterator]() {
					yield { choices: [{ delta: { content: "Test" } }] }
				}
			}

			mockClient.chat.completions.create.mockReturnValue(mockStream as unknown as ReturnType<typeof mockClient.chat.completions.create>)

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