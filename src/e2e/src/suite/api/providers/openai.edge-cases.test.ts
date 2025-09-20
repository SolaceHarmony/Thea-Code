import * as assert from 'assert'
import { expect } from 'chai'
import * as sinon from 'sinon'
/**
 * OpenAI provider edge case tests
 * Tests streaming/non-streaming, XmlMatcher reasoning, tool_calls to MCP conversion
 */

import OpenAI from "openai"
import { OpenAiHandler } from "../openai"
import type { ApiHandlerOptions   } from "../../../shared/api"
import type {
	NeutralConversationHistory, 
	NeutralMessageContent,
	NeutralMessage,
	NeutralContentBlock,
	NeutralTextContentBlock
} from "../../../shared/neutral-history"
import { XmlMatcher, XmlMatcherResult } from "../../../utils/xml-matcher"

// Mock OpenAI client
// Mock needs manual implementation
		chat: {
			completions: {
				create: sinon.stub()
			}
		}
	}
// Mock removed - needs manual implementation
// Mock cleanup
// Mock axios for any HTTP requests
// TODO: Mock setup needs manual migration for "axios"

// Mock XmlMatcher
// TODO: Use proxyquire for module mocking
		// Mock for "../../../utils/xml-matcher" needed here
	XmlMatcher: sinon.stub()
// Mock cleanup needed

// Mock conversion functions
// TODO: Use proxyquire for module mocking
		// Mock for "../../transform/neutral-openai-format" needed here
	convertToOpenAiHistory: sinon.stub((messages: NeutralMessage[]) => {
		// Simple mock conversion
		return messages.map((msg: NeutralMessage) => ({
			role: msg.role,
			content: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)
		}))
	}),
	convertToOpenAiContentBlocks: sinon.stub((content: string | NeutralMessageContent) => {
		if (typeof content === 'string') return content
		return content.map((block: NeutralContentBlock) => {
			if (block.type === 'text') return { type: 'text', text: (block as NeutralTextContentBlock).text }
			if (block.type === 'image') return { type: 'image_url', image_url: { url: 'data:image/png;base64,test' } }
			return block
		})
	})
// Mock cleanup needed

// Mock shared tool use functions
// TODO: Use proxyquire for module mocking
		// Mock for "../shared/tool-use" needed here
	hasToolCalls: sinon.stub(),
	ToolCallAggregator: sinon.stub().callsFake(() => ({
		processChunk: sinon.stub(),
		getCompletedToolCalls: sinon.stub().returns([])
	}))
// Mock cleanup needed

// Mock BaseProvider
// TODO: Use proxyquire for module mocking
		// Mock for "../base-provider" needed here
	BaseProvider: class MockBaseProvider {
		mcpIntegration: { registerTool: sinon.SinonStub; executeTool: sinon.SinonStub }
		
		constructor() {
			this.mcpIntegration = {
				registerTool: sinon.stub(),
				executeTool: sinon.stub()
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
// Mock cleanup
// Mock model capabilities
// TODO: Use proxyquire for module mocking
		// Mock for "../../../utils/model-capabilities" needed here
	supportsTemperature: sinon.stub((model: { reasoningEffort?: string }) => {
		// Mock that most models support temperature
		return model?.reasoningEffort !== 'extreme'
	}),
	hasCapability: sinon.stub()
// Mock cleanup needed

// Mock constants
// TODO: Use proxyquire for module mocking
		// Mock for "../constants" needed here
	ANTHROPIC_DEFAULT_MAX_TOKENS: 8192
// Mock cleanup needed

// Mock branded constants
// TODO: Use proxyquire for module mocking
		// Mock for "../../../shared/config/thea-config" needed here
	API_REFERENCES: {
		HOMEPAGE: "https://example.com",
		APP_TITLE: "Test App"
	}
// Mock cleanup
suite("OpenAiHandler - Edge Cases", () => {
	let handler: OpenAiHandler
	let mockClient: sinon.SinonStubbedInstance<OpenAI>
	let mockOptions: ApiHandlerOptions
	let consoleWarnSpy: sinon.SinonStub<void, [message?: unknown, ...optionalParams: unknown[]]>
	let mockXmlMatcher: sinon.SinonStubbedInstance<XmlMatcher>

	setup(() => {
		sinon.restore()
		
		// Setup console spy
		consoleWarnSpy = sinon.spy(console, 'warn').callsFake(() => undefined)

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
		;(handler.getModel as unknown as sinon.SinonStub) = sinon.stub().returns({
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
		const OpenAIMock = require("openai").default
		mockClient = OpenAIMock.mock.results[0]?.value || OpenAIMock()

		// Create mock XmlMatcher with update/final API
		mockXmlMatcher = {
			update: sinon.stub().returns([]),
			final: sinon.stub().returns([]),
			processChunk: sinon.stub().returns([])
		} as unknown as sinon.SinonStubbedInstance<XmlMatcher>

		// Mock XmlMatcher constructor to return our mock
		;(XmlMatcher as unknown as sinon.SinonStub).callsFake(() => mockXmlMatcher)
	})

	teardown(() => {
		sinon.restore()
	})

	suite("Streaming vs Non-Streaming", () => {
		test("should handle streaming mode with XmlMatcher for reasoning", async () => {
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

			mockClient.chat.completions.create.returns(mockStream as unknown as ReturnType<typeof mockClient.chat.completions.create>)
			
			// Mock XmlMatcher to extract thinking tags
			mockXmlMatcher.update.onFirstCall().returns([
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
			assert.ok(mockXmlMatcher.update.called)
			
			// Should include reasoning from reasoning_content
			assert.ok(results.some(x => JSON.stringify(x) === JSON.stringify({
				type: "reasoning",
				text: "More reasoning"
			})))
			
			// Should include usage
			assert.ok(results.some(x => JSON.stringify(x) === JSON.stringify({
				type: "usage",
				inputTokens: 10,
				outputTokens: 20
			})))
		})

		test("should handle non-streaming mode", async () => {
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

			mockClient.chat.completions.create.resolves(mockResponse as unknown as Awaited<ReturnType<typeof mockClient.chat.completions.create>>)

			const messages: NeutralConversationHistory = [
				{ role: "user", content: "Test" }
			]

			const stream = handler.createMessage("System", messages)
			const results: unknown[] = []
			
			for await (const chunk of stream) {
				results.push(chunk)
			}

			// Should return text and usage
		    assert.deepStrictEqual(results, [
			    { type: "text", text: "Response text" },
			    { type: "usage", inputTokens: 15, outputTokens: 25 }
		    ])
		})

		test("should handle streaming with no usage data", async () => {
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

			mockClient.chat.completions.create.returns(mockStream as unknown as ReturnType<typeof mockClient.chat.completions.create>)

			const messages: NeutralConversationHistory = []
			const stream = handler.createMessage("", messages)
			const results = []
			
			for await (const chunk of stream) {
				results.push(chunk)
			}

			// Should not have usage chunk
			assert.ok(!results.some(item => JSON.stringify(item) === JSON.stringify(// TODO: Object partial match - { type: "usage" })))
			)
		})
	})

	suite("Tool Calls to MCP Conversion", () => {
		test("should convert tool_calls to MCP tool_result", async () => {
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

			mockClient.chat.completions.create.returns(mockStream as unknown as ReturnType<typeof mockClient.chat.completions.create>)

			// Mock ToolCallAggregator to return completed calls
			const mockAggregator = {
				processChunk: sinon.stub(),
				getCompletedToolCalls: sinon.stub().returns([{
					id: "call_123",
					name: "calculator",
					arguments: { operation: "add", a: 2, b: 3 }
				}])
			}
			
			require("../shared/tool-use").ToolCallAggregator.callsFake(() => mockAggregator)

			const messages: NeutralConversationHistory = []
			const stream = handler.createMessage("", messages)
			const results = []
			
			for await (const chunk of stream) {
				results.push(chunk)
			}

			// Should convert to tool_result
			assert.ok(results.some(x => JSON.stringify(x) === JSON.stringify({
				type: "tool_result",
				id: "call_123",
				content: "Tool result for calculator"
			})))
		})

		test("should handle multiple tool calls in sequence", async () => {
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

			mockClient.chat.completions.create.returns(mockStream as unknown as ReturnType<typeof mockClient.chat.completions.create>)

			const mockAggregator = {
				processChunk: sinon.stub(),
				getCompletedToolCalls: sinon.stub()
					.onFirstCall().returns([{ id: "call_1", name: "tool1", arguments: {} }])
					.onFirstCall().returns([{ id: "call_2", name: "tool2", arguments: {} }])
			}
			
			require("../shared/tool-use").ToolCallAggregator.callsFake(() => mockAggregator)

			const messages: NeutralConversationHistory = []
			const stream = handler.createMessage("", messages)
			const results = []
			
			for await (const chunk of stream) {
				results.push(chunk)
			}

			// Should have both tool results
			expect(results.filter(r => r.type === "tool_result")).length, 2)
		})
	})

	suite("XmlMatcher Reasoning Extraction", () => {
		test("should extract thinking tags from content", async () => {
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

			mockClient.chat.completions.create.returns(mockStream as unknown as ReturnType<typeof mockClient.chat.completions.create>)
			
			mockXmlMatcher.update.returns([
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
			assert.ok(results.some(x => JSON.stringify(x) === JSON.stringify({ type: "thinking", text: "Let me analyze this" })))
			assert.ok(results.some(x => JSON.stringify(x) === JSON.stringify({ type: "text", text: "The answer is 42" })))
		})

		test("should call XmlMatcher.final() for remaining content", async () => {
			const mockStream = {
				*[Symbol.asyncIterator]() {
					yield {
						choices: [{
							delta: { content: "Partial <thinking>incomplete" }
						}]
					}
				}
			}

			mockClient.chat.completions.create.returns(mockStream as unknown as ReturnType<typeof mockClient.chat.completions.create>)
			
			mockXmlMatcher.update.returns([])
			mockXmlMatcher.final.returns([
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
			assert.ok(mockXmlMatcher.final.called)
			assert.ok(results.some(x => JSON.stringify(x) === JSON.stringify({ type: "thinking", text: "incomplete" })))
		})
	})

	suite("countTokens Edge Cases", () => {
		test("should count tokens using OpenAI API", async () => {
			mockClient.chat.completions.create.resolves({
				choices: [{ message: { content: "" } }],
				usage: { prompt_tokens: 42, completion_tokens: 0 }
			} as Awaited<ReturnType<typeof mockClient.chat.completions.create>>)

			const content: NeutralMessageContent = [
				{ type: "text", text: "Test content" }
			]

			const count = await handler.countTokens(content)

			assert.strictEqual(count, 42)
			assert.ok(mockClient.chat.completions.create.calledWith({
					model: "gpt-4",
					stream: false
}))
		})

		test("should fallback on API error", async () => {
			mockClient.chat.completions.create.rejects(new Error("API error"))

			const content: NeutralMessageContent = [
				{ type: "text", text: "Test" }
			]

			const count = await handler.countTokens(content)

			// Should use fallback
			assert.strictEqual(count, 100)
			assert.ok(consoleWarnSpy.calledWith(
				"OpenAI token counting failed, using fallback",
				sinon.match.instanceOf(Error))
			)
		})

		test("should handle mixed content types", async () => {
			mockClient.chat.completions.create.resolves({
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

			assert.strictEqual(count, 150)
		})
	})

	suite("Azure OpenAI Handling", () => {
		test("should detect Azure URLs and use AzureOpenAI client", () => {
			mockOptions.openAiBaseUrl = "https://myinstance.openai.azure.com/v1"
			
			const { AzureOpenAI } = require("openai")
			AzureOpenAI.callsFake(() => mockClient)

			handler = new OpenAiHandler(mockOptions)

			assert.ok(AzureOpenAI.calledWith({
					baseURL: "https://myinstance.openai.azure.com/v1",
					apiKey: "test-key"
}))
		})

		test("should use AzureOpenAI when openAiUseAzure flag is set", () => {
			mockOptions.openAiUseAzure = true
			
			const { AzureOpenAI } = require("openai")
			AzureOpenAI.callsFake(() => mockClient)

			handler = new OpenAiHandler(mockOptions)

			assert.ok(AzureOpenAI.called)
		})

		test("should handle invalid base URLs gracefully", () => {
			mockOptions.openAiBaseUrl = "not-a-valid-url"
			
			// Should not throw
			expect(() => new OpenAiHandler(mockOptions)).not.toThrow()
			
			// Should use regular OpenAI client
			assert.ok(OpenAI.called)
		})
	})

	suite("Model-Specific Handling", () => {
		test("should handle models without temperature support", async () => {
			// Mock a model that doesn't support temperature
			;(handler.getModel as unknown as sinon.SinonStub) = sinon.stub().returns({
				id: "o3-mini",
				info: {
					reasoningEffort: "extreme",
					maxTokens: 8192
				}
			})

			// Mock the O3 family handler
			const handlerWithO3 = handler as unknown as { handleO3FamilyMessage: sinon.SinonStub }
			handlerWithO3.handleO3FamilyMessage = sinon.stub().callsFake(async function* () {
				yield { type: "text", text: "O3 response" }
			})

			const messages: NeutralConversationHistory = []
			const stream = handler.createMessage("", messages)
			const results = []
			
			for await (const chunk of stream) {
				results.push(chunk)
			}

			// Should delegate to O3 handler
			assert.ok(handlerWithO3.handleO3FamilyMessage.called)
			assert.ok(results.some(x => JSON.stringify(x) === JSON.stringify({ type: "text", text: "O3 response" })))
		})

		test("should detect reasoning models by capability", async () => {
			// Mock a reasoning model
			;(handler.getModel as unknown as sinon.SinonStub) = sinon.stub().returns({
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

			mockClient.chat.completions.create.returns(mockStream as unknown as ReturnType<typeof mockClient.chat.completions.create>)

			const messages: NeutralConversationHistory = []
			const stream = handler.createMessage("", messages)
			const results = []
			
			for await (const chunk of stream) {
				results.push(chunk)
			}

			// Should handle reasoning content
			assert.ok(results.some(x => JSON.stringify(x) === JSON.stringify({
				type: "reasoning",
				text: "Reasoning output"
			})))
		})
	})

	suite("Default Values and Fallbacks", () => {
		test("should use default API key when not provided", () => {
			mockOptions.openAiApiKey = undefined
			
			handler = new OpenAiHandler(mockOptions)
			
			assert.ok(OpenAI.calledWith({
					apiKey: "not-provided"
}))
		})

		test("should use default base URL when not provided", () => {
			mockOptions.openAiBaseUrl = undefined
			
			handler = new OpenAiHandler(mockOptions)
			
			assert.ok(OpenAI.calledWith({
					baseURL: "https://api.openai.com/v1"
}))
		})

		test("should handle empty model ID", async () => {
			mockOptions.openAiModelId = ""
			handler = new OpenAiHandler(mockOptions)
			
			const mockStream = {
				*[Symbol.asyncIterator]() {
					yield { choices: [{ delta: { content: "Test" } }] }
				}
			}

			mockClient.chat.completions.create.returns(mockStream as unknown as ReturnType<typeof mockClient.chat.completions.create>)

			const messages: NeutralConversationHistory = []
			const stream = handler.createMessage("", messages)
			
			// Should not throw
			const results = []
			for await (const chunk of stream) {
				results.push(chunk)
			}
			
			assert.ok(results.length > 0)
		})
	})
// Mock cleanup
