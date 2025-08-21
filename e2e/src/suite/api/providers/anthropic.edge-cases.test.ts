import * as assert from 'assert'
import * as sinon from 'sinon'
import * as proxyquire from 'proxyquire'

/**
 * Anthropic provider edge case tests
 * Tests thinking budget clamping, tool_use to tool_result conversion, countTokens fallback
 */

import type { NeutralConversationHistory, NeutralMessageContent } from "../../../../../src/shared/neutral-history"
import { ApiHandlerOptions } from "../../../../../src/shared/api"

suite("AnthropicHandler - Edge Cases", () => {
	let AnthropicHandler: any
	let handler: any
	let mockClient: any
	let mockOptions: ApiHandlerOptions
	let consoleWarnStub: sinon.SinonStub
	let NeutralAnthropicClientStub: sinon.SinonStub
	let supportsThinkingStub: sinon.SinonStub
	let isThinkingModelStub: sinon.SinonStub
	let getBaseModelIdStub: sinon.SinonStub
	let getModelParamsStub: sinon.SinonStub

	setup(() => {
		// Reset mocks
		sinon.restore()
		
		// Setup console spy
		consoleWarnStub = sinon.stub(console, 'warn')

		// Create mock client
		mockClient = {
			createMessage: sinon.stub(),
			countTokens: sinon.stub()
		}

		// Create stubs for dependencies
		NeutralAnthropicClientStub = sinon.stub().returns(mockClient)
		
		supportsThinkingStub = sinon.stub().callsFake((model) => {
			return model?.supportsThinking === true
		})
		
		isThinkingModelStub = sinon.stub().callsFake((id) => {
			return typeof id === 'string' && id.includes('-thinking')
		})
		
		getBaseModelIdStub = sinon.stub().callsFake((id) => {
			if (typeof id === 'string' && id.includes('-thinking')) {
				return id.replace('-thinking', '')
			}
			return id
		})
		
		getModelParamsStub = sinon.stub().callsFake((options) => {
			const customMaxTokens = options.options?.modelMaxTokens
			const customTemperature = options.options?.modelTemperature
// Mock removed - needs manual implementation
		})

		// Mock BaseProvider
		class MockBaseProvider {
			mcpIntegration: any
			
			constructor() {
				this.mcpIntegration = {
					registerTool: sinon.stub(),
					executeTool: sinon.stub()
				}
			}
			
			registerTools() {
				// Mock method
			}
			
			async processToolUse(toolUse: any) {
				return `Tool result for ${toolUse.name}`
			}
			
			async countTokens(content: any) {
				// Mock fallback token counting
				return 100
			}
		}

		// Load AnthropicHandler with mocked dependencies
		const module = proxyquire('../../../../../src/api/providers/anthropic', {
			'../../services/anthropic': {
				NeutralAnthropicClient: NeutralAnthropicClientStub
			},
			'../../shared/api': {
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
				}
			},
			'./constants': {
				ANTHROPIC_DEFAULT_MAX_TOKENS: 8192
			},
			'../../utils/model-capabilities': {
				supportsThinking: supportsThinkingStub
			},
			'../../utils/model-pattern-detection': {
				isThinkingModel: isThinkingModelStub,
				getBaseModelId: getBaseModelIdStub
			},
			'../index': {
				getModelParams: getModelParamsStub
			},
			'./base-provider': {
				BaseProvider: MockBaseProvider
			}
		})
		
		AnthropicHandler = module.AnthropicHandler

		// Default options
		mockOptions = {
			apiKey: "test-key",
			apiModelId: "claude-3-5-sonnet-20241022",
			anthropicBaseUrl: "https://api.anthropic.com"
		}

		handler = new AnthropicHandler(mockOptions)
	})

	teardown(() => {
		sinon.restore()
	})

	suite("Thinking Budget Clamping", () => {
		test("should clamp thinking budget to 80% of max tokens", () => {
			// Set custom max tokens
			mockOptions.modelMaxTokens = 10000
			mockOptions.modelMaxThinkingTokens = 9000 // Too high, should be clamped
			mockOptions.apiModelId = "claude-3-5-sonnet-20241022-thinking"
			
			handler = new AnthropicHandler(mockOptions)
			const model = handler.getModel()
			
			// Should clamp to 80% of 10000 = 8000
			assert.deepStrictEqual(model.thinking, {
				type: "enabled",
				budget_tokens: 8000
			})
		})

		test("should ensure minimum thinking budget of 1024 tokens", () => {
			// Set very low max tokens
			mockOptions.modelMaxTokens = 1000
			mockOptions.apiModelId = "claude-3-5-sonnet-20241022-thinking"
			
			handler = new AnthropicHandler(mockOptions)
			const model = handler.getModel()
			
			// Should use minimum of 1024 despite 80% being 800
			assert.deepStrictEqual(model.thinking, {
				type: "enabled",
				budget_tokens: 1024
			})
		})

		test("should handle thinking model variants correctly", () => {
			mockOptions.apiModelId = "claude-3-5-haiku-20241022-thinking"
			
			handler = new AnthropicHandler(mockOptions)
			const model = handler.getModel()
			
			// Should detect thinking variant and enable thinking
			assert.strictEqual(model.virtualId, "claude-3-5-haiku-20241022-thinking")
			assert.strictEqual(model.id, "claude-3-5-haiku-20241022") // Base model ID
			assert.strictEqual(model.thinking?.type, "enabled")
		})

		test("should not enable thinking for non-thinking models", () => {
			mockOptions.apiModelId = "claude-2.1"
			
			handler = new AnthropicHandler(mockOptions)
			const model = handler.getModel()
			
			// Should not have thinking enabled
			assert.strictEqual(model.thinking, undefined)
		})

		test("should respect custom thinking budget within limits", () => {
			mockOptions.modelMaxTokens = 10000
			mockOptions.modelMaxThinkingTokens = 5000 // Within 80% limit
			mockOptions.apiModelId = "claude-3-5-sonnet-20241022-thinking"
			
			handler = new AnthropicHandler(mockOptions)
			const model = handler.getModel()
			
			// Should use the custom value
			assert.deepStrictEqual(model.thinking, {
				type: "enabled",
				budget_tokens: 5000
			})
		})
	})

	suite("Tool Use to Tool Result Conversion", () => {
		test("should convert tool_use chunks to tool_result", async () => {
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

			mockClient.createMessage.returns(mockStream())

			const messages: NeutralConversationHistory = [
				{ role: "user", content: "What is 2 + 3?" }
			]

			const stream = handler.createMessage("You are a helpful assistant", messages)
			const results = []
			
			for await (const chunk of stream) {
				results.push(chunk)
			}

			// Should have converted tool_use to tool_result
			assert.strictEqual(results.length, 3)
			assert.deepStrictEqual(results[0], { type: "text", text: "Let me help you with that." })
			assert.deepStrictEqual(results[1], {
				type: "tool_result",
				id: "tool-123",
				content: "Tool result for calculator"
			})
			assert.deepStrictEqual(results[2], { type: "text", text: "The result is above." })
		})

		test("should handle tool result as JSON string when not a string", async () => {
			// Override processToolUse to return an object
			handler.processToolUse = sinon.stub().resolves({ 
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

			mockClient.createMessage.returns(mockStream())

			const messages: NeutralConversationHistory = []
			const stream = handler.createMessage("", messages)
			const results = []
			
			for await (const chunk of stream) {
				results.push(chunk)
			}

			// Should JSON stringify non-string results
			assert.deepStrictEqual(results[0], {
				type: "tool_result",
				id: "tool-456",
				content: JSON.stringify({ answer: 5, explanation: "2 + 3 = 5" })
			})
		})

		test("should pass through non-tool_use chunks unchanged", async () => {
			const mockStream = async function* () {
				yield { type: "text", text: "Regular text" }
				yield { type: "thinking", text: "Internal thoughts" }
				yield { type: "usage", input_tokens: 10, output_tokens: 20 }
			}

			mockClient.createMessage.returns(mockStream())

			const messages: NeutralConversationHistory = []
			const stream = handler.createMessage("", messages)
			const results = []
			
			for await (const chunk of stream) {
				results.push(chunk)
			}

			// All chunks should pass through unchanged
			assert.deepStrictEqual(results, [
				{ type: "text", text: "Regular text" },
				{ type: "thinking", text: "Internal thoughts" },
				{ type: "usage", input_tokens: 10, output_tokens: 20 }
			])
		})
	})

	suite("countTokens Fallback", () => {
		test("should use client countTokens when successful", async () => {
			mockClient.countTokens.resolves(42)
			
			const content: NeutralMessageContent = [
				{ type: "text", text: "Test content" }
			]
			
			const count = await handler.countTokens(content)
			
			assert.strictEqual(count, 42)
			assert.ok(mockClient.countTokens.calledWith(
				"claude-3-5-sonnet-20241022",
				content
			))
			assert.ok(!consoleWarnStub.called)
		})

		test("should fallback to base implementation on error", async () => {
			mockClient.countTokens.rejects(new Error("API error"))
			
			const content: NeutralMessageContent = [
				{ type: "text", text: "Test content" }
			]
			
			const count = await handler.countTokens(content)
			
			// Should use fallback value from mocked BaseProvider
			assert.strictEqual(count, 100)
			assert.ok(consoleWarnStub.calledWith(
				"Anthropic token counting failed, using fallback",
				sinon.match.instanceOf(Error)
			))
		})

		test("should handle various content types in fallback", async () => {
			mockClient.countTokens.rejects(new Error("Network error"))
			
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
			assert.strictEqual(count, 100)
			assert.ok(consoleWarnStub.called)
		})
	})

	suite("completePrompt Helper", () => {
		test("should concatenate text chunks from stream", async () => {
			const mockStream = async function* () {
				yield { type: "text", text: "Part 1 " }
				yield { type: "text", text: "Part 2 " }
				yield { type: "thinking", text: "Ignored" }
				yield { type: "text", text: "Part 3" }
			}

			mockClient.createMessage.returns(mockStream())
			
			const result = await handler.completePrompt("Test prompt")
			
			// Should only concatenate text type chunks
			assert.strictEqual(result, "Part 1 Part 2 Part 3")
			assert.ok(mockClient.createMessage.calledWith({
				model: "claude-3-5-sonnet-20241022",
				systemPrompt: "",
				messages: [{ role: "user", content: "Test prompt" }],
				maxTokens: 8192,
				temperature: 0
			}))
		})

		test("should handle empty stream", async () => {
			const mockStream = async function* () {
				// Empty stream
			}

			mockClient.createMessage.returns(mockStream())
			
			const result = await handler.completePrompt("Test prompt")
			
			assert.strictEqual(result, "")
		})
	})

	suite("Model Selection Edge Cases", () => {
		test("should fallback to default model for invalid ID", () => {
			mockOptions.apiModelId = "invalid-model-id"
			
			handler = new AnthropicHandler(mockOptions)
			const model = handler.getModel()
			
			// Should use default model
			assert.strictEqual(model.id, "claude-3-5-sonnet-20241022")
		})

		test("should handle missing API key", () => {
			mockOptions.apiKey = undefined
			
			handler = new AnthropicHandler(mockOptions)
			
			// Should create client with empty string
			assert.ok(NeutralAnthropicClientStub.calledWith({
				apiKey: "",
				baseURL: "https://api.anthropic.com"
			}))
		})

		test("should pass custom base URL to client", () => {
			mockOptions.anthropicBaseUrl = "https://custom.api.com"
			
			handler = new AnthropicHandler(mockOptions)
			
			assert.ok(NeutralAnthropicClientStub.calledWith({
				apiKey: "test-key",
				baseURL: "https://custom.api.com"
			}))
		})
	})

	suite("Temperature and MaxTokens Handling", () => {
		test("should use custom temperature", () => {
			mockOptions.modelTemperature = 0.7
			
			handler = new AnthropicHandler(mockOptions)
			const model = handler.getModel()
			
			assert.strictEqual(model.temperature, 0.7)
		})

		test("should use custom max tokens", () => {
			mockOptions.modelMaxTokens = 4096
			
			handler = new AnthropicHandler(mockOptions)
			const model = handler.getModel()
			
			assert.strictEqual(model.maxTokens, 4096)
		})

		test("should apply default max tokens when not specified", () => {
			handler = new AnthropicHandler(mockOptions)
			const model = handler.getModel()
			
			// Should use ANTHROPIC_DEFAULT_MAX_TOKENS (8192)
			assert.strictEqual(model.maxTokens, 8192)
		})
	})
// Mock cleanup
