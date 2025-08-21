import * as assert from 'assert'
import * as sinon from 'sinon'
import * as proxyquire from 'proxyquire'

/**
 * Test to validate that all providers are properly enabled and functional
 * This test specifically addresses issue #107 - Provider Handler Re-enablement
 */
import { ApiConfiguration, ApiProvider, ModelInfo } from "../../../../src/shared/api"
import { NeutralMessageContent } from "../../../../src/shared/neutral-history"
import { ApiStream } from "../../../../src/api/transform/stream"

// Mock implementations
const mockMcpIntegration = {
	initialize: sinon.stub().resolves(undefined),
	registerTool: sinon.stub(),
	routeToolUse: sinon.stub().resolves("{}")
}

const mockFakeAI = {
	async *createMessage(): ApiStream {
		await Promise.resolve() // Add await to satisfy async requirement
		yield { type: "text" as const, text: "Mock response" }
	},
	getModel(): { id: string; info: ModelInfo } {
		return {
			id: "fake-ai-test",
			info: {
				maxTokens: 1000,
				contextWindow: 4000,
				supportsImages: false,
				supportsPromptCache: false,
				inputPrice: 0,
				outputPrice: 0,
				description: "Mock fake AI for testing",
			},
		}
	},
	async countTokens(content: NeutralMessageContent): Promise<number> {
		// Simple token count estimation
		const text = content.map((item) => (item.type === "text" ? item.text : "")).join(" ")
		return Promise.resolve(text.split(/\s+/).length)
	},
	async completePrompt(): Promise<string> {
		return Promise.resolve("Mock completion response")
	},
}

class MockMcpIntegration {
	initialize = mockMcpIntegration.initialize
	registerTool = mockMcpIntegration.registerTool
	routeToolUse = mockMcpIntegration.routeToolUse

	static getInstance = sinon.stub().returns(mockMcpIntegration)
}

// Import buildApiHandler with mocked dependencies
const { buildApiHandler } = proxyquire('../../../../src/api/index', {
	'../services/mcp/integration/McpIntegration': {
		McpIntegration: MockMcpIntegration
	}
})

suite("Provider Enablement Validation", () => {
	const baseConfig: Omit<ApiConfiguration, "apiProvider"> = {
		apiKey: "test-key",
		apiModelId: "test-model",
		// Additional required parameters for specific providers
		mistralApiKey: "test-mistral-key",
		requestyApiKey: "test-requesty-key",
		fakeAi: mockFakeAI,
	}

	suite("Provider instantiation", () => {
		const providers = [
			"openai",
			"ollama",
			"lmstudio",
			"openai-native",
			"deepseek",
			"vscode-lm",
			"mistral",
			"unbound",
			"requesty",
			"glama",
			"fake-ai",
		] as const

		providers.forEach((provider) => {
			test(`should successfully instantiate ${provider} provider`, () => {
				const config: ApiConfiguration = {
					...baseConfig,
					apiProvider: provider,
				}

				// Test that the provider can be instantiated without throwing
				const handler = buildApiHandler(config)
				assert.notStrictEqual(handler, undefined)
				assert.strictEqual(typeof handler.createMessage, "function")
				assert.strictEqual(typeof handler.getModel, "function")
				assert.strictEqual(typeof handler.countTokens, "function")
			})
		})

		test("should throw error for human-relay provider as documented", () => {
			const config: ApiConfiguration = {
				...baseConfig,
				apiProvider: "human-relay",
			}

			assert.throws(
				() => buildApiHandler(config),
				/HumanRelayHandler is not supported in this architecture/
			)
		})

		test("should default to anthropic for unknown provider", () => {
			const config: ApiConfiguration = {
				...baseConfig,
				apiProvider: "unknown-provider" as ApiProvider,
			}

			const handler = buildApiHandler(config)
			assert.notStrictEqual(handler, undefined)
			// Should not throw, defaults to AnthropicHandler
		})
	})

	suite("Provider model information", () => {
		const providers = [
			"openai",
			"ollama",
			"lmstudio",
			"openai-native",
			"deepseek",
			"vscode-lm",
			"mistral",
			"unbound",
			"requesty",
			"glama",
			"fake-ai",
		] as const

		providers.forEach((provider) => {
			test(`should return valid model info for ${provider}`, () => {
				const config: ApiConfiguration = {
					...baseConfig,
					apiProvider: provider,
				}

				const handler = buildApiHandler(config)
				const model = handler.getModel()

				assert.notStrictEqual(model, undefined)
				assert.notStrictEqual(model.id, undefined)
				assert.strictEqual(typeof model.id, "string")
				assert.notStrictEqual(model.info, undefined)
				assert.strictEqual(typeof model.info, "object")
			})
		})
	})

	suite("Provider token counting", () => {
		const providers = [
			"openai",
			"ollama",
			"lmstudio",
			"openai-native",
			"deepseek",
			"vscode-lm",
			"mistral",
			"unbound",
			"requesty",
			"glama",
			"fake-ai",
		] as const

		providers.forEach((provider) => {
			test(`should have working token counting for ${provider}`, async () => {
				const config: ApiConfiguration = {
					...baseConfig,
					apiProvider: provider,
				}

				const handler = buildApiHandler(config)

				// Test with simple text content
				const tokenCount = await handler.countTokens([{ type: "text", text: "Hello world" }])

				assert.strictEqual(typeof tokenCount, "number")
				// vscode-lm returns 0 in test environment, which is expected
				if (provider === "vscode-lm") {
					assert.ok(tokenCount >= 0, `Token count for ${provider} should be >= 0`)
				} else {
					assert.ok(tokenCount > 0, `Token count for ${provider} should be > 0`)
				}
			})
		})
	})
})
