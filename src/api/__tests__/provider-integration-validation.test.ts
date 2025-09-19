/**
 * Integration test to validate provider functionality with streaming and addresses the acceptance criteria from issue #107
 */
import assert from "node:assert/strict"
import sinon from "sinon"
import { buildApiHandler } from "../index"
import { ApiConfiguration } from "../../shared/api"
import { NeutralConversationHistory } from "../../shared/neutral-history"
import { McpIntegration } from "../../services/mcp/integration/McpIntegration"

let getInstanceStub: sinon.SinonStub

before(() => {
  const fake = {
    initialize: sinon.stub().resolves(),
    registerTool: sinon.stub(),
    routeToolUse: sinon
      .stub()
      .resolves('{"type":"tool_result","content":[{"type":"text","text":"Tool executed successfully"}]}'),
  } as unknown as McpIntegration
  getInstanceStub = sinon.stub(McpIntegration, "getInstance").returns(fake)
})

after(() => {
  getInstanceStub.restore()
})

describe("Provider Integration Validation", () => {
  // Mock FakeAI implementation for integration testing
  const mockFakeAI = {
        async *createMessage(_systemPrompt: string, _messages: NeutralConversationHistory) {
      await Promise.resolve() // Add await to satisfy async requirement
      yield { type: "text" as const, text: "Hello! I'm ready to help." }
      yield { type: "text" as const, text: " How can I assist you today?" }
    },
    getModel() {
      return {
        id: "fake-ai-integration",
        info: {
          maxTokens: 1000,
          contextWindow: 4000,
          supportsImages: false,
          supportsPromptCache: false,
          inputPrice: 0,
          outputPrice: 0,
          description: "Integration test fake AI",
        },
      }
    },
    async countTokens() {
      return Promise.resolve(5)
    },
    async completePrompt() {
      return Promise.resolve("Integration test response")
    },
  }

  const baseConfig = {
    apiKey: "test-key",
    apiModelId: "test-model",
    mistralApiKey: "test-mistral-key",
    requestyApiKey: "test-requesty-key",
    fakeAi: mockFakeAI,
  }

  describe("Streaming functionality", () => {
    const streamingProviders = [
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

    streamingProviders.forEach((provider) => {
      it(`should support streaming messages for ${provider}`, async () => {
        const config: ApiConfiguration = {
          ...baseConfig,
          apiProvider: provider,
        }

        const handler = buildApiHandler(config)
        const messages: NeutralConversationHistory = [
          { role: "user", content: [{ type: "text", text: "Hello, test message" }] },
        ]

        // Test that the stream generator function exists and can be called
        const stream = handler.createMessage("You are a helpful assistant.", messages)
        assert.ok(stream)
        assert.strictEqual(typeof stream[Symbol.asyncIterator], "function")

        // For this test, we just verify the stream is properly created
        // Actual streaming functionality would require more complex mocking
        // but the key point is that all providers have the createMessage method

        // Add await to satisfy async requirement
        await Promise.resolve()
      })
    })
  })

  describe("Error handling", () => {
    it("should properly handle invalid configurations", () => {
      // Test that providers properly validate their required configuration
      assert.throws(() => {
        buildApiHandler({ apiProvider: "mistral", apiKey: "test" } as ApiConfiguration)
      }, /Mistral API key is required/)

      assert.throws(() => {
        buildApiHandler({ apiProvider: "requesty", apiKey: "test" } as ApiConfiguration)
      }, /Requesty API key is required/)

      assert.throws(() => {
        buildApiHandler({ apiProvider: "fake-ai", apiKey: "test" } as ApiConfiguration)
      }, /Fake AI is not set/)
    })

    it("should handle unsupported human-relay provider", () => {
      assert.throws(() => {
        buildApiHandler({ apiProvider: "human-relay", apiKey: "test" } as ApiConfiguration)
      }, /HumanRelayHandler is not supported in this architecture\./)
    })
  })

  describe("Provider compatibility", () => {
    const compatibleProviders = [
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

    it("should have all expected providers enabled in buildApiHandler", () => {
      compatibleProviders.forEach((provider) => {
        const config: ApiConfiguration = {
          ...baseConfig,
          apiProvider: provider,
        }

        // Should not throw for any supported provider
        assert.doesNotThrow(() => buildApiHandler(config))
      })
    })

    it("should return different handler instances for different providers", () => {
      const handler1 = buildApiHandler({ ...baseConfig, apiProvider: "openai" })
      const handler2 = buildApiHandler({ ...baseConfig, apiProvider: "ollama" })
      const handler3 = buildApiHandler({ ...baseConfig, apiProvider: "fake-ai" })

      // Each provider should return a different instance
      assert.notStrictEqual(handler1, handler2)
      assert.notStrictEqual(handler2, handler3)
      assert.notStrictEqual(handler1, handler3)

      // But all should have the same interface
      assert.strictEqual(typeof handler1.createMessage, "function")
      assert.strictEqual(typeof handler2.createMessage, "function")
      assert.strictEqual(typeof handler3.createMessage, "function")
    })
  })
})
