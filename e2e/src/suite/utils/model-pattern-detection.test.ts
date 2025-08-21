import {
import * as assert from 'assert'
import * as sinon from 'sinon'
  isClaudeModel,
  isClaude37Model,
  isClaude35Model,
  isClaudeOpusModel,
  isClaudeHaikuModel,
  isClaude3SonnetModel,
  isThinkingModel,
  isDeepSeekR1Model,
  isO3MiniModel,
  setCapabilitiesFromModelId,
  getBaseModelId
} from "../model-pattern-detection"
import { ModelInfo } from "../../schemas"

suite("Model Pattern Detection", () => {
	let sandbox: sinon.SinonSandbox

	setup(() => {
		sandbox = sinon.createSandbox()
	})

	teardown(() => {
		sandbox.restore()
	})
  // Test model ID detection functions
  suite("Model ID detection functions", () => {
    suite("isClaudeModel", () => {
      test("should return true for Claude model IDs", () => {
        expect(isClaudeModel("claude-3-opus-20240229")).toBe(true)
        expect(isClaudeModel("anthropic/claude-3-sonnet")).toBe(true)
        expect(isClaudeModel("claude-3.5-sonnet")).toBe(true)
        expect(isClaudeModel("claude-3.7-haiku")).toBe(true)
      })

      test("should return false for non-Claude model IDs", () => {
        expect(isClaudeModel("gpt-4")).toBe(false)
        expect(isClaudeModel("gemini-pro")).toBe(false)
        expect(isClaudeModel("deepseek/deepseek-r1")).toBe(false)
        expect(isClaudeModel("o3-mini")).toBe(false)
      })
    })

    suite("isClaude37Model", () => {
      test("should return true for Claude 3.7 model IDs", () => {
        expect(isClaude37Model("claude-3.7-opus")).toBe(true)
        expect(isClaude37Model("anthropic/claude-3.7-sonnet")).toBe(true)
        expect(isClaude37Model("claude-3.7-haiku:thinking")).toBe(true)
      })

      test("should return false for non-Claude 3.7 model IDs", () => {
        expect(isClaude37Model("claude-3-opus")).toBe(false)
        expect(isClaude37Model("claude-3.5-sonnet")).toBe(false)
        expect(isClaude37Model("gpt-4")).toBe(false)
      })
    })

    suite("isClaude35Model", () => {
      test("should return true for Claude 3.5 model IDs", () => {
        expect(isClaude35Model("claude-3.5-opus")).toBe(true)
        expect(isClaude35Model("anthropic/claude-3.5-sonnet")).toBe(true)
        expect(isClaude35Model("claude-3.5-haiku:thinking")).toBe(true)
      })

      test("should return false for non-Claude 3.5 model IDs", () => {
        expect(isClaude35Model("claude-3-opus")).toBe(false)
        expect(isClaude35Model("claude-3.7-sonnet")).toBe(false)
        expect(isClaude35Model("gpt-4")).toBe(false)
      })
    })

    suite("isClaudeOpusModel", () => {
      test("should return true for Claude Opus model IDs", () => {
        expect(isClaudeOpusModel("claude-3-opus-20240229")).toBe(true)
        expect(isClaudeOpusModel("anthropic/claude-3.5-opus")).toBe(true)
        expect(isClaudeOpusModel("claude-3.7-opus:thinking")).toBe(true)
      })

      test("should return false for non-Opus model IDs", () => {
        expect(isClaudeOpusModel("claude-3-sonnet")).toBe(false)
        expect(isClaudeOpusModel("claude-3.5-haiku")).toBe(false)
        expect(isClaudeOpusModel("gpt-4")).toBe(false)
      })
    })

    suite("isClaudeHaikuModel", () => {
      test("should return true for Claude Haiku model IDs", () => {
        expect(isClaudeHaikuModel("claude-3-haiku")).toBe(true)
        expect(isClaudeHaikuModel("anthropic/claude-3.5-haiku")).toBe(true)
        expect(isClaudeHaikuModel("claude-3.7-haiku:thinking")).toBe(true)
      })

      test("should return false for non-Haiku model IDs", () => {
        expect(isClaudeHaikuModel("claude-3-sonnet")).toBe(false)
        expect(isClaudeHaikuModel("claude-3.5-opus")).toBe(false)
        expect(isClaudeHaikuModel("gpt-4")).toBe(false)
      })
    })

    suite("isClaude3SonnetModel", () => {
      test("should return true for Claude Sonnet model IDs", () => {
        expect(isClaude3SonnetModel("claude-3-sonnet-20240229")).toBe(true)
        expect(isClaude3SonnetModel("anthropic/claude-3.5-sonnet")).toBe(true)
        expect(isClaude3SonnetModel("claude-3.7-sonnet:thinking")).toBe(true)
      })

      test("should return false for non-Sonnet model IDs", () => {
        expect(isClaude3SonnetModel("claude-3-opus")).toBe(false)
        expect(isClaude3SonnetModel("claude-3.5-haiku")).toBe(false)
        expect(isClaude3SonnetModel("gpt-4")).toBe(false)
      })
    })

    suite("isThinkingModel", () => {
      test("should return true for thinking-enabled model IDs", () => {
        expect(isThinkingModel("claude-3-opus:thinking")).toBe(true)
        expect(isThinkingModel("anthropic/claude-3.5-sonnet:thinking")).toBe(true)
        expect(isThinkingModel("model-name-thinking")).toBe(true)
      })

      test("should return false for non-thinking model IDs", () => {
        expect(isThinkingModel("claude-3-opus")).toBe(false)
        expect(isThinkingModel("gpt-4")).toBe(false)
        expect(isThinkingModel("thinking-is-not-at-the-end")).toBe(false)
      })
    })

    suite("isDeepSeekR1Model", () => {
      test("should return true for DeepSeek R1 model IDs", () => {
        expect(isDeepSeekR1Model("deepseek/deepseek-r1")).toBe(true)
        expect(isDeepSeekR1Model("deepseek/deepseek-r1-v1.5")).toBe(true)
        expect(isDeepSeekR1Model("perplexity/sonar-reasoning")).toBe(true)
      })

      test("should return false for non-DeepSeek R1 model IDs", () => {
        expect(isDeepSeekR1Model("deepseek/deepseek-coder")).toBe(false)
        expect(isDeepSeekR1Model("perplexity/sonar")).toBe(false)
        expect(isDeepSeekR1Model("gpt-4")).toBe(false)
      })
    })

    suite("isO3MiniModel", () => {
      test("should return true for O3 Mini model IDs", () => {
        expect(isO3MiniModel("o3-mini")).toBe(true)
        expect(isO3MiniModel("openai/o3-mini")).toBe(true)
        expect(isO3MiniModel("o3-mini-v1")).toBe(true)
      })

      test("should return false for non-O3 Mini model IDs", () => {
        expect(isO3MiniModel("gpt-4")).toBe(false)
        expect(isO3MiniModel("claude-3-opus")).toBe(false)
        expect(isO3MiniModel("o3")).toBe(false)
      })
    })
  })

  // Test getBaseModelId function
  suite("getBaseModelId", () => {
    test("should remove thinking suffix", () => {
      expect(getBaseModelId("claude-3-opus:thinking")).toBe("claude-3-opus")
      expect(getBaseModelId("anthropic/claude-3.5-sonnet:thinking")).toBe("anthropic/claude-3.5-sonnet")
    })

    test("should return the original ID if no variant suffix is present", () => {
      expect(getBaseModelId("claude-3-opus")).toBe("claude-3-opus")
      expect(getBaseModelId("gpt-4")).toBe("gpt-4")
    })
  })

  // Test setCapabilitiesFromModelId function
  suite("setCapabilitiesFromModelId", () => {
    // Create a base model info object for testing
    const baseModelInfo: ModelInfo = {
      contextWindow: 16384,
      supportsPromptCache: false,
    }

    test("should set thinking capability for thinking models", () => {
      const result = setCapabilitiesFromModelId("claude-3-opus:thinking", baseModelInfo)
      assert.strictEqual(result.thinking, true)
    })

    test("should set prompt cache capability for Claude models", () => {
      const result = setCapabilitiesFromModelId("claude-3-opus", baseModelInfo)
      assert.strictEqual(result.supportsPromptCache, true)
    })

    test("should set cache pricing for Claude Opus models", () => {
      const result = setCapabilitiesFromModelId("claude-3-opus", baseModelInfo)
      assert.strictEqual(result.cacheWritesPrice, 18.75)
      assert.strictEqual(result.cacheReadsPrice, 1.5)
    })

    test("should set cache pricing for Claude Haiku models", () => {
      const result = setCapabilitiesFromModelId("claude-3-haiku", baseModelInfo)
      assert.strictEqual(result.cacheWritesPrice, 1.25)
      assert.strictEqual(result.cacheReadsPrice, 0.1)
    })

    test("should set default cache pricing for other Claude models", () => {
      const result = setCapabilitiesFromModelId("claude-3-sonnet", baseModelInfo)
      assert.strictEqual(result.cacheWritesPrice, 3.75)
      assert.strictEqual(result.cacheReadsPrice, 0.3)
    })

    test("should set computer use capability for Claude Sonnet models", () => {
      const result = setCapabilitiesFromModelId("claude-3-sonnet", baseModelInfo)
      assert.strictEqual(result.supportsComputerUse, true)
    })

    test("should not set computer use capability for older Claude Sonnet models", () => {
      const result = setCapabilitiesFromModelId("claude-3-sonnet-20240620", baseModelInfo)
      assert.strictEqual(result.supportsComputerUse, undefined)
    })

    test("should set max tokens for Claude 3.7 models based on thinking capability", () => {
      const thinkingResult = setCapabilitiesFromModelId("claude-3.7-opus:thinking", baseModelInfo)
      assert.strictEqual(thinkingResult.maxTokens, 64_000)

      const nonThinkingResult = setCapabilitiesFromModelId("claude-3.7-opus", baseModelInfo)
      assert.strictEqual(nonThinkingResult.maxTokens, 8192)
    })

    test("should set max tokens for Claude 3.5 models", () => {
      const result = setCapabilitiesFromModelId("claude-3.5-sonnet", baseModelInfo)
      assert.strictEqual(result.maxTokens, 8192)
    })

    test("should set temperature support to false for O3 Mini models", () => {
      const result = setCapabilitiesFromModelId("o3-mini", baseModelInfo)
      assert.strictEqual(result.supportsTemperature, false)
    })

    test("should set reasoning effort to high for DeepSeek R1 models", () => {
      const result = setCapabilitiesFromModelId("deepseek/deepseek-r1", baseModelInfo)
      assert.strictEqual(result.reasoningEffort, "high")
    })

    test("should not modify the original model info object", () => {
      setCapabilitiesFromModelId("claude-3-opus", baseModelInfo)
      assert.strictEqual(baseModelInfo.supportsPromptCache, false)
      assert.strictEqual(baseModelInfo.cacheWritesPrice, undefined)
    })
  })
})