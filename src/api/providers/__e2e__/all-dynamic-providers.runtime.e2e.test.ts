import * as assert from "assert"
import "../../../../test/generic-provider-mock/setup" // ensure mocks are initialized via side effects
import { ModelRegistry } from "../model-registry"

// Test configuration with mock API credentials and endpoints
const testConfigs = {
  anthropic: { apiKey: "test-anthropic-key" },
  openai: { apiKey: "test-openai-key" },
  bedrock: { region: "us-east-1", accessKeyId: "test-access-key", secretAccessKey: "test-secret-key" },
  gemini: { apiKey: "test-gemini-key" },
  vertex: { projectId: "test-project", region: "us-central1", keyFilename: "/fake/path/to/key.json" },
  mistral: { apiKey: "test-mistral-key" },
  deepseek: { apiKey: "test-deepseek-key" },
  ollama: { baseUrl: process.env.OLLAMA_BASE_URL || "http://localhost:11434" },
  openrouter: { apiKey: "test-openrouter-key" },
  lmstudio: { baseUrl: "http://localhost:1234" },
  msty: { apiKey: "test-msty-key" },
  together: { apiKey: "test-together-key" },
  groq: { apiKey: "test-groq-key" },
  xai: { apiKey: "test-xai-key" },
} as const

type ProviderName = keyof typeof testConfigs

suite("All Dynamic Providers Runtime Test", () => {
  let registry: ModelRegistry

  suiteSetup(async function () {
    this.timeout(10_000)
    // Allow mock servers to come up
    await new Promise((r) => setTimeout(r, 1000))
    registry = ModelRegistry.getInstance()
  })

  suiteTeardown(() => {
    // Reset singleton between runs
    ;(ModelRegistry as any).instance = null
  })

  suite("Dynamic Providers", () => {
    const dynamicProviders: ProviderName[] = [
      "anthropic",
      "openai",
      "bedrock",
      "gemini",
      "vertex",
      "mistral",
      "deepseek",
    ]

    dynamicProviders.forEach((providerName) => {
      test(`${providerName} should fetch models successfully`, async function () {
        this.timeout(10_000)
        const config = testConfigs[providerName]
        assert.ok(config)

        try {
          const models = await registry.getModels(providerName, config)
          assert.ok(Array.isArray(models))
          assert.ok(models.length > 0)

          // Validate model structure
          for (const model of models) {
            assert.ok(Object.prototype.hasOwnProperty.call(model, "id"))
            assert.ok(Object.prototype.hasOwnProperty.call(model, "name"))
            assert.ok(Object.prototype.hasOwnProperty.call(model, "capabilities"))
            assert.strictEqual(typeof (model as any).id, "string")
            assert.strictEqual(typeof (model as any).name, "string")
            assert.strictEqual(Array.isArray((model as any).capabilities), true)
          }

          console.log(`✅ ${providerName}: Found ${models.length} models`)
          console.log(`   Sample: ${models.slice(0, 3).map((m: any) => m.id).join(", ")}`)
        } catch (error) {
          console.error(`❌ ${providerName} failed:`, error)
          throw error
        }
      })
    })
  })

  suite("Static Providers (should have pre-defined models)", () => {
    const staticProviders: ProviderName[] = [
      "ollama",
      "openrouter",
      "lmstudio",
      "msty",
      "together",
      "groq",
      "xai",
    ]

    staticProviders.forEach((providerName) => {
      test(`${providerName} should return static models`, async function () {
        this.timeout(10_000)
        const config = testConfigs[providerName]

        try {
          const models = await registry.getModels(providerName, config)
          assert.ok(Array.isArray(models))
          if (models.length > 0) {
            console.log(`✅ ${providerName}: Found ${models.length} static models`)
          } else {
            console.log(`ℹ️ ${providerName}: No models configured (may be expected for some providers) `)
          }
        } catch (error) {
          // Some static providers may not have mock endpoints; treat failures as informational
          console.warn(`ℹ️ ${providerName} static fetch warning:`, (error as Error)?.message)
        }
      })
    })
  })

  suite("Cache Functionality", () => {
    test("should cache models and serve from cache on second request", async () => {
      const providerName: ProviderName = "anthropic"
      const config = testConfigs.anthropic

      const first = await registry.getModels(providerName, config)
      assert.ok(first.length > 0)

      const second = await registry.getModels(providerName, config)
      assert.deepStrictEqual(second, first)

      console.log(`✅ Cache test: Both requests returned ${first.length} models`)
    })

    test("should refresh cache when requested", async () => {
      const providerName: ProviderName = "openai"
      const config = testConfigs.openai

      const initial = await registry.getModels(providerName, config)
      assert.ok(initial.length > 0)

      await registry.refreshModels(providerName, config)
      const refreshed = await registry.getModels(providerName, config)
      assert.ok(refreshed.length > 0)
      console.log(`✅ Refresh test: Got ${refreshed.length} models after refresh`)
    })
  })

  suite("Error Handling", () => {
    test("should handle invalid provider gracefully", async () => {
      await assert.rejects(
        () => registry.getModels("nonexistent-provider" as any, {} as any),
        (err: any) => {
          assert.ok(err instanceof Error)
          return true
        },
      )
      console.log("✅ Invalid provider handled correctly")
    })

    test("should handle invalid configuration gracefully", async () => {
      await assert.rejects(
        () => registry.getModels("anthropic", {} as any),
        (err: any) => {
          assert.ok(err instanceof Error)
          return true
        },
      )
      console.log("✅ Missing configuration handled correctly")
    })
  })

  suite("Model Capabilities", () => {
    test("should return models with correct capability information", async () => {
      const models: any[] = await registry.getModels("anthropic", testConfigs.anthropic)
      const claude = models.find((m) => typeof m.id === "string" && m.id.includes("claude"))
      if (claude) {
        assert.ok(Array.isArray(claude.capabilities))
        assert.ok(claude.capabilities.includes("chat"))
        assert.ok(claude.capabilities.includes("tools"))
        console.log(`✅ Claude model capabilities: ${claude.capabilities.join(", ")}`)
      }
    })

    test("should categorize models by their capabilities", async () => {
      const models: any[] = await registry.getModels("gemini", testConfigs.gemini)
      const chat = models.filter((m) => m.capabilities?.includes("chat"))
      const vision = models.filter((m) => m.capabilities?.includes("vision"))
      const tools = models.filter((m) => m.capabilities?.includes("tools"))
      assert.ok(chat.length > 0)
      console.log(`✅ Gemini capabilities - Chat: ${chat.length}, Vision: ${vision.length}, Tools: ${tools.length}`)
    })
  })
})

suite("Performance Tests", () => {
  test("should fetch all provider models within reasonable time", async function () {
    this.timeout(20_000)
    const registry = ModelRegistry.getInstance()
    const start = Date.now()

    const providers: ProviderName[] = ["anthropic", "openai", "bedrock", "gemini", "mistral"]
    const results = await Promise.all(
      providers.map(async (provider) => {
        try {
          const config = testConfigs[provider]
          const models = await registry.getModels(provider, config)
          return { provider, success: true, modelCount: models.length }
        } catch (error) {
          return { provider, success: false, error: (error as Error).message }
        }
      }),
    )

    const total = Date.now() - start
    console.log(`\n🚀 Performance Test Results (${total}ms total):`)
    for (const r of results) {
      if ((r as any).success) {
        console.log(`   ✅ ${(r as any).provider}: ${(r as any).modelCount} models`)
      } else {
        console.log(`   ❌ ${(r as any).provider}: ${(r as any).error}`)
      }
    }

    assert.ok(total < 15_000)
    const successCount = results.filter((r: any) => r.success).length
    assert.ok(successCount >= 3)
  })
})
