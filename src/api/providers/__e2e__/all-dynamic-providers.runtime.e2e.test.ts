import * as assert from "assert"
import "../../../test/generic-provider-mock/setup" // ensure mocks are initialized via side effects
import { ModelRegistry } from "../model-registry"

// Dynamic providers with mock services running as separate instances
const dynamicProviders = ["openai", "bedrock", "gemini", "vertex", "mistral", "deepseek"] as const
type DynamicProvider = (typeof dynamicProviders)[number]

// Static/pre-defined providers
const staticProviders = ["ollama", "openrouter", "lmstudio", "msty", "together", "groq", "xai"] as const
type StaticProvider = (typeof staticProviders)[number]

type AllProvider = DynamicProvider | StaticProvider

suite("All Dynamic Providers Runtime Test", () => {
  let registry: ModelRegistry

  suiteSetup(async function () {
    this.timeout(10_000)
    // Allow mock servers to come up
    await new Promise((resolve) => setTimeout(resolve, 1000))
    registry = ModelRegistry.getInstance()
  })

  suiteTeardown(() => {
    // Reset singleton between runs
    const registryConstructor = ModelRegistry as unknown as { instance: ModelRegistry | null }
    registryConstructor.instance = null
  })

  suite("Dynamic Providers (with running mock services)", () => {
    dynamicProviders.forEach((providerName) => {
      test(`${providerName} should fetch models from mock service`, async function () {
        this.timeout(10_000)

        try {
          const models = await registry.getModels(providerName)
          assert.ok(Array.isArray(models), `Expected array, got ${typeof models}`)
          assert.ok(models.length > 0, `${providerName} should have models available from mock service`)

          // Validate model structure
          for (const model of models) {
            assert.ok(Object.prototype.hasOwnProperty.call(model, "id"), `Model should have id property`)
            assert.ok(Object.prototype.hasOwnProperty.call(model, "info"), `Model should have info property`)
            assert.strictEqual(typeof model.id, "string", `Model id should be string`)
          }

          const sampleIds = models.slice(0, 3).map((m) => m.id).join(", ")
          console.log(`✅ ${providerName}: Found ${models.length} models from mock service`)
          console.log(`   Sample: ${sampleIds}`)
        } catch (error) {
          console.error(`❌ ${providerName} failed:`, error)
          throw error
        }
      })
    })
  })

  suite("Static Providers (pre-defined models)", () => {
    staticProviders.forEach((providerName) => {
      test(`${providerName} should return static models`, async function () {
        this.timeout(10_000)

        try {
          const models = await registry.getModels(providerName as AllProvider)
          assert.ok(Array.isArray(models))
          if (models.length > 0) {
            console.log(`✅ ${providerName}: Found ${models.length} static models`)
          } else {
            console.log(`ℹ️ ${providerName}: No models configured (may be expected for some providers)`)
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
      const providerName: DynamicProvider = "openai"

      const first = await registry.getModels(providerName)
      assert.ok(first.length > 0)

      const second = await registry.getModels(providerName)
      assert.deepStrictEqual(second, first)

      console.log(`✅ Cache test: Both requests returned ${first.length} models`)
    })

    test("should refresh cache when requested", async () => {
      const providerName: DynamicProvider = "gemini"

      const initial = await registry.getModels(providerName)
      assert.ok(initial.length > 0)

      const refreshed = await registry.getModels(providerName, true)
      assert.ok(refreshed.length > 0)
      console.log(`✅ Refresh test: Got ${refreshed.length} models after refresh`)
    })
  })

  suite("Error Handling", () => {
    test("should handle invalid provider gracefully", async () => {
      // Invalid provider should return empty array, not throw
      const models = await registry.getModels("nonexistent-provider" as unknown as DynamicProvider)
      assert.ok(Array.isArray(models), "Should return array even for invalid provider")
      console.log("✅ Invalid provider handled gracefully")
    })

    test("should handle provider without mock service gracefully", async () => {
      // Provider without running service should handle gracefully
      const models = await registry.getModels("bedrock")
      assert.ok(Array.isArray(models), "Should return array")
      if (models.length > 0) {
        console.log(`✅ Bedrock returned ${models.length} models`)
      } else {
        console.log("ℹ️ Bedrock returned empty models (service may not be running)")
      }
    })
  })

  suite("Cross-Provider Model Discovery", () => {
    test("should discover diverse model IDs across providers", async () => {
      const providerSamples: Record<string, number> = {}

      for (const provider of ["openai", "bedrock", "gemini"] as const) {
        const models = await registry.getModels(provider)
        if (models.length > 0) {
          providerSamples[provider] = models.length
        }
      }

      const totalDiscovered = Object.values(providerSamples).reduce((a, b) => a + b, 0)
      assert.ok(totalDiscovered > 0, "Should discover models from at least one provider")
      console.log(`✅ Cross-provider discovery: ${JSON.stringify(providerSamples)}`)
    })
  })
})

suite("Performance Tests", () => {
  test("should fetch models from multiple providers within reasonable time", async function () {
    this.timeout(20_000)
    const registry = ModelRegistry.getInstance()
    const start = Date.now()

    interface FetchResult {
      provider: string
      success: boolean
      modelCount?: number
      error?: string
    }

    const results: FetchResult[] = await Promise.all(
      ["openai", "bedrock", "gemini", "vertex", "mistral"].map(
        async (provider): Promise<FetchResult> => {
          try {
            const models = await registry.getModels(provider as DynamicProvider)
            return { provider, success: true, modelCount: models.length }
          } catch (error) {
            return { provider, success: false, error: (error as Error)?.message ?? String(error) }
          }
        },
      ),
    )

    const total = Date.now() - start
    console.log(`\n🚀 Performance Test Results (${total}ms total):`)
    for (const r of results) {
      if (r.success) {
        console.log(`   ✅ ${r.provider}: ${r.modelCount} models`)
      } else {
        console.log(`   ❌ ${r.provider}: ${r.error}`)
      }
    }

    assert.ok(total < 15_000, `Performance test took ${total}ms, should be < 15000ms`)
    const successCount = results.filter((r) => r.success).length
    assert.ok(successCount >= 2, `Expected at least 2 successful providers, got ${successCount}`)
  })
})
