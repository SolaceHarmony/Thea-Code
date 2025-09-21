import * as assert from 'assert'

import { ModelRegistry } from '../model-registry'
import startProviderMocks from '../../../../test/generic-provider-mock/setup'
import stopProviderMocks from '../../../../test/generic-provider-mock/teardown'

// Test configuration with mock API credentials
const testConfigs = {
  anthropic: { apiKey: 'test-anthropic-key' },
  openai: { apiKey: 'test-openai-key' },
  bedrock: {
    region: 'us-east-1',
    accessKeyId: 'test-access-key',
    secretAccessKey: 'test-secret-key',
  },
  gemini: { apiKey: 'test-gemini-key' },
  vertex: {
    projectId: 'test-project',
    region: 'us-central1',
    keyFilename: '/fake/path/to/key.json',
  },
  mistral: { apiKey: 'test-mistral-key' },
  deepseek: { apiKey: 'test-deepseek-key' },
  ollama: { baseUrl: process.env.OLLAMA_BASE_URL || 'http://localhost:11434' },
  openrouter: { apiKey: 'test-openrouter-key' },
  lmstudio: { baseUrl: 'http://localhost:1234' },
  msty: { apiKey: 'test-msty-key' },
  together: { apiKey: 'test-together-key' },
  groq: { apiKey: 'test-groq-key' },
  xai: { apiKey: 'test-xai-key' },
} as const

describe('All Dynamic Providers Runtime Test', () => {
  let registry: ModelRegistry

  before(async () => {
    await startProviderMocks()
    // Allow the mock servers a moment to finish booting
    await new Promise((resolve) => setTimeout(resolve, 500))
    registry = ModelRegistry.getInstance()
  })

  after(async () => {
    await stopProviderMocks()
    ;(ModelRegistry as unknown as { instance?: ModelRegistry }).instance = undefined
  })

  describe('Dynamic Providers', () => {
    const dynamicProviders: Array<keyof typeof testConfigs> = [
      'anthropic',
      'openai',
      'bedrock',
      'gemini',
      'vertex',
      'mistral',
      'deepseek',
    ]

    dynamicProviders.forEach((providerName) => {
      it(`${providerName} should fetch models successfully`, async function () {
        this.timeout(10000)
        const config = testConfigs[providerName]
        assert.ok(config)

        const models = await registry.getModels(providerName, config)
        assert.ok(Array.isArray(models))
        assert.ok(models.length > 0)

        for (const model of models) {
          assert.ok(model.id)
          assert.ok(model.name)
          assert.ok(Array.isArray(model.capabilities))
        }
      })
    })
  })

  describe('Static Providers', () => {
    const staticProviders: Array<keyof typeof testConfigs> = [
      'ollama',
      'openrouter',
      'lmstudio',
      'msty',
      'together',
      'groq',
      'xai',
    ]

    staticProviders.forEach((providerName) => {
      it(`${providerName} should return static models`, async () => {
        const config = testConfigs[providerName]
        assert.ok(config)

        const models = await registry.getModels(providerName, config)
        assert.ok(Array.isArray(models))
      })
    })
  })

  describe('Cache Functionality', () => {
    it('caches models and serves from cache on subsequent request', async () => {
      const providerName: keyof typeof testConfigs = 'anthropic'
      const config = testConfigs[providerName]

      const firstRequest = await registry.getModels(providerName, config)
      assert.ok(firstRequest.length > 0)

      const secondRequest = await registry.getModels(providerName, config)
      assert.ok(secondRequest.length > 0)
    })
  })
})
