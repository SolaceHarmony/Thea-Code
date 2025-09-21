import * as assert from 'assert'
import * as sinon from 'sinon'

import {
  ModelRegistry,
  StaticModelProvider,
  migrateStaticModels,
} from '../model-registry'
import type { ModelProvider, ModelListing } from '../model-registry'
import type { ModelInfo } from '../../../schemas'

function createMockProvider(models: ModelListing[] = []): ModelProvider & {
  listModels: sinon.SinonStub
  getModelInfo: sinon.SinonStub
  getDefaultModelId: sinon.SinonStub
} {
  return {
    listModels: sinon.stub().resolves(models),
    getModelInfo: sinon.stub().resolves(models[0]?.info ?? null),
    getDefaultModelId: sinon.stub().returns(models[0]?.id ?? ''),
    configure: sinon.stub(),
  }
}

describe('ModelRegistry', () => {
  let registry: ModelRegistry
  let mockProvider: ReturnType<typeof createMockProvider>

  beforeEach(() => {
    ModelRegistry.reset()
    registry = ModelRegistry.getInstance()
    mockProvider = createMockProvider()
  })

  afterEach(() => {
    sinon.restore()
    ModelRegistry.reset()
  })

  describe('Singleton pattern', () => {
    it('returns the same instance', () => {
      const first = ModelRegistry.getInstance()
      const second = ModelRegistry.getInstance()
      assert.strictEqual(first, second)
    })

    it('creates a new instance after reset', () => {
      const first = ModelRegistry.getInstance()
      ModelRegistry.reset()
      const second = ModelRegistry.getInstance()
      assert.notStrictEqual(first, second)
    })
  })

  describe('Provider registration', () => {
    it('registers and checks provider presence', () => {
      registry.registerProvider('test', mockProvider)
      assert.strictEqual(registry.hasProvider('test'), true)
    })

    it('unregisters a provider', () => {
      registry.registerProvider('test', mockProvider)
      registry.unregisterProvider('test')
      assert.strictEqual(registry.hasProvider('test'), false)
    })

    it('lists registered providers', () => {
      registry.registerProvider('provider1', mockProvider)
      registry.registerProvider('provider2', mockProvider)
      const providers = registry.getProviderNames()
      assert.ok(providers.includes('provider1'))
      assert.ok(providers.includes('provider2'))
    })
  })

  describe('Model listing', () => {
    const listings: ModelListing[] = [
      {
        id: 'model-1',
        info: {
          maxTokens: 4096,
          contextWindow: 8192,
          supportsImages: true,
          inputPrice: 1,
          outputPrice: 2,
        } as ModelInfo,
        displayName: 'Model 1',
      },
      {
        id: 'model-2',
        info: {
          maxTokens: 8192,
          contextWindow: 16384,
          supportsImages: false,
          inputPrice: 2,
          outputPrice: 4,
        } as ModelInfo,
        displayName: 'Model 2',
      },
    ]

    beforeEach(() => {
      mockProvider = createMockProvider(listings)
      registry.registerProvider('test', mockProvider)
    })

    it('fetches models from provider', async () => {
      const models = await registry.getModels('test')
      assert.deepStrictEqual(models, listings)
      assert.strictEqual(mockProvider.listModels.callCount, 1)
    })

    it('caches models until invalidated', async () => {
      await registry.getModels('test')
      await registry.getModels('test')
      assert.strictEqual(mockProvider.listModels.callCount, 1)
    })

    it('forces refresh when requested', async () => {
      await registry.getModels('test')
      await registry.getModels('test', true)
      assert.strictEqual(mockProvider.listModels.callCount, 2)
    })

    it('returns empty array for unknown provider', async () => {
      const models = await registry.getModels('missing')
      assert.deepStrictEqual(models, [])
    })

    it('falls back to cached data on provider error', async () => {
      await registry.getModels('test')
      mockProvider.listModels.rejects(new Error('API error'))
      const models = await registry.getModels('test', true)
      assert.deepStrictEqual(models, listings)
    })
  })

  describe('Model info lookup', () => {
    const listings: ModelListing[] = [
      {
        id: 'model-1',
        info: { maxTokens: 4096, contextWindow: 8192 } as ModelInfo,
      },
    ]

    beforeEach(() => {
      mockProvider = createMockProvider(listings)
      registry.registerProvider('test', mockProvider)
    })

    it('returns cached model info', async () => {
      const info = await registry.getModelInfo('test', 'model-1')
      assert.ok(info)
      assert.strictEqual(info?.maxTokens, 4096)
    })
  })

  describe('Default model handling', () => {
    const listings: ModelListing[] = [
      {
        id: 'model-1',
        info: { maxTokens: 4096, contextWindow: 8192 } as ModelInfo,
      },
    ]

    beforeEach(() => {
      mockProvider = createMockProvider(listings)
      mockProvider.getDefaultModelId.returns('default-model')
      registry.registerProvider('test', mockProvider)
    })

    it('returns provider default', async () => {
      const id = await registry.getDefaultModelId('test')
      assert.strictEqual(id, 'default-model')
    })

    it('falls back to first model on error', async () => {
      mockProvider.getDefaultModelId.throws(new Error('error'))
      const id = await registry.getDefaultModelId('test')
      assert.strictEqual(id, 'model-1')
    })

    it('returns empty string for unknown provider', async () => {
      const id = await registry.getDefaultModelId('missing')
      assert.strictEqual(id, '')
    })
  })

  describe('Cache management', () => {
    beforeEach(() => {
      mockProvider = createMockProvider([])
      registry.registerProvider('test1', mockProvider)
      registry.registerProvider('test2', mockProvider)
    })

    it('clears cache for specific provider', async () => {
      await registry.getModels('test1')
      await registry.getModels('test2')
      registry.clearCache('test1')
      await registry.getModels('test1')
      await registry.getModels('test2')
      assert.strictEqual(mockProvider.listModels.callCount, 3)
    })

    it('clears all caches', async () => {
      await registry.getModels('test1')
      await registry.getModels('test2')
      registry.clearCache()
      await registry.getModels('test1')
      await registry.getModels('test2')
      assert.strictEqual(mockProvider.listModels.callCount, 4)
    })

    it('respects custom TTL', async () => {
      await registry.getModels('test1')
      registry.setCacheTTL('test1', 1)
      await new Promise((resolve) => setTimeout(resolve, 5))
      await registry.getModels('test1')
      assert.strictEqual(mockProvider.listModels.callCount, 2)
    })
  })

  describe('StaticModelProvider', () => {
    it('wraps static models', async () => {
      const staticModels = {
        'static-1': { maxTokens: 1000, contextWindow: 2000 } as ModelInfo,
        'static-2': { maxTokens: 2000, contextWindow: 4000 } as ModelInfo,
      }
      const provider = new StaticModelProvider(staticModels, 'static-1')
      const models = await provider.listModels()
      assert.strictEqual(models.length, 2)
      const info = await provider.getModelInfo('static-2')
      assert.strictEqual(info?.maxTokens, 2000)
      const defaultId = provider.getDefaultModelId()
      assert.strictEqual(defaultId, 'static-1')
    })
  })

  describe('migrateStaticModels', () => {
    it('converts static models to listings', () => {
      const staticModels = {
        'model-a': { maxTokens: 1000, contextWindow: 2000 } as ModelInfo,
        'model-b': { maxTokens: 2000, contextWindow: 4000 } as ModelInfo,
      }
      const listings = migrateStaticModels(staticModels)
      assert.strictEqual(listings.length, 2)
      assert.deepStrictEqual(listings[0], {
        id: 'model-a',
        info: staticModels['model-a'],
        displayName: 'model-a',
        deprecated: false,
        releaseDate: undefined,
      })
    })
  })
})
