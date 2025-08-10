const assert = require('assert')
const { loadTheaModule } = require('../helpers/thea-loader')

describe('Roo-migration: utils/model-capabilities', function () {
  const mod = loadTheaModule('src/utils/model-capabilities.ts')
  const fullFeaturedModel = {
    maxTokens: 8192,
    contextWindow: 200_000,
    supportsPromptCache: true,
    supportsComputerUse: true,
    supportsImages: true,
    supportsTemperature: true,
    thinking: true,
    reasoningEffort: 'high',
    inputPrice: 3.0,
    outputPrice: 15.0,
  }
  const limitedModel = {
    maxTokens: 4096,
    contextWindow: 8192,
    supportsPromptCache: false,
    supportsComputerUse: false,
    supportsImages: false,
    supportsTemperature: false,
    thinking: false,
    reasoningEffort: 'low',
    inputPrice: 1.0,
    outputPrice: 5.0,
  }
  const partiallyDefinedModel = { contextWindow: 16384, supportsPromptCache: true }

  it('supportsComputerUse', function () {
    assert.strictEqual(mod.supportsComputerUse(fullFeaturedModel), true)
    assert.strictEqual(mod.supportsComputerUse(limitedModel), false)
    assert.strictEqual(mod.supportsComputerUse(partiallyDefinedModel), false)
  })

  it('supportsPromptCaching', function () {
    assert.strictEqual(mod.supportsPromptCaching(fullFeaturedModel), true)
    assert.strictEqual(mod.supportsPromptCaching(limitedModel), false)
    assert.strictEqual(mod.supportsPromptCaching({ contextWindow: 16384 }), false)
  })

  it('supportsImages', function () {
    assert.strictEqual(mod.supportsImages(fullFeaturedModel), true)
    assert.strictEqual(mod.supportsImages(limitedModel), false)
    assert.strictEqual(mod.supportsImages(partiallyDefinedModel), false)
  })

  it('supportsThinking', function () {
    assert.strictEqual(mod.supportsThinking(fullFeaturedModel), true)
    assert.strictEqual(mod.supportsThinking(limitedModel), false)
    assert.strictEqual(mod.supportsThinking(partiallyDefinedModel), false)
  })

  it('supportsTemperature', function () {
    assert.strictEqual(mod.supportsTemperature(fullFeaturedModel), true)
    assert.strictEqual(mod.supportsTemperature(limitedModel), false)
    assert.strictEqual(mod.supportsTemperature(partiallyDefinedModel), true)
  })

  it('getMaxTokens', function () {
    assert.strictEqual(mod.getMaxTokens(fullFeaturedModel), 8192)
    assert.strictEqual(mod.getMaxTokens(partiallyDefinedModel), 4096)
    assert.strictEqual(mod.getMaxTokens(partiallyDefinedModel, 2048), 2048)
  })

  it('getReasoningEffort', function () {
    assert.strictEqual(mod.getReasoningEffort(fullFeaturedModel), 'high')
    assert.strictEqual(mod.getReasoningEffort(partiallyDefinedModel), undefined)
  })

  it('hasCapability', function () {
    assert.strictEqual(mod.hasCapability(fullFeaturedModel, 'computerUse'), true)
    assert.strictEqual(mod.hasCapability(limitedModel, 'computerUse'), false)
    assert.strictEqual(mod.hasCapability(fullFeaturedModel, 'promptCache'), true)
    assert.strictEqual(mod.hasCapability(limitedModel, 'promptCache'), false)
    assert.strictEqual(mod.hasCapability(fullFeaturedModel, 'images'), true)
    assert.strictEqual(mod.hasCapability(limitedModel, 'images'), false)
    assert.strictEqual(mod.hasCapability(fullFeaturedModel, 'thinking'), true)
    assert.strictEqual(mod.hasCapability(limitedModel, 'thinking'), false)
    assert.strictEqual(mod.hasCapability(fullFeaturedModel, 'temperature'), true)
    assert.strictEqual(mod.hasCapability(limitedModel, 'temperature'), false)
  })

  it('getContextWindowSize', function () {
    assert.strictEqual(mod.getContextWindowSize(fullFeaturedModel), 200_000)
    assert.strictEqual(mod.getContextWindowSize({ supportsPromptCache: true }), 8192)
    assert.strictEqual(mod.getContextWindowSize({ supportsPromptCache: true }, 4096), 4096)
  })
})
