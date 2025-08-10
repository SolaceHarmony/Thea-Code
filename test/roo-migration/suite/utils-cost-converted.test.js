const assert = require('assert')
const { loadTheaModule } = require('../helpers/thea-loader')

describe('Roo-migration: utils/cost', function () {
  const { calculateApiCostAnthropic, calculateApiCostOpenAI } = loadTheaModule('src/utils/cost.ts')
  const mockModelInfo = {
    maxTokens: 8192,
    contextWindow: 200_000,
    supportsPromptCache: true,
    inputPrice: 3.0,
    outputPrice: 15.0,
    cacheWritesPrice: 3.75,
    cacheReadsPrice: 0.3,
  }

  describe('calculateApiCostAnthropic', function () {
    it('basic input/output costs', function () {
      const cost = calculateApiCostAnthropic(mockModelInfo, 1000, 500)
      assert.strictEqual(cost, 0.0105)
    })
    it('handles cache writes', function () {
      const cost = calculateApiCostAnthropic(mockModelInfo, 1000, 500, 2000)
      assert.ok(Math.abs(cost - 0.018) < 1e-6)
    })
    it('handles cache reads', function () {
      const cost = calculateApiCostAnthropic(mockModelInfo, 1000, 500, undefined, 3000)
      assert.strictEqual(cost, 0.0114)
    })
    it('handles all components together', function () {
      const cost = calculateApiCostAnthropic(mockModelInfo, 1000, 500, 2000, 3000)
      assert.strictEqual(cost, 0.0189)
    })
    it('missing prices → 0', function () {
      const model = { maxTokens: 8192, contextWindow: 200_000, supportsPromptCache: true }
      const cost = calculateApiCostAnthropic(model, 1000, 500, 2000, 3000)
      assert.strictEqual(cost, 0)
    })
    it('zero tokens → 0', function () {
      const cost = calculateApiCostAnthropic(mockModelInfo, 0, 0, 0, 0)
      assert.strictEqual(cost, 0)
    })
  })

  describe('calculateApiCostOpenAI', function () {
    it('basic input/output costs', function () {
      const cost = calculateApiCostOpenAI(mockModelInfo, 1000, 500)
      assert.strictEqual(cost, 0.0105)
    })
    it('handles cache writes', function () {
      const cost = calculateApiCostOpenAI(mockModelInfo, 3000, 500, 2000)
      assert.ok(Math.abs(cost - 0.018) < 1e-6)
    })
    it('handles cache reads', function () {
      const cost = calculateApiCostOpenAI(mockModelInfo, 4000, 500, undefined, 3000)
      assert.strictEqual(cost, 0.0114)
    })
    it('handles all components together', function () {
      const cost = calculateApiCostOpenAI(mockModelInfo, 6000, 500, 2000, 3000)
      assert.strictEqual(cost, 0.0189)
    })
    it('missing prices → 0', function () {
      const model = { maxTokens: 8192, contextWindow: 200_000, supportsPromptCache: true }
      const cost = calculateApiCostOpenAI(model, 1000, 500, 2000, 3000)
      assert.strictEqual(cost, 0)
    })
    it('zero tokens → 0', function () {
      const cost = calculateApiCostOpenAI(mockModelInfo, 0, 0, 0, 0)
      assert.strictEqual(cost, 0)
    })
  })
})
