const assert = require('assert')
const { loadTheaModule } = require('../helpers/thea-loader')

describe('Roo-migration: utils/model-pattern-detection', function () {
  const mod = loadTheaModule('src/utils/model-pattern-detection.ts')

  it('detects model families and variants', function () {
    assert.strictEqual(mod.isClaudeModel('anthropic/claude-3.5-sonnet'), true)
    assert.strictEqual(mod.isClaude35Model('anthropic/claude-3.5-sonnet'), true)
    assert.strictEqual(mod.isClaude37Model('anthropic/claude-3.7-sonnet:thinking'), true)
    assert.strictEqual(mod.isClaude3SonnetModel('anthropic/claude-3.7-sonnet'), true)
    assert.strictEqual(mod.isClaudeOpusModel('anthropic/opus-vast'), true)
    assert.strictEqual(mod.isClaudeHaikuModel('anthropic/haiku-lite'), true)
    assert.strictEqual(mod.isThinkingModel('anthropic/claude-3.7-sonnet:thinking'), true)
    assert.strictEqual(mod.isDeepSeekR1Model('deepseek/deepseek-r1'), true)
    assert.strictEqual(mod.isO3MiniModel('openai/o3-mini'), true)
  })

  it('setCapabilitiesFromModelId mutates a copy as expected', function () {
    const base = { maxTokens: 0 }
    const updated = mod.setCapabilitiesFromModelId('anthropic/claude-3.7-sonnet:thinking', base)
    // Should not mutate original
    assert.strictEqual(base.maxTokens, 0)
    // Should set thinking, computerUse, cache prices, and max tokens per rules
    assert.strictEqual(updated.thinking, true)
    assert.strictEqual(updated.supportsComputerUse, true)
    assert.strictEqual(typeof updated.cacheWritesPrice, 'number')
    assert.strictEqual(typeof updated.cacheReadsPrice, 'number')
    assert.strictEqual(updated.maxTokens > 0, true)
  })

  it('getBaseModelId strips variant suffix', function () {
    assert.strictEqual(mod.getBaseModelId('anthropic/claude-3.7-sonnet:thinking'), 'anthropic/claude-3.7-sonnet')
  })
})
