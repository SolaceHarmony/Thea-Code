const assert = require('assert')
const { loadTheaModule } = require('../helpers/thea-loader')

describe('Roo-migration: api/transform/neutral-ollama-format', function () {
  const mod = loadTheaModule('src/api/transform/neutral-ollama-format.ts')
  const { convertToOllamaHistory, convertToOllamaContentBlocks, convertToNeutralHistoryFromOllama } = mod

  describe('convertToOllamaHistory', function () {
    it('converts a simple user message', function () {
      const neutralHistory = [
        { role: 'user', content: [{ type: 'text', text: 'What is the capital of France?' }] },
      ]
      const expected = [{ role: 'user', content: 'What is the capital of France?' }]
      assert.deepStrictEqual(convertToOllamaHistory(neutralHistory), expected)
    })

    it('converts system and user messages', function () {
      const neutralHistory = [
        { role: 'system', content: [{ type: 'text', text: 'You are a helpful assistant.' }] },
        { role: 'user', content: [{ type: 'text', text: 'What is the capital of France?' }] },
      ]
      const expected = [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'What is the capital of France?' },
      ]
      assert.deepStrictEqual(convertToOllamaHistory(neutralHistory), expected)
    })

    it('converts multi-turn conversations', function () {
      const neutralHistory = [
        { role: 'user', content: [{ type: 'text', text: 'What is the capital of France?' }] },
        { role: 'assistant', content: [{ type: 'text', text: 'The capital of France is Paris.' }] },
        { role: 'user', content: [{ type: 'text', text: 'What is its population?' }] },
      ]
      const expected = [
        { role: 'user', content: 'What is the capital of France?' },
        { role: 'assistant', content: 'The capital of France is Paris.' },
        { role: 'user', content: 'What is its population?' },
      ]
      assert.deepStrictEqual(convertToOllamaHistory(neutralHistory), expected)
    })

    it('joins multiple text blocks with newlines', function () {
      const neutralHistory = [
        { role: 'user', content: [ { type: 'text', text: 'Hello' }, { type: 'text', text: 'World' } ] },
      ]
      const expected = [{ role: 'user', content: 'Hello\n\nWorld' }]
      assert.deepStrictEqual(convertToOllamaHistory(neutralHistory), expected)
    })

    it('handles string content', function () {
      const neutralHistory = [ { role: 'user', content: 'What is the capital of France?' } ]
      const expected = [ { role: 'user', content: 'What is the capital of France?' } ]
      assert.deepStrictEqual(convertToOllamaHistory(neutralHistory), expected)
    })

    it('ignores non-text content blocks (logs may be emitted)', function () {
      const neutralHistory = [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'Look at this image:' },
            { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'base64data' } },
          ],
        },
      ]
      const expected = [{ role: 'user', content: 'Look at this image:' }]
      assert.deepStrictEqual(convertToOllamaHistory(neutralHistory), expected)
    })

    it('handles empty content array', function () {
      const neutralHistory = [ { role: 'user', content: [] } ]
      const expected = [ { role: 'user', content: '' } ]
      assert.deepStrictEqual(convertToOllamaHistory(neutralHistory), expected)
    })

    it('handles unknown roles by defaulting to user', function () {
      const neutralHistory = [ { role: 'tool', content: [{ type: 'text', text: 'Tool result' }] } ]
      const result = convertToOllamaHistory(neutralHistory)
      assert.strictEqual(result[0].role, 'user')
      assert.strictEqual(result[0].content, 'Tool result')
    })
  })

  describe('convertToOllamaContentBlocks', function () {
    it('converts a single text block to string', function () {
      const content = [{ type: 'text', text: 'Hello, world!' }]
      assert.strictEqual(convertToOllamaContentBlocks(content), 'Hello, world!')
    })

    it('joins multiple text blocks with newlines', function () {
      const content = [ { type: 'text', text: 'Hello' }, { type: 'text', text: 'World' } ]
      assert.strictEqual(convertToOllamaContentBlocks(content), 'Hello\n\nWorld')
    })

    it('extracts only text from mixed blocks', function () {
      const content = [
        { type: 'text', text: 'Look at this image:' },
        { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'base64data' } },
        { type: 'text', text: 'What do you think?' },
      ]
      assert.strictEqual(convertToOllamaContentBlocks(content), 'Look at this image:\n\nWhat do you think?')
    })

    it('handles empty array', function () {
      assert.strictEqual(convertToOllamaContentBlocks([]), '')
    })
  })

  describe('convertToNeutralHistoryFromOllama', function () {
    it('converts a simple user message', function () {
      const ollama = [ { role: 'user', content: 'What is the capital of France?' } ]
      const expected = [ { role: 'user', content: [ { type: 'text', text: 'What is the capital of France?' } ] } ]
      assert.deepStrictEqual(convertToNeutralHistoryFromOllama(ollama), expected)
    })

    it('converts system and user messages', function () {
      const ollama = [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'What is the capital of France?' },
      ]
      const expected = [
        { role: 'system', content: [ { type: 'text', text: 'You are a helpful assistant.' } ] },
        { role: 'user', content: [ { type: 'text', text: 'What is the capital of France?' } ] },
      ]
      assert.deepStrictEqual(convertToNeutralHistoryFromOllama(ollama), expected)
    })

    it('converts multi-turn conversations', function () {
      const ollama = [
        { role: 'user', content: 'What is the capital of France?' },
        { role: 'assistant', content: 'The capital of France is Paris.' },
        { role: 'user', content: 'What is its population?' },
      ]
      const expected = [
        { role: 'user', content: [ { type: 'text', text: 'What is the capital of France?' } ] },
        { role: 'assistant', content: [ { type: 'text', text: 'The capital of France is Paris.' } ] },
        { role: 'user', content: [ { type: 'text', text: 'What is its population?' } ] },
      ]
      assert.deepStrictEqual(convertToNeutralHistoryFromOllama(ollama), expected)
    })

    it('handles array content by mapping to text blocks', function () {
      const ollama = [ { role: 'user', content: ['Hello', 'World'] } ]
      const result = convertToNeutralHistoryFromOllama(ollama)
      assert.strictEqual(result[0].role, 'user')
      const content = result[0].content
      assert.ok(Array.isArray(content))
      assert.strictEqual(content.length, 2)
      assert.strictEqual(content[0].type, 'text')
      assert.strictEqual(content[0].text, 'Hello')
      assert.strictEqual(content[1].type, 'text')
      assert.strictEqual(content[1].text, 'World')
    })

    it('handles unknown roles by defaulting to user', function () {
      const ollama = [ { role: 'function', content: 'Function result' } ]
      const result = convertToNeutralHistoryFromOllama(ollama)
      assert.strictEqual(result[0].role, 'user')
    })

    it('handles empty content', function () {
      const ollama = [ { role: 'user', content: '' } ]
      const expected = [ { role: 'user', content: [ { type: 'text', text: '' } ] } ]
      assert.deepStrictEqual(convertToNeutralHistoryFromOllama(ollama), expected)
    })

    it('stringifies non-string, non-array content', function () {
      const ollama = [ { role: 'user', content: { foo: 'bar' } } ]
      const result = convertToNeutralHistoryFromOllama(ollama)
      const content = result[0].content
      assert.ok(Array.isArray(content))
      assert.strictEqual(content.length, 1)
      assert.strictEqual(content[0].type, 'text')
      assert.strictEqual(content[0].text, JSON.stringify({ foo: 'bar' }))
    })
  })
})
