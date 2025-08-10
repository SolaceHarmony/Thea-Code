const assert = require('assert')
const { loadTheaModule } = require('../helpers/thea-loader')

describe('Roo-migration: api/transform/neutral-mistral-format', function () {
  const mod = loadTheaModule('src/api/transform/neutral-mistral-format.ts')
  const { convertToMistralMessages } = mod

  it('converts simple text messages for user and assistant roles', function () {
    const neutral = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi there!' },
    ]
    const out = convertToMistralMessages(neutral)
    assert.strictEqual(out.length, 2)
    assert.deepStrictEqual(out[0], { role: 'user', content: 'Hello' })
    assert.deepStrictEqual(out[1], { role: 'assistant', content: 'Hi there!' })
  })

  it('handles user messages with image content', function () {
    const neutral = [
      {
        role: 'user',
        content: [
          { type: 'text', text: 'What is in this image?' },
          { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: 'base64data' } },
        ],
      },
    ]
    const out = convertToMistralMessages(neutral)
    assert.strictEqual(out.length, 1)
    assert.strictEqual(out[0].role, 'user')
    const content = out[0].content
    assert.ok(Array.isArray(content))
    assert.strictEqual(content.length, 2)
    assert.deepStrictEqual(content[0], { type: 'text', text: 'What is in this image?' })
    assert.deepStrictEqual(content[1], {
      type: 'image_url',
      imageUrl: { url: 'data:image/jpeg;base64,base64data' },
    })
  })

  it('handles user messages with only tool results (no messages produced)', function () {
    const neutral = [
      {
        role: 'user',
        content: [
          { type: 'tool_result', tool_use_id: 'weather-123', content: [{ type: 'text', text: 'Current temperature in London: 20\u00b0C' }] },
        ],
      },
    ]
    const out = convertToMistralMessages(neutral)
    assert.strictEqual(out.length, 0)
  })

  it('handles user messages with mixed content (text, image, and tool results)', function () {
    const neutral = [
      {
        role: 'user',
        content: [
          { type: 'text', text: "Here's the weather data and an image:" },
          { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'imagedata123' } },
          { type: 'tool_result', tool_use_id: 'weather-123', content: [{ type: 'text', text: 'Current temperature in London: 20\u00b0C' }] },
        ],
      },
    ]
    const out = convertToMistralMessages(neutral)
    assert.strictEqual(out.length, 1)
    assert.strictEqual(out[0].role, 'user')
    const userContent = out[0].content
    assert.ok(Array.isArray(userContent))
    assert.strictEqual(userContent.length, 2)
    assert.deepStrictEqual(userContent[0], { type: 'text', text: "Here's the weather data and an image:" })
    assert.deepStrictEqual(userContent[1], { type: 'image_url', imageUrl: { url: 'data:image/png;base64,imagedata123' } })
  })

  it('handles assistant messages with text content', function () {
    const neutral = [
      { role: 'assistant', content: [ { type: 'text', text: "I'll help you with that question." } ] },
    ]
    const out = convertToMistralMessages(neutral)
    assert.strictEqual(out.length, 1)
    assert.strictEqual(out[0].role, 'assistant')
    assert.strictEqual(out[0].content, "I'll help you with that question.")
  })

  it('handles assistant messages with tool use', function () {
    const neutral = [
      { role: 'assistant', content: [ { type: 'text', text: 'Let me check the weather for you.' }, { type: 'tool_use', id: 'weather-123', name: 'get_weather', input: { city: 'London' } } ] },
    ]
    const out = convertToMistralMessages(neutral)
    assert.strictEqual(out.length, 1)
    assert.strictEqual(out[0].role, 'assistant')
    assert.strictEqual(out[0].content, 'Let me check the weather for you.')
  })

  it('handles multiple text blocks in assistant messages by joining with newlines', function () {
    const neutral = [
      { role: 'assistant', content: [ { type: 'text', text: 'First paragraph of information.' }, { type: 'text', text: 'Second paragraph with more details.' } ] },
    ]
    const out = convertToMistralMessages(neutral)
    assert.strictEqual(out.length, 1)
    assert.strictEqual(out[0].role, 'assistant')
    assert.strictEqual(out[0].content, 'First paragraph of information.\nSecond paragraph with more details.')
  })

  it('handles a conversation with mixed message types', function () {
    const neutral = [
      { role: 'user', content: [ { type: 'text', text: "What's in this image?" }, { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: 'imagedata' } } ] },
      { role: 'assistant', content: [ { type: 'text', text: 'This image shows a landscape with mountains.' }, { type: 'tool_use', id: 'search-123', name: 'search_info', input: { query: 'mountain types' } } ] },
      { role: 'user', content: [ { type: 'tool_result', tool_use_id: 'search-123', content: [ { type: 'text', text: 'Found information about different mountain types.' } ] } ] },
      { role: 'assistant', content: 'Based on the search results, I can tell you more about the mountains in the image.' },
    ]
    const out = convertToMistralMessages(neutral)
    assert.strictEqual(out.length, 3)
    assert.strictEqual(out[0].role, 'user')
    const userContent = out[0].content
    assert.ok(Array.isArray(userContent))
    assert.strictEqual(userContent.length, 2)
    assert.strictEqual(out[1].role, 'assistant')
    assert.strictEqual(out[1].content, 'This image shows a landscape with mountains.')
    assert.deepStrictEqual(out[2], { role: 'assistant', content: 'Based on the search results, I can tell you more about the mountains in the image.' })
  })

  it('handles empty content in assistant messages', function () {
    const neutral = [
      { role: 'assistant', content: [ { type: 'tool_use', id: 'search-123', name: 'search_info', input: { query: 'test query' } } ] },
    ]
    const out = convertToMistralMessages(neutral)
    assert.strictEqual(out.length, 1)
    assert.strictEqual(out[0].role, 'assistant')
    assert.strictEqual(out[0].content, undefined)
  })
})
