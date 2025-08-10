const assert = require('assert')
const { loadTheaModule } = require('../helpers/thea-loader')

describe('Roo-migration: api/transform/neutral-openai-format', function () {
  const mod = loadTheaModule('src/api/transform/neutral-openai-format.ts')
  const { convertToOpenAiHistory } = mod

  it('converts simple text messages', function () {
    const neutralMessages = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi there!' },
    ]
    const openAiMessages = convertToOpenAiHistory(neutralMessages)
    assert.strictEqual(openAiMessages.length, 2)
    assert.deepStrictEqual(openAiMessages[0], { role: 'user', content: 'Hello' })
    assert.deepStrictEqual(openAiMessages[1], { role: 'assistant', content: 'Hi there!' })
  })

  it('handles messages with image content', function () {
    const neutralMessages = [
      {
        role: 'user',
        content: [
          { type: 'text', text: 'What is in this image?' },
          {
            type: 'image',
            source: { type: 'base64', media_type: 'image/jpeg', data: 'base64data' },
          },
        ],
      },
    ]
    const openAiMessages = convertToOpenAiHistory(neutralMessages)
    assert.strictEqual(openAiMessages.length, 1)
    const msg = openAiMessages[0]
    assert.strictEqual(msg.role, 'user')
    const content = msg.content
    assert.ok(Array.isArray(content))
    assert.strictEqual(content.length, 2)
    assert.deepStrictEqual(content[0], { type: 'text', text: 'What is in this image?' })
    assert.deepStrictEqual(content[1], {
      type: 'image_url',
      image_url: { url: 'data:image/jpeg;base64,base64data' },
    })
  })

  it('handles assistant messages with tool use', function () {
    const neutralMessages = [
      {
        role: 'assistant',
        content: [
          { type: 'text', text: 'Let me check the weather.' },
          { type: 'tool_use', id: 'weather-123', name: 'get_weather', input: { city: 'London' } },
        ],
      },
    ]
    const openAiMessages = convertToOpenAiHistory(neutralMessages)
    assert.strictEqual(openAiMessages.length, 1)
    const assistantMessage = openAiMessages[0]
    assert.strictEqual(assistantMessage.role, 'assistant')
    assert.strictEqual(assistantMessage.content, 'Let me check the weather.')
    assert.ok(Array.isArray(assistantMessage.tool_calls))
    assert.strictEqual(assistantMessage.tool_calls.length, 1)
    assert.deepStrictEqual(assistantMessage.tool_calls[0], {
      id: 'weather-123',
      type: 'function',
      function: { name: 'get_weather', arguments: JSON.stringify({ city: 'London' }) },
    })
  })

  it('handles user messages with tool results', function () {
    const neutralMessages = [
      {
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: 'weather-123',
            content: [{ type: 'text', text: 'Current temperature in London: 20\u00b0C' }],
          },
        ],
      },
    ]
    const openAiMessages = convertToOpenAiHistory(neutralMessages)
    assert.strictEqual(openAiMessages.length, 1)
    const toolMessage = openAiMessages[0]
    assert.strictEqual(toolMessage.role, 'tool')
    assert.strictEqual(toolMessage.tool_call_id, 'weather-123')
    assert.strictEqual(toolMessage.content, 'Current temperature in London: 20\u00b0C')
  })
})
