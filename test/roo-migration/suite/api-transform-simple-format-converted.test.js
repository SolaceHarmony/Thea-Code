const assert = require('assert')
const { loadTheaModule } = require('../helpers/thea-loader')

describe('Roo-migration: api/transform/simple-format', function () {
  const mod = loadTheaModule('src/api/transform/simple-format.ts')
  const { convertToSimpleContent, convertToSimpleMessages } = mod

  describe('convertToSimpleContent', function () {
    it('returns string content as-is', function () {
      const content = 'Hello world'
      assert.strictEqual(convertToSimpleContent(content), 'Hello world')
    })

    it('extracts text from text blocks', function () {
      const content = [
        { type: 'text', text: 'Hello' },
        { type: 'text', text: 'world' },
      ]
      assert.strictEqual(convertToSimpleContent(content), 'Hello\nworld')
    })

    it('converts image blocks to descriptive text', function () {
      const content = [
        { type: 'text', text: "Here's an image:" },
        {
          type: 'image',
          source: {
            type: 'base64',
            media_type: 'image/png',
            data: 'base64data',
          },
        },
      ]
      assert.strictEqual(convertToSimpleContent(content), "Here's an image:\n[Image: image/png]")
    })

    it('converts tool use blocks to descriptive text', function () {
      const content = [
        { type: 'text', text: 'Using a tool:' },
        {
          type: 'tool_use',
          id: 'tool-1',
          name: 'read_file',
          input: { path: 'test.txt' },
        },
      ]
      assert.strictEqual(convertToSimpleContent(content), 'Using a tool:\n[Tool Use: read_file]')
    })

    it('handles string tool result content', function () {
      const content = [
        { type: 'text', text: 'Tool result:' },
        {
          type: 'tool_result',
          tool_use_id: 'tool-1',
          content: [{ type: 'text', text: 'Result text' }],
        },
      ]
      assert.strictEqual(convertToSimpleContent(content), 'Tool result:\nResult text')
    })

    it('handles array tool result content with text and images', function () {
      const content = [
        {
          type: 'tool_result',
          tool_use_id: 'tool-1',
          content: [
            { type: 'text', text: 'Result 1' },
            {
              type: 'image',
              source: {
                type: 'base64',
                media_type: 'image/jpeg',
                data: 'base64data',
              },
            },
            { type: 'text', text: 'Result 2' },
          ],
        },
      ]
      assert.strictEqual(convertToSimpleContent(content), 'Result 1\n[Image: image/jpeg]\nResult 2')
    })

    it('filters out empty strings', function () {
      const content = [
        { type: 'text', text: 'Hello' },
        { type: 'text', text: '' },
        { type: 'text', text: 'world' },
      ]
      assert.strictEqual(convertToSimpleContent(content), 'Hello\nworld')
    })
  })

  describe('convertToSimpleMessages', function () {
    it('converts messages with string content', function () {
      const messages = [
        { role: 'user', content: 'Hello' },
        { role: 'assistant', content: 'Hi there' },
      ]
      assert.deepStrictEqual(convertToSimpleMessages(messages), [
        { role: 'user', content: 'Hello' },
        { role: 'assistant', content: 'Hi there' },
      ])
    })

    it('converts messages with complex content', function () {
      const messages = [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'Look at this:' },
            {
              type: 'image',
              source: {
                type: 'base64',
                media_type: 'image/png',
                data: 'base64data',
              },
            },
          ],
        },
        {
          role: 'assistant',
          content: [
            { type: 'text', text: 'I see the image' },
            {
              type: 'tool_use',
              id: 'tool-1',
              name: 'analyze_image',
              input: { data: 'base64data' },
            },
          ],
        },
      ]

      assert.deepStrictEqual(convertToSimpleMessages(messages), [
        { role: 'user', content: 'Look at this:\n[Image: image/png]' },
        { role: 'assistant', content: 'I see the image\n[Tool Use: analyze_image]' },
      ])
    })
  })
})
