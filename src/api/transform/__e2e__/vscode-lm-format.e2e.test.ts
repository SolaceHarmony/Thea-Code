import * as assert from 'assert'
import type { NeutralConversationHistory } from '../../../shared/neutral-history'

import { convertToVsCodeLmMessages, convertToAnthropicRole } from '../vscode-lm-format'

suite('convertToVsCodeLmMessages', () => {
  test('converts simple string messages', () => {
    const messages: NeutralConversationHistory = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi there' }
    ]

    const result = convertToVsCodeLmMessages(messages)

    assert.strictEqual(result.length, 2)
    assert.strictEqual(result[0].content[0].value, 'Hello')
    assert.strictEqual(result[1].content[0].value, 'Hi there')
    assert.strictEqual(convertToAnthropicRole(result[0].role), 'user')
    assert.strictEqual(convertToAnthropicRole(result[1].role), 'assistant')
  })

  test('handles complex user messages with tool results', () => {
    const messages: NeutralConversationHistory = [
      {
        role: 'user',
        content: [
          { type: 'text', text: 'Here is the result:' },
          {
            type: 'tool_result',
            tool_use_id: 'tool-1',
            content: [{ type: 'text', text: 'Tool output' }]
          }
        ]
      }
    ]

    const result = convertToVsCodeLmMessages(messages)

    assert.strictEqual(result.length, 1)
    assert.strictEqual(result[0].content.length, 2)
    const [toolResult, textContent] = result[0].content
    assert.strictEqual(toolResult.constructor.name, 'LanguageModelToolResultPart')
    assert.strictEqual(textContent.constructor.name, 'LanguageModelTextPart')
  })

  test('handles complex assistant messages with tool calls', () => {
    const messages: NeutralConversationHistory = [
      {
        role: 'assistant',
        content: [
          { type: 'text', text: 'Let me help you with that.' },
          {
            type: 'tool_use',
            id: 'tool-1',
            name: 'calculator',
            input: { operation: 'add', numbers: [2, 2] }
          }
        ]
      }
    ]

    const result = convertToVsCodeLmMessages(messages)

    assert.strictEqual(result.length, 1)
    assert.strictEqual(result[0].content.length, 2)
    const [toolCall, textContent] = result[0].content
    assert.strictEqual(toolCall.constructor.name, 'LanguageModelToolCallPart')
    assert.strictEqual(textContent.constructor.name, 'LanguageModelTextPart')
  })

  test('handles image blocks with placeholders', () => {
    const messages: NeutralConversationHistory = [
      {
        role: 'user',
        content: [
          { type: 'text', text: 'Look at this:' },
          {
            type: 'image',
            source: { type: 'base64', media_type: 'image/png', data: 'base64data' }
          }
        ]
      }
    ]

    const result = convertToVsCodeLmMessages(messages)

    assert.strictEqual(result.length, 1)
    const imagePlaceholder = result[0].content[1]
    assert.ok(String(imagePlaceholder.value).includes('[Image (base64): image/png not supported by VSCode LM API]'))
  })
})

suite('convertToAnthropicRole', () => {
  test('maps roles and unknown → null', () => {
    const assistantRole = convertToVsCodeLmMessages([{ role: 'assistant', content: 'x' }])[0].role
    const userRole = convertToVsCodeLmMessages([{ role: 'user', content: 'y' }])[0].role
    assert.strictEqual(convertToAnthropicRole(assistantRole), 'assistant')
    assert.strictEqual(convertToAnthropicRole(userRole), 'user')
    assert.strictEqual(convertToAnthropicRole('unknown' as any), null)
  })
})

suite('asObjectSafe via convertToVsCodeLmMessages', () => {
  test('parses JSON strings in tool_use input', () => {
    const messages: NeutralConversationHistory = [
      {
        role: 'assistant',
        content: [
          {
            type: 'tool_use',
            id: '1',
            name: 'test',
            input: { jsonString: '{"foo": "bar"}' }
          }
        ]
      }
    ]
    const result = convertToVsCodeLmMessages(messages)
    const toolCall = result[0].content[0]
    assert.deepStrictEqual(toolCall.input, { jsonString: '{"foo": "bar"}' })
  })

  test('handles invalid JSON by returning empty object', () => {
    const messages: NeutralConversationHistory = [
      {
        role: 'assistant',
        content: [
          {
            type: 'tool_use',
            id: '2',
            name: 'test',
            input: { invalidJson: '{invalid}' }
          }
        ]
      }
    ]
    const result = convertToVsCodeLmMessages(messages)
    const toolCall = result[0].content[0]
    assert.deepStrictEqual(toolCall.input, { invalidJson: '{invalid}' })
  })

  test('clones object inputs', () => {
    const obj = { a: 1 }
    const messages: NeutralConversationHistory = [
      {
        role: 'assistant',
        content: [{ type: 'tool_use', id: '3', name: 'test', input: obj }]
      }
    ]
    const result = convertToVsCodeLmMessages(messages)
    const toolCall = result[0].content[0]
    assert.deepStrictEqual(toolCall.input, obj)
    assert.notStrictEqual(toolCall.input, obj)
  })
})

