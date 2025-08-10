const assert = require('assert')
const { loadTheaModule } = require('../helpers/thea-loader')

// Use the local stubbed 'vscode' module provided under test/roo-migration/node_modules
const vscode = require('vscode')

describe('Roo-migration: api/transform/vscode-lm-format', function () {
  const mod = loadTheaModule('src/api/transform/vscode-lm-format.ts')
  const { convertToVsCodeLmMessages, convertToAnthropicRole } = mod

  it('converts simple string messages', function () {
    const messages = [ { role: 'user', content: 'Hello' }, { role: 'assistant', content: 'Hi there' } ]
    const result = convertToVsCodeLmMessages(messages)
    assert.strictEqual(result.length, 2)
    assert.strictEqual(convertToAnthropicRole(result[0].role), 'user')
    assert.strictEqual(result[0].content[0].value, 'Hello')
    assert.strictEqual(convertToAnthropicRole(result[1].role), 'assistant')
    assert.strictEqual(result[1].content[0].value, 'Hi there')
  })

  it('handles complex user messages with tool results', function () {
    const messages = [ { role: 'user', content: [ { type: 'text', text: 'Here is the result:' }, { type: 'tool_result', tool_use_id: 'tool-1', content: [ { type: 'text', text: 'Tool output' } ] } ] } ]
    const result = convertToVsCodeLmMessages(messages)
    assert.strictEqual(result.length, 1)
    assert.strictEqual(convertToAnthropicRole(result[0].role), 'user')
    assert.strictEqual(result[0].content.length, 2)
    const [toolResult, textContent] = result[0].content
    assert.ok(toolResult instanceof vscode.LanguageModelToolResultPart)
    assert.ok(textContent instanceof vscode.LanguageModelTextPart)
  })

  it('handles complex assistant messages with tool calls', function () {
    const messages = [ { role: 'assistant', content: [ { type: 'text', text: 'Let me help you with that.' }, { type: 'tool_use', id: 'tool-1', name: 'calculator', input: { operation: 'add', numbers: [2, 2] } } ] } ]
    const result = convertToVsCodeLmMessages(messages)
    assert.strictEqual(result.length, 1)
    assert.strictEqual(convertToAnthropicRole(result[0].role), 'assistant')
    assert.strictEqual(result[0].content.length, 2)
    const [toolCall, textContent] = result[0].content
    assert.ok(toolCall instanceof vscode.LanguageModelToolCallPart)
    assert.ok(textContent instanceof vscode.LanguageModelTextPart)
  })

  it('handles image blocks with placeholders', function () {
    const messages = [ { role: 'user', content: [ { type: 'text', text: 'Look at this:' }, { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'base64data' } } ] } ]
    const result = convertToVsCodeLmMessages(messages)
    assert.strictEqual(result.length, 1)
    const imagePlaceholder = result[0].content[1]
    assert.ok(String(imagePlaceholder.value).includes('[Image (base64): image/png not supported by VSCode LM API]'))
  })

  it('convertToAnthropicRole maps roles and unknown → null', function () {
    assert.strictEqual(convertToAnthropicRole(vscode.LanguageModelChatMessageRole.Assistant), 'assistant')
    assert.strictEqual(convertToAnthropicRole(vscode.LanguageModelChatMessageRole.User), 'user')
    assert.strictEqual(convertToAnthropicRole('unknown'), null)
  })
})
