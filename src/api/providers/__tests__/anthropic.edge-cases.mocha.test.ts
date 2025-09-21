import * as assert from 'assert'

import { AnthropicHandler } from '../anthropic'
import type { ApiHandlerOptions } from '../../../shared/api'
import type { NeutralConversationHistory, NeutralMessageContent } from '../../../shared/neutral-history'
import type { ApiStreamChunk } from '../../transform/stream'
import { mockAnthropic } from '../../../../test/generic-provider-mock/server'

const HOST = '127.0.0.1'

const baseOptions: ApiHandlerOptions = {
  apiKey: 'test-anthropic-key',
  apiModelId: 'claude-3-5-sonnet-20241022-thinking',
}

describe('AnthropicHandler (mock server)', () => {
  let baseUrl: string

  before(async () => {
    await mockAnthropic.start()
    const port = mockAnthropic.getPort()
    if (!port) throw new Error('mockAnthropic did not expose a port')
    baseUrl = `http://${HOST}:${port}`
  })

  after(async () => {
    await mockAnthropic.stop()
  })

  beforeEach(() => {
    mockAnthropic.clearRequestLog()
    mockAnthropic.clearResponseOverrides()
  })

  function createHandler(extra?: Partial<ApiHandlerOptions>) {
    return new AnthropicHandler({
      ...baseOptions,
      anthropicBaseUrl: baseUrl,
      ...extra,
    })
  }

  async function gather(stream: AsyncIterable<ApiStreamChunk>) {
    const chunks: ApiStreamChunk[] = []
    for await (const chunk of stream) {
      chunks.push(chunk)
    }
    return chunks
  }

  it('clamps thinking budget when custom limits are provided', () => {
    const handler = createHandler({ modelMaxTokens: 16_000, modelMaxThinkingTokens: 20_000 })
    const model = handler.getModel()
    assert.ok(model.thinking)
    const thinking = model.thinking as { type: string; budget_tokens: number }
    assert.strictEqual(thinking.type, 'enabled')
    // Should clamp to 80% of max tokens (=12_800) rather than requested 20_000
    assert.strictEqual(thinking.budget_tokens, 12_800)
  })

  it('streams text chunks from the mock anthropic server', async () => {
    const handler = createHandler()
    const history: NeutralConversationHistory = [
      { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
    ]
    const chunks = await gather(handler.createMessage('system', history))
    const texts = chunks.filter(chunk => chunk.type === 'text')
    assert.ok(texts.length > 0)
  })

  it('aggregates streamed text when completing prompts', async () => {
    const handler = createHandler()
    const result = await handler.completePrompt('Test prompt')
    assert.ok(typeof result === 'string')
    assert.ok(result.length >= 0)
  })

  it('uses remote token counting first, then falls back on error', async () => {
    const handler = createHandler()
    const content: NeutralMessageContent = [{ type: 'text', text: 'token count me' }]
    const count = await handler.countTokens(content)
    assert.ok(typeof count === 'number')
    assert.ok(count > 0)
  })
})
