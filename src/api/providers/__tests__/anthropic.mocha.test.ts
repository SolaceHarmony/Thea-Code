import * as assert from 'assert'

import { AnthropicHandler } from '../anthropic'
import type { ApiHandlerOptions } from '../../../shared/api'
import type { NeutralConversationHistory } from '../../../shared/neutral-history'
import type { ApiStreamChunk } from '../../transform/stream'
import { mockAnthropic } from '../../../../test/generic-provider-mock/server'

const HOST = '127.0.0.1'

const baseOptions: ApiHandlerOptions = {
  apiKey: 'test-anthropic-key',
  apiModelId: 'claude-3-5-sonnet-20241022',
}

describe('AnthropicHandler', () => {
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

  async function drain(stream: AsyncIterable<ApiStreamChunk>) {
    const chunks: ApiStreamChunk[] = []
    for await (const chunk of stream) {
      chunks.push(chunk)
    }
    return chunks
  }

  it('initializes with provided options', () => {
    const handler = createHandler()
    assert.ok(handler instanceof AnthropicHandler)
    assert.strictEqual(handler.getModel().id, baseOptions.apiModelId)
  })

  it('accepts custom base URL and streams responses', async () => {
    const handler = createHandler({ anthropicBaseUrl: baseUrl })
    const messages: NeutralConversationHistory = [
      { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
    ]
    const chunks = await drain(handler.createMessage('system', messages))
    assert.ok(chunks.some((chunk) => chunk.type === 'text'))
  })

  it('aggregates completion text', async () => {
    const handler = createHandler()
    const result = await handler.completePrompt('Test prompt')
    assert.ok(typeof result === 'string')
  })

  it('counts tokens using the mock endpoint', async () => {
    const handler = createHandler()
    const count = await handler.countTokens('Hello world')
    assert.ok(count > 0)
  })
})
