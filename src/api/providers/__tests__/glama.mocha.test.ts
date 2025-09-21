import * as assert from 'assert'

import { GlamaHandler } from '../glama'
import { OpenAiHandler } from '../openai'
import type { ApiHandlerOptions } from '../../../shared/api'
import type { NeutralConversationHistory } from '../../../shared/neutral-history'
import type { ApiStreamChunk } from '../../transform/stream'
import { mockOpenAI } from '../../../../test/generic-provider-mock/server'

const HOST = '127.0.0.1'

describe('GlamaHandler', () => {
  let baseUrl: string

  before(async () => {
    await mockOpenAI.start()
    const port = mockOpenAI.getPort()
    if (!port) throw new Error('mockOpenAI did not expose a port')
    baseUrl = `http://${HOST}:${port}/v1`
  })

  after(async () => {
    await mockOpenAI.stop()
  })

  beforeEach(() => {
    mockOpenAI.clearRequestLog()
    mockOpenAI.clearResponseOverrides()
  })

  function createHandler(extra?: Partial<ApiHandlerOptions>) {
    const options: ApiHandlerOptions = {
      apiModelId: 'openai/gpt-4',
      glamaModelId: 'openai/gpt-4',
      glamaApiKey: 'test-api-key',
      ...extra,
    }
    const handler = new GlamaHandler(options)
    // Replace the internal OpenAI handler so it targets our mock server
    ;(handler as unknown as { openAiHandler: OpenAiHandler }).openAiHandler = new OpenAiHandler({
      ...options,
      openAiApiKey: options.glamaApiKey,
      openAiModelId: options.glamaModelId,
      openAiBaseUrl: baseUrl,
    })
    return handler
  }

  async function collect(stream: AsyncIterable<ApiStreamChunk>) {
    const chunks: ApiStreamChunk[] = []
    for await (const chunk of stream) {
      chunks.push(chunk)
    }
    return chunks
  }

  it('streams text and usage from the mock server', async () => {
    const handler = createHandler()
    const messages: NeutralConversationHistory = [{ role: 'user', content: 'Hello!' }]
    const chunks = await collect(handler.createMessage('system', messages))
    assert.ok(chunks.some((chunk) => chunk.type === 'text'))
    assert.ok(chunks.some((chunk) => chunk.type === 'usage'))
  })

  it('aggregates completion text', async () => {
    const handler = createHandler()
    const result = await handler.completePrompt('Test prompt')
    assert.ok(typeof result === 'string')
    assert.ok(result.length >= 0)
  })

  it('returns provider model info', () => {
    const handler = createHandler()
    const model = handler.getModel()
    assert.strictEqual(model.id, 'openai/gpt-4')
  })
})
