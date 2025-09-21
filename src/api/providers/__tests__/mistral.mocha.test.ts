import * as assert from 'assert'

import { MistralHandler } from '../mistral'
import type { ApiHandlerOptions } from '../../../shared/api'
import type { NeutralConversationHistory, NeutralMessageContent } from '../../../shared/neutral-history'
import type { ApiStreamChunk } from '../../transform/stream'
import { mockMistral } from '../../../../test/generic-provider-mock/server'

const HOST = '127.0.0.1'

const baseOptions: ApiHandlerOptions = {
  mistralApiKey: 'test-key',
  apiModelId: 'codestral-latest',
  mistralCodestralUrl: 'https://codestral.mistral.ai',
}

describe('MistralHandler (mock server)', () => {
  let baseUrl: string

  before(async () => {
    await mockMistral.start()
    const port = mockMistral.getPort()
    if (!port) throw new Error('mockMistral did not expose a port')
    baseUrl = `http://${HOST}:${port}`
  })

  after(async () => {
    await mockMistral.stop()
  })

  beforeEach(() => {
    mockMistral.clearRequestLog()
    mockMistral.clearResponseOverrides()
  })

  function createHandler(extra?: Partial<ApiHandlerOptions>) {
    return new MistralHandler({
      ...baseOptions,
      mistralCodestralUrl: baseUrl,
      ...extra,
    })
  }

  async function collect(stream: AsyncIterable<ApiStreamChunk>) {
    const chunks: ApiStreamChunk[] = []
    for await (const chunk of stream) {
      chunks.push(chunk)
    }
    return chunks
  }

  it('streams converted messages from the mock server', async () => {
    const handler = createHandler()
    const history: NeutralConversationHistory = [
      { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
    ]
    const chunks = await collect(handler.createMessage('system', history))
    const textChunks = chunks.filter(chunk => chunk.type === 'text')
    assert.ok(textChunks.length > 0)
  })

  it('delegates token counting to the base provider', async () => {
    const handler = createHandler()
    const content: NeutralMessageContent = [{ type: 'text', text: 'Hello' }]
    const result = await handler.countTokens(content)
    assert.ok(result >= 0)
  })

  it('completes prompts using the mock client', async () => {
    const handler = createHandler()
    const result = await handler.completePrompt('Test prompt')
    assert.ok(typeof result === 'string')
  })

  it('wraps completion errors with context', async () => {
    mockMistral.setResponseOverride('chat_codestral-latest', { error: { message: 'API Error' } })
    const handler = createHandler()

    await assert.rejects(
      handler.completePrompt('prompt'),
      (error: unknown) => error instanceof Error && error.message === 'Mistral completion error: API Error',
    )
  })
})
