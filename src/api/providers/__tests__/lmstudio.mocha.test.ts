import * as assert from 'assert'

import { LmStudioHandler, getLmStudioModels } from '../lmstudio'
import type { ApiHandlerOptions } from '../../../shared/api'
import type { NeutralConversationHistory } from '../../../shared/neutral-history'
import type { ApiStreamChunk } from '../../transform/stream'
import { mockOpenAI } from '../../../../test/generic-provider-mock/server'

const HOST = '127.0.0.1'

const defaultOptions: ApiHandlerOptions = {
  apiModelId: 'local-model',
  lmStudioModelId: 'local-model',
}

describe('LmStudioHandler (mock server)', () => {
  let baseUrl: string

  before(async () => {
    await mockOpenAI.start()
    const port = mockOpenAI.getPort()
    if (!port) throw new Error('mockOpenAI did not expose a port')
    baseUrl = `http://${HOST}:${port}`
  })

  after(async () => {
    await mockOpenAI.stop()
  })

  beforeEach(() => {
    mockOpenAI.clearRequestLog()
    mockOpenAI.clearResponseOverrides()
  })

  function createHandler(extra?: Partial<ApiHandlerOptions>) {
    return new LmStudioHandler({
      ...defaultOptions,
      lmStudioBaseUrl: baseUrl,
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

  it('initializes with provided options', () => {
    const handler = createHandler()
    assert.ok(handler instanceof LmStudioHandler)
    assert.strictEqual(handler.getModel().id, defaultOptions.lmStudioModelId)
  })

  it('streams responses from createMessage', async () => {
    const handler = createHandler()
    const messages: NeutralConversationHistory = [
      { role: 'user', content: [{ type: 'text', text: 'Hello!' }] },
    ]

    const chunks = await collect(handler.createMessage('system', messages))
    const textChunks = chunks.filter(chunk => chunk.type === 'text')
    assert.ok(textChunks.length > 0)
  })

  it('wraps API errors with a helpful message', async () => {
    mockOpenAI.setResponseOverride('chat_local-model', { error: { code: 500, message: 'API Error' } })
    const handler = createHandler()

    await assert.rejects(
      handler.completePrompt('Test prompt'),
      (error: unknown) =>
        error instanceof Error &&
        error.message.includes('Please check the LM Studio developer logs'),
    )
  })

  it('aggregates streamed text when completing prompts', async () => {
    const handler = createHandler()
    const result = await handler.completePrompt('Hello from test')
    assert.ok(typeof result === 'string')
    assert.ok(result.length > 0)
  })

  it('fetches model metadata from mock server', async () => {
    const models = await getLmStudioModels(baseUrl)
    assert.ok(Array.isArray(models))
  })
})
