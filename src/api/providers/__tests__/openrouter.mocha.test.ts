import * as assert from 'assert'

import { OpenRouterHandler } from '../openrouter'
import type { ApiHandlerOptions, ModelInfo } from '../../../shared/api'
import type { NeutralConversationHistory } from '../../../shared/neutral-history'
import type { ApiStreamChunk } from '../../transform/stream'
import { mockOpenAI } from '../../../../test/generic-provider-mock/server'

const HOST = '127.0.0.1'

const mockOpenRouterModelInfo: ModelInfo = {
  maxTokens: 1000,
  contextWindow: 2000,
  supportsPromptCache: true,
  inputPrice: 0.01,
  outputPrice: 0.02,
}

const baseOptions: ApiHandlerOptions = {
  openRouterApiKey: 'test-key',
  openRouterModelId: 'test-model',
  openRouterModelInfo: mockOpenRouterModelInfo,
}

describe('OpenRouterHandler (networked)', () => {
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
    return new OpenRouterHandler({
      ...baseOptions,
      openRouterBaseUrl: `${baseUrl}/v1`,
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

  it('sends requests through the mock server with expected headers', async () => {
    const handler = createHandler()
    const messages: NeutralConversationHistory = [{ role: 'user', content: 'hello' }]

    await drain(handler.createMessage('system', messages))

    const [entry] = mockOpenAI.getRequestLog() as Array<{
      method: string
      path: string
      headers: Record<string, string>
      body: unknown
    }>
    assert.ok(entry)
    assert.strictEqual(entry.method, 'POST')
    assert.strictEqual(entry.path, '/v1/chat/completions')
    assert.strictEqual(entry.headers['x-title'], 'Thea Code')
    assert.strictEqual(entry.headers['http-referer'], 'https://solaceofharmony.ai')
  })

  it('returns model info sourced from options', () => {
    const handler = createHandler()
    const model = handler.getModel()
    assert.strictEqual(model.id, baseOptions.openRouterModelId)
    assert.deepStrictEqual(model.info, baseOptions.openRouterModelInfo)
  })

  it('streams text chunks from the mock response', async () => {
    const handler = createHandler()
    const chunks = await drain(handler.createMessage('system', [{ role: 'user', content: 'hello' }]))
    const textChunks = chunks.filter(chunk => chunk.type === 'text')
    assert.ok(textChunks.length > 0)
    const combined = textChunks.map(chunk => chunk.text ?? '').join('')
    assert.ok(combined.length > 0)
  })

  it('includes middle-out transform when enabled', async () => {
    const handler = createHandler({ openRouterUseMiddleOutTransform: true })
    await drain(handler.createMessage('system', []))
    const [entry] = mockOpenAI.getRequestLog() as Array<{
      body: Record<string, unknown>
    }>
    const body = entry.body
    assert.deepStrictEqual(body.transforms, ['middle-out'])
  })

  it('adds cache-control metadata for Claude models', async () => {
    const handler = createHandler({ openRouterModelId: 'anthropic/claude-3.5-sonnet' })
    const messages: NeutralConversationHistory = [
      { role: 'user', content: 'message 1' },
      { role: 'assistant', content: 'response 1' },
      { role: 'user', content: 'message 2' },
    ]
    await drain(handler.createMessage('system', messages))
    const body = (mockOpenAI.getRequestLog()[0] as { body: { messages: Array<Record<string, unknown>> } }).body
    const system = body.messages.find(msg => msg.role === 'system') as { content: Array<Record<string, unknown>> }
    assert.ok(system)
    assert.ok(system.content.some(part => 'cache_control' in part))
  })

  it('propagates API errors from the mock server', async () => {
    mockOpenAI.setResponseOverride('chat_test-model', { error: { code: 500, message: 'API Error' } })
    const handler = createHandler()
    await assert.rejects(
      (async () => {
        for await (const chunk of handler.createMessage('system', [])) {
          void chunk
        }
      })(),
      (error: unknown) => error instanceof Error && error.message === 'OpenRouter API Error 500: API Error',
    )
  })

  it('returns completion text for non-streaming requests', async () => {
    const handler = createHandler()
    const result = await handler.completePrompt('prompt')
    assert.ok(typeof result === 'string')
    assert.ok(result.length > 0)
  })

  it('throws when completion response contains an error', async () => {
    mockOpenAI.setResponseOverride('chat_test-model', { error: { code: 429, message: 'Rate limit' } })
    const handler = createHandler()
    await assert.rejects(
      handler.completePrompt('prompt'),
      (error: unknown) => error instanceof Error && error.message === 'OpenRouter API Error 429: Rate limit',
    )
  })
})
