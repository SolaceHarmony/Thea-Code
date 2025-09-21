import * as assert from 'assert'
import * as sinon from 'sinon'

import { AwsBedrockHandler } from '../bedrock'
import type { ApiHandlerOptions } from '../../../shared/api'
import type { NeutralConversationHistory } from '../../../shared/neutral-history'
import type { ApiStreamChunk } from '../../transform/stream'
import { mockBedrock } from '../../../../test/generic-provider-mock/server'
import { BedrockRuntimeClient } from '@aws-sdk/client-bedrock-runtime'

const HOST = '127.0.0.1'

describe('AwsBedrockHandler', () => {
  let baseUrl: string
  let sendStub: sinon.SinonStub

  before(async () => {
    await mockBedrock.start()
    const port = mockBedrock.getPort()
    if (!port) throw new Error('mockBedrock did not expose a port')
    baseUrl = `http://${HOST}:${port}`
    process.env.AWS_BEDROCK_ENDPOINT = baseUrl
  })

  after(async () => {
    delete process.env.AWS_BEDROCK_ENDPOINT
    await mockBedrock.stop()
  })

  beforeEach(() => {
    sendStub = sinon.stub(BedrockRuntimeClient.prototype, 'send').resolves({ stream: undefined } as never)
  })

  afterEach(() => {
    sinon.restore()
    mockBedrock.clearRequestLog()
    mockBedrock.clearResponseOverrides()
  })

  function createHandler(extra?: Partial<ApiHandlerOptions>) {
    return new AwsBedrockHandler({
      apiModelId: 'anthropic.claude-v2',
      awsAccessKey: 'key',
      awsSecretKey: 'secret',
      awsRegion: 'us-east-1',
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

  it('converts neutral history to Bedrock format and streams response', async () => {
    const events = [
      { metadata: { usage: { inputTokens: 10, outputTokens: 5 } } },
      { contentBlockDelta: { delta: { text: 'Test response' } } },
    ]
    sendStub.resolves({ stream: events[Symbol.iterator]() } as never)

    const handler = createHandler()
    const history: NeutralConversationHistory = [{ role: 'user', content: [{ type: 'text', text: 'Hello' }] }]
    const chunks = await collect(handler.createMessage('system', history))
    assert.ok(chunks.some(chunk => chunk.type === 'text'))
  })

  it('delegates token counting to base provider when remote call fails', async () => {
    const handler = createHandler()
    const count = await handler.countTokens('Hello world')
    assert.ok(count >= 0)
  })

  it('completes prompts using mocked response', async () => {
    sendStub.resolves({ output: new TextEncoder().encode(JSON.stringify({ content: 'Test completion' })) } as never)
    const handler = createHandler()
    const result = await handler.completePrompt('Test prompt')
    assert.strictEqual(result, 'Test completion')
  })
})
