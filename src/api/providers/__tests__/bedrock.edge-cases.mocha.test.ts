import * as assert from 'assert'
import * as sinon from 'sinon'

import { AwsBedrockHandler } from '../bedrock'
import type { ApiHandlerOptions } from '../../../shared/api'
import type { ApiStreamChunk } from '../../transform/stream'
import type { NeutralConversationHistory } from '../../../shared/neutral-history'
import { mockBedrock } from '../../../../test/generic-provider-mock/server'
import { BedrockRuntimeClient } from '@aws-sdk/client-bedrock-runtime'

const HOST = '127.0.0.1'

function asStream(events: Array<Record<string, unknown>>): AsyncIterable<Record<string, unknown>> {
  let index = 0
  return {
    [Symbol.asyncIterator]() {
      return {
        next(): Promise<IteratorResult<Record<string, unknown>>> {
          if (index < events.length) {
            return Promise.resolve({ value: events[index++], done: false })
          }
          return Promise.resolve({ value: undefined as unknown as Record<string, unknown>, done: true })
        },
      }
    },
  }
}

describe('AwsBedrockHandler edge cases', () => {
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
    sendStub = sinon.stub(BedrockRuntimeClient.prototype, 'send').resolves({ stream: asStream([]) } as never)
  })

  afterEach(() => {
    sinon.restore()
    mockBedrock.clearRequestLog()
    mockBedrock.clearResponseOverrides()
  })

  function createHandler(extra?: Partial<ApiHandlerOptions>) {
    return new AwsBedrockHandler({
      apiModelId: 'anthropic.claude-3-sonnet-20240229-v1:0',
      awsRegion: 'us-east-1',
      ...extra,
    })
  }

  it('validates custom ARNs', () => {
    const handler = createHandler({ awsCustomArn: 'arn:aws:bedrock:us-east-1:123456789012:inference-profile/anthropic.claude-3-sonnet-20240229-v1:0' })
    assert.strictEqual(handler.getModel().id, 'arn:aws:bedrock:us-east-1:123456789012:inference-profile/anthropic.claude-3-sonnet-20240229-v1:0')
  })

  it('streams usage information from the mock server', async () => {
    const events = [
      { metadata: { usage: { inputTokens: 10, outputTokens: 5 } } },
      { contentBlockDelta: { delta: { text: 'Response' } } },
      { messageStop: {} },
    ]
    sendStub.resolves({ stream: asStream(events) } as never)

    const handler = createHandler()
    const history: NeutralConversationHistory = [{ role: 'user', content: [{ type: 'text', text: 'Hello' }] }]
    const chunks: ApiStreamChunk[] = []
    for await (const chunk of handler.createMessage('system', history)) {
      chunks.push(chunk)
    }
    assert.ok(chunks.some((chunk) => chunk.type === 'usage'))
  })

  it('handles throttling errors by surfacing the exception', async () => {
    const throttled = new Error('ThrottlingException')
    throttled.name = 'ThrottlingException'
    sendStub.rejects(throttled)

    const handler = createHandler()
    const history: NeutralConversationHistory = [{ role: 'user', content: [{ type: 'text', text: 'Hi' }] }]
    const stream = handler.createMessage('system', history)
    await assert.rejects(async () => {
      for await (const chunk of stream) {
        void chunk
      }
    })
  })

  it('supports various credentials options', () => {
    const direct = createHandler({ awsAccessKey: 'key', awsSecretKey: 'secret' })
    assert.ok(direct)
    const session = createHandler({ awsAccessKey: 'key', awsSecretKey: 'secret', awsSessionToken: 'token' })
    assert.ok(session)
  })
})
