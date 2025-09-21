import * as assert from 'assert'
import * as sinon from 'sinon'

import { AwsBedrockHandler, StreamEvent } from '../bedrock'
import type { ApiHandlerOptions } from '../../../shared/api'
import type { ApiStreamChunk } from '../../transform/stream'
import { BedrockRuntimeClient } from '@aws-sdk/client-bedrock-runtime'
import { mockBedrock } from '../../../../test/generic-provider-mock/server'

const HOST = '127.0.0.1'

function asStream(events: StreamEvent[]): AsyncIterable<StreamEvent> {
  let index = 0
  return {
    [Symbol.asyncIterator]() {
      return {
        next(): Promise<IteratorResult<StreamEvent>> {
          if (index < events.length) {
            return Promise.resolve({ value: events[index++], done: false })
          }
          return Promise.resolve({ value: undefined as unknown as StreamEvent, done: true })
        },
      }
    },
  }
}

describe('AwsBedrockHandler invokedModelId behaviour', () => {
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
      apiModelId: 'anthropic.claude-3-5-sonnet-20241022-v2:0',
      awsAccessKey: 'key',
      awsSecretKey: 'secret',
      awsRegion: 'us-east-1',
      ...extra,
    })
  }

 it('updates cost model when invokedModelId is present', async () => {
   const handler = createHandler({
     awsCustomArn: 'arn:aws:bedrock:us-west-2:699475926481:default-prompt-router/anthropic.claude:1',
   })

   sendStub.resolves({
     stream: asStream([
       {
         trace: {
           promptRouter: {
             invokedModelId: 'arn:aws:bedrock:us-west-2:699475926481:inference-profile/us.anthropic.claude-3-5-sonnet-20240620-v1:0',
           },
         },
       },
       { metadata: { usage: { inputTokens: 100, outputTokens: 200 } } },
     ]),
   } as never)

    const chunks: ApiStreamChunk[] = []
    for await (const chunk of handler.createMessage('system', [{ role: 'user', content: 'hello' }])) {
      chunks.push(chunk)
    }

    const updatedModel = handler.getModel()
    assert.strictEqual(updatedModel.id, 'anthropic.claude-3-5-sonnet-20240620-v1:0')
    assert.ok(chunks.some((chunk) => chunk.type === 'usage'))
  })

  it('leaves cost model untouched when invokedModelId is absent', async () => {
    const handler = createHandler()
    sendStub.resolves({
      stream: asStream([
        { contentBlockDelta: { delta: { text: 'Hello' } } },
        { metadata: { usage: { inputTokens: 10, outputTokens: 5 } } },
      ]),
    } as never)

    for await (const chunk of handler.createMessage('system', [{ role: 'user', content: 'hi' }])) {
      void chunk
    }

    const model = handler.getModel()
    assert.strictEqual(model.id, 'anthropic.claude-3-5-sonnet-20241022-v2:0')
  })
})
