import * as assert from 'assert'
import * as sinon from 'sinon'

import { AwsBedrockHandler } from '../bedrock'
import type { ApiHandlerOptions } from '../../../shared/api'
import type { ApiStreamChunk } from '../../transform/stream'
import { mockBedrock } from '../../../../test/generic-provider-mock/server'
import type { ConverseCommandInput, ConverseCommandOutput } from '@aws-sdk/client-bedrock-runtime'
import { BedrockRuntimeClient } from '@aws-sdk/client-bedrock-runtime'

const HOST = '127.0.0.1'

const customArn = 'arn:aws:bedrock:us-east-1:123456789012:foundation-model/anthropic.claude-3-sonnet-20240229-v1:0'

const baseOptions: ApiHandlerOptions = {
  apiModelId: 'custom-arn',
  awsCustomArn: customArn,
  awsRegion: 'us-west-2',
}

describe('AwsBedrockHandler with custom ARN', () => {
  let baseUrl: string
  let sendStub: sinon.SinonStub<[unknown], Promise<ConverseCommandOutput>>

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
    sendStub = sinon.stub(BedrockRuntimeClient.prototype, 'send').resolves({
      output: new TextEncoder().encode(JSON.stringify({ content: 'Test response' })),
    } as ConverseCommandOutput)
  })

  afterEach(() => {
    sinon.restore()
    mockBedrock.clearRequestLog()
    mockBedrock.clearResponseOverrides()
  })

  function createHandler(extra?: Partial<ApiHandlerOptions>) {
    return new AwsBedrockHandler({
      ...baseOptions,
      awsRegion: baseOptions.awsRegion,
      ...extra,
    })
  }

  it('uses the custom ARN as model id', () => {
    const handler = createHandler()
    const model = handler.getModel()
    assert.strictEqual(model.id, customArn)
    assert.ok(model.info.contextWindow)
  })

  it('extracts region from ARN and overrides provided region', () => {
    const handler = createHandler()
    const clientRegion = (handler as unknown as { client: { config: { region?: string } } }).client.config.region
    assert.strictEqual(clientRegion, 'us-east-1')
  })

  it('validates ARN format and surfaces errors', async () => {
    const handler = createHandler({ awsCustomArn: 'invalid-arn-format' })
    await assert.rejects(
      handler.completePrompt('test'),
      (error: unknown) => error instanceof Error && error.message.includes('Invalid ARN format'),
    )
  })

  it('completes a prompt using mocked response', async () => {
    const handler = createHandler()
    const response = await handler.completePrompt('test prompt')
    assert.strictEqual(response, 'Test response')

    // Ensure the SDK command was invoked with expected payload
    assert.strictEqual(sendStub.called, true)
    const input = sendStub.firstCall.args[0] as { input: ConverseCommandInput }
    assert.strictEqual((input.input as ConverseCommandInput).modelId, customArn)
  })
})
