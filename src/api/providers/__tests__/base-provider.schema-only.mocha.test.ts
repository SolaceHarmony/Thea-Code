import * as assert from 'assert'
import * as sinon from 'sinon'

import { BaseProvider } from '../base-provider'
import { McpIntegration } from '../../../services/mcp/integration/McpIntegration'
import { ApiStream } from '../../transform/stream'
import type { ModelInfo } from '../../../shared/api'
import type { NeutralConversationHistory } from '../../../shared/neutral-history'

class TestProvider extends BaseProvider {
  createMessage(_systemPrompt: string, _messages: NeutralConversationHistory): ApiStream {
    throw new Error('not implemented')
  }

  getModel(): { id: string; info: ModelInfo } {
    return {
      id: 'test-model',
      info: {
        maxTokens: 100000,
        contextWindow: 100000,
        supportsImages: false,
        supportsPromptCache: false,
        inputPrice: 0,
        outputPrice: 0,
      },
    }
  }

  exposeRegisterTools() {
    this.registerTools()
  }
}

describe('BaseProvider tool registration', () => {
  let registerToolStub: sinon.SinonStub
  let routeToolUseStub: sinon.SinonStub
  let getInstanceStub: sinon.SinonStub

  beforeEach(() => {
    const mockIntegration = {
      initialize: sinon.stub().resolves(),
      registerTool: sinon.stub(),
      executeTool: sinon.stub(),
      routeToolUse: sinon.stub().resolves('ok'),
    } as unknown as McpIntegration & { registerTool: sinon.SinonStub; routeToolUse: sinon.SinonStub }

    registerToolStub = (mockIntegration.registerTool as unknown as sinon.SinonStub)
    routeToolUseStub = (mockIntegration.routeToolUse as unknown as sinon.SinonStub)

    getInstanceStub = sinon.stub(McpIntegration, 'getInstance').returns(mockIntegration)
  })

  afterEach(() => {
    sinon.restore()
  })

  it('registers the default tool schemas', () => {
    const provider = new TestProvider()
    provider.exposeRegisterTools()

    const names = registerToolStub.getCalls().map((call) => call.args[0].name)
    assert.deepStrictEqual(names, [
      'read_file',
      'write_to_file',
      'list_files',
      'search_files',
      'apply_diff',
      'insert_content',
      'search_and_replace',
      'ask_followup_question',
    ])
  })

  it('delegates tool use processing to MCP', async () => {
    const provider = new TestProvider()
    const result = await (provider as unknown as { processToolUse: (input: unknown) => Promise<unknown> }).processToolUse({ name: 'tool' })
    assert.strictEqual(result, 'ok')
    assert.strictEqual(routeToolUseStub.calledWithMatch({ name: 'tool' }), true)
  })

  it('initializes MCP integration once per provider instance', () => {
    void new TestProvider()
    assert.strictEqual(getInstanceStub.called, true)
  })
})
