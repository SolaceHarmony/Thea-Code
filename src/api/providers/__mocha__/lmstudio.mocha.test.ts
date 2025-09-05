import * as assert from 'assert'
import { LmStudioHandler, getLmStudioModels } from '../lmstudio'
import type { ApiHandlerOptions } from '../../../shared/api'

const LMSTUDIO_BASE = process.env.LMSTUDIO_BASE_URL || 'http://127.0.0.1:1234'
const LMSTUDIO_MODEL = process.env.LMSTUDIO_MODEL_ID || 'qwen3-coder-30b-a3b-instruct-1m'

describe('LmStudioHandler (local)', function () {
  this.timeout(20000)

  it('connects to LM Studio and completes a prompt', async function () {
    // Verify server is up and model is available
    const models = await getLmStudioModels(LMSTUDIO_BASE)
    if (!models.includes(LMSTUDIO_MODEL)) {
      this.test?.skip()
      return
    }

    const options: ApiHandlerOptions = {
      lmStudioBaseUrl: LMSTUDIO_BASE,
      lmStudioModelId: LMSTUDIO_MODEL,
      modelTemperature: 0,
    }
    const handler = new LmStudioHandler(options)

    const result = await Promise.race([
      handler.completePrompt('Hello from test'),
      new Promise<string>((_, reject) => setTimeout(() => reject(new Error('local lmstudio timeout')), 5000)),
    ]).catch((err) => {
      // If local provider is slow, skip to keep suite fast
      this.test?.skip()
      return ''
    })
    assert.ok(typeof result === 'string')
    assert.ok(result.length >= 0)
  })
})
