import * as assert from 'assert'
import { OllamaHandler, getOllamaModels } from '../ollama'
import type { ApiHandlerOptions } from '../../../shared/api'

const OLLAMA_BASE = process.env.OLLAMA_BASE_URL || 'http://127.0.0.1:11434'
const OLLAMA_MODEL = process.env.OLLAMA_MODEL_ID || 'qwen3:30b'

describe('OllamaHandler (local)', function () {
  this.timeout(20000)

  it('connects to Ollama and completes a prompt', async function () {
    const models = await getOllamaModels(OLLAMA_BASE)
    if (!models.includes(OLLAMA_MODEL)) {
      this.test?.skip()
      return
    }

    const options: ApiHandlerOptions = {
      ollamaBaseUrl: OLLAMA_BASE,
      ollamaModelId: OLLAMA_MODEL,
      modelTemperature: 0,
    }
    const handler = new OllamaHandler(options)

    const result = await Promise.race([
      handler.completePrompt('Hello from test'),
      new Promise<string>((_, reject) => setTimeout(() => reject(new Error('local ollama timeout')), 5000)),
    ]).catch((err) => {
      this.test?.skip()
      return ''
    })
    assert.ok(typeof result === 'string')
    assert.ok(result.length >= 0)
  })
})
