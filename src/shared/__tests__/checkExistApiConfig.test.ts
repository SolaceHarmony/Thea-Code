import assert from "node:assert/strict"
import { checkExistKey } from "../checkExistApiConfig"
import type { ProviderSettings } from "../../schemas"

describe("checkExistApiConfig", () => {
  it("returns false when config is undefined", () => {
    assert.strictEqual(checkExistKey(undefined), false)
  })

  it("returns true for human-relay and fake-ai regardless of keys", () => {
    assert.strictEqual(checkExistKey({ apiProvider: "human-relay" } as ProviderSettings), true)
    assert.strictEqual(checkExistKey({ apiProvider: "fake-ai" } as ProviderSettings), true)
  })

  it("returns true if any secret key exists", () => {
    const config = { apiProvider: "openrouter", openRouterApiKey: "sk-123" } as ProviderSettings
    assert.strictEqual(checkExistKey(config), true)
  })

  it("returns true if any non-secret config exists (awsRegion, vertexProjectId, etc.)", () => {
    const a: ProviderSettings = { apiProvider: "bedrock", awsRegion: "us-east-1" }
    const b: ProviderSettings = { apiProvider: "vertex", vertexProjectId: "proj" }
    const c: ProviderSettings = { apiProvider: "ollama", ollamaModelId: "llama3" }
    const d: ProviderSettings = { apiProvider: "lmstudio", lmStudioModelId: "foo" }
    const e: ProviderSettings = { apiProvider: "vscode-lm", vsCodeLmModelSelector: { vendor: "copilot" } }

    assert.strictEqual(checkExistKey(a), true)
    assert.strictEqual(checkExistKey(b), true)
    assert.strictEqual(checkExistKey(c), true)
    assert.strictEqual(checkExistKey(d), true)
    assert.strictEqual(checkExistKey(e), true)
  })

  it("returns false when no secret or non-secret config is set", () => {
    const config = { apiProvider: "openai" } as ProviderSettings
    assert.strictEqual(checkExistKey(config), false)
  })
})
