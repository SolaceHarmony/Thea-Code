import { expect } from "chai"
import { checkExistKey } from "../checkExistApiConfig"
import type { ProviderSettings } from "../../schemas"

describe("checkExistApiConfig", () => {
  it("returns false when config is undefined", () => {
    expect(checkExistKey(undefined)).to.equal(false)
  })

  it("returns true for human-relay and fake-ai regardless of keys", () => {
    expect(checkExistKey({ apiProvider: "human-relay" } as ProviderSettings)).to.equal(true)
    expect(checkExistKey({ apiProvider: "fake-ai" } as ProviderSettings)).to.equal(true)
  })

  it("returns true if any secret key exists", () => {
    const config = { apiProvider: "openrouter", openRouterApiKey: "sk-123" } as ProviderSettings
    expect(checkExistKey(config)).to.equal(true)
  })

  it("returns true if any non-secret config exists (awsRegion, vertexProjectId, etc.)", () => {
    const a: ProviderSettings = { apiProvider: "bedrock", awsRegion: "us-east-1" }
    const b: ProviderSettings = { apiProvider: "vertex", vertexProjectId: "proj" }
    const c: ProviderSettings = { apiProvider: "ollama", ollamaModelId: "llama3" }
    const d: ProviderSettings = { apiProvider: "lmstudio", lmStudioModelId: "foo" }
    const e: ProviderSettings = { apiProvider: "vscode-lm", vsCodeLmModelSelector: { vendor: "copilot" } }

    expect(checkExistKey(a)).to.equal(true)
    expect(checkExistKey(b)).to.equal(true)
    expect(checkExistKey(c)).to.equal(true)
    expect(checkExistKey(d)).to.equal(true)
    expect(checkExistKey(e)).to.equal(true)
  })

  it("returns false when no secret or non-secret config is set", () => {
    const config = { apiProvider: "openai" } as ProviderSettings
    expect(checkExistKey(config)).to.equal(false)
  })
})
