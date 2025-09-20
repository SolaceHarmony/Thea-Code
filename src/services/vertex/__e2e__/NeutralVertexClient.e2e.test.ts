import * as assert from 'assert'
import * as sinon from 'sinon'
import * as proxyquire from 'proxyquire'
import type {
  NeutralVertexClaudeResponse,
  NeutralVertexGeminiResponse,
} from "../types";
import type { NeutralMessageContent } from "../../../shared/neutral-history";

// Mock google-auth-library to avoid real GCP auth, keep shapes minimal but correct
// Mock google-auth-library
const googleAuthMock = {
  GoogleAuth: class {
    getClient() {
      return Promise.resolve({
        getAccessToken: () =>
          Promise.resolve({
            token: "test-token",
            res: { data: { expiry_date: Date.now() + 60 * 60 * 1000 } },
          }),
      });
    }
  },
};

// Mock tiktoken lite to keep token counting predictable/lightweight in tests
// Mock tiktoken
const tiktokenMock = {
  Tiktoken: class {
    static specialTokenRegex = /<\|[^|]+\|>/g;
    // Accept config arg for compatibility with real constructor

    constructor(_config?: unknown) {}
    encode(text: string): number[] {
      return text.split(/\s+/).filter(Boolean).map((_, i) => i);
    }
  },
};

// Mock the o200k_base ranks (constructor arg); we don't need real data
// Mock o200k_base ranks
const o200kBaseMock = {};

suite("NeutralVertexClient", () => {
  const projectId = "proj-123";
  const region = "us-central1";

  let NeutralVertexClient: typeof import("../NeutralVertexClient").NeutralVertexClient;
  let fetchStub: sinon.SinonStub;
  let sandbox: sinon.SinonSandbox;

  setup(() => {
    sandbox = sinon.createSandbox();
    
    // Stub global fetch
    fetchStub = sandbox.stub(globalThis, "fetch");
    fetchStub.resolves(
      new Response(
        JSON.stringify({ content: [{ type: "text", text: "ok" }] }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      ) as unknown as Response
    );

    // Load NeutralVertexClient with mocked dependencies
    const module = proxyquire
      .noCallThru()('../NeutralVertexClient', {
        'google-auth-library': googleAuthMock,
        'js-tiktoken/lite': tiktokenMock,
        'js-tiktoken/ranks/o200k_base': o200kBaseMock,
      }) as unknown as typeof import('../NeutralVertexClient');
    
    NeutralVertexClient = module.NeutralVertexClient;
  });

  teardown(() => {
    sandbox.restore();
  });

  const makeClient = () => new NeutralVertexClient({ projectId, region });

  test("completeClaudePrompt returns text and calls correct endpoint with auth", async () => {
    const client = makeClient();
    const model = "claude-3-5-sonnet@20240620";

    const payload: NeutralVertexClaudeResponse = {
      content: [{ type: "text", text: "Hello Claude" }],
    };
    fetchStub.onFirstCall().resolves(
      new Response(JSON.stringify(payload), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }) as unknown as Response
    );

    const text = await client.completeClaudePrompt("Hi", model, 42, 0.1);
    assert.strictEqual(text, "Hello Claude");

    // Verify URL and headers/body
    assert.strictEqual(fetchStub.callCount, 1);
  const [url, init] = fetchStub.firstCall.args as [string, { headers?: Record<string, string>; body?: string }];
    assert.strictEqual(url, 
      `https://${region}-aiplatform.googleapis.com/v1/projects/${projectId}/locations/${region}/publishers/anthropic/models/${model}:generateContent`
    );
    const headers = init.headers ?? {};
    assert.strictEqual(headers["Authorization"], "Bearer test-token");
    assert.strictEqual(headers["Content-Type"], "application/json");

    type ClaudeRequestBody = {
      model: string;
      max_tokens?: number;
      temperature?: number;
      system?: unknown;
      messages?: unknown;
      stream: boolean;
    };
    const parsed: unknown = JSON.parse(init.body ?? "{}");
    const isClaudeRequest = (v: unknown): v is ClaudeRequestBody =>
      typeof v === "object" && v !== null &&
      typeof (v as Record<string, unknown>).model === "string" &&
      typeof (v as Record<string, unknown>).stream === "boolean";
    assert.strictEqual(isClaudeRequest(parsed), true);
    if (isClaudeRequest(parsed)) {
      assert.strictEqual(parsed.stream, false);
      assert.strictEqual(parsed.model, model);
      assert.strictEqual(parsed.max_tokens, 42);
      assert.strictEqual(parsed.temperature, 0.1);
    }
  });

  test("completeGeminiPrompt concatenates text parts and hits correct endpoint", async () => {
    const client = makeClient();
    const model = "gemini-1.5-pro";

    const payload: NeutralVertexGeminiResponse = {
      candidates: [
        {
          content: { parts: [{ text: "Hello " }, { text: "Gemini" }] },
        },
      ],
    };
    fetchStub.onFirstCall().resolves(
      new Response(JSON.stringify(payload), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }) as unknown as Response
    );

    const text = await client.completeGeminiPrompt("Hi", model, 128, 0.2);
    assert.strictEqual(text, "Hello Gemini");

    assert.strictEqual(fetchStub.callCount, 1);
  const [url, init] = fetchStub.firstCall.args as [string, { body?: string }];
    assert.strictEqual(url, 
      `https://${region}-aiplatform.googleapis.com/v1/projects/${projectId}/locations/${region}/publishers/google/models/${model}:generateContent`
    );
    type GeminiRequestBody = {
      model: string;
      contents?: unknown;
      generationConfig?: { maxOutputTokens?: number; temperature?: number };
      systemInstruction?: unknown;
      stream: boolean;
    };
    const parsed: unknown = JSON.parse(init.body ?? "{}");
    const isGeminiRequest = (v: unknown): v is GeminiRequestBody =>
      typeof v === "object" && v !== null &&
      typeof (v as Record<string, unknown>).model === "string" &&
      typeof (v as Record<string, unknown>).stream === "boolean";
    assert.strictEqual(isGeminiRequest(parsed), true);
    if (isGeminiRequest(parsed)) {
      assert.strictEqual(parsed.stream, false);
      assert.strictEqual(parsed.model, model);
      assert.strictEqual(parsed.generationConfig?.maxOutputTokens, 128);
      assert.strictEqual(parsed.generationConfig?.temperature, 0.2);
    }
  });

  test("countTokens counts only text blocks", () => {
    const client = makeClient();
    const content1: NeutralMessageContent = [{ type: "text", text: "one two three" }];
    const content2: NeutralMessageContent = [
      { type: "text", text: "alpha beta gamma" },
      { type: "text", text: "delta" },
    ];
    const count1 = client.countTokens("any", content1);
    const count2 = client.countTokens("any", content2);
    assert.ok(count1 > 0);
    assert.ok(count2 > count1);
  });

  test("completeClaudePrompt throws on HTTP error with status in message", async () => {
    const client = makeClient();
    const model = "claude-3-5-sonnet@20240620";

    fetchStub.onFirstCall().resolves(new Response("Server error", { status: 500 }) as unknown as Response);

    await assert.rejects(
      () => client.completeClaudePrompt("Hi", model),
      /Vertex AI Claude API error: 500/
    );
  });

  test("completeGeminiPrompt throws on HTTP error with status in message", async () => {
    const client = makeClient();
    const model = "gemini-1.5-pro";

    fetchStub.onFirstCall().resolves(new Response("Forbidden", { status: 403 }) as unknown as Response);

    await assert.rejects(
      () => client.completeGeminiPrompt("Hi", model),
      /Vertex AI Gemini API error: 403/
    );
  });
});
