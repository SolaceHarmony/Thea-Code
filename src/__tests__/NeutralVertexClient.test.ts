import { strict as assert } from "assert";
import { NeutralVertexClient } from "../services/vertex/NeutralVertexClient";
import type {
  NeutralVertexClaudeResponse,
} from "../services/vertex/types";
import type { NeutralMessageContent } from "../shared/neutral-history";

describe("NeutralVertexClient", () => {
  const projectId = "proj-123";
  const region = "us-central1";
  let originalFetch: typeof global.fetch;
  let fetchCalls: Array<{ url: string; init: RequestInit }> = [];

  beforeEach(() => {
    fetchCalls = [];
    originalFetch = global.fetch;
    
    // Stub fetch to track calls and return test responses
    global.fetch = ((input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = input instanceof Request ? input.url : String(input);
      fetchCalls.push({ url, init: init ?? {} });
      
      // Return anthropic response
      const payload: NeutralVertexClaudeResponse = {
        content: [{ type: "text", text: "test response" }],
      };
      return Promise.resolve(new Response(JSON.stringify(payload), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }));
    }) as typeof global.fetch;
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  const makeClient = () => new NeutralVertexClient({ projectId, region });

  it("countTokens counts only text blocks", () => {
    const client = makeClient();
    const content1: NeutralMessageContent = [{ type: "text", text: "one two three" }];
    const content2: NeutralMessageContent = [
      { type: "text", text: "alpha beta gamma" },
      { type: "text", text: "delta" },
    ];
    const count1 = client.countTokens("any", content1);
    const count2 = client.countTokens("any", content2);
    
    assert(count1 > 0, "count1 should be greater than 0");
    assert(count2 > count1, "count2 should be greater than count1");
  });
});
