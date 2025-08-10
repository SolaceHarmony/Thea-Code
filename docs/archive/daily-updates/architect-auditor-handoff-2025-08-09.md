# Architect/Auditor Handoff Plan — 2025-08-09

This plan is designed for execution without running code now. It aligns with the neutral, provider‑agnostic architecture and MCP‑first tool routing.

References:
- Findings and test plan: `docs/code-audit-findings-2025-08-09.md`
- Broader recommendations: `neutral_client_implementation_recommendations.md`

## Guiding principles
- Preserve neutral types and unified MCP routing end to end.
- Add tests before behavior changes; gate cleanup behind green CI and coverage thresholds.
- Prefer small, reviewable PRs with explicit acceptance criteria.

## Scope
- In: Coverage and resilience for MCP conversions/routing, transports, providers, and utilities; lifecycle/error path validation; gated SDK cleanup.
- Out: Large refactors, UI changes, unrelated features.

## Workstreams and acceptance criteria

1) Coverage baseline and mapping
- Deliver: Per‑file Jest+Mocha coverage, list of files <70%.
- Accept: Reports published in CI; prioritized list recorded in docs.

2) Converters and format detection (highest leverage)
- Files: McpConverters, json‑xml‑bridge, McpToolRouter.
- Tests: xml/json/openai→neutral conversions; mcpToXml escaping (text/image/url/base64/tool_use/tool_result/unknown); JsonMatcher buffer cap/partial chunks; FormatDetector heuristics.
- Accept: New tests pass; ≥85% lines/branches for these files; no regressions.

3) Router + executor lifecycle and round‑trips
- Files: McpToolRouter, McpToolExecutor, McpToolRegistry.
- Tests: detectFormat for XML/JSON/OpenAI/neutral; round‑trips preserve tool_use_id; init/shutdown repeats; pendingRegistrations flushed; events forwarded; neutral error envelopes for failures.
- Accept: Parity across formats; no listener/state leaks; ≥80% coverage.

4) Transports and embedded provider hardening
- Files: EmbeddedMcpProvider, SseTransport, StdioTransport.
- Tests: SSE dynamic import mocked; __JEST_TEARDOWN__ guard; getPort/underlying transport exposure; port=0 assigns/reassigns on restart; started/stopped events; Stdio fallback/mock stderr/handlers.
- Accept: Lifecycle tests stable; ≥80% coverage.

5) Provider conformance and edge cases
- Anthropic: thinking budget clamp; tool_use → tool_result; countTokens fallback logs.
- OpenAI: streaming/non‑streaming; XmlMatcher reasoning; tool_calls → MCP routing.
- Ollama: delegates tool detection to OpenAI handler; HybridMatcher reasoning; tool_result shapes.
- Vertex: Claude/Gemini via NeutralVertexClient; thinking variants; completePrompt helpers.
- Bedrock: ARN validation; cross‑region inference; creds modes; error/usage yield semantics.
- Glama: OpenAI‑compatible path; temperature gate; non‑streaming completion.
- Accept: 2–3 focused tests per provider key behaviors; ≥75% coverage of core logic paths.

6) Misc correctness
- BaseProvider registerTools schemas only; execution via MCP provider.
- attemptCompletionTool partial/final flows; approvals; telemetry/events; Neutral* filtering; feedback assembly.
- TheaTaskHistory IO; delete cleanup order; export path.
- port‑utils retries/timeouts; test‑mode fast path.
- Logging/i18n test guards.
- Accept: Documented intended flows; teardown clean; logs suppressed after teardown.

7) Coverage and CI policy
- Thresholds: 80% statements/branches for MCP core/transports/providers; 70% overall.
- CI: Run Mocha+Jest, publish HTML artifacts, fail on threshold regressions.
- Accept: Thresholds enforced; artifacts available in pipeline.

8) Dependency cleanup (gated)
- Confirm no live imports of vendor SDKs; remove `@anthropic-ai/sdk` once green; retain AWS SDK for Bedrock intentionally.
- Accept: Build/tests green post‑removal; docs/changelog updated to reflect NeutralAnthropicClient usage.

## Execution notes (deferred)
- Do not run commands now. When authorized, land tests per workstream in small PRs with: files changed, cov delta, risks, rollback.

## Risks and mitigations
- SSE lifecycle flakes → retries/guards, deterministic test ports, re‑randomization confirmed.
- Provider mocks drift → shared fixtures for tool_calls and neutral blocks; snapshots for critical shapes.
- Over‑fitting tests → assert public contracts, not internals.

## Communication cadence
- Async updates referencing the two docs above, including: completed items, coverage deltas, next two targets.

---

## Appendix A — Ground truth (verified from code, 2025-08-09)

These are concrete, per-file facts observed in the codebase (no execution). Useful for implementers to target tests and validate logic.

### MCP core
- src/services/mcp/core/McpToolRouter.ts
	- Singleton with getInstance(config?), forwards provider events (tool-registered/unregistered, started/stopped).
	- detectFormat(content) returns one of XML/JSON/OPENAI/NEUTRAL based on string/object heuristics (OpenAI when function_call/tool_calls present).
	- routeToolUse({ format, content }): convertToMcp → executeTool → convertFromMcp; on error, returns neutral error wrapped and converted back to original format.
	- convertToMcp: XML via McpConverters.xmlToMcp, JSON via jsonToMcp, OpenAI via openAiToMcp, NEUTRAL validates required fields (type/id/name/input).
	- convertFromMcp: returns XML/JSON/OpenAI/NEUTRAL via McpConverters counterparts; NEUTRAL passes through object.

- src/services/mcp/core/McpToolExecutor.ts
	- Singleton; initialize() creates EmbeddedMcpProvider via static create() and starts it; forwards started/stopped and tool events; wraps provider.emit to ensure forwarding in some test contexts.
	- Queues registrations when not initialized; flushes after start; unregister mirrors provider and registry.
	- executeToolFromNeutralFormat calls provider.executeTool(name, input) and returns neutral tool_result with status and optional error mapping.
	- getServerUrl() proxies provider.getServerUrl().

- src/services/mcp/core/McpToolRegistry.ts
	- In-memory registry with EventEmitter; register/unregister emit events; executeTool(name, args) invokes stored handler and wraps errors with tool name.

### Transports and embedded provider
- src/services/mcp/providers/EmbeddedMcpProvider.ts
	- Static create accepts SSE/stdio config; server constructed via @modelcontextprotocol/sdk/server/mcp.js McpServer (lazy import, error logged when missing).
	- start(): choose StdioTransport or SseTransport; start transport; register handlers for tools/resources/templates; connect server to underlying SDK transport; resolve port with retries; waitForPortInUse in non-test; sets serverUrl and emits started; stop() closes transport, clears state, re-randomizes port in tests.
	- Dynamic port (0) path uses findAvailablePort and ensures different port on restart in tests.

- src/services/mcp/transport/SseTransport.ts
	- Lazy imports @modelcontextprotocol/sdk/server/streamableHttp.js; __JEST_TEARDOWN__ short-circuit; wraps underlying transport; getUnderlyingTransport/getPort accessors; error logs suppressed in tests.

- src/services/mcp/transport/StdioTransport.ts
	- Imports @modelcontextprotocol/sdk/server/stdio.js when available; otherwise uses MockStdioServerTransport; exposes stderr and onerror/onclose bindings.

### Providers
- src/api/providers/openai.ts
	- Uses OpenAI/AzureOpenAI client; convertToOpenAiHistory adds system prompt when missing; supportsTemperature gate for o3-family special handling; streams with include_usage.
	- XmlMatcher emits reasoning from <think>; tool_calls gathered via shared/tool-use and executed via processToolUse, yielding tool_result chunks.

- src/api/providers/ollama.ts
	- Uses OpenAI client against Ollama base URL; delegates tool-call detection to OpenAI handler; HybridMatcher used solely for reasoning; tool_result yielded with stringified content as needed.

- src/api/providers/anthropic.ts
	- Uses NeutralAnthropicClient (fetch-based) with neutral history; thinking budget clamped; tool_use chunks routed via processToolUse; countTokens delegates to client with fallback logging.

- src/api/providers/vertex.ts
	- NeutralVertexClient handles Claude/Gemini; thinking variants normalized; tool_use routed; completePrompt helpers for both families.

- src/api/providers/bedrock.ts
	- AWS SDK client; validateBedrockArn() performs regex validation and region checks; cross-region inference prefixes model ID; yields usage and errors during ConverseStream.

- src/api/providers/glama.ts
	- Extends OpenAiCompatibleHandler; delegates createMessage; supportsTemperature via capabilities; non-streaming completePrompt implemented via neutral API.

### Shared utilities and contracts
- src/shared/neutral-history.ts
	- Defines text/image/image_url/image_base64/tool_use/tool_result blocks; tool_result links via tool_use_id; message role supports user/assistant/system/tool.

- src/api/providers/shared/tool-use.ts
	- extractToolCallsFromDelta() parses OpenAI delta for tool_calls and function_call; ToolCallAggregator collects multi-part tool call arguments; hasToolCalls() convenience.

- src/utils/json-xml-bridge.ts
	- JsonMatcher extracts JSON objects from streaming text with buffer cap; FormatDetector distinguishes json/xml/unknown; supports tool_use and thinking types.

- src/utils/port-utils.ts
	- findAvailablePort() with preferred ranges and retry count; waitForPortInUse() supports friendly logging labels; test-mode fast paths detected via JEST_WORKER_ID.

- src/utils/logging/index.ts
	- logger resolves to noop in Jest (JEST_WORKER_ID present) or CompactLogger otherwise.

- src/__mocks__/jest.setup.ts and test/globalTeardown.ts
	- Wrap console to suppress logs after teardown; sets __JEST_TEARDOWN__; orchestrates teardown of mock servers (mcp, ollama, openai).

---

## Appendix B — Contracts & I/O (signatures, behaviors)

All signatures based on source; names/types simplified for clarity. Errors return typed neutral error objects unless specified.

### McpToolRouter
- detectFormat(content: string | object): ToolUseFormat
	- Returns: XML | JSON | OPENAI | NEUTRAL via heuristics.
- routeToolUse(req: { format: ToolUseFormat; content: string | object }): { format; content }
	- Flow: convertToMcp → executeTool → convertFromMcp.
	- Errors: returns same format with NeutralToolResult { type: 'tool_result', status: 'error', tool_use_id: 'error', content:[{type:'text',text:msg}] }.
- convertToMcp(req): NeutralToolUseRequest
	- XML: McpConverters.xmlToMcp(string)
	- JSON: McpConverters.jsonToMcp(object|string)
	- OPENAI: McpConverters.openAiToMcp(object)
	- NEUTRAL: validates required fields (type,id,name,input) and casts.
- convertFromMcp(result, format): { format, content }
	- XML/JSON/OPENAI via McpConverters; NEUTRAL passes result object.

### McpToolExecutor
- initialize(): Promise<void>
	- Side effects: creates EmbeddedMcpProvider, starts, wires events, flushes pending registrations.
- shutdown(): Promise<void>
	- Stops provider; resets isInitialized.
- registerTool(def: ToolDefinition): void
	- Before init: queues and triggers initialize(); After: registers with provider + registry.
- unregisterTool(name: string): boolean
	- Unregisters from provider and registry; returns true only if both succeed.
- executeToolFromNeutralFormat(req: NeutralToolUseRequest): Promise<NeutralToolResult>
	- Success: { type:'tool_result', tool_use_id:req.id, content, status:'success' }
	- Error: same with status:'error' and error.message.
- getServerUrl(): URL | undefined

### EmbeddedMcpProvider (high-level)
- static create(options): Promise<EmbeddedMcpProvider>
	- Options: { type:'sse', config?: SseTransportConfig } | { type:'stdio', config: StdioTransportConfig } | SseTransportConfig
- start(): Promise<void>
	- Creates transport, start(), registerHandlers(), connect(server↔transport), determine port (SSE), waitForPortInUse (non-test), set serverUrl, emit 'started'.
- stop(): Promise<void>
	- Close transport, clear serverUrl/state, emit 'stopped'.
- registerToolDefinition(def), unregisterTool(name)
	- Emits tool-registered/unregistered.
- getServerUrl(): URL | undefined

### SseTransport
- start(): Promise<void>
	- Lazy imports StreamableHTTPServerTransport; guards on __JEST_TEARDOWN__.
- close(): Promise<void>
- getUnderlyingTransport(): unknown | undefined
- getPort(): number | undefined
- onerror/onclose setters: forward to underlying transport.

### StdioTransport
- start(): Promise<void>
	- Uses StdioServerTransport if available, else mock.
- close(): Promise<void>
- stderr getter; onerror/onclose setters.

### Converters (via McpConverters)
- xmlToMcp(xml: string): NeutralToolUseRequest
- jsonToMcp(data: object|string): NeutralToolUseRequest
- openAiToMcp(obj: object): NeutralToolUseRequest
- mcpToXml(result: NeutralToolResult): string
- mcpToJson(result: NeutralToolResult): object
- mcpToOpenAi(result: NeutralToolResult): object
	- Error mapping: error.status/message preserved in content/error fields where applicable.

### Provider tool-use contract (via BaseProvider.processToolUse)
- Input: tool_use { id, name, input }
- Behavior: routes through MCP integration; returns tool_result content used to emit streaming chunk { type:'tool_result', id, content } in providers.

### OpenAI tool-call utilities
- extractToolCallsFromDelta(delta): Array<{ id, function: { name, arguments } }>
- hasToolCalls(delta): boolean
- ToolCallAggregator: accumulates partial arguments per tool_call id.

### Neutral clients
- NeutralAnthropicClient
	- createMessage({ model, systemPrompt, messages, maxTokens, temperature, thinking }): AsyncIterable<ApiStreamChunk>
		- Emits: usage, text, reasoning, tool_use chunks; '[DONE]' terminates.
	- countTokens(model, content): Promise<number>
- NeutralVertexClient (inferred from usages)
	- createClaudeMessage/createGeminiMessage; completeClaudePrompt/completeGeminiPrompt.

### Utilities
- port-utils
	- findAvailablePort(start, host, ranges, retries, silent?): Promise<number>
	- waitForPortInUse(port, host, retryMs, timeoutMs, label?, attempts?): Promise<void>
- logging
	- logger: noop in Jest; CompactLogger otherwise.

---

## Appendix C — Generic mock server contract (provider-compatible)

Goal: a single mock service that emulates key behaviors across providers for integration tests without hitting real APIs. Shapes align with current code paths to maximize fidelity.

### Common
- Base behavior
	- Auth: accept any static bearer key; echo model in response.
	- Determinism: respond with fixed text/usage unless parameters opt-in to variants.
	- Latency: configurable via header `X-Mock-Latency: ms`.

### OpenAI-compatible (OpenAI, OpenRouter, Glama, Ollama via /v1)
- Endpoints
	- POST /v1/chat/completions
	- POST /v1/models (optional list)
- Request (subset)
	- { model: string, messages: [{role, content}], temperature?: number, stream?: boolean, tools?: [...], tool_choice?: ... }
- Response (non-stream)
	- { id, object:"chat.completion", model, choices:[{ index:0, message:{ role:"assistant", content:string, tool_calls?:[...] }, finish_reason:"stop" }], usage:{ prompt_tokens, completion_tokens, total_tokens } }
- Streaming
	- Server-Sent Events lines starting with "data: {json}\n" and terminating with "data: [DONE]\n".
	- Emit sequence: usage chunk (if desired), delta chunks with {choices:[{delta:{content?:string, tool_calls?:[{id,type:"function",function:{name,arguments}}]}}]}.
	- Tool calls: split arguments across multiple deltas per OpenAI behavior; ensure ids stable.
- Error
	- 401/403 on missing/invalid key; 400 for bad payload; 429 retry-after; 500 generic.

### Anthropic-compatible (NeutralAnthropicClient)
- Endpoints
	- POST /v1/messages (streaming) — SSE
	- POST /v1/messages/count_tokens
- Headers
	- x-api-key; anthropic-version: 2023-06-01
- Request (subset)
	- { model, system?: [{type:"text",text,cache_control?}], messages:[{role,content:[...]}], max_tokens, temperature?, thinking? }
- Streaming events (each line: `data: {json}`)
	- message_start: { message: { usage: { input_tokens, output_tokens, cache_creation_input_tokens?, cache_read_input_tokens? } } }
	- content_block_start: { content_block: { type:"text", text } | { type:"thinking", thinking } | { type:"tool_use", id,name,input } }
	- content_block_delta: { delta: { type:"text_delta", text } | { type:"thinking_delta", thinking } }
	- message_delta: { usage: { output_tokens } }
	- message_stop (optional)
	- [DONE]
- Count tokens
	- Request: { model, messages:[{ role:"user", content:[...] }] }
	- Response: { input_tokens: number }
- Error
	- 4xx/5xx with text body.

### Vertex-compatible (Claude/Gemini)
- Endpoints (emulated neutral client surface)
	- POST /vertex/claude/messages (streaming like Anthropic above)
	- POST /vertex/gemini/messages (streaming with OpenAI-like deltas or unified neutral chunks)
- Requirements
	- Accept neutral history; emit tool_use chunks and text/reasoning; usage optional.

### Bedrock-compatible
- Endpoints (approximate)
	- POST /bedrock/converse (non-stream) and /bedrock/converse/stream (chunked JSON events)
- Stream events (subset)
	- messageStart, contentBlockStart {start:{text}}, contentBlockDelta {delta:{text}}, metadata { usage:{inputTokens,outputTokens} }, messageStop { stopReason }
- Error
	- Invalid ARN or region mismatch simulated via input fields; include warning text in response; allow success path with cross-region prefixing.

### Tool calls and MCP integration hooks
- When tools present in input, mock should emit tool_calls (OpenAI) or tool_use blocks (Anthropic/neutral) early in stream.
- After tool_result (client-side), mock may continue with regular text content to simulate resumed generation.

### Config toggles (via headers or model suffix)
- Reasoning on/off; emit <think> or thinking deltas.
- Force tool_call vs plain text.
- Inject error at Nth chunk; simulate 429 with Retry-After.
- Prompt cache counters: include cache_creation_input_tokens/cache_read_input_tokens.

