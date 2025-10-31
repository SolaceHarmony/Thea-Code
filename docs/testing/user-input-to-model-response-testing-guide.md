# User Input to Model Response Testing Guide

This document provides a comprehensive guide for testing the user input to model response flow in Thea Code, including patterns, best practices, and infrastructure usage.

## Overview

The user input to model response flow is a critical path in Thea Code that involves:

1. **User Input** → User message from webview or command
2. **Message Processing** → TheaTask processes and prepares message for API
3. **API Request** → Handler sends message to AI model
4. **Model Response** → Model returns text, tool calls, or errors
5. **Response Processing** → TheaTask handles response and updates UI
6. **Tool Execution** → If tool calls, execute and send results back
7. **Continuation** → Model processes tool results and continues

## Test Coverage Strategy

### Unit Tests (Fast, Isolated)

**Location**: `src/core/__tests__/user-input-to-response.test.ts`

**Purpose**: Test the API handler message flow in isolation using the OpenAI mock

**What to Test**:
- Simple user text messages
- Multi-turn conversations
- Tool calls and tool results
- Streaming responses
- Token usage tracking
- Error handling (API errors, rate limiting, invalid responses)
- System prompt handling
- Different content types

**Key Pattern**:
```typescript
import openaiSetup, { openAIMock } from "../../../test/openai-mock/setup"
import { openaiTeardown } from "../../../test/openai-mock/teardown"

beforeEach(async () => {
  await openaiTeardown()
  await openaiSetup()
  
  // Setup custom endpoint
  (openAIMock as any)!.addCustomEndpoint("POST", "/v1/chat/completions", 
    function (_uri: any, body: any) {
      return [200, {
        id: "test-id",
        choices: [{
          message: { role: "assistant", content: "Response" },
          finish_reason: "stop"
        }],
        usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 }
      }]
    }
  )
})

afterEach(async () => {
  await openaiTeardown()
})
```

### Integration Tests (Moderate Speed, Multiple Components)

**Location**: `src/core/__tests__/thea-task-message-flow.test.ts`

**Purpose**: Test complete conversation flows with history management and context

**What to Test**:
- Complete multi-turn task conversations
- Context maintenance across tool executions
- Error recovery in conversations
- Token accumulation across requests
- Message history order and persistence
- Conversation with mixed content types

### E2E Tests (Slow, Full System)

**Location**: `src/core/__e2e__/user-input-model-response.e2e.test.ts`

**Purpose**: Test with VS Code extension loaded in extension host

**What to Test** (when infrastructure is ready):
- Full user message to response flow through extension API
- Tool execution with real VS Code APIs (file operations, terminal commands)
- Streaming with webview updates
- Error handling with user notifications
- Multi-task scenarios
- Resource cleanup verification

**Current State**: 
- Basic extension activation tests ✅
- Command registration verification ✅
- Configuration access tests ✅
- Full flow tests ⚠️ (needs infrastructure)

## OpenAI Mock Infrastructure

### Setup and Teardown

```typescript
import openaiSetup, { openAIMock } from "../../../test/openai-mock/setup"
import { openaiTeardown } from "../../../test/openai-mock/teardown"

beforeEach(async () => {
  await openaiSetup()  // Starts nock interception
})

afterEach(async () => {
  await openaiTeardown()  // Cleans up nock
})
```

### Custom Endpoints

See the full documentation file for detailed examples of:
- Non-streaming responses
- Streaming responses
- Tool use responses
- Error responses
- Capturing request data
- Dynamic responses

## Testing Patterns

The guide includes comprehensive patterns for:
- Simple message exchange
- Multi-turn conversations
- Tool execution flow
- Error handling and retry
- Token usage tracking

## Test Checklist

### ✅ Unit Tests (Completed)
- [x] Simple user text messages
- [x] Multi-turn conversations
- [x] Tool calls and results
- [x] Error handling
- [x] Streaming responses
- [x] Token tracking

### ✅ Integration Tests (Completed)
- [x] Complete multi-turn conversations
- [x] Context maintenance
- [x] Error recovery
- [x] Message history management

### ⚠️ E2E Tests (Infrastructure Needed)
- [x] Extension activation
- [x] Command registration
- [ ] Full flow with webview (needs infrastructure)
- [ ] Tool execution with VS Code APIs (needs infrastructure)

## Running Tests

```bash
# Run all unit tests
npm run test:unit

# Run specific test file
NODE_ENV=test THEA_DISABLE_MCP_SDK=1 \
  npx mocha -r tsx -r src/test/mocha-global.ts \
  "src/core/__tests__/user-input-to-response.test.ts"

# Run E2E tests
npm run test:e2e
```

## Best Practices

1. Use OpenAI Mock for deterministic unit tests
2. Test error cases thoroughly
3. Test full conversations, not just single exchanges
4. Verify token tracking
5. Keep tests focused and independent
6. Clean up mocks properly
7. Document test patterns

## References

- OpenAI Mock: `test/openai-mock/`
- Provider Tests: `src/api/providers/__tests__/`
- TheaTask Tests: `src/core/__e2e__/TheaTask.e2e.test.ts`
- Test Checklist: `MASTER_TEST_CHECKLIST.md`

For complete examples and detailed patterns, see the full guide in this file.
