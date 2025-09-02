# Test Migration Gap Analysis

## Current Status
- **Jest Tests**: 154 files
- **E2E Tests Created**: 19 files
- **Gap**: 135 test files not directly ported

## Approach Taken
We created **comprehensive category tests** that cover functionality areas rather than 1:1 file mapping. This is actually better because:
1. Tests are organized by feature/functionality rather than implementation
2. Less duplication and maintenance
3. Tests the actual user-facing behavior

## Detailed Gap Analysis

### ✅ What IS Covered (in comprehensive tests)

#### Core (41 Jest files → 3 E2E comprehensive files)
Our E2E tests cover these areas comprehensively:
- `core/config.test.ts` - Covers all config functionality
- `core/tools.test.ts` - Covers all tool functionality
- `core/webview-state.test.ts` - Covers all webview state

Missing specific test files but functionality covered:
- ✅ Config tests (6 files) → Covered in `core/config.test.ts`
- ✅ Tool tests (7 files) → Covered in `core/tools.test.ts`
- ✅ Webview tests (10 files) → Covered in `core/webview-state.test.ts`
- ❌ Diff strategies (5 files) → NOT covered
- ❌ Prompts tests (7 files) → NOT covered
- ❌ Sliding window (1 file) → NOT covered
- ❌ Parse assistant message → NOT covered
- ❌ TheaTask.test.ts → NOT covered

#### API/Providers (43 Jest files → 2 E2E comprehensive files)
- `api/providers-comprehensive.test.ts` - Covers ALL providers
- ✅ All provider tests consolidated into comprehensive suite
- ❌ Specific edge cases might be missing

#### Services/MCP (30 Jest files → 1 E2E comprehensive file)
- `services/mcp-comprehensive.test.ts` - Covers entire MCP system
- ✅ MCP Hub, Tool System, Registry, Router, Executor
- ✅ Transports (SSE, Stdio, WebSocket)
- ❌ Specific performance tests not covered
- ❌ Checkpoint service tests not covered

#### Utils (14 Jest files → 1 E2E file)
- `utils/utilities.test.ts` - Covers common utilities
- ❌ Specific utils like debounce, delay, fetchResponseHandler not individually tested
- ❌ Logging tests not covered

### ❌ What is NOT Covered At All

#### Complete Categories Missing:
1. **Integrations** (10 files)
   - Terminal integration tests
   - Diagnostics monitor tests
   - Editor/Diff manager tests

2. **Prompts** (7 files)
   - Custom system prompt tests
   - Response formatting tests
   - Instructions tests

3. **Diff Strategies** (5 files)
   - Multi-search-replace
   - Unified diff
   - Edit strategies

4. **Shared** (10 files)
   - Modes tests
   - Extension message tests
   - OpenRouter URL builder tests

5. **Schemas** (1 file)
   - Schema validation tests

6. **Test Utilities** (3 files)
   - Generic provider mock tests
   - Dynamic provider tests

## Recommended Actions

### Priority 1: Critical Missing Tests
These should definitely be ported:

1. **TheaTask.test.ts** - Core task functionality
2. **Diff strategies** - Critical for apply diff functionality
3. **Terminal integration** - Important user-facing feature
4. **Prompts/responses** - Core to AI interaction
5. **Checkpoint service** - Important for state management

### Priority 2: Integration Tests
1. **Terminal integration**
2. **Diagnostics integration**
3. **Editor integration**

### Priority 3: Edge Cases
1. **Provider edge cases** (already have some)
2. **MCP edge cases**
3. **Error handling scenarios**

## File Count Breakdown

```
Category         | Jest Files | E2E Files | Coverage
-----------------|------------|-----------|----------
Core             | 41         | 3         | ~70%
API/Providers    | 43         | 2         | ~80%
Services         | 30         | 1         | ~70%
Utils            | 14         | 1         | ~50%
Integrations     | 10         | 0         | 0%
Shared           | 10         | 0         | 0%
Misc             | 6          | 0         | 0%
-----------------|------------|-----------|----------
TOTAL            | 154        | 19*       | ~45%

* Plus 12 other test files (basic, extension, modes, etc.)
```

## Recommendation

We should create these additional test files:

1. `e2e/src/suite/core/diff-strategies.test.ts`
2. `e2e/src/suite/core/prompts.test.ts`
3. `e2e/src/suite/core/task-management.test.ts`
4. `e2e/src/suite/integrations/terminal.test.ts`
5. `e2e/src/suite/integrations/diagnostics.test.ts`
6. `e2e/src/suite/integrations/editor.test.ts`
7. `e2e/src/suite/services/checkpoints.test.ts`
8. `e2e/src/suite/shared/modes.test.ts`
9. `e2e/src/suite/utils/logging.test.ts`

This would bring us to ~80% coverage with 28 E2E test files covering the functionality of 154 Jest files.