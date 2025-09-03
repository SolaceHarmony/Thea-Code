# Test Migration Plan

## Overview
Migrating from Jest to VSCode Extension Test Framework (Mocha + @vscode/test-electron)

## Test Organization Structure

```
e2e/src/suite/
├── minimal.test.ts        ✅ Example minimal test
├── basic.test.ts          ✅ Basic extension tests  
├── extension.test.ts      ✅ Extension activation tests
├── utilities.test.ts      ✅ Utility function tests (NEW)
├── api/
│   └── providers.test.ts  ✅ API provider tests (NEW)
├── core/
│   ├── config.test.ts     🔄 Configuration tests (TODO)
│   ├── tools.test.ts      🔄 Tool tests (TODO)
│   └── webview.test.ts    🔄 Webview tests (TODO)
├── services/
│   ├── mcp.test.ts        🔄 MCP tests (TODO)
│   └── telemetry.test.ts  🔄 Telemetry tests (TODO)
└── integration/
    ├── modes.test.ts      ✅ Mode switching tests
    ├── task.test.ts       ✅ Task execution tests
    └── subtasks.test.ts   ✅ Subtask tests
```

## Migration Status

### ✅ Completed
- [x] getNonce utility test → utilities.test.ts
- [x] Basic test structure setup

### 🔄 In Progress
- [ ] API provider tests structure

### 📋 Priority Tests to Migrate

#### High Priority (Core Functionality)
1. [ ] Core configuration tests
2. [ ] Model registry tests
3. [ ] Provider handler tests (OpenAI, Anthropic, etc.)
4. [ ] MCP integration tests

#### Medium Priority (Features)
1. [ ] Tool tests (applyDiff, executeCommand, etc.)
2. [ ] Webview state management
3. [ ] Task history and stack
4. [ ] Custom modes

#### Low Priority (Utilities)
1. [ ] Logging tests
2. [ ] Path utilities
3. [ ] XML/JSON utilities

## Migration Guidelines

### Converting Jest to Mocha

#### Assertions
```javascript
// Jest
expect(value).toBe(expected)
expect(value).toEqual(expected)
expect(value).toMatch(/pattern/)
expect(fn).toThrow()

// Mocha with assert
assert.strictEqual(value, expected)
assert.deepStrictEqual(value, expected)
assert.match(value, /pattern/)
assert.throws(fn)
```

#### Test Structure
```javascript
// Jest
describe("Suite", () => {
  beforeEach(() => {})
  it("test", () => {})
})

// Mocha
suite("Suite", () => {
  setup(() => {})
  test("test", () => {})
})
```

#### Async Tests
```javascript
// Jest
it("async test", async () => {
  await someAsyncFunction()
})

// Mocha
test("async test", async function() {
  this.timeout(5000) // Set timeout if needed
  await someAsyncFunction()
})
```

### Best Practices

1. **Test Independence**: Each test should be independent and not rely on other tests
2. **Proper Cleanup**: Use `teardown()` or `suiteTeardown()` to clean up resources
3. **Timeouts**: Set appropriate timeouts for async operations
4. **Mocking**: Use stubs/spies sparingly; prefer testing real implementations where possible
5. **Categories**: Organize tests by functionality (api/, core/, services/, integration/)

## Running Tests

```bash
# Run all e2e tests
npm run test:e2e

# Run specific test file (from e2e directory)
npm test -- --grep "Utility Functions"

# Debug tests in VSCode
# Use "Extension Tests (watch)" launch configuration
```

## Notes

- The e2e tests run in a real VSCode instance, so they have access to all VSCode APIs
- Tests should avoid external dependencies (network calls, file system operations outside temp)
- Focus on testing public APIs and user-facing functionality
- Unit tests for pure functions can go in utilities.test.ts
- Integration tests that require extension context go in integration/

## Cleanup TODO

Once migration is complete:
- [ ] Remove all Jest test files from src/
- [ ] Remove Jest dependencies from package.json
- [ ] Remove Jest configuration files
- [ ] Update CI/CD to use new test commands
- [ ] Update documentation