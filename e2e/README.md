# Thea Code E2E Test Suite

## Overview

This directory contains the end-to-end (E2E) test suite for the Thea Code VSCode extension. These tests run in a real VSCode environment using the official `@vscode/test-electron` framework with Mocha as the test runner.

## Why E2E Tests?

- **Real Environment**: Tests run in actual VSCode, not mocked
- **Integration Testing**: Tests real interactions between components
- **User Perspective**: Tests features as users would experience them
- **VSCode Test Explorer**: Full integration with VSCode's built-in test runner
- **Clean & Simple**: No complex mocking or stubbing required

## Test Structure

```
e2e/
├── src/
│   ├── suite/
│   │   ├── basic.test.ts         # Basic extension functionality
│   │   ├── commands.test.ts      # Command registration and execution
│   │   ├── configuration.test.ts # Settings and configuration
│   │   ├── extension.test.ts     # Extension activation
│   │   ├── mcp.test.ts          # Model Context Protocol tests
│   │   ├── modes.test.ts        # Mode switching functionality
│   │   ├── providers.test.ts    # API provider tests
│   │   ├── webview.test.ts      # Webview panel tests
│   │   ├── task.test.ts         # Task execution
│   │   ├── subtasks.test.ts     # Subtask handling
│   │   └── utilities.test.ts    # Utility functions
│   ├── runTest.ts               # Test runner configuration
│   └── thea-constants.ts        # Extension constants
└── out/                         # Compiled JavaScript output
```

## Running Tests

### From Command Line

```bash
# Run all e2e tests
npm run test:e2e

# From the e2e directory
npm test

# Compile tests only
npm run compile

# Watch mode (compile on change)
npm run watch
```

### From VSCode

1. Open the Test Explorer (Testing icon in sidebar)
2. Navigate to the test you want to run
3. Click the play button to run individual tests or suites
4. Use the debug button to debug tests with breakpoints

### Using Launch Configuration

1. Press `F5` or go to Run and Debug
2. Select "Extension Tests (watch)" from the dropdown
3. Tests will run in a new VSCode window

## Writing Tests

### Test Structure

```typescript
import * as assert from "assert"
import * as vscode from "vscode"
import { EXTENSION_ID } from "../thea-constants"

suite("Feature Name", () => {
    let extension: vscode.Extension<any> | undefined

    suiteSetup(async function() {
        this.timeout(30000) // Allow time for extension activation
        extension = vscode.extensions.getExtension(EXTENSION_ID)
        if (!extension) {
            assert.fail("Extension not found")
        }
        if (!extension.isActive) {
            await extension.activate()
        }
    })

    test("should do something", async function() {
        this.timeout(10000) // Set timeout for async operations
        
        // Your test code here
        assert.ok(true, "Test passed")
    })
})
```

### Best Practices

1. **Use Descriptive Names**: Test names should clearly describe what they test
2. **Set Appropriate Timeouts**: Network operations need longer timeouts
3. **Clean Up After Tests**: Use `teardown()` to restore state
4. **Test User Workflows**: Focus on real user scenarios
5. **Avoid External Dependencies**: Don't rely on external services when possible
6. **Use Skip for WIP Tests**: Use `test.skip()` for tests under development

### Assertion Examples

```typescript
// Basic assertions
assert.ok(value, "Value should be truthy")
assert.strictEqual(actual, expected, "Values should be equal")
assert.deepStrictEqual(obj1, obj2, "Objects should be deeply equal")

// Pattern matching
assert.match(string, /pattern/, "String should match pattern")

// Error testing
assert.throws(() => dangerousFunction(), "Should throw error")
assert.rejects(async () => asyncFunction(), "Should reject promise")

// Array/Collection testing
assert.ok(array.includes(item), "Array should contain item")
assert.strictEqual(array.length, expected, "Array should have expected length")
```

## Test Categories

### Core Tests
- **basic.test.ts**: Extension presence and basic functionality
- **extension.test.ts**: Extension activation and lifecycle
- **commands.test.ts**: Command registration and execution

### Feature Tests
- **configuration.test.ts**: Settings management
- **modes.test.ts**: Mode switching (Ask, Edit, Code, etc.)
- **webview.test.ts**: Webview panel functionality
- **providers.test.ts**: API provider integration

### Integration Tests
- **mcp.test.ts**: Model Context Protocol integration
- **task.test.ts**: Task execution and management
- **subtasks.test.ts**: Subtask handling

### Utility Tests
- **utilities.test.ts**: Helper functions and utilities

## Environment Variables

Create a `.env.local` file in the e2e directory for test configuration:

```bash
# API Keys (optional, for integration tests)
OPENROUTER_API_KEY=sk-or-v1-...
ANTHROPIC_API_KEY=sk-ant-...

# Test Configuration
TEST_TIMEOUT=60000
SKIP_SLOW_TESTS=false
```

## Debugging Tests

1. Set breakpoints in your test files
2. Use VSCode's Test Explorer debug button
3. Or use the "Extension Tests (watch)" launch configuration
4. Check the Debug Console for output

## Migration from Jest

We're migrating from Jest to this E2E framework. Key differences:

| Jest | Mocha (E2E) |
|------|------------|
| `describe()` | `suite()` |
| `it()` | `test()` |
| `beforeEach()` | `setup()` |
| `afterEach()` | `teardown()` |
| `beforeAll()` | `suiteSetup()` |
| `afterAll()` | `suiteTeardown()` |
| `expect().toBe()` | `assert.strictEqual()` |
| `expect().toEqual()` | `assert.deepStrictEqual()` |
| `expect().toMatch()` | `assert.match()` |

## Continuous Integration

The E2E tests can be run in CI environments:

```yaml
# Example GitHub Actions
- name: Run E2E Tests
  run: |
    npm run build
    xvfb-run -a npm run test:e2e
```

## Troubleshooting

### Tests Not Running
- Ensure the extension compiles: `npm run build`
- Check for TypeScript errors: `npm run compile`
- Verify extension ID in `thea-constants.ts`

### Timeouts
- Increase timeout for slow operations
- Check network connectivity for API tests
- Ensure extension activates properly

### Debugging Tips
- Use `console.log()` for quick debugging
- Check the Output panel in VSCode
- Look at the Extension Host log

## Contributing

When adding new tests:
1. Choose the appropriate test file or create a new one
2. Follow the existing patterns and structure
3. Add descriptive test names and failure messages
4. Update this README if adding new test categories
5. Ensure tests pass locally before committing

## Future Improvements

- [ ] Add performance benchmarks
- [ ] Implement test coverage reporting
- [ ] Add visual regression tests for webview
- [ ] Create test data fixtures
- [ ] Add automated test generation for commands
- [ ] Implement parallel test execution
- [ ] Add test result reporting dashboard