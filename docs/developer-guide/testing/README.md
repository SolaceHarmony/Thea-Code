# Thea Code Testing Guide

**Status:** Published
**Last Updated:** 2025-08-10
**Category:** Developer Guide

## Overview

This guide provides comprehensive information about testing in Thea Code, including unit tests, integration tests, end-to-end tests, and benchmarks.

## Table of Contents

- [Test Infrastructure](#test-infrastructure)
- [Running Tests](#running-tests)
- [Test Coverage](#test-coverage)
- [Unit Testing](#unit-testing)
- [Integration Testing](#integration-testing)
- [End-to-End Testing](#end-to-end-testing)
- [Benchmark Testing](#benchmark-testing)
- [Writing Tests](#writing-tests)
- [Common Issues](#common-issues)

## Test Infrastructure

### Test Framework

Thea Code uses the following testing tools:
- **Jest** - Primary test runner for unit and integration tests
- **Mocha** - Used for VSCode extension integration tests
- **Playwright** - Browser automation testing
- **Custom benchmark harness** - For performance testing

### Directory Structure

```
tests/
├── src/__tests__/           # Unit tests (co-located with source)
├── e2e/                     # End-to-end VSCode integration tests
├── test/                    # Test utilities and mock servers
│   ├── generic-provider-mock/
│   ├── mcp-mock-server/
│   ├── openai-mock/
│   └── roo-migration/
└── benchmark/               # Performance benchmarks
```

## Running Tests

### All Tests
```bash
npm test
```

### Unit Tests Only
```bash
npm run test:unit
```

### Integration Tests
```bash
npm run test:integration
```

### End-to-End Tests
```bash
npm run test:e2e
```

### Coverage Report
```bash
npm run test:coverage
```

### Watch Mode
```bash
npm run test:watch
```

## Test Coverage

### Current Coverage Status

The project maintains a comprehensive test coverage checklist in `MASTER_TEST_CHECKLIST.md`, generated automatically by:

```bash
npm run generate:test-checklist
```

### Coverage Goals
- Overall coverage: >80%
- Critical paths: >95%
- New code: 100%

### Viewing Coverage Reports

After running coverage tests, view the HTML report:
```bash
open coverage/lcov-report/index.html
```

## Unit Testing

### Writing Unit Tests

Unit tests are co-located with source files using the `__tests__` directory pattern:

```typescript
// src/utils/__tests__/path.test.ts
import { arePathsEqual, formatPath } from '../path';

describe('Path Utilities', () => {
  describe('arePathsEqual', () => {
    it('should handle case-insensitive comparison on Windows', () => {
      // Test implementation
    });
  });
});
```

### Mocking

Common mocks are provided in `src/__mocks__/`:
- `vscode.js` - VSCode API mocks
- `@modelcontextprotocol/sdk` - MCP SDK mocks
- `fs/promises.ts` - File system mocks

### Best Practices

1. **Test naming**: Use descriptive names that explain what is being tested
2. **Arrange-Act-Assert**: Structure tests clearly
3. **One assertion per test**: Keep tests focused
4. **Mock external dependencies**: Isolate unit tests

## Integration Testing

### Provider Testing

Test API providers with mock servers:

```typescript
// src/api/providers/__tests__/ollama.test.ts
describe('Ollama Provider', () => {
  beforeAll(async () => {
    // Start mock server
    await startMockOllamaServer();
  });

  afterAll(async () => {
    // Cleanup
    await stopMockOllamaServer();
  });

  test('should handle streaming responses', async () => {
    // Test implementation
  });
});
```

### MCP Integration Testing

Test MCP (Model Context Protocol) integration:

```typescript
// src/services/mcp/__tests__/McpIntegration.test.ts
describe('MCP Integration', () => {
  test('should register and execute tools', async () => {
    // Test MCP tool registration and execution
  });
});
```

## End-to-End Testing

### VSCode Extension Tests

The `e2e/` directory contains full VSCode extension integration tests:

```typescript
// e2e/src/suite/extension.test.ts
suite('Extension Test Suite', () => {
  test('Extension should activate', async () => {
    const extension = vscode.extensions.getExtension('SolaceHarmony.thea-code');
    assert.ok(extension);
    await extension.activate();
  });
});
```

### Running E2E Tests

```bash
# Run VSCode integration tests
npm run test:e2e

# With specific VSCode version
npm run test:e2e -- --vscode-version 1.85.0
```

## Benchmark Testing

### Benchmark Harness

The benchmark suite tests performance across multiple languages:

```bash
# Build and start Docker environment
npm run docker:start

# Run specific benchmark
npm run docker:benchmark -- -e exercises/javascript/binary

# Run all benchmarks for a language
npm run cli -- run javascript all

# Run all benchmarks
npm run cli -- run all
```

### Supported Languages

- C++
- Go
- Java
- JavaScript
- Python
- Rust

### Writing Benchmarks

Create benchmark exercises in `benchmark/exercises/[language]/`:

```typescript
// benchmark/exercises/javascript/example.js
export const exercise = {
  name: 'Example Exercise',
  prompt: 'Implement a function that...',
  validate: (result) => {
    // Validation logic
    return result === expectedOutput;
  }
};
```

## Writing Tests

### Test Template

```typescript
import { describe, test, expect, beforeEach, afterEach } from '@jest/globals';

describe('ComponentName', () => {
  let component: ComponentType;

  beforeEach(() => {
    // Setup
    component = new ComponentType();
  });

  afterEach(() => {
    // Cleanup
    jest.clearAllMocks();
  });

  describe('methodName', () => {
    test('should handle normal case', () => {
      // Arrange
      const input = 'test';
      
      // Act
      const result = component.method(input);
      
      // Assert
      expect(result).toBe('expected');
    });

    test('should handle edge case', () => {
      // Test edge cases
    });

    test('should handle error case', () => {
      // Test error handling
      expect(() => component.method(null)).toThrow();
    });
  });
});
```

### Async Testing

```typescript
test('should handle async operations', async () => {
  const result = await asyncFunction();
  expect(result).toBeDefined();
});

test('should handle streaming', async () => {
  const stream = getStream();
  const chunks = [];
  
  for await (const chunk of stream) {
    chunks.push(chunk);
  }
  
  expect(chunks).toHaveLength(3);
});
```

## Common Issues

### Port Conflicts

Tests use dynamic port assignment to avoid conflicts:

```typescript
import { findAvailablePort } from '../utils/port-utils';

const port = await findAvailablePort(10000);
server.listen(port);
```

### Timeout Issues

Adjust timeouts for slow operations:

```typescript
test('long running operation', async () => {
  // Test implementation
}, 30000); // 30 second timeout
```

### Mock Server Issues

Ensure mock servers are properly started and stopped:

```typescript
beforeAll(async () => {
  await waitForPortAvailable(mockPort);
  await startMockServer();
  await waitForPortInUse(mockPort);
});

afterAll(async () => {
  await stopMockServer();
});
```

### Environment Variables

Set required environment variables for tests:

```bash
# .env.test
OPENROUTER_API_KEY=test-key
ANTHROPIC_API_KEY=test-key
```

## CI/CD Integration

### GitHub Actions

Tests run automatically on:
- Pull requests
- Push to main branch
- Release tags

### Pre-commit Hooks

Install pre-commit hooks:

```bash
npm run prepare
```

This runs tests before commits to catch issues early.

## Test Improvements

### Recent Improvements

1. **Port Management** - Dynamic port assignment to prevent conflicts
2. **Timeout Handling** - Proper timeout configuration for async operations
3. **Mock Servers** - Improved mock server setup and teardown
4. **Coverage Tracking** - Automated coverage reporting

### Planned Improvements

- [ ] Increase coverage to >90%
- [ ] Add performance regression tests
- [ ] Implement visual regression testing for webview
- [ ] Add mutation testing

## Related Documentation

- [Contributing Guide](../contributing.md)
- [Architecture Overview](../architecture/README.md)
- [API Reference](../api-reference/README.md)

## Support

For test-related issues:
1. Check the [Common Issues](#common-issues) section
2. Review existing [GitHub Issues](https://github.com/SolaceHarmony/Thea-Code/issues)
3. Ask in [Discord](https://discord.gg/EmberHarmony)

---

**Changelog:**
- 2025-08-10: Consolidated testing documentation from multiple sources