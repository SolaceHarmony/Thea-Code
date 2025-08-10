# Thea Code Developer Guide

**Status:** Published
**Last Updated:** 2025-08-10
**Category:** Developer Guide

## Overview

This guide provides comprehensive documentation for developers contributing to Thea Code or extending its functionality.

## Table of Contents

### Getting Started
- [Contributing Guide](./contributing.md) - How to contribute to Thea Code
- [Development Setup](#development-setup) - Set up your development environment
- [Project Structure](#project-structure) - Understanding the codebase

### Architecture
- [Architecture Overview](./architecture/README.md) - System design and components
- [Neutral Client Architecture](./architecture/neutral-client.md) - Provider abstraction layer
- [API Handlers](./architecture/api-handlers.md) - API integration patterns
- [MCP System](./architecture/mcp-system.md) - Model Context Protocol implementation
- [Webview Architecture](./architecture/webview.md) - UI and state management

### API Reference
- [Extension API](./api-reference/README.md) - Public API for extensions
- [Provider API](./api-reference/providers.md) - Creating custom providers
- [Tool API](./api-reference/tools.md) - Implementing custom tools
- [Types & Interfaces](./api-reference/types.md) - TypeScript definitions

### Testing
- [Testing Guide](./testing/README.md) - Comprehensive testing documentation
- [Unit Tests](./testing/unit-tests.md) - Writing and running unit tests
- [Integration Tests](./testing/integration-tests.md) - Provider and system testing
- [Benchmarks](./testing/benchmarks.md) - Performance testing

### Migration
- [Migration Guide](./migration/README.md) - Upgrading from older versions
- [Dynamic Models Migration](./migration/dynamic-models.md) - Model system updates

## Development Setup

### Prerequisites

- Node.js 18+ and npm 9+
- VSCode 1.85.0+
- Git
- Docker (for benchmarks)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/SolaceHarmony/Thea-Code.git
cd Thea-Code
```

2. **Install dependencies**
```bash
npm install
```

3. **Build the extension**
```bash
npm run build
```

4. **Run in development mode**
```bash
npm run watch
```

5. **Launch VSCode with extension**
Press `F5` in VSCode or:
```bash
code --extensionDevelopmentPath=.
```

### Development Workflow

1. **Create a feature branch**
```bash
git checkout -b feature/your-feature
```

2. **Make changes and test**
```bash
npm run test
npm run lint
```

3. **Build and verify**
```bash
npm run build
npm run package
```

4. **Submit pull request**
- Follow PR template
- Ensure tests pass
- Update documentation

## Project Structure

```
Thea-Code/
├── src/                    # Source code
│   ├── api/               # API providers and handlers
│   ├── core/              # Core functionality
│   ├── services/          # Service layer (MCP, browser, etc.)
│   ├── integrations/      # VSCode integrations
│   ├── shared/            # Shared utilities
│   └── extension.ts       # Extension entry point
├── webview-ui/            # React webview application
│   ├── src/
│   │   ├── components/    # UI components
│   │   ├── context/       # React context
│   │   └── App.tsx        # Main app component
├── test/                  # Test infrastructure
│   ├── generic-provider-mock/
│   ├── mcp-mock-server/
│   └── openai-mock/
├── docs/                  # Documentation
├── benchmark/             # Performance benchmarks
└── package.json          # Project configuration
```

### Key Components

#### API Layer (`src/api/`)
- **BaseProvider** - Abstract provider class
- **Provider implementations** - Anthropic, OpenAI, Ollama, etc.
- **Transform utilities** - Message format conversion

#### Core System (`src/core/`)
- **TheaTask** - Task management
- **Tool system** - Tool definitions and execution
- **Prompt system** - System prompts and instructions
- **Configuration** - Settings and mode management

#### Services (`src/services/`)
- **MCP** - Model Context Protocol implementation
- **Browser** - Browser automation
- **Terminal** - Terminal integration
- **Checkpoints** - State management

#### UI Layer (`webview-ui/`)
- **React components** - Chat, settings, history
- **State management** - Extension state context
- **Communication** - Message passing with extension

## Development Guidelines

### Code Style

- **TypeScript** - Use TypeScript for all new code
- **ESLint** - Follow project ESLint configuration
- **Prettier** - Auto-format with Prettier
- **Naming** - Use descriptive, consistent naming

### Best Practices

1. **Type Safety**
```typescript
// ✅ Good - Explicit types
interface ToolResult {
  success: boolean;
  output?: string;
  error?: Error;
}

// ❌ Bad - Any type
function processTool(result: any) { }
```

2. **Error Handling**
```typescript
// ✅ Good - Proper error handling
try {
  const result = await riskyOperation();
  return { success: true, data: result };
} catch (error) {
  logger.error('Operation failed', error);
  return { success: false, error };
}
```

3. **Async/Await**
```typescript
// ✅ Good - Clean async/await
const data = await fetchData();
const processed = await processData(data);

// ❌ Bad - Callback hell
fetchData((data) => {
  processData(data, (processed) => {
    // ...
  });
});
```

### Testing Requirements

- **Unit tests** for all utilities and pure functions
- **Integration tests** for API providers
- **E2E tests** for critical user flows
- **Minimum 80% code coverage**

### Documentation Requirements

- **JSDoc comments** for public APIs
- **README files** for new features
- **Update existing docs** when changing behavior
- **Include examples** in documentation

## Common Development Tasks

### Adding a New Provider

1. Create provider class extending `BaseProvider`
2. Implement required methods
3. Add tests in `__tests__` directory
4. Update provider factory
5. Document in user guide

See [Provider Implementation Guide](./architecture/api-handlers.md)

### Adding a New Tool

1. Define tool in `src/core/tools/`
2. Add tool schema
3. Implement tool handler
4. Register in tool system
5. Add tests

See [Tool Implementation Guide](./api-reference/tools.md)

### Modifying the UI

1. Edit components in `webview-ui/src/components/`
2. Update styles if needed
3. Test in different themes
4. Ensure accessibility

See [UI Development Guide](./architecture/webview.md)

## Debugging

### Extension Debugging

1. Set breakpoints in VSCode
2. Press `F5` to launch debug session
3. Use Debug Console for output
4. Check Extension Host logs

### Webview Debugging

1. Open Developer Tools: `Ctrl/Cmd + Shift + P` → "Developer: Toggle Developer Tools"
2. Navigate to Console tab
3. Use React DevTools if installed

### Common Issues

- **Module not found** - Run `npm install`
- **Build errors** - Check TypeScript errors with `npm run typecheck`
- **Test failures** - Run specific test with `npm test -- [test-name]`
- **Port conflicts** - Check for running servers

## Release Process

1. **Version bump**
```bash
npm version patch|minor|major
```

2. **Update CHANGELOG.md**
- Add version section
- List changes
- Credit contributors

3. **Create release PR**
- Follow PR template
- Ensure CI passes

4. **Publish**
```bash
npm run package
vsce publish
```

## Resources

### Internal Documentation
- [Architecture Diagrams](./architecture/README.md#diagrams)
- [API Reference](./api-reference/README.md)
- [Test Coverage Report](./testing/README.md#coverage)

### External Resources
- [VSCode Extension API](https://code.visualstudio.com/api)
- [Model Context Protocol](https://github.com/modelcontextprotocol/specification)
- [TypeScript Documentation](https://www.typescriptlang.org/docs/)

### Community
- [Discord Server](https://discord.gg/EmberHarmony)
- [GitHub Discussions](https://github.com/SolaceHarmony/Thea-Code/discussions)
- [Contributing Guide](./contributing.md)

## Support

For development questions:
1. Check this guide and related documentation
2. Search existing [GitHub Issues](https://github.com/SolaceHarmony/Thea-Code/issues)
3. Ask in the [Discord #development channel](https://discord.gg/EmberHarmony)

---

**Changelog:**
- 2025-08-10: Initial comprehensive developer guide