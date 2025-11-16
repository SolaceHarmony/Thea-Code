# Thea Code - Copilot Instructions

This document provides guidance for GitHub Copilot when working on the Thea Code repository.

## Project Overview

**Thea Code** is an AI-powered autonomous coding agent that runs as a Visual Studio Code extension. It's a community fork of Roo Code that provides an AI-driven development assistant capable of reading/writing files, running terminal commands, automating browser actions, and integrating with multiple AI providers through the Model Context Protocol (MCP).

### Key Technologies
- **Language**: TypeScript (strict mode)
- **Runtime**: Node.js (v20+)
- **Frontend**: React + Vite (webview-ui directory)
- **Platform**: VS Code Extension API
- **Architecture**: MCP (Model Context Protocol) for universal tool access across AI providers
- **Testing**: Mocha for unit tests, VS Code Test API for integration tests

## Repository Structure

```
/src                    - Core extension source code
  /api                  - AI provider integrations (OpenAI, Anthropic, Ollama, etc.)
  /core                 - Core functionality (prompts, commands, agent logic)
  /services             - Services (MCP, browser automation, git, etc.)
  /integrations         - External integrations
  /shared               - Shared utilities and types
  /utils                - Utility functions
  /__tests__            - Unit tests
  /e2e                  - E2E test suite
/webview-ui             - React-based UI (Vite + TypeScript)
/docs                   - Documentation and architectural notes
/test                   - Additional test infrastructure
/scripts                - Build and utility scripts
/locales                - Internationalization files
```

## Build and Test Commands

### Development
```bash
npm run install:all     # Install all dependencies (root, webview-ui, e2e)
npm run dev             # Start webview development server with HMR
npm run watch:esbuild   # Watch extension code changes
npm run watch:tsc       # Watch TypeScript compilation
```

### Building
```bash
npm run build           # Full build (lint + clean + build extension + build webview)
npm run build:extension # Build extension only
npm run build:webui     # Build webview UI only
npm run package         # Create .vsix package
```

### Testing
```bash
npm run test:unit       # Run unit tests (Mocha)
npm run test:integration # Run integration tests
npm run test:e2e        # Run end-to-end tests
npm run test:all        # Run all integration tests
npm run test:types      # Type check without emitting
```

### Linting and Formatting
```bash
npm run lint            # Lint all files (must pass with 0 warnings)
npm run lint:fix        # Auto-fix linting issues
npm run lint:src        # Lint source files only
npm run lint:tests      # Lint test files only
npm run pretty          # Format all files with Prettier
npm run pretty:check    # Check formatting without writing
```

## Coding Standards and Conventions

### TypeScript
- **Strict mode enabled**: All TypeScript strict checks are enforced
- **No implicit any**: Avoid using `any` type; use proper types or `unknown`
- **No unsafe rules**: Do not suppress TypeScript safety rules without justification
- **Module resolution**: Uses `Bundler` mode (modern ESM)
- **Target**: ES2022 with ESNext features

### Testing Conventions
- Use Node's built-in `assert/strict` and `sinon` for unit tests
- Avoid chai/expect-style assertion chains; prefer straightforward assertions
- Always restore stubs/spies in `afterEach` using `sinon.restore()`
- Do not suppress unsafe TypeScript rules; address violations or justify with narrowly-scoped inline comments

### Code Style
- Use ESLint configuration defined in `eslint.config.mjs`
- Format code with Prettier (config in `.prettierrc.json`)
- **Zero warnings policy**: All code must pass linting with 0 warnings
- Use `camelCase` for variables and functions
- Use `PascalCase` for classes and types
- Prefer `const` over `let` when possible

### File Organization
- Keep related functionality together in the same directory
- Use `__tests__` subdirectories for unit tests co-located with source
- Use descriptive file names that match their exported content
- Avoid circular dependencies

## Development Workflow

### Setting Up Local Environment
1. Clone the repository
2. Run `npm run install:all` to install all dependencies
3. Run `npm run dev` to start the webview development server
4. Press F5 in VS Code to launch extension development host

### Making Changes
1. Create focused, single-purpose pull requests
2. Run linting and tests before committing: `npm run lint && npm run test:unit`
3. Ensure all tests pass and no new warnings are introduced
4. Update tests when modifying functionality
5. Add JSDoc comments for public APIs

### Pull Request Guidelines
- Keep PRs focused on a single feature or bug fix
- Write clear, descriptive commit messages
- Reference related issues using #issue-number
- Include tests for new features
- Add screenshots for UI changes
- Ensure CI checks pass (linting, formatting, tests)

## Key Architectural Concepts

### Model Context Protocol (MCP)
- Provides universal tool access across all 16+ AI providers
- Automatic format conversion (XML, JSON, OpenAI function calls)
- Located in `/src/services/mcp/`
- See `docs/architectural_notes/tool_use/mcp/` for detailed documentation

### AI Provider Integration
- Unified interface for OpenAI, Anthropic, Ollama, and others
- Located in `/src/api/providers/`
- Each provider implements capability detection and tool formatting
- Support for streaming responses and function calling

### Extension Architecture
- Main extension entry: `/src/extension.ts`
- Webview communication through message passing
- State management via VS Code context and secret storage
- Command registration in `package.json`

## Common Tasks

### Adding a New AI Provider
1. Create provider class in `/src/api/providers/`
2. Implement required interfaces (streaming, tool use, capability detection)
3. Add provider configuration in relevant config files
4. Add tests in provider's `__mocha__` directory
5. Update documentation

### Adding New MCP Tools
1. Define tool schema in appropriate MCP provider
2. Implement tool handler
3. Add format conversion if needed
4. Test with multiple AI providers
5. Update tool documentation

### Modifying UI
1. Navigate to `/webview-ui/src/`
2. Make React component changes
3. Run `npm run dev` for hot module replacement
4. Test in extension development host
5. Build with `npm run build:webui`

## Important Files

- `package.json` - Extension manifest and dependencies
- `README.md` - Project overview and getting started
- `CONTRIBUTING.md` - Contribution guidelines and roadmap
- `CHANGELOG.md` - Version history and release notes
- `CODE_OF_CONDUCT.md` - Community standards
- `tsconfig.json` - TypeScript configuration
- `eslint.config.mjs` - ESLint configuration
- `.prettierrc.json` - Prettier configuration

## Testing Strategy

### Unit Tests
- Test individual functions and modules in isolation
- Use mocks and stubs for external dependencies
- Located alongside source in `__tests__` directories
- Run with `npm run test:unit`

### Integration Tests
- Test interaction between components
- Use VS Code Test API for extension-specific tests
- Located in `/src/e2e/`
- Run with `npm run test:integration`

### End-to-End Tests
- Test full user workflows
- Require VS Code test environment
- Run with `npm run test:e2e`

## Known Constraints

### Testing Environment
- Unit tests run with MCP SDK disabled via environment variables
- Some features require VS Code extension host
- Browser automation tests require Playwright setup

### Build System
- Uses esbuild for fast compilation
- Webview built separately with Vite
- No source maps in production builds
- Package size optimization enabled

## Security Considerations

- Never commit secrets or API keys
- Use VS Code secret storage for sensitive data
- Report security vulnerabilities privately via GitHub Security Advisories
- Follow secure coding practices for extension APIs

## Internationalization

- Translation files located in `/locales/`
- Use i18next for string translation
- Support for 12+ languages
- Always use translation keys, never hardcoded strings in UI

## Community and Support

- **Discord**: https://discord.gg/EmberHarmony
- **Reddit**: https://www.reddit.com/r/TheaCode/
- **Issues**: https://github.com/SolaceHarmony/Thea-Code/issues
- **Discussions**: https://github.com/SolaceHarmony/Thea-Code/discussions

## Documentation

For more detailed information, see:
- Architecture & MCP Guide: `/docs/`
- Migration Guide: `/docs/architectural_notes/MIGRATION_GUIDE.md`
- MCP Implementation: `/docs/architectural_notes/tool_use/mcp/mcp_comprehensive_guide.md`
- Test Checklist: `MASTER_TEST_CHECKLIST.md`

## License

Apache 2.0 © 2025 Solace Project
