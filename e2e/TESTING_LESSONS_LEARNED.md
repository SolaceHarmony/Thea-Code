# VS Code Extension Testing: Critical Lessons Learned

## The Extension Host Timeout Problem

### Problem Summary
When running VS Code extension tests with `@vscode/test-electron`, the extension host becomes unresponsive and times out after 3-4 seconds during test initialization. This prevents tests from running even though the test framework itself works correctly.

### Root Cause: Module-Level Code in extension.ts
The primary cause is heavy module-level code in the main `extension.ts` file. Module-level code executes immediately when the file is imported, before the `activate()` function is even called.

## What NOT to Do

```typescript
// ❌ BAD: Heavy imports at module level
import * as vscode from "vscode"
import * as dotenvx from "@dotenvx/dotenvx"  // Heavy module
import * as path from "path"

// ❌ BAD: Module-level code execution
console.log('[Extension] Loading...')  // Runs immediately on import

// ❌ BAD: Module-level initialization
try {
  const envPath = path.join(__dirname, "..", ".env")
  dotenvx.config({ path: envPath })  // Runs before activate()
} catch (e) {
  console.warn("Failed to load environment variables:", e)
}

export function activate(context: vscode.ExtensionContext) {
  // Too late - timeout already happened during module loading
}
```

## What TO Do Instead

```typescript
// ✅ GOOD: Only lightweight imports at module level
import * as vscode from "vscode"
import { EXTENSION_NAME } from "./shared/config/constants" // Simple constants OK

// ✅ GOOD: Type-only imports don't affect runtime
import type { MyProvider } from "./providers/MyProvider"

export async function activate(context: vscode.ExtensionContext) {
  // ✅ GOOD: Detect test mode early
  const isTestMode = process.env.NODE_ENV === 'test'
  
  if (isTestMode) {
    // ✅ GOOD: Minimal activation for tests
    console.log('Test mode: skipping heavy initialization')
    return { testMode: true }
  }
  
  // ✅ GOOD: Use dynamic imports for heavy modules
  try {
    // Load heavy dependencies only when needed
    const dotenvx = await import("@dotenvx/dotenvx")
    const path = await import("path")
    
    // Now safe to do heavy initialization
    const envPath = path.join(__dirname, "..", ".env")
    dotenvx.config({ path: envPath })
    
    // Continue with normal activation...
    const { MyProvider } = await import("./providers/MyProvider")
    const provider = new MyProvider()
    
  } catch (error) {
    console.error('Failed to initialize:', error)
  }
}
```

## Key Principles

### 1. Module-Level Code Must Be Minimal
- Only import lightweight modules at the top level
- No file I/O, network calls, or heavy computations at module level
- No console.log or other side effects outside functions

### 2. Use Dynamic Imports for Heavy Dependencies
- Load heavy modules inside `activate()` using `await import()`
- This defers loading until after the extension host is ready
- Particularly important for modules that do filesystem or network operations

### 3. Implement Test Mode Detection
- Check for test environment variables early in `activate()`
- Skip heavy initialization when in test mode
- Return a minimal API that satisfies test requirements

### 4. Type-Only Imports Are Safe
- TypeScript's `import type` statements don't affect runtime
- Use these freely for type definitions without performance impact

## Test Infrastructure Setup

### Critical Files and Their Roles

1. **e2e/src/runTest.ts**
   - Entry point for test execution
   - Configures VS Code download and launch parameters
   - Must set correct workspace path (use empty test workspace, not project root)

2. **e2e/src/suite/index.ts**
   - Mocha test runner configuration
   - Sets up global test functions (suite/test for TDD, describe/it for BDD)
   - Handles test discovery and execution

3. **e2e/src/suite/setup.ts**
   - Extension activation for tests
   - Sets up global test API
   - Must handle both full and minimal extension APIs

### Common Pitfalls

1. **Variable Declaration Order**: Ensure variables are declared before use in all test files
2. **Workspace Loading**: Don't load the entire project as test workspace - use an empty folder
3. **Timeout Configuration**: Extension host needs generous timeouts (600+ seconds) for initial setup
4. **Test Exit Handling**: Extension host may hang after tests complete - this is normal

## Debugging Tips

1. **Enable Verbose Logging**:
   ```bash
   export ELECTRON_ENABLE_LOGGING=1
   export VSCODE_VERBOSE_LOGGING=true
   export VSCODE_LOG_LEVEL=trace
   ```

2. **Check for Module Loading**:
   - Add console.log at module level to identify what loads when
   - If you see module-level logs before activation, that's the problem

3. **Progressive Simplification**:
   - Start with ultra-minimal extension.ts that just returns a simple object
   - Gradually add functionality back to identify what causes timeouts

4. **Monitor Extension Host**:
   - Look for "Extension host (LocalProcess pid: XXXX) is unresponsive" in logs
   - This indicates module-level code is taking too long

## Summary

The golden rule: **Keep extension.ts module-level code absolutely minimal**. Move ALL heavy initialization into the `activate()` function and use dynamic imports. This ensures the extension module loads instantly, giving the extension host time to properly initialize before your heavy code runs.

This approach solves the timeout issue while preserving all functionality - nothing is lost, only reorganized for better performance and testability.