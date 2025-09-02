# Jest to E2E Test Migration Tracker

## Summary
- **Total Jest Tests**: 154 files
- **Migration Status**: 0/154 (0%)
- **Target**: 100% migration to E2E framework

## Migration Batches

### Batch 1: Core Tools (7 files)
- [ ] core/tools/__tests__/askFollowupQuestionTool.test.ts
- [ ] core/tools/__tests__/applyDiffTool.test.ts
- [ ] core/tools/__tests__/attemptCompletionTool.flow.test.ts
- [ ] core/tools/__tests__/executeCommandTool.test.ts
- [ ] core/tools/__tests__/browserTool.test.ts
- [ ] core/tools/__tests__/computer-use-browser.test.ts
- [ ] core/tools/__tests__/listFilesTool.test.ts

### Batch 2: Core Config (7 files)
- [ ] core/config/__tests__/importExport.test.ts
- [ ] core/config/__tests__/ContextProxy.test.ts
- [ ] core/config/__tests__/CustomModesSettings.test.ts
- [ ] core/config/__tests__/CustomModesManager.test.ts
- [ ] core/config/__tests__/ModeConfig.test.ts
- [ ] core/config/__tests__/ProviderSettingsManager.test.ts
- [ ] core/config/__tests__/GlobalConfigManager.test.ts

### Batch 3: Core Webview (10 files)
- [ ] core/webview/__tests__/TheaTaskStack.test.ts
- [ ] core/webview/__tests__/TheaTaskHistory.test.ts
- [ ] core/webview/__tests__/getNonce.test.ts ✅ (already done)
- [ ] core/webview/__tests__/TheaMcpManager.test.ts
- [ ] core/webview/__tests__/TheaApiManager.test.ts
- [ ] core/webview/__tests__/TheaStateManager.test.ts
- [ ] core/webview/__tests__/TheaCacheManager.test.ts
- [ ] core/webview/__tests__/TheaCombinedManager.test.ts
- [ ] core/webview/__tests__/TheaToolCallManager.test.ts
- [ ] core/webview/__tests__/TheaTaskExecutor.test.ts

### Batch 4: API Providers (43 files)
- [ ] api/providers/__tests__/openai.test.ts
- [ ] api/providers/__tests__/anthropic.test.ts
- [ ] api/providers/__tests__/vertex.test.ts
- [ ] api/providers/__tests__/bedrock.test.ts
- [ ] api/providers/__tests__/openrouter.test.ts
- [ ] api/providers/__tests__/vscode-lm.test.ts
- [ ] api/providers/__tests__/model-registry.test.ts
- [ ] ... (36 more provider tests)

### Batch 5: Services/MCP (30 files)
- [ ] services/mcp/__tests__/McpHub.test.ts
- [ ] services/mcp/__tests__/UnifiedMcpToolSystem.test.ts
- [ ] services/mcp/core/__tests__/McpToolExecutor.test.ts
- [ ] services/mcp/core/__tests__/McpToolRegistry.test.ts
- [ ] services/mcp/core/__tests__/McpToolRouter.test.ts
- [ ] ... (25 more MCP tests)

### Batch 6: Utils (14 files)
- [ ] utils/__tests__/debounce.test.ts
- [ ] utils/__tests__/delay.test.ts
- [ ] utils/__tests__/fetchResponseHandler.test.ts
- [ ] utils/__tests__/path.test.ts
- [ ] utils/__tests__/getSymbolsFromOpenFiles.test.ts
- [ ] utils/logging/__tests__/CompactLogger.test.ts
- [ ] utils/logging/__tests__/CompactTransport.test.ts
- [ ] ... (7 more util tests)

### Batch 7: Integrations (10 files)
- [ ] integrations/terminal/__tests__/TerminalRegistry.test.ts
- [ ] integrations/terminal/__tests__/Terminal.test.ts
- [ ] integrations/diagnostics/__tests__/DiagnosticsMonitor.test.ts
- [ ] integrations/diagnostics/__tests__/createDiagnosticCollection.test.ts
- [ ] integrations/editor/__tests__/DiffManager.test.ts
- [ ] ... (5 more integration tests)

### Batch 8: Shared & Misc (13 files)
- [ ] shared/__tests__/modes.test.ts
- [ ] shared/__tests__/ExtensionMessage.test.ts
- [ ] shared/__tests__/OpenRouterUrlBuilder.test.ts
- [ ] __tests__/NeutralVertexClient.test.ts
- [ ] test/generic-provider-mock/__tests__/all-providers-runtime.test.ts
- [ ] ... (8 more misc tests)

## Migration Strategy

1. Start with simple utility tests (easy wins)
2. Move to core functionality tests
3. Tackle provider tests (largest group)
4. Finish with integration tests
5. Remove Jest completely

## Files to Delete After Migration
- jest.config.js
- jest.setup.js
- All __tests__ directories
- All *.test.ts files in src/

## Package.json Changes Needed
- Remove jest dependencies
- Remove @types/jest
- Update test scripts to use E2E only