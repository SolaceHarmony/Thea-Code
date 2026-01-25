# Security Audit Summary

## Overview
This document captures current security-relevant concerns discovered while reviewing the Thea Code VS Code extension on September 26, 2025.

## Findings

### 1. Sensitive State Logged to Console *(resolved)*
- Location: `src/core/webview/TheaProvider.ts:878-889`.
- The entire state payload (including API credentials, task transcripts, and other sensitive configuration values) was logged before each call to `postMessageToWebview`.
- Impact: Credentials and user data appeared in the VS Code DevTools console and any host-level logging, creating an immediate data-leak risk.
- Status: The verbose `console.log` call has been removed so no state snapshot is emitted. If we introduce diagnostics in the future, they should redact secrets and remain opt-in.

### 2. Webview Content Security Policy Blocks Required Domains *(addressed for Requesty)*
- Location: `src/core/webview/TheaProvider.ts:700-718` (static HTML) and `webview-ui/src/components/ui/hooks/useRequestyKeyInfo.ts:21-27`.
- The CSP originally only allowed connections to OpenRouter and PostHog, but the UI fetched Requesty account data from `https://api.requesty.ai`.
- Impact: Requests were blocked, breaking the Requesty configuration flow.
- Status: The CSP `connect-src` list now explicitly whitelists `https://api.requesty.ai`. Longer term, the React webview will be retired and the CSP can be tightened again.

### 3. Global File-System Watcher Without Filtering *(partially mitigated)*
- Location: `src/integrations/workspace/WorkspaceTracker.ts:29-110`, invoked by `TheaProvider.resolveWebviewView`.
- A blanket watcher (`**`) starts as soon as the webview is resolved, regardless of workspace size or task requirements.
- Impact: Elevated risk of exhausting OS watcher limits, leaking file metadata, and causing performance regressions on large or sensitive repositories.
- Status: The tracker now ignores common bulky directories (e.g. `node_modules`, `.git`, `dist`, etc.) both during the initial crawl and in subsequent change events. Further work is still recommended to defer watcher creation until a task actually requires workspace context.

## Suggested Next Steps
1. Strip sensitive logging and introduce a sanitized diagnostics system.
2. Revisit the CSP to balance strictness with required functionality; document allowed domains.
3. Implement a scoped, permission-aware workspace tracking strategy to minimize exposure and resource usage.
