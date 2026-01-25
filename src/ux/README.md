# Thea Code UX Migration

This directory houses the new UX surface that replaces the legacy webview-based
implementation. The goal is to express every interaction through native VS Code
APIs (chat participants, command palette, QuickPick, status bar, etc.) so we can
retire the deprecated Webview UI Toolkit and React bundle entirely.

## Current components

- `chat/registerChatParticipant.ts` — Registers the initial chat participant and
  streams placeholder responses. Real task orchestration will be migrated here in
  follow-up work.
- `index.ts` — Entry point used by `extension.ts` to register native UX features.

## TODO

- Migrate “New Task” behaviour so chat requests spin up real `TheaTask`
  instances and stream progress through `ChatResponseStream` parts.
- Surface approval prompts with chat command buttons / QuickPick instead of
  webview messages.
- Render history and checkpoint previews via native chat components (markdown,
  file tree, anchors).
- Port MCP server management to QuickPick workflows.
- Remove the `webview-ui` build once the native UX reaches feature parity.
