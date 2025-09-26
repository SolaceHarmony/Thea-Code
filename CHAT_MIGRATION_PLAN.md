# Chat Migration Plan

This document maps the existing webview-driven functionality in Thea Code to the modern Visual Studio Code chat and native UI APIs. It focuses on replacing custom HTML/React surfaces—especially those built on the now end-of-life Webview UI Toolkit—with first-class extension APIs.

## 1. Activation & Participant Registration

| Current responsibility | Replacement API | Notes |
| --- | --- | --- |
| Registering commands, wiring webview providers inside `activate` | `vscode.chat.createChatParticipant` (Chat Participant API) | Define chat participants in `package.json` under `contributes.chatParticipants`, then create them during activation. Each participant exposes a `ChatRequestHandler` that drives multi-turn conversations.
| Manually instantiating `TheaProvider` and sidebar view (`registerWebviewViewProvider`) | Same chat participant registration + optional `contributes.chatCommands` | Chat participants become the primary entry point. Additional slash commands map to `chatCommands` contributions if needed.

## 2. Request Handling & State Orchestration

| Current responsibility | Replacement API | Notes |
| --- | --- | --- |
| Processing `WebviewMessage` events in `webviewMessageHandler` | `vscode.ChatRequestHandler` implementation | The handler receives `vscode.ChatRequest`, `ChatContext`, `ChatResponseStream`, and `CancellationToken`. Use the stream to emit results, progress, and references without maintaining a separate message bus.
| Streaming agent replies via custom messaging | `ChatResponseStream.markdown()`, `ChatResponseStream.progress()`, `ChatResponseStream.push(new ChatResponseProgressPart(...))`, `ChatResponseStream.push(new ChatResponseReferencePart(...))` | Supports incremental markdown, progress updates, reference lists, and command buttons directly in the chat panel.
| Triggering follow-up prompts or buttons | `ChatResponseStream.button(new vscode.ChatResponseCommandButtonPart(command))` and `ChatFollowupProvider` | Command buttons and suggested follow-ups are rendered natively by VS Code.

## 3. Model Invocation & Tooling

| Current responsibility | Replacement API | Notes |
| --- | --- | --- |
| Building prompts and forwarding to LLMs manually | `vscode.LanguageModelChatMessage` helpers and `request.model.sendRequest(...)` | Chat handlers can compose `LanguageModelChatMessage.User/Assistant` objects and dispatch through the selected chat model. Tool calling metadata flows through the same API.
| Managing per-request tools in custom JSON | `chatUtils.sendChatParticipantRequest` (from official samples) or direct tool invocation via `vscode.lm.tools` | Offers built-in integration with registered language model tools and tagging.

## 4. UI Rendering & Theming

| Current responsibility | Replacement API | Notes |
| --- | --- | --- |
| Assembling HTML via `getHtmlContent`/`getHMRHtmlContent`, serving React bundle | No HTML generation required | VS Code renders the chat UI automatically. Only participant metadata (icon, description) is configured. Any supplementary UI (quick picks, notifications) should use `vscode.window` primitives.
| Applying theme colors via CSS variables | Native chat UI honors VS Code themes | When custom UI is unavoidable (e.g., legacy diff view), rely on CSS variables directly, but the main chat interface requires no styling work.

## 5. User Interaction Helpers

| Current responsibility | Replacement API | Notes |
| --- | --- | --- |
| Webview `postMessage` exchange for approvals (`askResponse`, `newTask`, etc.) | `ChatResponseStream` command buttons + `vscode.window.showQuickPick` / `showInformationMessage` | Use command buttons for structured approvals and native dialogs for binary decisions or configuration changes.
| Custom context pickers in React (file search, mode switching) | `vscode.window.showQuickPick`, `showInputBox`, `showOpenDialog` | These functions deliver consistent, accessible UI without custom components.
| Status updates via custom banners or React state | `ChatResponseStream.progress` or `vscode.window.withProgress` | Emits progress directly in chat or the command palette.

## 6. File & Workspace Operations

| Current responsibility | Replacement API | Notes |
| --- | --- | --- |
| Pushing workspace metadata into webview (`workspaceUpdated`) | Access `context.history`, `vscode.workspace.fs`, and respond with markdown lists or `ChatResponseFileTreePart` | The chat response can include file trees/references, eliminating the need for a persistent client-side store.
| Launching diff viewers via custom scheme | Continue using `vscode.commands.executeCommand('vscode.diff', ...)` | Diff and editor commands remain valid and can be invoked from chat handlers or command buttons.

## 7. Telemetry & Lifecycle

| Current responsibility | Replacement API | Notes |
| --- | --- | --- |
| Telemetry tied to webview lifecycle | Keep existing telemetry services, trigger events from chat handlers | Activation remains similar; only the UI surface changes.
| Cleaning up webview instances (`dispose`) | Dispose chat participant (`ChatParticipant.dispose()`) when extension deactivates | Ensures resources like tool registrations and event emitters are released.

## Implementation Roadmap

1. **Register chat participant(s)** in `package.json` and create them in `activate`.
2. **Refactor `webviewMessageHandler` logic** into one or more `ChatRequestHandler` implementations. Reuse existing domain services (tasks, MCP integration) behind the handler.
3. **Map UI affordances**:
   - Replace React components with chat response parts, command buttons, and follow-up prompts.
   - Substitute any remaining dialogs or pickers with `vscode.window` APIs.
4. **Retire webview infrastructure**: remove `TheaProvider`, `webview-ui` bundle, and custom CSP logic once chat flow parity is achieved.
5. **Gradually migrate commands** so buttons like “New Task” or “Explain Code” invoke chat commands or participants directly.
6. **Validate telemetry and state management** to ensure no sensitive data is logged or leaked during the transition.

### Immediate Follow-up Tasks

- ✅ Initial task streaming and approve/reject/respond buttons now live in `src/ux/chat/registerChatParticipant.ts`.
- ✅ History browsing now exposed via `showHistoryQuickPick` in `registerCommands`.
- ✅ Prompt and MCP panels replaced with native QuickPick flows.
- Continue replacing residual `webviewMessageHandler` message types (settings updates, MCP toggles, diff previews) with chat commands or native prompts.
- Move checkpoint previews into `ChatResponseFileTreePart`/`ChatResponseAnchorPart` outputs.
- Delete the `webview-ui` React bundle once the chat path reaches feature parity, then remove associated build tooling from `package.json`.

This staged approach migrates Thea Code away from the deprecated Webview UI Toolkit while embracing the officially supported chat ecosystem in VS Code 1.103+.
