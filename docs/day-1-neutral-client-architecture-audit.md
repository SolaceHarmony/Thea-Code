# Day 1 — Neutral Client Architecture Audit & Validation Report

Date: 2025-08-09
Owner: PM

## Scope
Evaluate the current VS Code extension’s Neutral Client approach and its readiness to add a Vertex AI implementation without leaking provider specifics. Audit focuses on boundaries, contracts, streaming (SSE) patterns, error/telemetry posture, configuration, and i18n. No code changes; observations are based on repository structure and typical VS Code/TypeScript conventions.

## Current Architecture (summary)
- Workspace structure suggests a layered design: `src/core`, `src/services`, `src/integrations`, `src/utils`, `src/i18n`, and `src/extension.ts` as the activation entry point.
- Likely a “neutral” abstraction in `core` or `services` with provider adapters under `integrations`.
- Build via `tsc`/`esbuild`, tests via `jest`, and i18n catalogs under `locales/`.
- Webview UI present (`webview-ui/`), implying streaming updates to UI are possible.

Assumptions
- A Neutral Client interface defines provider-agnostic methods (e.g., completion/chat) and emits streaming events.
- Credentials and provider configuration are managed via VS Code Secrets API and configuration settings.

## Contracts & Boundaries (Neutral Client expectations)
- Single entry surface for generation: request DTO (prompt, model, options), response DTO (content, metadata, finish reason).
- Streaming lifecycle: onStart → onChunk → onComplete/onError; caller can cancel.
- Strict provider isolation: provider adapters map neutral DTOs to vendor requests and back; no vendor types leak upward.
- Observability: standardized telemetry hooks (counters/timers/errors) at the neutral layer.
- Error taxonomy: auth, network, rate-limit, server, client-usage, timeout, malformed-response; retry policy only for retryable classes.
- Configuration: per-user/workspace settings (timeouts, retries, model defaults, region/project), plus feature flags.
- i18n: all user-visible messages localized through a central i18n service.

## Findings
### Strengths
- Clear separation of concerns implied by directory layout (core/services/integrations).
- Existing i18n and webview indicate maturity for user feedback and streaming UI updates.
- Testing and build tooling in place (jest, esbuild/tsc) to support adapter growth.

### Gaps
- Streaming contract may not be explicitly documented (chunk shape, heartbeats, backpressure behavior, cancellation semantics).
- Error taxonomy and retry/circuit-breaker rules may be implicit rather than codified.
- Telemetry schema (events, counters, timers, sampling) not visibly standardized across providers.
- Configuration/flags for provider enablement and rollout may be scattered or undocumented.

### Anti-patterns (risks to check for)
- Provider-specific logic creeping into neutral types or call sites.
- UI coupling to vendor response shapes (e.g., token counters, finish reasons).
- Ad-hoc SSE handling without buffering, heartbeat tolerance, or cancel propagation.
- Retry logic applied to non-retryable errors (auth/4xx semantic errors).

## Risks & Dependencies (with mitigations)
- Rate limits and quotas (mitigation: backoff + jitter, queueing, user messaging; circuit breaker on persistent failures).
- Auth token churn/expiry (mitigation: token cache + refresh, clear user remediation flow, Secrets API usage).
- SSE flakiness (mitigation: heartbeat detection, bounded buffers, cancel + reconnect strategies, idempotent resume where possible).
- Privacy/telemetry (mitigation: redact prompts by default, opt-in telemetry, sample errors, avoid PII).
- Localization gaps (mitigation: central error mapping to i18n keys, fallback locales, translation backlog tracking).

## Recommendations (ranked)
1) Document the Neutral Client contract (DTOs + streaming lifecycle) in prose and add examples; make it the single source of truth.
2) Define and publish the error taxonomy with retry/circuit-breaker rules and example flows; enforce via tests.
3) Standardize telemetry schema (counters/timers/errors) and privacy posture; ensure provider adapters log through a neutral sink.
4) Create a configuration and feature-flag matrix covering scopes (user/workspace), defaults, and rollout toggles for providers.
5) Provide an SSE handling guideline: heartbeats, backpressure caps, chunk validation, cancel semantics, reconnect policies.
6) Introduce a provider adapter checklist (auth, request shaping, response mapping, streaming, errors, telemetry, i18n) to gate new providers.

## Acceptance Verifications
- [ ] Written Neutral Client contract (methods, DTOs, events) reviewed and approved.
- [ ] Streaming lifecycle documented with cancellation and backpressure behaviors.
- [ ] Error taxonomy and retry/circuit-breaker rules published with example flows.
- [ ] Telemetry event list, counters/timers, and privacy policy defined and adopted by all providers.
- [ ] Config keys and feature flags enumerated with scopes/defaults and rollout guidance.
- [ ] Provider adapter checklist created and added to contributor docs.

Notes
- This artifact is code-free and intended to guide Day 6 API contract and Day 7 error/telemetry specs.
