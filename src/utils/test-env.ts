// Utility to detect test environment in a framework-agnostic way (Mocha-compatible)
export function isTestEnv(): boolean {
  // Prefer explicit NODE_ENV flag
  if (process.env.NODE_ENV === 'test') return true

  // Mocha does not set a standard env var like Jest; check common globals instead
  const anyGlobal = globalThis as unknown as Record<string, unknown>
  if (typeof anyGlobal.describe === 'function' && typeof anyGlobal.it === 'function') return true

  // Support CI runners that might set MOCHA_WORKER_ID
  if (typeof process.env.MOCHA_WORKER_ID !== 'undefined') return true

  return false
}
