/**
 * Safe error utilities to avoid unsafe member access on unknown errors.
 */

export type ErrorLike = {
  name?: unknown
  message?: unknown
  stack?: unknown
  code?: unknown
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null
}

export function isError(value: unknown): value is Error {
  return value instanceof Error
}

export function isErrorLike(value: unknown): value is ErrorLike {
  return isRecord(value)
}

/**
 * Extract a safe, human-friendly error message without violating no-unsafe-member-access.
 */
export function getErrorMessage(err: unknown): string {
  if (isError(err)) return err.message
  if (isErrorLike(err)) {
    const msg = err.message
    if (typeof msg === "string" && msg.length > 0) return msg
    // Fallbacks based on common shapes
    if (typeof err.code === "string") return `Error code: ${err.code}`
  }
  try {
    return JSON.stringify(err)
  } catch {
    return String(err)
  }
}

/**
 * Extract a string/number error code if present.
 */
export function getErrorCode(err: unknown): string | number | undefined {
  if (isErrorLike(err)) {
    const code = (err as Record<string, unknown>).code
    if (typeof code === "string" || typeof code === "number") return code
  }
  return undefined
}

/**
 * Convert an unknown error to an Error instance for throwing/logging contexts.
 */
export function toError(err: unknown): Error {
  if (isError(err)) return err
  const message = getErrorMessage(err)
  return new Error(message)
}
