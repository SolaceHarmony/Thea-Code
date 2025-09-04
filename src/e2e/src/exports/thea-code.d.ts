// Minimal shim for e2e compilation.
// The real `TheaCodeAPI` types live in the main project; e2e only needs a lightweight
// shape at compile-time.

export interface TheaCodeAPI {
  // Minimal surface used by tests during setup
  isTestMode?: boolean
  // Use index signature with unknown to avoid any
  [key: string]: unknown
}
