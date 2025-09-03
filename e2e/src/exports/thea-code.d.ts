// Minimal shim for e2e compilation.
// The real `TheaCodeAPI` types live in the main project; e2e only needs a lightweight
// shape at compile-time.

export interface TheaCodeAPI {
  // Minimal surface used by tests during setup
  isTestMode?: boolean
  // Use index signature to allow any additional properties without strict typing
  [key: string]: any
}
