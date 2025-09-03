/*
 * E2E test runner entrypoint
 * This file is intentionally small: it chooses the smoke runner or the full
 * suite runner and delegates to the exported `run()` function. It is compiled
 * to ../out/src/e2e/runTest.js by the e2e compile step and invoked by
 * `node ../out/src/e2e/runTest.js` from the e2e package scripts.
 */

type RunnerModule = { run: () => void | Promise<void> }

async function main(): Promise<void> {
  try {
    const smokeOnly = process.env.E2E_SMOKE_ONLY === "1"
    const skipSetup = process.env.E2E_SKIP_SETUP === "1"

    // Choose the runner module.
  const runnerPath = smokeOnly ? "./suite/smokeIndex" : "./suite/index"
  const mod: unknown = await import(runnerPath)
  const runner: Partial<RunnerModule> = (typeof mod === "object" && mod !== null ? (mod as Partial<RunnerModule>) : {})

  if (!runner || typeof runner.run !== "function") {
      console.error(`[e2e/runTest] Runner module ${runnerPath} did not export run()`)
      process.exit(2)
    }

    // Expose a debug hint when skipping setup (suite/index will check E2E_SKIP_SETUP)
    if (skipSetup) {
      console.log("[e2e/runTest] E2E_SKIP_SETUP=1 detected; setup will be skipped by suite runner")
    }

  await runner.run()
    // Successful completion
    process.exit(0)
  } catch (err) {
    // Ensure errors are visible in CI logs
    console.error("[e2e/runTest] Unhandled error:", err)
    process.exit(1)
  }
}

void main()
