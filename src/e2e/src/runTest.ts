import * as path from "path"
import * as fs from "fs"
import { runTests } from "@vscode/test-electron"

async function main() {
  // Extension under test (repo root)
  const extensionDevelopmentPath = path.resolve(__dirname, "../../../")
  
  // Isolated test sandbox under repo/.vscode-test
  const testRoot = path.resolve(extensionDevelopmentPath, ".vscode-test")
  const userDataDir = path.join(testRoot, "user-data")
  const extensionsDir = path.join(testRoot, "extensions")
  const crashDir = path.join(testRoot, "crashes")
  const debugLog = path.join(testRoot, "thea-e2e.log")
  ;[userDataDir, extensionsDir, crashDir].forEach((p) => fs.mkdirSync(p, { recursive: true }))
  try { fs.writeFileSync(debugLog, "[runner] E2E start\n") } catch {}
  
  // Mocha adapter (compiled JS)
  const defaultTestsPath = path.resolve(__dirname, "./suite/index")
  const smokeTestsPath = path.resolve(__dirname, "./suite/smokeIndex")
  const extensionTestsPath = process.env.E2E_SMOKE_INDEX === "1" ? smokeTestsPath : defaultTestsPath
  try { fs.appendFileSync(debugLog, `[runner] testsPath=${extensionTestsPath} smokeIndex=${process.env.E2E_SMOKE_INDEX}\n`) } catch {}

  // Use an empty workspace to avoid loading the full project
  const workspacePath = path.resolve(__dirname, "../test-fixtures/empty-workspace")

  try {
    await runTests({
      extensionDevelopmentPath,
      extensionTestsPath,
      launchArgs: [
        "--disable-gpu",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--log=trace",
        "--skip-welcome",
        "--skip-release-notes",
        "--skip-getting-started",
        "--disable-workspace-trust",
        "--crash-reporter-directory",
        crashDir,
        "--user-data-dir",
        userDataDir,
        "--extensions-dir",
        extensionsDir,
        workspacePath,
      ],
      extensionTestsEnv: {
        THEA_E2E: "1",
        NODE_ENV: "test",
        ELECTRON_ENABLE_LOGGING: "1",
        ELECTRON_ENABLE_STACK_DUMPING: "1",
        VSCODE_LOG_LEVEL: "trace",
        VSCODE_VERBOSE_LOGGING: "true",
        E2E_TEST_GLOB: process.env.E2E_TEST_GLOB,
        E2E_SKIP_SETUP: process.env.E2E_SKIP_SETUP,
        E2E_SMOKE_ONLY: process.env.E2E_SMOKE_ONLY,
        E2E_SMOKE_INDEX: process.env.E2E_SMOKE_INDEX,
      },
    })
  } catch (err) {
    console.error("E2E failed to run:", err)
    process.exit(1)
  }
}

void main()