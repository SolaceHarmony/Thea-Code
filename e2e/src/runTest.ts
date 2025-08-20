import * as path from "path"
import * as fs from "fs"
import { runTests } from "@vscode/test-electron"

async function main() {
  // Extension under test (repo root)
  const extensionDevelopmentPath = path.resolve(__dirname, "../../")
  // Mocha adapter (compiled JS)
  const extensionTestsPath = path.resolve(__dirname, "./suite/index")

  // Isolated test sandbox under repo/.vscode-test
  const testRoot = path.resolve(extensionDevelopmentPath, ".vscode-test")
  const userDataDir = path.join(testRoot, "user-data")
  const extensionsDir = path.join(testRoot, "extensions")
  const crashDir = path.join(testRoot, "crashes")
  ;[userDataDir, extensionsDir, crashDir].forEach((p) => fs.mkdirSync(p, { recursive: true }))

  // Open the repo root as the workspace
  const workspacePath = extensionDevelopmentPath

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
      },
    })
  } catch (err) {
    console.error("E2E failed to run:", err)
    process.exit(1)
  }
}

main()