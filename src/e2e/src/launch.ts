import * as path from "path"
import * as fs from "fs"
import { execSync, spawn } from "child_process"
import { runTests, downloadAndUnzipVSCode, resolveCliArgsFromVSCodeExecutablePath } from "@vscode/test-electron"

async function main() {
  try {
    // This file is compiled to e2e/out/suite/launch.js
  const compiledDir = __dirname // e2e/out/suite
  const repoRoot = path.resolve(compiledDir, "..", "..", "..") // -> repo root

  const extensionDevelopmentPath = repoRoot
  const extensionTestsPath = path.resolve(compiledDir, "suite", "index.js")

    // Create an isolated sandbox workspace and user data dirs
    const testRoot = path.resolve(repoRoot, ".vscode-test")
    const workspaceDir = process.env.E2E_WORKSPACE_DIR ?? path.join(testRoot, "workspace")
    const userDataDir = path.join(testRoot, "user-data")
    const extensionsDir = path.join(testRoot, "extensions")

    // Ensure directories exist and the workspace is empty each run
    await fs.promises.rm(workspaceDir, { recursive: true, force: true })
    await fs.promises.mkdir(workspaceDir, { recursive: true })
    await fs.promises.mkdir(userDataDir, { recursive: true })
    await fs.promises.mkdir(extensionsDir, { recursive: true })

  const launchArgs = [
      `--user-data-dir=${userDataDir}`,
      `--extensions-dir=${extensionsDir}`,
      `--enable-proposed-api=SolaceHarmony.thea-code`,
      "--disable-workspace-trust",
      "--skip-release-notes",
      "--skip-welcome",
      workspaceDir,
    ]

    // Sanity checks and diagnostics
    if (!fs.existsSync(extensionTestsPath)) {
      console.error(`[e2e/launch] extensionTestsPath not found: ${extensionTestsPath}`)
      console.error(`[e2e/launch] compiledDir=${compiledDir}`)
      process.exit(2)
    }
    console.log(`[e2e/launch] extensionDevelopmentPath=${extensionDevelopmentPath}`)
    console.log(`[e2e/launch] extensionTestsPath=${extensionTestsPath}`)
    console.log(`[e2e/launch] workspaceDir=${workspaceDir}`)
    console.log(`[e2e/launch] userDataDir=${userDataDir}`)
    console.log(`[e2e/launch] extensionsDir=${extensionsDir}`)

    // Ensure extension is built so latest E2E sandbox detection is present
    const distEntry = path.resolve(repoRoot, "dist", "extension.js")
    const forceRebuild = process.env.E2E_FORCE_REBUILD === "1"
    if (forceRebuild || !fs.existsSync(distEntry)) {
      console.log(`[e2e/launch] Building extension (force=${forceRebuild}, distExists=${fs.existsSync(distEntry)})`)
      execSync("npm run build:extension", { cwd: repoRoot, stdio: "inherit" })
    }

    const env = {
      ...process.env,
      ELECTRON_ENABLE_LOGGING: "1",
      ELECTRON_ENABLE_STACK_DUMPING: "1",
      // Run full extension for true E2E by default
      THEA_E2E: process.env.THEA_E2E ?? "1",
      // Prefer sandboxed, workspace-local config during tests
      THEA_PREFER_LOCAL_CONFIG: process.env.THEA_PREFER_LOCAL_CONFIG ?? "1",
      NODE_ENV: process.env.NODE_ENV ?? "test",
      E2E_SMOKE_ONLY: process.env.E2E_SMOKE_ONLY ?? "0",
      // Default to full test discovery; can override to 1 for targeted runs
      E2E_DIRECT_TEST: process.env.E2E_DIRECT_TEST ?? "0",
      E2E_TEST_GLOB: process.env.E2E_TEST_GLOB ?? "**/*.test.js",
    }

    try {
      // Prefer invoking the VS Code CLI shim to avoid Insiders app binary rejecting args on macOS
      const vscodeExecutablePath = await downloadAndUnzipVSCode({ version: "insiders", extensionDevelopmentPath })
      const [cli] = resolveCliArgsFromVSCodeExecutablePath(vscodeExecutablePath)

      const args = [
        ...launchArgs,
        "--no-sandbox",
        "--disable-gpu-sandbox",
        "--disable-updates",
        "--skip-welcome",
        "--skip-release-notes",
        "--disable-workspace-trust",
        `--extensionDevelopmentPath=${extensionDevelopmentPath}`,
        `--extensionTestsPath=${extensionTestsPath}`,
      ]

      console.log(`[e2e/launch] Spawning VS Code CLI: ${cli}`)
      const shell = process.platform === "win32"
      await new Promise<void>((resolve, reject) => {
        // Pass only our args to avoid duplicate options becoming arrays in yargs
        const child = spawn(shell ? `"${cli}"` : cli, [...args], {
          env,
          stdio: "inherit",
          shell,
        })
        child.on("error", reject)
        child.on("exit", (code, signal) => {
          console.log(`[e2e/launch] VS Code exited with ${code ?? signal}`)
          if (code === 0) {
            resolve()
          } else {
            reject(new Error(`VS Code exited with ${code ?? signal}`))
          }
        })
      })
    } catch (cliErr) {
      console.warn(`[e2e/launch] CLI spawn failed (${cliErr instanceof Error ? cliErr.message : String(cliErr)}); falling back to test-electron`)
      await runTests({
        extensionDevelopmentPath,
        extensionTestsPath,
        version: "insiders",
        launchArgs,
        extensionTestsEnv: env,
      })
    }
  } catch (err) {
    console.error("[e2e] VS Code test launch failed:", err)
    process.exit(1)
  }
}

void main()
