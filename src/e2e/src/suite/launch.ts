import * as path from "path"
import * as fs from "fs"
import { execSync } from "child_process"
import { runTests, downloadAndUnzipVSCode, resolveCliArgsFromVSCodeExecutablePath } from "@vscode/test-electron"
import { spawn } from "child_process"

async function main() {
  try {
    // This file compiles to e2e/out/suite/launch.js
    const compiledDir = __dirname // e2e/out/suite
    const repoRoot = path.resolve(compiledDir, "..", "..", "..") // -> repo root

    const extensionDevelopmentPath = repoRoot
    // Tests compile under out/suite/suite/** based on tsconfig rootDir/outDir
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
      "--disable-extensions",
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
    // Extra breadcrumb to detect hangs before VS Code starts
    try { fs.appendFileSync(path.join(testRoot, "e2e-launch.log"), `[launch] using tests at ${extensionTestsPath}\n`) } catch {}

  // Always (re)build the extension so latest E2E sandbox detection is present
  console.log(`[e2e/launch] Building extension`)
  execSync("npm run build:extension", { cwd: repoRoot, stdio: "inherit" })

    const env = {
      ...process.env,
      ELECTRON_ENABLE_LOGGING: "1",
      ELECTRON_ENABLE_STACK_DUMPING: "1",
      THEA_E2E: "1",
      NODE_ENV: process.env.NODE_ENV ?? "test",
      E2E_SMOKE_ONLY: process.env.E2E_SMOKE_ONLY ?? "0",
      E2E_DIRECT_TEST: process.env.E2E_DIRECT_TEST ?? "1",
      // index.ts expects a glob relative to out/suite
      E2E_TEST_GLOB: process.env.E2E_TEST_GLOB ?? "selected/**/*.test.js",
    }

    // Workaround Insiders macOS binary arg parsing: invoke the CLI shim instead
    // of the Electron binary returned by runTests(), which rejects product args.
    try {
      const vscodeExecutablePath = await downloadAndUnzipVSCode({ version: "insiders", extensionDevelopmentPath })
      const [cli, ...cliBaseArgs] = resolveCliArgsFromVSCodeExecutablePath(vscodeExecutablePath)

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
        const child = spawn(shell ? `"${cli}"` : cli, [...cliBaseArgs, ...args], {
          env,
          stdio: "inherit",
          shell,
        })
        child.on("error", reject)
        child.on("exit", (code, signal) => {
          console.log(`[e2e/launch] VS Code exited with ${code ?? signal}`)
          code === 0 ? resolve() : reject(new Error(`VS Code exited with ${code ?? signal}`))
        })
      })
    } catch (cliErr) {
      console.warn(`[e2e/launch] CLI spawn failed (${cliErr instanceof Error ? cliErr.message : String(cliErr)}); falling back to test-electron`)
      // Fallback to the standard helper
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
