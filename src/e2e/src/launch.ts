import * as path from "path"
import * as fs from "fs"
import { execSync, spawn } from "child_process"
import { runTests, downloadAndUnzipVSCode, resolveCliArgsFromVSCodeExecutablePath } from "@vscode/test-electron"

async function main() {
  try {
    // Some environments (including certain CI/dev shells) set this, which makes
    // the VS Code Electron binary behave like `node` and reject VS Code args.
    if (process.env.ELECTRON_RUN_AS_NODE) {
      console.log(`[e2e/launch] Clearing ELECTRON_RUN_AS_NODE (was ${process.env.ELECTRON_RUN_AS_NODE})`)
      delete process.env.ELECTRON_RUN_AS_NODE
    }

    // Preflight: kill lingering test/mock processes to avoid port conflicts or duplicate mocks
    // Can be disabled via THEA_AUTO_KILL=0
    const autoKill = (process.env.THEA_AUTO_KILL ?? "1") === "1"
    if (autoKill) {
      try {
        console.log("[e2e/launch] Preflight cleanup: killing lingering test/mock processes")
        if (process.platform === "win32") {
          // Best-effort on Windows. Silently ignore errors.
          const winKill = (pattern: string) => {
            try { execSync(`wmic process where \"CommandLine like '%${pattern.replace(/"/g, "\\\"")}%'\" call terminate`, { stdio: "ignore" }) } catch { }
          }
            ;["mocha -r tsx", "uv tool uvx", "markitdown-mcp", "imagesorcery-mcp"].forEach(winKill)
        } else {
          // POSIX/macOS
          const pkill = (p: string) => { try { execSync(`pkill -f "${p}"`, { stdio: "ignore" }) } catch { } }
            ;["mocha -r tsx", "uv tool uvx", "markitdown-mcp", "imagesorcery-mcp"].forEach(pkill)
        }
      } catch {
        // Ignore any cleanup errors
      }
    }

    // This file is compiled to e2e/out/suite/launch.js
    const compiledDir = __dirname // e2e/out/suite
    // Go up 4 levels: suite -> out -> e2e -> src -> repo root
    const repoRoot = path.resolve(compiledDir, "..", "..", "..", "..", "..", "..", "..") // -> repo root

    const extensionDevelopmentPath = repoRoot
    const extensionTestsPath = path.resolve(compiledDir, "suite", "index.js")

    console.log(`[e2e/launch] Resolved repoRoot: ${repoRoot}`)
    if (!fs.existsSync(path.join(repoRoot, "package.json"))) {
      console.error(`[e2e/launch] CRITICAL: package.json not found at ${repoRoot}`)
      // Try to find where it is
      let curr = compiledDir
      while (curr !== path.dirname(curr)) {
        if (fs.existsSync(path.join(curr, "package.json"))) {
          console.log(`[e2e/launch] Found package.json at ${curr}`)
          break
        }
        curr = path.dirname(curr)
      }
    }

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
      // Silence MCP port wait logs and skip confirmation checks during tests
      THEA_SKIP_MCP_PORT_WAIT: process.env.THEA_SKIP_MCP_PORT_WAIT ?? "1",
      THEA_SILENT_MCP_LOGS: process.env.THEA_SILENT_MCP_LOGS ?? "1",
      NODE_ENV: process.env.NODE_ENV ?? "test",
      E2E_SMOKE_ONLY: process.env.E2E_SMOKE_ONLY ?? "0",
      // Default to full test discovery; can override to 1 for targeted runs
      E2E_DIRECT_TEST: process.env.E2E_DIRECT_TEST ?? "0",
      E2E_TEST_GLOB: process.env.E2E_TEST_GLOB ?? "**/*.test.js",
    }

    try {
      console.log(`[e2e/launch] Starting tests using runTests`)
      await runTests({
        extensionDevelopmentPath,
        extensionTestsPath,
        version: "insiders",
        launchArgs: [
          ...launchArgs,
          "--no-sandbox",
          "--disable-gpu-sandbox"
        ],
        extensionTestsEnv: env,
      })
    } catch (err) {
      console.error(`[e2e/launch] runTests failed: ${err}`)
      throw err
    }
  } catch (err) {
    console.error("[e2e] VS Code test launch failed:", err)
    process.exit(1)
  }
}

void main()
