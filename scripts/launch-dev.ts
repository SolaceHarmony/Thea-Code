/**
 * Interactive Dev Launcher
 * 
 * Launches VS Code in the same E2E test configuration but keeps it running
 * for manual testing and log capture. This allows you to interact with the
 * extension while capturing console output.
 * 
 * Usage:
 *   npm run dev:launch
 * 
 * Environment variables:
 *   THEA_AUTO_KILL=0    - Skip killing lingering processes
 *   E2E_WORKSPACE_DIR   - Custom workspace directory
 */

import * as path from "path"
import * as fs from "fs"
import { execSync } from "child_process"
import { downloadAndUnzipVSCode, resolveCliArgsFromVSCodeExecutablePath } from "@vscode/test-electron"
import { spawn } from "child_process"

async function main() {
    try {
        // Preflight: kill lingering test/mock processes
        const autoKill = (process.env.THEA_AUTO_KILL ?? "1") === "1"
        if (autoKill) {
            try {
                console.log("[dev-launcher] Preflight cleanup: killing lingering processes")
                if (process.platform === "win32") {
                    const winKill = (pattern: string) => {
                        try { execSync(`wmic process where "CommandLine like '%${pattern.replace(/"/g, "\\\"")}%'" call terminate`, { stdio: "ignore" }) } catch { }
                    }
                        ;["mocha -r tsx", "uv tool uvx", "markitdown-mcp", "imagesorcery-mcp"].forEach(winKill)
                } else {
                    const pkill = (p: string) => { try { execSync(`pkill -f "${p}"`, { stdio: "ignore" }) } catch { } }
                        ;["mocha -r tsx", "uv tool uvx", "markitdown-mcp", "imagesorcery-mcp"].forEach(pkill)
                }
            } catch {
                // Ignore cleanup errors
            }
        }

        const repoRoot = path.resolve(__dirname, "..")
        const extensionDevelopmentPath = repoRoot

        console.log(`[dev-launcher] Repo root: ${repoRoot}`)

        // Create isolated sandbox workspace and user data dirs
        const testRoot = path.resolve(repoRoot, ".vscode-test")
        const workspaceDir = process.env.E2E_WORKSPACE_DIR ?? path.join(testRoot, "dev-workspace")
        const userDataDir = path.join(testRoot, "dev-user-data")
        const extensionsDir = path.join(testRoot, "dev-extensions")

        // Create directories (keep existing data for persistence)
        await fs.promises.mkdir(workspaceDir, { recursive: true })
        await fs.promises.mkdir(userDataDir, { recursive: true })
        await fs.promises.mkdir(extensionsDir, { recursive: true })

        // Ensure extension is built
        const distEntry = path.resolve(repoRoot, "dist", "extension.js")
        if (!fs.existsSync(distEntry)) {
            console.log("[dev-launcher] Building extension...")
            execSync("npm run build:extension", { cwd: repoRoot, stdio: "inherit" })
        }

        // Download VS Code Insiders (same as E2E tests)
        console.log("[dev-launcher] Downloading VS Code Insiders...")
        const vscodeExecutablePath = await downloadAndUnzipVSCode("insiders")
        const resolvedArgs = resolveCliArgsFromVSCodeExecutablePath(vscodeExecutablePath)

        console.log("[dev-launcher] Resolved args:", JSON.stringify(resolvedArgs, null, 2))

        const cli = resolvedArgs[0]
        // Skip the first element (cli) and flatten any nested arrays, ensuring all are strings
        const cliArgs = resolvedArgs.slice(1).flat(Infinity).filter(arg => typeof arg === 'string')

        // Don't add user-data-dir or extensions-dir - they're already in cliArgs
        // Just override them and add our custom args
        const launchArgs = [
            `--user-data-dir=${userDataDir}`,
            `--extensions-dir=${extensionsDir}`,
            `--extensionDevelopmentPath=${extensionDevelopmentPath}`,
            `--enable-proposed-api=SolaceHarmony.thea-code`,
            "--disable-workspace-trust",
            "--skip-release-notes",
            "--skip-welcome",
            "--wait", // Keep process attached for logging
            workspaceDir,
        ]

        console.log("[dev-launcher] Final launch args:", launchArgs)

        console.log("[dev-launcher] Launching VS Code...")
        console.log(`[dev-launcher] Workspace: ${workspaceDir}`)
        console.log(`[dev-launcher] User data: ${userDataDir}`)
        console.log(`[dev-launcher] Extensions: ${extensionsDir}`)
        console.log("")
        console.log("=".repeat(80))
        console.log("VS Code Dev Instance Launched!")
        console.log("=".repeat(80))
        console.log("")
        console.log("Environment:")
        console.log(`  THEA_E2E=1 (E2E mode enabled)`)
        console.log("")
        console.log("Logs will appear below. Press Ctrl+C to stop.")
        console.log("")
        console.log("-".repeat(80))
        console.log("")

        // Spawn VS Code with environment variables (matching E2E test env)
        const vscode = spawn(cli, launchArgs, {
            env: {
                ...process.env,
                ELECTRON_ENABLE_LOGGING: "1",
                ELECTRON_ENABLE_STACK_DUMPING: "1",
                THEA_E2E: "1", // Enable E2E mode
                THEA_PREFER_LOCAL_CONFIG: "1", // Use workspace-local config
                THEA_SKIP_MCP_PORT_WAIT: "1", // Skip MCP port wait
                THEA_SILENT_MCP_LOGS: "1", // Silence MCP logs
                NODE_ENV: "test",
                VSCODE_LOG_LEVEL: "trace", // Enable verbose logging
            },
            stdio: "inherit", // Pipe all output to console
        })

        // Handle process exit
        vscode.on("exit", (code, signal) => {
            console.log("")
            console.log("-".repeat(80))
            console.log(`[dev-launcher] VS Code exited with code ${code} and signal ${signal}`)
            process.exit(code ?? 0)
        })

        // Handle Ctrl+C
        process.on("SIGINT", () => {
            console.log("")
            console.log("[dev-launcher] Received SIGINT, shutting down...")
            vscode.kill("SIGTERM")
        })

    } catch (error) {
        console.error("[dev-launcher] Error:", error)
        process.exit(1)
    }
}

void main()
