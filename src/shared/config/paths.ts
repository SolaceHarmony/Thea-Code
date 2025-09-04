import * as path from 'path'
import type * as vscode from 'vscode'
import { EXTENSION_CONFIG_DIR, EXTENSION_DISPLAY_NAME } from './thea-config'
import * as os from 'os'

/** Returns true if running in E2E/test mode. */
export const isTestMode = () => process.env.THEA_E2E === '1' || process.env.NODE_ENV === 'test'

/** Returns true if we should prefer local (workspace/globalStorage) config over home dir. */
export const preferLocalConfig = () => process.env.THEA_PREFER_LOCAL_CONFIG === '1' || isTestMode()

/**
 * Compute the root folder to store Thea config files in a sandboxed way.
 * Priority:
 * 1) When preferLocalConfig() is true, use workspace root if provided, else VS Code globalStorageUri.
 * 2) Otherwise, fall back to user home platform-specific folders used previously.
 */
export function getPreferredConfigRoot(opts: {
  context?: vscode.ExtensionContext
  workspaceRoot?: string
} = {}): string {
  const { context, workspaceRoot } = opts
  if (preferLocalConfig()) {
    const base = workspaceRoot || context?.globalStorageUri?.fsPath || process.cwd()
    return path.join(base, EXTENSION_CONFIG_DIR)
  }

  // Platform-specific default outside of tests (legacy behavior)
  if (process.platform === 'win32') {
    return path.join(os.homedir(), 'AppData', 'Roaming', String(EXTENSION_DISPLAY_NAME))
  }
  if (process.platform === 'darwin') {
    return path.join(os.homedir(), 'Documents', String(EXTENSION_DISPLAY_NAME))
  }
  return path.join(os.homedir(), '.local', 'share', String(EXTENSION_DISPLAY_NAME))
}

/** Preferred MCP servers directory respecting sandboxing rules. */
export function getPreferredMcpServersDir(opts: {
  context?: vscode.ExtensionContext
  workspaceRoot?: string
} = {}): string {
  const root = getPreferredConfigRoot(opts)
  return path.join(root, 'mcp')
}

