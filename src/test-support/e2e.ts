import * as assert from "assert"
import * as vscode from "vscode"

/** Ensure the extension is activated and return its Extension object. */
export async function ensureActivated(extensionId: string): Promise<vscode.Extension<unknown>> {
  const ext = vscode.extensions.getExtension(extensionId)
  assert.ok(ext, `Extension ${extensionId} not found`)
  if (!ext.isActive) {
    await ext.activate()
  }
  assert.ok(ext.isActive, `Extension ${extensionId} did not activate`)
  return ext
}

/** Return the first workspace folder or throw if none. */
export function getWorkspaceFolder(): vscode.WorkspaceFolder {
  const folder = vscode.workspace.workspaceFolders?.[0]
  assert.ok(folder, "workspace folder not found")
  return folder
}
