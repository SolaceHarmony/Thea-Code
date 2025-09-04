import * as path from "path"
import os from "os"
import * as vscode from "vscode"

export const getWorkspacePath = (defaultCwdPath = "") => {
  const cwdPath = vscode.workspace.workspaceFolders?.map((folder) => folder.uri.fsPath).at(0) || defaultCwdPath
  const currentFileUri = vscode.window.activeTextEditor?.document.uri
  if (currentFileUri) {
    const workspaceFolder = vscode.workspace.getWorkspaceFolder(currentFileUri)
    return workspaceFolder?.uri.fsPath || cwdPath
  }
  return cwdPath
}

// Convenience re-export for parity if needed elsewhere
export const getDesktopPath = () => path.join(os.homedir(), "Desktop")

