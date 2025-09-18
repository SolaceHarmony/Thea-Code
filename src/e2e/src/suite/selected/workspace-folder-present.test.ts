import * as assert from "assert"
import * as vscode from "vscode"

suite("Workspace Folder", () => {
  test("has a workspace folder opened", () => {
    const folders = vscode.workspace.workspaceFolders
    assert.ok(folders && folders.length >= 1, "A workspace folder should be open for e2e tests")
  })
})
