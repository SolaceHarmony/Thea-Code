import * as assert from "assert"
import * as vscode from "vscode"

suite("URI joinPath", () => {
  test("joinPath builds child path and fsPath is valid", () => {
    const root = vscode.workspace.workspaceFolders?.[0]
    assert.ok(root, "workspace not found")
    const child = vscode.Uri.joinPath(root.uri, "a", "b", "c.txt")
    assert.ok(child.fsPath.endsWith("a" + require('path').sep + "b" + require('path').sep + "c.txt"))
  })
})
