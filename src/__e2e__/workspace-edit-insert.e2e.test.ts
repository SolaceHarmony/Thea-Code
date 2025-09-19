import * as assert from "assert"
import * as vscode from "vscode"

suite("WorkspaceEdit Insert", () => {
  test("insert text into a new file via WorkspaceEdit", async function () {
    this.timeout(30000)
    const folder = vscode.workspace.workspaceFolders?.[0]
    assert.ok(folder, "workspace folder not found")

    const uri = vscode.Uri.joinPath(folder.uri, "e2e-edit.txt")
    // Ensure file exists
    await vscode.workspace.fs.writeFile(uri, new TextEncoder().encode(""))

    const doc = await vscode.workspace.openTextDocument(uri)
    const edit = new vscode.WorkspaceEdit()
    edit.insert(uri, new vscode.Position(0, 0), "hello world")
    const applied = await vscode.workspace.applyEdit(edit)
    assert.ok(applied, "edit should be applied")

    const refreshed = await vscode.workspace.openTextDocument(uri)
    assert.strictEqual(refreshed.getText(), "hello world")
  })
})
