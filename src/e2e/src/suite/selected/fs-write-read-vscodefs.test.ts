import * as assert from "assert"
import * as vscode from "vscode"

suite("FS write/read via vscode.workspace.fs", () => {
  test("write a file in workspace and read it back with VS Code FS", async function () {
    this.timeout(30000)
    const folder = vscode.workspace.workspaceFolders?.[0]
    assert.ok(folder, "workspace folder not found")

    const uri = vscode.Uri.joinPath(folder.uri, "e2e-vscodefs.txt")
    const encoder = new TextEncoder()
    const content = `hello ${Date.now()}`
    await vscode.workspace.fs.writeFile(uri, encoder.encode(content))

    const data = await vscode.workspace.fs.readFile(uri)
    const text = new TextDecoder().decode(data)
    assert.strictEqual(text, content)
  })
})
