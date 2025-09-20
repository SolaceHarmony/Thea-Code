import * as assert from "assert"
import * as vscode from "vscode"
import * as fs from "fs"
import * as path from "path"

suite("FS write/read via Node", () => {
  test("write a file in workspace and read it back", function () {
    this.timeout(30000)
    const folder = vscode.workspace.workspaceFolders?.[0]
    assert.ok(folder, "workspace folder not found")

    const filePath = path.join(folder.uri.fsPath, "e2e-node.txt")
    const content = `hello ${Date.now()}`
    fs.writeFileSync(filePath, content, "utf8")

    const data = fs.readFileSync(filePath, "utf8")
    assert.strictEqual(data, content)
  })
})
