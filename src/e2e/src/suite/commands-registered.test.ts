import * as assert from "assert"
import * as vscode from "vscode"
import { CMD } from "../thea-constants"

suite("Commands", () => {
  test("core commands are registered", async () => {
    const all = await vscode.commands.getCommands(true)
    const expected = [
      CMD.plusButtonClicked,
      CMD.historyButtonClicked,
      CMD.settingsButtonClicked,
    ]
    const missing = expected.filter((c) => !all.includes(c))
    assert.deepStrictEqual(missing, [], `Missing commands: ${missing.join(", ")}`)
  })
})
