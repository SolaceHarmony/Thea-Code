const vscode = require("vscode")
let ctx

async function activate(context) {
  ctx = context
  // Lightweight: do nothing else; expose context for tests
  return { getContext: () => ctx }
}

async function deactivate() {}

module.exports = { activate, deactivate }
