const assert = require('assert')
const { loadTheaModule } = require('../helpers/thea-loader')

// Patch the local vscode stub workspaceFolders for these tests
const vscode = require('../node_modules/vscode')

describe('Roo-migration: utils/pathUtils', function () {
  const { isPathOutsideWorkspace } = loadTheaModule('src/utils/pathUtils.ts')

  beforeEach(() => {
    vscode.workspace.workspaceFolders = [
      { uri: { fsPath: '/ws/a' } },
      { uri: { fsPath: '/ws/b' } },
    ]
  })

  it('returns false for paths inside a workspace folder', function () {
    vscode.workspace.workspaceFolders = [ { uri: { fsPath: '/ws/a' } } ]
    assert.strictEqual(isPathOutsideWorkspace('/ws/a/src/file.ts'), false)
    assert.strictEqual(isPathOutsideWorkspace('/ws/a'), false)
  })

  it('returns true for paths outside all workspace folders', function () {
    assert.strictEqual(isPathOutsideWorkspace('/other/path/file.ts'), true)
    assert.strictEqual(isPathOutsideWorkspace('/ws/ax/file.ts'), true)
  })

  it('handles multiple workspace folders', function () {
    assert.strictEqual(isPathOutsideWorkspace('/ws/b/docs/readme.md'), false)
    assert.strictEqual(isPathOutsideWorkspace('/ws/bb/docs'), true)
  })

  it('returns true when no workspace folders are set', function () {
    vscode.workspace.workspaceFolders = null
    assert.strictEqual(isPathOutsideWorkspace('/anything'), true)
  })
})
