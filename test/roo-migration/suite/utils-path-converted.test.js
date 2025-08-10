const assert = require('assert')
const os = require('os')
const path = require('path')
const { loadTheaModule } = require('../helpers/thea-loader')

describe('Roo-migration: utils/path', function () {
  const mod = loadTheaModule('src/utils/path.ts')
  const { arePathsEqual, getReadablePath } = mod

  it('toPosix extension works', function () {
  // String with backslashes: C:\\Users\\test\\file.txt
  const windowsPath = 'C:\\Users\\test\\file.txt'
    assert.strictEqual(windowsPath.toPosix(), 'C:/Users/test/file.txt')
  })

  it('arePathsEqual normalizes and compares', function () {
    const p1 = path.normalize('/Users/test/Proj/../Proj').replace(/\\/g,'/')
    const p2 = '/Users/test/Proj'
    assert.strictEqual(arePathsEqual(p1, p2), true)
  })

  it('getReadablePath returns relative within cwd and absolute outside', function () {
    const home = os.homedir()
    const cwd = path.join(home, 'project')
    const inside = path.join(cwd, 'src', 'a.txt')
    const outside = path.join(home, 'other', 'b.txt')
    assert.strictEqual(getReadablePath(cwd, inside), 'src/a.txt')
    assert.strictEqual(getReadablePath(cwd, outside), outside.toPosix())
  })
})
