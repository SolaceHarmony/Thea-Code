const assert = require('assert')
const { loadTheaModule } = require('../helpers/thea-loader')

describe('Roo-migration: utils/shell', function () {
  const { getShell } = loadTheaModule('src/utils/shell.ts')

  it('returns a shell string without throwing', function () {
    const s = getShell()
    assert.strictEqual(typeof s, 'string')
    assert.ok(s.length > 0)
  })
})
