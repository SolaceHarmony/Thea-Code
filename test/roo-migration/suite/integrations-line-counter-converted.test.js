const assert = require('assert')
const fs = require('fs')
const os = require('os')
const path = require('path')
const { loadTheaModule } = require('../helpers/thea-loader')

describe('Roo-migration: integrations/misc/line-counter', function () {
  const { countFileLines } = loadTheaModule('src/integrations/misc/line-counter.ts')

  it('counts lines in a small file', async function () {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'lc-'))
    const file = path.join(tmp, 'a.txt')
    fs.writeFileSync(file, 'a\n\nccc\n') // 3 lines
    const n = await countFileLines(file)
    assert.strictEqual(n, 3)
  })

  it('throws if file missing', async function () {
    await assert.rejects(() => countFileLines('/no/such/file'), /File not found/)
  })
})
