const assert = require('assert')
const fs = require('fs')
const os = require('os')
const path = require('path')
const { loadTheaModule } = require('../helpers/thea-loader')

describe('Roo-migration: integrations/misc/read-lines', function () {
  const { readLines } = loadTheaModule('src/integrations/misc/read-lines.ts')

  it('reads a range inclusive', async function () {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'rl-'))
    const file = path.join(tmp, 'a.txt')
    fs.writeFileSync(file, 'L0\nL1\nL2\nL3\nL4\n')
    const s = await readLines(file, 3, 1)
    assert.strictEqual(s, 'L1\nL2\nL3')
  })

  it('reads to end when endLine is undefined', async function () {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'rl-'))
    const file = path.join(tmp, 'b.txt')
    fs.writeFileSync(file, 'A\nB\nC')
    const s = await readLines(file, undefined, 1)
    assert.strictEqual(s, 'B\nC')
  })

  it('rejects on invalid ranges', async function () {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'rl-'))
    const file = path.join(tmp, 'c.txt')
    fs.writeFileSync(file, 'X\nY')
    await assert.rejects(() => readLines(file, 0, 2), /startLine \(2\) must be less than or equal to endLine \(0\)/)
    await assert.rejects(() => readLines(file, -1, 0), /Invalid endLine/)
    await assert.rejects(() => readLines(file, 0, -1), /Invalid startLine/)
  })

  it('rejects when out of range', async function () {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'rl-'))
    const file = path.join(tmp, 'd.txt')
    fs.writeFileSync(file, 'X\nY')
    await assert.rejects(() => readLines(file, undefined, 5), /Line with index 5 does not exist/)
  })
})
