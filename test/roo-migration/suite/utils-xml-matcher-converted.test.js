const assert = require('assert')
const { loadTheaModule } = require('../helpers/thea-loader')

describe('Roo-migration: utils/xml-matcher', function () {
  const { XmlMatcher } = loadTheaModule('src/utils/xml-matcher.ts')

  it('matches simple thinking block', function () {
    const m = new XmlMatcher('think')
    const chunks = m.update('<think>data</think>')
    assert.deepStrictEqual(chunks, [{ matched: true, data: 'data' }])
  })

  it('streams across chunks', function () {
    const m = new XmlMatcher('think')
  const parts = []
  parts.push(...m.update('<thi'))
  parts.push(...m.update('nk>abc</'))
  parts.push(...m.update('think>'))
  // Implementation may split matched chunks; assert on concatenated matched data
  const matched = parts.filter(p => p.matched).map(p => p.data).join('')
  assert.strictEqual(matched, 'abc')
  })
})
