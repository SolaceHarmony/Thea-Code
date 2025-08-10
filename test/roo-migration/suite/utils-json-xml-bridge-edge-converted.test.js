const assert = require('assert')
const { loadTheaModule } = require('../helpers/thea-loader')

describe('Roo-migration: utils/json-xml-bridge edge cases', function () {
  const bridge = loadTheaModule('src/utils/json-xml-bridge.ts')
  const { JsonMatcher, FormatDetector } = bridge

  function extractMatched(results) {
    return results.filter(r => r.matched).map(r => r.data)
  }

  it('buffer capacity and nested braces', function () {
    const m = new JsonMatcher('test')
    const large = '{"type":"test","data":"' + 'x'.repeat(100000) + '"}'
    const res = m.update(large)
    const objs = extractMatched(res)
    assert.ok(objs.length >= 1)
    assert.strictEqual(objs[0].type, 'test')
  })

  it('partial chunks assemble into object', function () {
    const m = new JsonMatcher('test')
    m.update('{"type"')
    m.update(':"test","name"')
    const res = m.update(':"ok"}')
    const objs = extractMatched(res)
    assert.strictEqual(objs.length, 1)
    assert.deepStrictEqual(objs[0], { type: 'test', name: 'ok' })
  })

  it('multiple objects in stream', function () {
    const m = new JsonMatcher('test')
    const res = m.update('a {"type":"test","first":1} b {"type":"test","second":2} c {"type":"test","third":3}')
    const objs = extractMatched(res)
    assert.strictEqual(objs.length, 3)
    assert.deepStrictEqual(objs[0], { type: 'test', first: 1 })
    assert.deepStrictEqual(objs[1], { type: 'test', second: 2 })
    assert.deepStrictEqual(objs[2], { type: 'test', third: 3 })
  })

  it('escaped quotes handled', function () {
    const m = new JsonMatcher('test')
    const res = m.update('{"type":"test","text":"String with \\"escaped\\" quotes"}')
    const objs = extractMatched(res)
    assert.strictEqual(objs.length, 1)
    assert.strictEqual(objs[0].text, 'String with "escaped" quotes')
  })

  it('thinking extracts content', function () {
    const m = new JsonMatcher('thinking')
    const res = m.update('pfx {"type":"thinking","content":"Processing..."} sfx')
    const objs = extractMatched(res)
    assert.strictEqual(objs.length, 1)
    assert.strictEqual(objs[0], 'Processing...')
  })

  it('malformed JSON is skipped, valid kept', function () {
    const m = new JsonMatcher('test')
    const res = m.update('{"type":"test","valid":true} {broken json} {"type":"test","another":"valid"}')
    const objs = extractMatched(res)
    assert.strictEqual(objs.length, 2)
    assert.deepStrictEqual(objs[0], { type: 'test', valid: true })
    assert.deepStrictEqual(objs[1], { type: 'test', another: 'valid' })
  })

  it('arrays at top level are not matched', function () {
    const m = new JsonMatcher('test')
    const res = m.update('[{"x":1},{"x":2}]')
    assert.strictEqual(extractMatched(res).length, 0)
  })

  it('incomplete -> complete object across updates', function () {
    const m = new JsonMatcher('test')
    m.update('{"type":"test","complete":true}')
    m.update(' {"type":"test","incomplete":')
    const res = m.update('false}')
    const objs = extractMatched(res)
    assert.strictEqual(objs.length >= 1, true)
    assert.deepStrictEqual(objs[0], { type: 'test', incomplete: false })
  })

  it('ignores non-matching types', function () {
    const m = new JsonMatcher('test')
    const res = m.update('{"type":"other","data":1} {"type":"test","data":2} {"type":"different","data":3}')
    const objs = extractMatched(res)
    assert.strictEqual(objs.length, 1)
    assert.deepStrictEqual(objs[0], { type: 'test', data: 2 })
  })

  it('rapid small chunks still match', function () {
    const m = new JsonMatcher('test')
    const json = '{"type":"test","value":123}'
    let all = []
    for (const ch of json) {
      all = all.concat(m.update(ch))
    }
    const objs = extractMatched(all)
    assert.strictEqual(objs.length, 1)
    assert.deepStrictEqual(objs[0], { type: 'test', value: 123 })
  })

  it('spill behavior when exceeding max length', function () {
    const m = new JsonMatcher('test')
    const huge = '{"type":"test","data":"' + 'x'.repeat(300000) + '"'
    const res = m.update(huge)
    const nonMatched = res.filter(r => !r.matched)
    assert.ok(nonMatched.length > 0)
  })

  it('FormatDetector recognizes xml/json/unknown across variants', function () {
    const d = new FormatDetector()
    assert.strictEqual(d.detectFormat('<tool>v</tool>'), 'xml')
    assert.strictEqual(d.detectFormat('{"type":"tool_use"}'), 'json')
    assert.strictEqual(d.detectFormat('plain text'), 'unknown')
    // arrays stay unknown
    assert.strictEqual(d.detectFormat('[1,2,3]'), 'unknown')
  })
})
