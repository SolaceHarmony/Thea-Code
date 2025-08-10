const assert = require('assert')
const { loadTheaModule } = require('../helpers/thea-loader')

describe('Roo-migration: utils/json-xml-bridge', function () {
  const bridge = loadTheaModule('src/utils/json-xml-bridge.ts')
  const {
    JsonMatcher,
    FormatDetector,
    jsonThinkingToXml,
    xmlThinkingToJson,
    jsonToolUseToXml,
    xmlToolUseToJson,
    jsonToolResultToXml,
    xmlToolResultToJson,
    openAiFunctionCallToNeutralToolUse,
    neutralToolUseToOpenAiFunctionCall,
    HybridMatcher,
  } = bridge

  it('JsonMatcher detects thinking blocks and preserves surrounding text', function () {
    const m = new JsonMatcher('thinking')
    const res = m.update('before {"type":"thinking","content":"R"} after')
    assert.strictEqual(res.length, 3)
    assert.deepStrictEqual(res[0], { matched: false, data: 'before ' })
    assert.deepStrictEqual(res[1], { matched: true, data: 'R', type: 'thinking' })
    assert.deepStrictEqual(res[2], { matched: false, data: ' after' })
  })

  it('FormatDetector detects xml/json/unknown', function () {
    const d = new FormatDetector()
    assert.strictEqual(d.detectFormat('<think>x</think>'), 'xml')
    assert.strictEqual(d.detectFormat('{"type":"thinking"}'), 'json')
    assert.strictEqual(d.detectFormat('plain'), 'unknown')
  })

  it('jsonThinkingToXml / xmlThinkingToJson roundtrip', function () {
    const xml = jsonThinkingToXml({ type: 'thinking', content: 'abc' })
    assert.strictEqual(xml, '<think>abc</think>')
    const json = xmlThinkingToJson(xml)
    assert.strictEqual(json, '{"type":"thinking","content":"abc"}')
  })

  it('jsonToolUseToXml / xmlToolUseToJson', function () {
    const xml = jsonToolUseToXml({ type: 'tool_use', name: 'read_file', id: 'id1', input: { path: 'p', n: 1 } })
    assert.strictEqual(xml, '<read_file>\n<path>p</path>\n<n>1</n>\n</read_file>')
    const back = JSON.parse(xmlToolUseToJson(xml))
    assert.strictEqual(back.type, 'tool_use')
    assert.strictEqual(back.name, 'read_file')
    assert.strictEqual(back.input.path, 'p')
  })

  it('jsonToolResultToXml / xmlToolResultToJson', function () {
    const xml = jsonToolResultToXml({
      type: 'tool_result', tool_use_id: 'tid', status: 'success', content: [{ type: 'text', text: 'hi' }]
    })
    assert.strictEqual(xml, '<tool_result tool_use_id="tid" status="success">\nhi\n</tool_result>')
    const back = JSON.parse(xmlToolResultToJson(xml))
    assert.strictEqual(back.type, 'tool_result')
    assert.strictEqual(back.tool_use_id, 'tid')
    assert.strictEqual(back.content[0].text, 'hi')
  })

  it('OpenAI function call conversions', function () {
    const nu = openAiFunctionCallToNeutralToolUse({ function_call: { name: 'read_file', arguments: '{"path":"x"}', id: 'c1' } })
    assert.strictEqual(nu.type, 'tool_use'); assert.strictEqual(nu.name, 'read_file'); assert.strictEqual(nu.id, 'c1')
    const back = neutralToolUseToOpenAiFunctionCall(nu)
    assert.strictEqual(back.function_call.name, 'read_file')
  })

  it('HybridMatcher basics', function () {
    const hm = new HybridMatcher('think', 'thinking')
    const res = hm.update('a <think>t</think> b')
    assert.ok(res.length > 0)
  })
})
