const assert = require('assert')
const { loadTheaModule } = require('../helpers/thea-loader')

describe('Roo-migration: api/transform/r1-format', function () {
  const mod = loadTheaModule('src/api/transform/r1-format.ts')
  const { convertToR1Format } = mod

  it('converts basic text messages', function () {
    const input = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi there' },
    ]
    const expected = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi there' },
    ]
    assert.deepStrictEqual(convertToR1Format(input), expected)
  })

  it('merges consecutive messages with same role', function () {
    const input = [
      { role: 'user', content: 'Hello' },
      { role: 'user', content: 'How are you?' },
      { role: 'assistant', content: 'Hi!' },
      { role: 'assistant', content: "I'm doing well" },
    ]
    const expected = [
      { role: 'user', content: 'Hello\nHow are you?' },
      { role: 'assistant', content: "Hi!\nI'm doing well" },
    ]
    assert.deepStrictEqual(convertToR1Format(input), expected)
  })

  it('handles image content', function () {
    const input = [
      { role: 'user', content: [ { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: 'base64data' } } ] },
    ]
    const out = convertToR1Format(input)
    assert.strictEqual(out.length, 1)
    assert.strictEqual(out[0].role, 'user')
    assert.ok(Array.isArray(out[0].content))
    const parts = out[0].content
    assert.strictEqual(parts.length, 1)
    assert.strictEqual(parts[0].type, 'image_url')
    const url = typeof parts[0].image_url === 'string' ? parts[0].image_url : parts[0].image_url.url
    assert.strictEqual(url, 'data:image/jpeg;base64,base64data')
  })

  it('handles mixed text and image content', function () {
    const input = [
      { role: 'user', content: [
        { type: 'text', text: 'Check this image:' },
        { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: 'base64data' } },
      ] },
    ]
    const out = convertToR1Format(input)
    assert.strictEqual(out.length, 1)
    assert.strictEqual(out[0].role, 'user')
    const parts = out[0].content
    assert.ok(Array.isArray(parts))
    assert.strictEqual(parts.length, 2)
    assert.deepStrictEqual(parts[0], { type: 'text', text: 'Check this image:' })
    const url = typeof parts[1].image_url === 'string' ? parts[1].image_url : parts[1].image_url.url
    assert.strictEqual(parts[1].type, 'image_url')
    assert.strictEqual(url, 'data:image/jpeg;base64,base64data')
  })

  it('merges mixed content messages with same role', function () {
    const input = [
      { role: 'user', content: [
        { type: 'text', text: 'First image:' },
        { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: 'image1' } },
      ] },
      { role: 'user', content: [
        { type: 'text', text: 'Second image:' },
        { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'image2' } },
      ] },
    ]
    const out = convertToR1Format(input)
    assert.strictEqual(out.length, 1)
    assert.strictEqual(out[0].role, 'user')
    const parts = out[0].content
    assert.ok(Array.isArray(parts))
    assert.strictEqual(parts.length, 4)
    assert.deepStrictEqual(parts[0], { type: 'text', text: 'First image:' })
    assert.strictEqual(parts[1].type, 'image_url')
    let url = typeof parts[1].image_url === 'string' ? parts[1].image_url : parts[1].image_url.url
    assert.strictEqual(url, 'data:image/jpeg;base64,image1')
    assert.deepStrictEqual(parts[2], { type: 'text', text: 'Second image:' })
    assert.strictEqual(parts[3].type, 'image_url')
    url = typeof parts[3].image_url === 'string' ? parts[3].image_url : parts[3].image_url.url
    assert.strictEqual(url, 'data:image/png;base64,image2')
  })

  it('handles empty messages array', function () {
    assert.deepStrictEqual(convertToR1Format([]), [])
  })

  it('handles messages with empty content', function () {
    const input = [ { role: 'user', content: '' }, { role: 'assistant', content: '' } ]
    const expected = [ { role: 'user', content: '' }, { role: 'assistant', content: '' } ]
    assert.deepStrictEqual(convertToR1Format(input), expected)
  })

  it('handles unknown parts by inserting placeholder text when merging', function () {
    const input = [
      { role: 'user', content: [ { type: 'text', text: 'hello' }, { type: 'unknown', foo: 'bar' } ] },
      { role: 'user', content: [ { type: 'text', text: 'world' } ] },
    ]
    const out = convertToR1Format(input)
    assert.strictEqual(out.length, 1)
    const first = out[0]
    assert.strictEqual(first.role, 'user')
    const expectContains = (s) => {
      assert.ok(s.includes('hello'))
      assert.ok(s.includes('world'))
      assert.ok(s.includes('[Unsupported block type:'))
      assert.ok(!s.includes('foo'))
    }
    if (typeof first.content === 'string') {
      expectContains(first.content)
    } else {
      const textPart = first.content.find((p) => p.type === 'text')
      const text = textPart ? textPart.text : ''
      expectContains(text)
    }
  })
})
