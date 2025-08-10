const assert = require('assert')
const { loadTheaModule } = require('../helpers/thea-loader')

describe('Roo-migration: shared/array', function () {
  it('findLastIndex returns last index matching predicate', function () {
    const { findLastIndex } = loadTheaModule('src/shared/array.ts')
    const arr = [1, 2, 3, 2]
    const idx = findLastIndex(arr, x => x === 2)
    assert.strictEqual(idx, 3)
  })

  it('findLastIndex returns -1 when no match', function () {
    const { findLastIndex } = loadTheaModule('src/shared/array.ts')
    const arr = [1, 2, 3]
    const idx = findLastIndex(arr, x => x === 4)
    assert.strictEqual(idx, -1)
  })

  it('findLast returns last element matching predicate', function () {
    const { findLast } = loadTheaModule('src/shared/array.ts')
    const arr = ['a', 'b', 'c', 'b']
    const val = findLast(arr, x => x === 'b')
    assert.strictEqual(val, 'b')
  })

  it('findLast returns undefined when no match', function () {
    const { findLast } = loadTheaModule('src/shared/array.ts')
    const arr = []
    const val = findLast(arr, x => x > 0)
    assert.strictEqual(val, undefined)
  })
})
