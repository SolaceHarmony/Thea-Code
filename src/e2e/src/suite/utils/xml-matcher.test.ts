import * as assert from 'assert'
import * as path from 'path'
import * as fs from 'fs'
import * as sinon from 'sinon'

function findRepoRoot(startDir: string): string {
  let dir = startDir
  for (let i = 0; i < 10; i++) {
    try {
      const pkg = JSON.parse(fs.readFileSync(path.join(dir, 'package.json'), 'utf8'))
      if (pkg && pkg.name === 'thea-code') return dir
    } catch {}
    const parent = path.dirname(dir)
    if (parent === dir) break
    dir = parent
  }
  return path.resolve(startDir, '../../../../..')
}

const repoRoot = findRepoRoot(__dirname)
// eslint-disable-next-line @typescript-eslint/no-var-requires
const { XmlMatcher } = require(path.join(repoRoot, 'out', 'utils', 'xml-matcher.js'))

suite("XmlMatcher", () => {
	let sandbox: sinon.SinonSandbox

	setup(() => {
		sandbox = sinon.createSandbox()
	})

	teardown(() => {
		sandbox.restore()
	})
	test("only match at position 0", () => {
		const matcher = new XmlMatcher("think")
		const chunks = [...matcher.update("<think>data</think>"), ...matcher.final()]
		assert.strictEqual(chunks.length, 1)
		assert.deepStrictEqual(chunks, [
			{
				matched: true,
				data: "data",
			},
		])
	})
	test("tag with space", () => {
		const matcher = new XmlMatcher("think")
		const chunks = [...matcher.update("< think >data</ think >"), ...matcher.final()]
		assert.strictEqual(chunks.length, 1)
		assert.deepStrictEqual(chunks, [
			{
				matched: true,
				data: "data",
			},
		])
	})

	test("invalid tag", () => {
		const matcher = new XmlMatcher("think")
		const chunks = [...matcher.update("< think 1>data</ think >"), ...matcher.final()]
		assert.strictEqual(chunks.length, 1)
		assert.deepStrictEqual(chunks, [
			{
				matched: false,
				data: "< think 1>data</ think >",
			},
		])
	})

	test("anonymous tag", () => {
		const matcher = new XmlMatcher("think")
		const chunks = [...matcher.update("<>data</>"), ...matcher.final()]
		assert.strictEqual(chunks.length, 1)
		assert.deepStrictEqual(chunks, [
			{
				matched: false,
				data: "<>data</>",
			},
		])
	})

	test("streaming push", () => {
		const matcher = new XmlMatcher("think")
		const chunks = [
			...matcher.update("<thi"),
			...matcher.update("nk"),
			...matcher.update(">dat"),
			...matcher.update("a</"),
			...matcher.update("think>"),
		]
		assert.strictEqual(chunks.length, 2)
		assert.deepStrictEqual(chunks, [
			{
				matched: true,
				data: "dat",
			},
			{
				matched: true,
				data: "a",
			},
		])
	})

	test("nested tag", () => {
		const matcher = new XmlMatcher("think")
		const chunks = [...matcher.update("<think>X<think>Y</think>Z</think>"), ...matcher.final()]
		assert.strictEqual(chunks.length, 1)
		assert.deepStrictEqual(chunks, [
			{
				matched: true,
				data: "X<think>Y</think>Z",
			},
		])
	})

	test("nested invalid tag", () => {
		const matcher = new XmlMatcher("think")
		const chunks = [...matcher.update("<think>X<think>Y</thxink>Z</think>"), ...matcher.final()]
		assert.strictEqual(chunks.length, 2)
		assert.deepStrictEqual(chunks, [
			{
				matched: true,
				data: "X<think>Y</thxink>Z",
			},
			{
				matched: true,
				data: "</think>",
			},
		])
	})

	test("Wrong matching position", () => {
		const matcher = new XmlMatcher("think")
		const chunks = [...matcher.update("1<think>data</think>"), ...matcher.final()]
		assert.strictEqual(chunks.length, 1)
		assert.deepStrictEqual(chunks, [
			{
				matched: false,
				data: "1<think>data</think>",
			},
		])
	})

	test("Unclosed tag", () => {
		const matcher = new XmlMatcher("think")
		const chunks = [...matcher.update("<think>data"), ...matcher.final()]
		assert.strictEqual(chunks.length, 1)
		assert.deepStrictEqual(chunks, [
			{
				matched: true,
				data: "data",
			},
		])
	})
// Mock cleanup

})
