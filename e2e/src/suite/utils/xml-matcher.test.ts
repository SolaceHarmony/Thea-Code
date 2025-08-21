import { XmlMatcher } from "../xml-matcher"
import * as assert from 'assert'
import * as sinon from 'sinon'

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
})
