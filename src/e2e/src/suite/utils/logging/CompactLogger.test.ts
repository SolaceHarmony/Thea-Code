// __tests__/CompactLogger.test.ts
import { CompactLogger } from "../CompactLogger"
import { MockTransport } from "./MockTransport"
import { LogLevel } from "../types"

import * as assert from 'assert'
import * as sinon from 'sinon'
suite("CompactLogger", () => {
	let transport: MockTransport
	let logger: CompactLogger

	setup(() => {
		transport = new MockTransport()
		logger = new CompactLogger(transport)

	teardown(() => {
		transport.clear()

	suite("Log Levels", () => {
		const levels: LogLevel[] = ["debug", "info", "warn", "error", "fatal"]

		levels.forEach((level) => {
			test(`${level} level logs correctly`, () => {
				const message = `test ${level} message`
				;(logger[level] as (message: string) => void)(message)

				assert.strictEqual(transport.entries.length, 1)
				expect(transport.entries[0]).toMatchObject({
					l: level,
					m: message,
				assert.ok(transport.entries[0].t > 0)

	suite("Metadata Handling", () => {
		test("logs with simple metadata", () => {
			const meta = { ctx: "test", userId: "123" }
			logger.info("test message", meta)

			expect(transport.entries[0]).toMatchObject({
				m: "test message",
				c: "test",
				d: { userId: "123" },

		test("handles undefined metadata", () => {
			logger.info("test message")

			expect(transport.entries[0]).toMatchObject({
				m: "test message",
			assert.strictEqual(transport.entries[0].d, undefined)

		test("strips empty metadata", () => {
			logger.info("test message", { ctx: "test" })

			expect(transport.entries[0]).toMatchObject({
				m: "test message",
				c: "test",
			assert.strictEqual(transport.entries[0].d, undefined)

	suite("Error Handling", () => {
		test("handles Error objects in error level", () => {
			const error = new Error("test error")
			logger.error(error)

			expect(transport.entries[0]).toMatchObject({
				l: "error",
				m: "test error",
				c: "error",
				d: {
					error: {
						name: "Error",
						message: "test error",
						stack: error.stack,
					},
				},

		test("handles Error objects in fatal level", () => {
			const error = new Error("test fatal")
			logger.fatal(error)

			expect(transport.entries[0]).toMatchObject({
				l: "fatal",
				m: "test fatal",
				c: "fatal",
				d: {
					error: {
						name: "Error",
						message: "test fatal",
						stack: error.stack,
					},
				},

		test("handles Error objects with custom metadata", () => {
			const error = new Error("test error")
			const meta = { ctx: "custom", userId: "123" }
			logger.error(error, meta)

			expect(transport.entries[0]).toMatchObject({
				l: "error",
				m: "test error",
				c: "custom",
				d: {
					userId: "123",
					error: {
						name: "Error",
						message: "test error",
						stack: error.stack,
					},
				},

	suite("Child Loggers", () => {
		test("creates child logger with inherited metadata", () => {
			const parentMeta = { ctx: "parent", traceId: "123" }
			const childMeta = { ctx: "child", userId: "456" }

			const parentLogger = new CompactLogger(transport, parentMeta)
			const childLogger = parentLogger.child(childMeta)

			childLogger.info("test message")

			expect(transport.entries[0]).toMatchObject({
				m: "test message",
				c: "child",
				d: {
					traceId: "123",
					userId: "456",
				},

		test("child logger respects parent context when not overridden", () => {
			const parentLogger = new CompactLogger(transport, { ctx: "parent" })
			const childLogger = parentLogger.child({ userId: "123" })

			childLogger.info("test message")

			expect(transport.entries[0]).toMatchObject({
				m: "test message",
				c: "parent",
				d: { userId: "123" },

	suite("Lifecycle", () => {
		test("closes transport on logger close", () => {
			logger.close()
			assert.strictEqual(transport.closed, true)

	suite("Timestamp Handling", () => {
		setup(() => {
			jest.useFakeTimers()

		teardown(() => {
			jest.useRealTimers()

		test("generates increasing timestamps", () => {
			const now = Date.now()
			jest.setSystemTime(now)

			logger.info("first")
			jest.setSystemTime(now + 10)
			logger.info("second")

			assert.ok(transport.entries[0].t < transport.entries[1].t)

	suite("Message Handling", () => {
		test("handles empty string messages", () => {
			logger.info("")
			expect(transport.entries[0]).toMatchObject({
				m: "",
				l: "info",

	suite("Metadata Edge Cases", () => {
		test("handles metadata with undefined values", () => {
			const meta = {
				ctx: "test",
				someField: undefined,
				validField: "value",

			logger.info("test", meta)

			expect(transport.entries[0].d).toMatchObject({
				someField: undefined,
				validField: "value",

		test("handles metadata with null values", () => {
			logger.info("test", { ctx: "test", nullField: null })
			expect(transport.entries[0].d).toMatchObject({ nullField: null })

		test("maintains metadata value types", () => {
			const meta = {
				str: "string",
				num: 123,
				bool: true,
				arr: [1, 2, 3],
				obj: { nested: true },

			logger.info("test", meta)
			assert.deepStrictEqual(transport.entries[0].d, meta)

	suite("Child Logger Edge Cases", () => {
		test("deeply nested child loggers maintain correct metadata inheritance", () => {
			const root = new CompactLogger(transport, { ctx: "root", rootVal: 1 })
			const child1 = root.child({ level1: "a" })
			const child2 = child1.child({ level2: "b" })
			const child3 = child2.child({ ctx: "leaf" })

			child3.info("test")

			expect(transport.entries[0]).toMatchObject({
				c: "leaf",
				d: {
					rootVal: 1,
					level1: "a",
					level2: "b",
				},

		test("child logger with empty metadata inherits parent metadata unchanged", () => {
			const parent = new CompactLogger(transport, { ctx: "parent", data: "value" })
			const child = parent.child({})

			child.info("test")

			expect(transport.entries[0]).toMatchObject({
				c: "parent",
				d: { data: "value" },

	suite("Error Handling Edge Cases", () => {
		test("handles custom error types", () => {
			class CustomError extends Error {
				constructor(
					message: string,
					public code: string,
				) {
					super(message)
					this.name = "CustomError"

			const error = new CustomError("custom error", "ERR_CUSTOM")
			logger.error(error)

			expect(transport.entries[0]).toMatchObject({
				m: "custom error",
				d: {
					error: {
						name: "CustomError",
						message: "custom error",
						stack: error.stack,
					},
				},

		test("handles errors without stack traces", () => {
			const error = new Error("test")
			delete error.stack

			logger.error(error)

			expect(transport.entries[0].d).toMatchObject({
				error: {
					name: "Error",
					message: "test",
					stack: undefined,
				},

	suite("Timestamp Generation", () => {
		setup(() => {
			jest.useFakeTimers()

		teardown(() => {
			jest.useRealTimers()

		test("uses current timestamp for entries", () => {
			const baseTime = 1000000000000
			jest.setSystemTime(baseTime)

			logger.info("test")
			assert.strictEqual(transport.entries[0].t, baseTime)

		test("timestamps reflect time progression", () => {
			const baseTime = 1000000000000
			jest.setSystemTime(baseTime)

			logger.info("first")
			jest.setSystemTime(baseTime + 100)
			logger.info("second")

			assert.strictEqual(transport.entries.length, 2)
			assert.strictEqual(transport.entries[0].t, baseTime)
			assert.strictEqual(transport.entries[1].t, baseTime + 100)
