import * as assert from 'assert'
import * as sinon from 'sinon'
import * as proxyquire from 'proxyquire'
import { LogLevel, CompactLogEntry, LogMeta } from '../../../../../src/utils/logging/types'

suite("CompactLogger", () => {
	let CompactLogger: any
	let MockTransport: any
	let transport: any
	let logger: any
	let clock: sinon.SinonFakeTimers

	setup(() => {
		// Create mock transport class
		MockTransport = class {
			public entries: CompactLogEntry[] = []
			public closed = false

			log(entry: CompactLogEntry): void {
				this.entries.push(entry)
			}

			close(): void {
				this.closed = true
			}

			clear(): void {
				this.entries = []
				this.closed = false
			}
		}

		// Load CompactLogger with mocked dependencies
		const module = proxyquire('../../../../../src/utils/logging/CompactLogger', {
			'./CompactTransport': {
				CompactTransport: MockTransport
			}
		})
		
		CompactLogger = module.CompactLogger

		transport = new MockTransport()
		logger = new CompactLogger(transport)
	})

	teardown(() => {
		transport.clear()
		if (clock) {
			clock.restore()
		}
	})

	suite("Log Levels", () => {
		const levels: LogLevel[] = ["debug", "info", "warn", "error", "fatal"]

		levels.forEach((level) => {
			test(`${level} level logs correctly`, () => {
				const message = `test ${level} message`
				;(logger[level] as (message: string) => void)(message)

				assert.strictEqual(transport.entries.length, 1)
				assert.strictEqual(transport.entries[0].l, level)
				assert.strictEqual(transport.entries[0].m, message)
				assert.ok(transport.entries[0].t > 0)
			})
		})
	})

	suite("Metadata Handling", () => {
		test("logs with simple metadata", () => {
			const meta = { ctx: "test", userId: "123" }
			logger.info("test message", meta)

			assert.strictEqual(transport.entries[0].m, "test message")
			assert.strictEqual(transport.entries[0].c, "test")
			assert.deepStrictEqual(transport.entries[0].d, { userId: "123" })
		})

		test("handles undefined metadata", () => {
			logger.info("test message")

			assert.strictEqual(transport.entries[0].m, "test message")
			assert.strictEqual(transport.entries[0].d, undefined)
		})

		test("strips empty metadata", () => {
			logger.info("test message", { ctx: "test" })

			assert.strictEqual(transport.entries[0].m, "test message")
			assert.strictEqual(transport.entries[0].c, "test")
			assert.strictEqual(transport.entries[0].d, undefined)
		})
	})

	suite("Error Handling", () => {
		test("handles Error objects in error level", () => {
			const error = new Error("test error")
			logger.error(error)

			assert.strictEqual(transport.entries[0].l, "error")
			assert.strictEqual(transport.entries[0].m, "test error")
			assert.strictEqual(transport.entries[0].c, "error")
			assert.strictEqual(transport.entries[0].d.error.name, "Error")
			assert.strictEqual(transport.entries[0].d.error.message, "test error")
			assert.ok(transport.entries[0].d.error.stack)
		})

		test("handles Error objects in fatal level", () => {
			const error = new Error("test fatal")
			logger.fatal(error)

			assert.strictEqual(transport.entries[0].l, "fatal")
			assert.strictEqual(transport.entries[0].m, "test fatal")
			assert.strictEqual(transport.entries[0].c, "fatal")
			assert.strictEqual(transport.entries[0].d.error.name, "Error")
			assert.strictEqual(transport.entries[0].d.error.message, "test fatal")
			assert.ok(transport.entries[0].d.error.stack)
		})

		test("handles Error objects with custom metadata", () => {
			const error = new Error("test error")
			const meta = { ctx: "custom", userId: "123" }
			logger.error(error, meta)

			assert.strictEqual(transport.entries[0].l, "error")
			assert.strictEqual(transport.entries[0].m, "test error")
			assert.strictEqual(transport.entries[0].c, "custom")
			assert.strictEqual(transport.entries[0].d.userId, "123")
			assert.strictEqual(transport.entries[0].d.error.name, "Error")
			assert.strictEqual(transport.entries[0].d.error.message, "test error")
			assert.ok(transport.entries[0].d.error.stack)
		})
	})

	suite("Child Loggers", () => {
		test("creates child logger with inherited metadata", () => {
			const parentMeta = { ctx: "parent", traceId: "123" }
			const childMeta = { ctx: "child", userId: "456" }

			const parentLogger = new CompactLogger(transport, parentMeta)
			const childLogger = parentLogger.child(childMeta)

			childLogger.info("test message")

			assert.strictEqual(transport.entries[0].m, "test message")
			assert.strictEqual(transport.entries[0].c, "child")
			assert.strictEqual(transport.entries[0].d.traceId, "123")
			assert.strictEqual(transport.entries[0].d.userId, "456")
		})

		test("child logger respects parent context when not overridden", () => {
			const parentLogger = new CompactLogger(transport, { ctx: "parent" })
			const childLogger = parentLogger.child({ userId: "123" })

			childLogger.info("test message")

			assert.strictEqual(transport.entries[0].m, "test message")
			assert.strictEqual(transport.entries[0].c, "parent")
			assert.strictEqual(transport.entries[0].d.userId, "123")
		})
	})

	suite("Lifecycle", () => {
		test("closes transport on logger close", () => {
			logger.close()
			assert.strictEqual(transport.closed, true)
		})
	})

	suite("Timestamp Handling", () => {
		setup(() => {
			clock = sinon.useFakeTimers()
		})

		teardown(() => {
			clock.restore()
		})

		test("generates increasing timestamps", () => {
			const now = Date.now()
			clock.setSystemTime(now)

			logger.info("first")
			clock.setSystemTime(now + 10)
			logger.info("second")

			assert.ok(transport.entries[0].t < transport.entries[1].t)
		})

		test("uses current timestamp for entries", () => {
			const baseTime = 1000000000000
			clock.setSystemTime(baseTime)

			logger.info("test")
			assert.strictEqual(transport.entries[0].t, baseTime)
		})

		test("timestamps reflect time progression", () => {
			const baseTime = 1000000000000
			clock.setSystemTime(baseTime)

			logger.info("first")
			clock.setSystemTime(baseTime + 100)
			logger.info("second")

			assert.strictEqual(transport.entries.length, 2)
			assert.strictEqual(transport.entries[0].t, baseTime)
			assert.strictEqual(transport.entries[1].t, baseTime + 100)
		})
	})

	suite("Message Handling", () => {
		test("handles empty string messages", () => {
			logger.info("")
			assert.strictEqual(transport.entries[0].m, "")
			assert.strictEqual(transport.entries[0].l, "info")
		})
	})

	suite("Metadata Edge Cases", () => {
		test("handles metadata with undefined values", () => {
			const meta = {
				ctx: "test",
				someField: undefined,
				validField: "value",
			}

			logger.info("test", meta)

			assert.strictEqual(transport.entries[0].d.someField, undefined)
			assert.strictEqual(transport.entries[0].d.validField, "value")
		})

		test("handles metadata with null values", () => {
			logger.info("test", { ctx: "test", nullField: null })
			assert.strictEqual(transport.entries[0].d.nullField, null)
		})

		test("maintains metadata value types", () => {
			const meta = {
				str: "string",
				num: 123,
				bool: true,
				arr: [1, 2, 3],
				obj: { nested: true },
			}

			logger.info("test", meta)
			assert.deepStrictEqual(transport.entries[0].d, meta)
		})
	})

	suite("Child Logger Edge Cases", () => {
		test("deeply nested child loggers maintain correct metadata inheritance", () => {
			const root = new CompactLogger(transport, { ctx: "root", rootVal: 1 })
			const child1 = root.child({ level1: "a" })
			const child2 = child1.child({ level2: "b" })
			const child3 = child2.child({ ctx: "leaf" })

			child3.info("test")

			assert.strictEqual(transport.entries[0].c, "leaf")
			assert.strictEqual(transport.entries[0].d.rootVal, 1)
			assert.strictEqual(transport.entries[0].d.level1, "a")
			assert.strictEqual(transport.entries[0].d.level2, "b")
		})

		test("child logger with empty metadata inherits parent metadata unchanged", () => {
			const parent = new CompactLogger(transport, { ctx: "parent", data: "value" })
			const child = parent.child({})

			child.info("test")

			assert.strictEqual(transport.entries[0].c, "parent")
			assert.strictEqual(transport.entries[0].d.data, "value")
		})
	})

	suite("Error Handling Edge Cases", () => {
		test("handles custom error types", () => {
			class CustomError extends Error {
				constructor(
					message: string,
					public code: string,
				) {
					super(message)
					this.name = "CustomError"
				}
			}

			const error = new CustomError("custom error", "ERR_CUSTOM")
			logger.error(error)

			assert.strictEqual(transport.entries[0].m, "custom error")
			assert.strictEqual(transport.entries[0].d.error.name, "CustomError")
			assert.strictEqual(transport.entries[0].d.error.message, "custom error")
			assert.ok(transport.entries[0].d.error.stack)
		})

		test("handles errors without stack traces", () => {
			const error = new Error("test")
			delete error.stack

			logger.error(error)

			assert.strictEqual(transport.entries[0].d.error.name, "Error")
			assert.strictEqual(transport.entries[0].d.error.message, "test")
			assert.strictEqual(transport.entries[0].d.error.stack, undefined)
		})
	})
// Mock cleanup
