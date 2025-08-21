import * as assert from 'assert'
import * as sinon from 'sinon'
/**
 * port-utils retry and timeout tests
 * Tests retry logic, exponential backoff, timeout handling, and abort signals
 */

import { 
	isPortAvailable, 
	findAvailablePort, 
	waitForPortAvailable, 
	waitForPortInUse 
} from "../port-utils"
import { logger } from "../logging"

// Mock tcp-port-used module
// TODO: Use proxyquire for module mocking
		// Mock for "tcp-port-used" needed here
	check: sinon.stub(),
	waitUntilFree: sinon.stub(),
	waitUntilUsed: sinon.stub()
// Mock cleanup needed

// Mock logger
// TODO: Use proxyquire for module mocking
		// Mock for "../logging" needed here
	logger: {
		info: sinon.stub(),
		warn: sinon.stub(),
		error: sinon.stub()
	}
// Mock cleanup

suite("port-utils - Retry and Timeout Tests", () => {
	let mockTcpPortUsed: any
	let mockLogger: sinon.SinonStubbedInstance<typeof logger>
	let originalTestWorkerId: string | undefined
	let originalNodeEnv: string | undefined
	let consoleErrorSpy: sinon.SinonStub
	let consoleWarnSpy: sinon.SinonStub
	let consoleLogSpy: sinon.SinonStub

	setup(() => {
		// Clear all mocks
		sinon.restore()
		
		// Setup console spies
		consoleErrorSpy = sinon.spy(console, 'error').callsFake()
		consoleWarnSpy = sinon.spy(console, 'warn').callsFake()
		consoleLogSpy = sinon.spy(console, 'log').callsFake()

		// Get mocked module
		mockTcpPortUsed = require("tcp-port-used")
		mockLogger = logger as sinon.SinonStubbedInstance<typeof logger>
		
		// Save and clear environment variables to simulate non-test environment
		originalTestWorkerId = process.env.MOCHA_WORKER_ID
		originalNodeEnv = process.env.NODE_ENV
		delete process.env.MOCHA_WORKER_ID
		delete process.env.NODE_ENV
		
		// Clear global test framework reference
		// (typically not needed for Mocha)
	})

	teardown(() => {
		// Restore environment variables
		if (originalTestWorkerId) {
			process.env.MOCHA_WORKER_ID = originalTestWorkerId
		}
		if (originalNodeEnv) {
			process.env.NODE_ENV = originalNodeEnv
		}
		
		// No need to restore global mocha
		
		sinon.restore()
	})

	suite("isPortAvailable", () => {
		test("should handle tcp-port-used errors gracefully", async () => {
			mockTcpPortUsed.check.rejects(new Error("Network error"))
			
			const result = await isPortAvailable(3000)
			
			assert.strictEqual(result, false)
			assert.ok(consoleErrorSpy.calledWith(
				"Error checking port 3000 availability:",
				sinon.match.instanceOf(Error))
			)
		})

		test("should return true when port is available", async () => {
			mockTcpPortUsed.check.resolves(false) // false = not in use
			
			const result = await isPortAvailable(3000)
			
			assert.strictEqual(result, true)
			assert.ok(mockTcpPortUsed.check.calledWith(3000, "localhost"))
		})

		test("should return false when port is in use", async () => {
			mockTcpPortUsed.check.resolves(true) // true = in use
			
			const result = await isPortAvailable(3000)
			
			assert.strictEqual(result, false)
		})
	})

	suite("findAvailablePort", () => {
		test("should retry up to maxAttempts times", async () => {
			// All ports unavailable
			mockTcpPortUsed.check.resolves(true)
			
			try {
				await findAvailablePort(3000, 'localhost', undefined, 5)
				assert.fail('Should have thrown an error')
			} catch (error) {
				assert.ok(error instanceof Error)
				assert.strictEqual(error.message, "No available ports found after 5 attempts")
			}
			
			// Should have checked 5 sequential ports plus last resort ports
			// The function tries maxAttempts sequential ports then tries last resort ports
			assert.ok(mockTcpPortUsed.check.mock.calls.length >= 5)
		})

		test("should switch to random port search after 20 sequential attempts", async () => {
			let callCount = 0
			mockTcpPortUsed.check.callsFake(async (port: number) => {
				callCount++
				// Make port available on 21st attempt
				return callCount < 21
			})
			
			const result = await findAvailablePort(3000)
			
			// Should have switched to random search
			assert.ok(consoleLogSpy.calledWith(
				// TODO: String contains check - "Switching to random port search"))
			)
			
			// Should eventually find a port
			assert.ok(result >= 1024)
			assert.ok(result <= 65535)
		})

		test("should try last resort ports when regular search fails", async () => {
			let callCount = 0
			mockTcpPortUsed.check.callsFake(async (port: number) => {
				callCount++
				// Make port 8080 available (a last resort port)
				return port !== 8080
			})
			
			const result = await findAvailablePort(3000, 'localhost', undefined, 10)
			
			assert.strictEqual(result, 8080)
			assert.ok(consoleLogSpy.calledWith("Trying last resort ports..."))
			assert.ok(consoleLogSpy.calledWith("Found available last resort port 8080"))
		})

		test("should handle preferred ranges with retries", async () => {
			let checkCount = 0
			mockTcpPortUsed.check.callsFake(async () => {
				checkCount++
				// Fail first 5 attempts in preferred range, then succeed
				return checkCount <= 5
			})
			
			const result = await findAvailablePort(
				3000, 
				'localhost',
				[[10000, 10100]]
			)
			
			// Should find port in preferred range
			assert.ok(result >= 10000)
			assert.ok(result <= 10100)
			assert.ok(consoleLogSpy.calledWith(
				// TODO: String contains check - "Found available port"))
			)
		})

		test("should validate and skip invalid port ranges", async () => {
			mockTcpPortUsed.check.resolves(false) // Port available
			
			await findAvailablePort(
				3000,
				'localhost',
				[[500, 1000], [70000, 80000], [5000, 4000]] // All invalid ranges
			)
			
			assert.ok(consoleWarnSpy.calledWith(
				// TODO: String contains check - "Invalid port range"))
			)
		})

		test("should suppress logs in silent mode", async () => {
			mockTcpPortUsed.check.resolves(false) // Port available
			
			await findAvailablePort(3000, 'localhost', undefined, 100, true)
			
			assert.ok(!consoleLogSpy.called)
			assert.ok(!consoleWarnSpy.called)
		})
	})

	suite("waitForPortAvailable", () => {
		test("should retry with exponential backoff", async () => {
			let attempts = 0
			mockTcpPortUsed.waitUntilFree.callsFake(async () => {
				attempts++
				if (attempts < 3) {
					throw new Error("Port still in use")
				}
			})
			
			const startTime = Date.now()
			await waitForPortAvailable(3000, 'localhost', 100, 30000, 'test-service', 5)
			const duration = Date.now() - startTime
			
			// Should have retried with increasing delays
			assert.strictEqual(attempts, 3)
			assert.ok(mockLogger.warn.calledWith(
				// TODO: String contains check - "Retry 1/5")),
				sinon.match.object
			)
			assert.ok(mockLogger.warn.calledWith(
				// TODO: String contains check - "Retry 2/5")),
				sinon.match.object
			)
			
			// Should have taken some time due to retries
			assert.ok(duration > 200) // At least 2 retry delays
		})

		test("should throw after maxRetries exceeded", async () => {
			mockTcpPortUsed.waitUntilFree.rejects(new Error("Port in use"))
			
			try {
				await waitForPortAvailable(3000, 'localhost', 50, 1000, 'test-service', 3)
				assert.fail('Should have thrown an error')
			} catch (error) {
				assert.ok(error instanceof Error)
				assert.strictEqual(error.message, "Timeout waiting for test-service on port 3000 to become available after 3 attempts")
			}
			
			assert.ok(mockLogger.error.calledWith(
				// TODO: String contains check - "Timeout waiting")),
				sinon.match.object
			)
		})

		test("should respect abort signal", async () => {
			const controller = new AbortController()
			
			// Abort immediately
			controller.abort()
			
			await waitForPortAvailable(
				3000, 
				'localhost', 
				100, 
				30000, 
				'test-service', 
				5,
				controller.signal
			)
			
			// Should return immediately without throwing
			assert.ok(!mockTcpPortUsed.waitUntilFree.called)
		})

		test("should abort during retry delay", async () => {
			const controller = new AbortController()
			let attempts = 0
			
			mockTcpPortUsed.waitUntilFree.callsFake(async () => {
				attempts++
				if (attempts === 1) {
					// Abort after first attempt
					setTimeout(() => controller.abort(), 50)
					throw new Error("Port in use")
				}
			})
			
			await waitForPortAvailable(
				3000,
				'localhost',
				200, // Longer retry time
				30000,
				'test-service',
				5,
				controller.signal
			)
			
			// Should have attempted once then aborted
			assert.strictEqual(attempts, 1)
		})

		test("should skip in test environment", async () => {
			// Restore test environment
			process.env.MOCHA_WORKER_ID = "1"
			
			await waitForPortAvailable(3000)
			
			// Should not call any methods
			assert.ok(!mockTcpPortUsed.waitUntilFree.called)
			assert.ok(!mockLogger.info.called)
		})

		test("should calculate attempt timeout correctly", async () => {
			mockTcpPortUsed.waitUntilFree.resolves(undefined)
			
			await waitForPortAvailable(3000, 'localhost', 200, 60000) // 60 second total timeout
			
			// Should use max 10 seconds per attempt
			assert.ok(mockTcpPortUsed.waitUntilFree.calledWith(
				3000,
				'localhost',
				200,
				10000 // Min of timeOutMs/3 and 10000
			))
		})
	})

	suite("waitForPortInUse", () => {
		test("should retry with exponential backoff and jitter", async () => {
			let attempts = 0
			const retryTimes: number[] = []
			
			mockTcpPortUsed.waitUntilUsed.callsFake(async (
				port: number,
				host: string,
				retryTime: number
			) => {
				attempts++
				retryTimes.push(retryTime)
				if (attempts < 3) {
					throw new Error("Port not yet in use")
				}
			})
			
			await waitForPortInUse(3000, 'localhost', 100, 30000, 'test-server', 5)
			
			// First retry should be base time
			assert.strictEqual(retryTimes[0], 100)
			
			// Subsequent retries should increase
			assert.ok(retryTimes[1] > 100)
			assert.ok(retryTimes[2] > retryTimes[1])
			
			// Should log retries
			assert.strictEqual(mockLogger.warn.callCount, 2)
		})

		test("should throw after maxRetries exceeded", async () => {
			mockTcpPortUsed.waitUntilUsed.rejects(new Error("Port not in use"))
			
			try {
				await waitForPortInUse(3000, 'localhost', 50, 1000, 'test-server', 2)
				assert.fail('Should have thrown an error')
			} catch (error) {
				assert.ok(error instanceof Error)
				assert.strictEqual(error.message, "Timeout waiting for test-server on port 3000 to be ready after 2 attempts")
			}
			
			assert.ok(mockLogger.error.calledWith(
				// TODO: String contains check - "Timeout waiting")),
				sinon.match.object
			)
		})

		test("should handle silent mode for fast checks", async () => {
			mockTcpPortUsed.waitUntilUsed.resolves(undefined)
			
			await waitForPortInUse(3000, 'localhost', 100, 1000, 'test-server', 2, true)
			
			// Should use short timeout for fast checks
			assert.ok(mockTcpPortUsed.waitUntilUsed.calledWith(
				3000,
				'localhost',
				100,
				250 // Min of timeOutMs and 250 for silent mode with low retries
			))
			
			// Should not log in silent mode
			assert.ok(!mockLogger.info.called)
			assert.ok(!mockLogger.warn.called)
		})

		test("should handle silent mode with failures", async () => {
			mockTcpPortUsed.waitUntilUsed.rejects(new Error("Port not in use"))
			
			// Should not throw in silent mode with low retries
			await waitForPortInUse(3000, 'localhost', 100, 1000, 'test-server', 1, true)
			
			// Should not log errors in silent mode
			assert.ok(!mockLogger.error.called)
		})

		test("should respect abort signal during initial check", async () => {
			const controller = new AbortController()
			controller.abort()
			
			await waitForPortInUse(
				3000,
				'localhost',
				100,
				30000,
				'test-server',
				5,
				false,
				controller.signal
			)
			
			// Should return immediately
			assert.ok(!mockTcpPortUsed.waitUntilUsed.called)
		})

		test("should abort during retry with cleanup", async () => {
			const controller = new AbortController()
			let attempts = 0
			
			mockTcpPortUsed.waitUntilUsed.callsFake(async () => {
				attempts++
				if (attempts === 1) {
					// Schedule abort during retry delay
					setTimeout(() => controller.abort(), 100)
					throw new Error("Not ready")
				}
			})
			
			await waitForPortInUse(
				3000,
				'localhost',
				300, // Longer than abort timeout
				30000,
				'test-server',
				5,
				false,
				controller.signal
			)
			
			// Should have tried once then aborted
			assert.strictEqual(attempts, 1)
			assert.ok(mockLogger.warn.calledWith(
				// TODO: String contains check - "Retry 1/5")),
				sinon.match.object
			)
		})

		test("should skip in test environment with NODE_ENV", async () => {
			process.env.NODE_ENV = 'test'
			
			await waitForPortInUse(3000)
			
			assert.ok(!mockTcpPortUsed.waitUntilUsed.called)
		})

		test("should skip in test environment with global mocha", async () => {
			(globalThis as any).mocha = true
			
			await waitForPortInUse(3000)
			
			assert.ok(!mockTcpPortUsed.waitUntilUsed.called)
		})

		test("should cap retry time at maximum", async () => {
			let lastRetryTime = 0
			mockTcpPortUsed.waitUntilUsed.callsFake(async (
				port: number,
				host: string,
				retryTime: number
			) => {
				lastRetryTime = retryTime
				throw new Error("Not ready")
			})
			
			try {
				await waitForPortInUse(3000, 'localhost', 1500, 30000, 'test-server', 3)
			} catch {
				// Expected to fail
			}
			
			// Retry time should be capped at 2000ms even with exponential backoff
			assert.ok(lastRetryTime <= 2000)
		})
	})

	suite("Edge Cases", () => {
		test("should handle network timeouts gracefully", async () => {
			mockTcpPortUsed.check.callsFake(() => 
				new Promise((_, reject) => 
					setTimeout(() => reject(new Error("ETIMEDOUT")), 100)
				)
			)
			
			const result = await isPortAvailable(3000)
			
			assert.strictEqual(result, false)
			assert.ok(consoleErrorSpy.calledWith(
				// TODO: String contains check - "Error checking port")),
				sinon.match.instanceOf(Error)
			)
		})

		test("should handle port range boundaries", async () => {
			mockTcpPortUsed.check.resolves(false)
			
			// Test with port at upper boundary
			const result = await findAvailablePort(65535, 'localhost', undefined, 1)
			
			assert.strictEqual(result, 65535)
		})

		test("should handle concurrent retry operations", async () => {
			let resolveCount = 0
			mockTcpPortUsed.waitUntilUsed.callsFake(async () => {
				resolveCount++
				if (resolveCount === 1) {
					throw new Error("Not ready")
				}
			})
			
			// Start two concurrent wait operations
			const promises = [
				waitForPortInUse(3000, 'localhost', 100, 5000, 'server1', 3),
				waitForPortInUse(3001, 'localhost', 100, 5000, 'server2', 3)
			]
			
			await Promise.all(promises)
			
			// Both should complete successfully
			assert.ok(mockLogger.info.calledWith(
				// TODO: String contains check - "server1 on port 3000 is now ready")),
				sinon.match.object
			)
			assert.ok(mockLogger.info.calledWith(
				// TODO: String contains check - "server2 on port 3001 is now ready")),
				sinon.match.object
			)
		})

		test("should handle immediate success without retries", async () => {
			mockTcpPortUsed.waitUntilFree.resolves(undefined)
			
			await waitForPortAvailable(3000, 'localhost', 200, 30000, 'test-service')
			
			// Should succeed on first attempt
			assert.ok(!mockLogger.warn.called)
			assert.ok(mockLogger.info.calledWith(
				"test-service on port 3000 is now available",
				sinon.match.object)
			)
		})

		test("should handle jitter in retry delays", async () => {
			const delays: number[] = []
			sinon.spy(global, 'setTimeout').callsFake((fn: any, delay?: number) => {
				if (delay) delays.push(delay)
				fn()
// Mock return block needs context
// 				return {} as NodeJS.Timeout
// 			})
// 			
// 			mockTcpPortUsed.waitUntilUsed.rejects(new Error("Not ready"))
// 			
// 			try {
// 				await waitForPortInUse(3000, 'localhost', 100, 1000, 'test', 3)
// 			} catch {
// 				// Expected to fail
// 			}
// 			
			// Delays should include jitter (not exactly the same)
			assert.ok(delays.length > 0)
			// Verify delays are not all identical (jitter is applied)
			const uniqueDelays = new Set(delays)
			assert.ok(uniqueDelays.size > 1)
		})
	})
// Mock cleanup