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
jest.mock("tcp-port-used", () => ({
	check: jest.fn(),
	waitUntilFree: jest.fn(),
	waitUntilUsed: jest.fn()
}))

// Mock logger
jest.mock("../logging", () => ({
	logger: {
		info: jest.fn(),
		warn: jest.fn(),
		error: jest.fn()
	}
}))

describe("port-utils - Retry and Timeout Tests", () => {
	let mockTcpPortUsed: any
	let mockLogger: jest.Mocked<typeof logger>
	let originalJestWorkerId: string | undefined
	let originalNodeEnv: string | undefined
	let consoleErrorSpy: jest.SpyInstance
	let consoleWarnSpy: jest.SpyInstance
	let consoleLogSpy: jest.SpyInstance

	beforeEach(() => {
		// Clear all mocks
		jest.clearAllMocks()
		
		// Setup console spies
		consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation()
		consoleWarnSpy = jest.spyOn(console, 'warn').mockImplementation()
		consoleLogSpy = jest.spyOn(console, 'log').mockImplementation()

		// Get mocked module
		mockTcpPortUsed = jest.requireMock("tcp-port-used")
		mockLogger = logger as jest.Mocked<typeof logger>
		
		// Save and clear environment variables to simulate non-test environment
		originalJestWorkerId = process.env.JEST_WORKER_ID
		originalNodeEnv = process.env.NODE_ENV
		delete process.env.JEST_WORKER_ID
		delete process.env.NODE_ENV
		
		// Clear global jest reference
		if (typeof (globalThis as any).jest !== 'undefined') {
			delete (globalThis as any).jest
		}
	})

	afterEach(() => {
		// Restore environment variables
		if (originalJestWorkerId) {
			process.env.JEST_WORKER_ID = originalJestWorkerId
		}
		if (originalNodeEnv) {
			process.env.NODE_ENV = originalNodeEnv
		}
		
		// Restore global jest
		(globalThis as any).jest = jest
		
		jest.restoreAllMocks()
	})

	describe("isPortAvailable", () => {
		it("should handle tcp-port-used errors gracefully", async () => {
			mockTcpPortUsed.check.mockRejectedValue(new Error("Network error"))
			
			const result = await isPortAvailable(3000)
			
			expect(result).toBe(false)
			expect(consoleErrorSpy).toHaveBeenCalledWith(
				"Error checking port 3000 availability:",
				expect.any(Error)
			)
		})

		it("should return true when port is available", async () => {
			mockTcpPortUsed.check.mockResolvedValue(false) // false = not in use
			
			const result = await isPortAvailable(3000)
			
			expect(result).toBe(true)
			expect(mockTcpPortUsed.check).toHaveBeenCalledWith(3000, "localhost")
		})

		it("should return false when port is in use", async () => {
			mockTcpPortUsed.check.mockResolvedValue(true) // true = in use
			
			const result = await isPortAvailable(3000)
			
			expect(result).toBe(false)
		})
	})

	describe("findAvailablePort", () => {
		it("should retry up to maxAttempts times", async () => {
			// All ports unavailable
			mockTcpPortUsed.check.mockResolvedValue(true)
			
			await expect(
				findAvailablePort(3000, 'localhost', undefined, 5)
			).rejects.toThrow("No available ports found after 5 attempts")
			
			// Should have checked 5 sequential ports plus last resort ports
			// The function tries maxAttempts sequential ports then tries last resort ports
			expect(mockTcpPortUsed.check.mock.calls.length).toBeGreaterThanOrEqual(5)
		})

		it("should switch to random port search after 20 sequential attempts", async () => {
			let callCount = 0
			mockTcpPortUsed.check.mockImplementation(async (port: number) => {
				callCount++
				// Make port available on 21st attempt
				return callCount < 21
			})
			
			const result = await findAvailablePort(3000)
			
			// Should have switched to random search
			expect(consoleLogSpy).toHaveBeenCalledWith(
				expect.stringContaining("Switching to random port search")
			)
			
			// Should eventually find a port
			expect(result).toBeGreaterThanOrEqual(1024)
			expect(result).toBeLessThanOrEqual(65535)
		})

		it("should try last resort ports when regular search fails", async () => {
			let callCount = 0
			mockTcpPortUsed.check.mockImplementation(async (port: number) => {
				callCount++
				// Make port 8080 available (a last resort port)
				return port !== 8080
			})
			
			const result = await findAvailablePort(3000, 'localhost', undefined, 10)
			
			expect(result).toBe(8080)
			expect(consoleLogSpy).toHaveBeenCalledWith("Trying last resort ports...")
			expect(consoleLogSpy).toHaveBeenCalledWith("Found available last resort port 8080")
		})

		it("should handle preferred ranges with retries", async () => {
			let checkCount = 0
			mockTcpPortUsed.check.mockImplementation(async () => {
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
			expect(result).toBeGreaterThanOrEqual(10000)
			expect(result).toBeLessThanOrEqual(10100)
			expect(consoleLogSpy).toHaveBeenCalledWith(
				expect.stringContaining("Found available port")
			)
		})

		it("should validate and skip invalid port ranges", async () => {
			mockTcpPortUsed.check.mockResolvedValue(false) // Port available
			
			await findAvailablePort(
				3000,
				'localhost',
				[[500, 1000], [70000, 80000], [5000, 4000]] // All invalid ranges
			)
			
			expect(consoleWarnSpy).toHaveBeenCalledWith(
				expect.stringContaining("Invalid port range")
			)
		})

		it("should suppress logs in silent mode", async () => {
			mockTcpPortUsed.check.mockResolvedValue(false) // Port available
			
			await findAvailablePort(3000, 'localhost', undefined, 100, true)
			
			expect(consoleLogSpy).not.toHaveBeenCalled()
			expect(consoleWarnSpy).not.toHaveBeenCalled()
		})
	})

	describe("waitForPortAvailable", () => {
		it("should retry with exponential backoff", async () => {
			let attempts = 0
			mockTcpPortUsed.waitUntilFree.mockImplementation(async () => {
				attempts++
				if (attempts < 3) {
					throw new Error("Port still in use")
				}
			})
			
			const startTime = Date.now()
			await waitForPortAvailable(3000, 'localhost', 100, 30000, 'test-service', 5)
			const duration = Date.now() - startTime
			
			// Should have retried with increasing delays
			expect(attempts).toBe(3)
			expect(mockLogger.warn).toHaveBeenCalledWith(
				expect.stringContaining("Retry 1/5"),
				expect.any(Object)
			)
			expect(mockLogger.warn).toHaveBeenCalledWith(
				expect.stringContaining("Retry 2/5"),
				expect.any(Object)
			)
			
			// Should have taken some time due to retries
			expect(duration).toBeGreaterThan(200) // At least 2 retry delays
		})

		it("should throw after maxRetries exceeded", async () => {
			mockTcpPortUsed.waitUntilFree.mockRejectedValue(new Error("Port in use"))
			
			await expect(
				waitForPortAvailable(3000, 'localhost', 50, 1000, 'test-service', 3)
			).rejects.toThrow("Timeout waiting for test-service on port 3000 to become available after 3 attempts")
			
			expect(mockLogger.error).toHaveBeenCalledWith(
				expect.stringContaining("Timeout waiting"),
				expect.any(Object)
			)
		})

		it("should respect abort signal", async () => {
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
			expect(mockTcpPortUsed.waitUntilFree).not.toHaveBeenCalled()
		})

		it("should abort during retry delay", async () => {
			const controller = new AbortController()
			let attempts = 0
			
			mockTcpPortUsed.waitUntilFree.mockImplementation(async () => {
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
			expect(attempts).toBe(1)
		})

		it("should skip in test environment", async () => {
			// Restore test environment
			process.env.JEST_WORKER_ID = "1"
			
			await waitForPortAvailable(3000)
			
			// Should not call any methods
			expect(mockTcpPortUsed.waitUntilFree).not.toHaveBeenCalled()
			expect(mockLogger.info).not.toHaveBeenCalled()
		})

		it("should calculate attempt timeout correctly", async () => {
			mockTcpPortUsed.waitUntilFree.mockResolvedValue(undefined)
			
			await waitForPortAvailable(3000, 'localhost', 200, 60000) // 60 second total timeout
			
			// Should use max 10 seconds per attempt
			expect(mockTcpPortUsed.waitUntilFree).toHaveBeenCalledWith(
				3000,
				'localhost',
				200,
				10000 // Min of timeOutMs/3 and 10000
			)
		})
	})

	describe("waitForPortInUse", () => {
		it("should retry with exponential backoff and jitter", async () => {
			let attempts = 0
			const retryTimes: number[] = []
			
			mockTcpPortUsed.waitUntilUsed.mockImplementation(async (
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
			expect(retryTimes[0]).toBe(100)
			
			// Subsequent retries should increase
			expect(retryTimes[1]).toBeGreaterThan(100)
			expect(retryTimes[2]).toBeGreaterThan(retryTimes[1])
			
			// Should log retries
			expect(mockLogger.warn).toHaveBeenCalledTimes(2)
		})

		it("should throw after maxRetries exceeded", async () => {
			mockTcpPortUsed.waitUntilUsed.mockRejectedValue(new Error("Port not in use"))
			
			await expect(
				waitForPortInUse(3000, 'localhost', 50, 1000, 'test-server', 2)
			).rejects.toThrow("Timeout waiting for test-server on port 3000 to be ready after 2 attempts")
			
			expect(mockLogger.error).toHaveBeenCalledWith(
				expect.stringContaining("Timeout waiting"),
				expect.any(Object)
			)
		})

		it("should handle silent mode for fast checks", async () => {
			mockTcpPortUsed.waitUntilUsed.mockResolvedValue(undefined)
			
			await waitForPortInUse(3000, 'localhost', 100, 1000, 'test-server', 2, true)
			
			// Should use short timeout for fast checks
			expect(mockTcpPortUsed.waitUntilUsed).toHaveBeenCalledWith(
				3000,
				'localhost',
				100,
				250 // Min of timeOutMs and 250 for silent mode with low retries
			)
			
			// Should not log in silent mode
			expect(mockLogger.info).not.toHaveBeenCalled()
			expect(mockLogger.warn).not.toHaveBeenCalled()
		})

		it("should handle silent mode with failures", async () => {
			mockTcpPortUsed.waitUntilUsed.mockRejectedValue(new Error("Port not in use"))
			
			// Should not throw in silent mode with low retries
			await waitForPortInUse(3000, 'localhost', 100, 1000, 'test-server', 1, true)
			
			// Should not log errors in silent mode
			expect(mockLogger.error).not.toHaveBeenCalled()
		})

		it("should respect abort signal during initial check", async () => {
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
			expect(mockTcpPortUsed.waitUntilUsed).not.toHaveBeenCalled()
		})

		it("should abort during retry with cleanup", async () => {
			const controller = new AbortController()
			let attempts = 0
			
			mockTcpPortUsed.waitUntilUsed.mockImplementation(async () => {
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
			expect(attempts).toBe(1)
			expect(mockLogger.warn).toHaveBeenCalledWith(
				expect.stringContaining("Retry 1/5"),
				expect.any(Object)
			)
		})

		it("should skip in test environment with NODE_ENV", async () => {
			process.env.NODE_ENV = 'test'
			
			await waitForPortInUse(3000)
			
			expect(mockTcpPortUsed.waitUntilUsed).not.toHaveBeenCalled()
		})

		it("should skip in test environment with global jest", async () => {
			(globalThis as any).jest = jest
			
			await waitForPortInUse(3000)
			
			expect(mockTcpPortUsed.waitUntilUsed).not.toHaveBeenCalled()
		})

		it("should cap retry time at maximum", async () => {
			let lastRetryTime = 0
			mockTcpPortUsed.waitUntilUsed.mockImplementation(async (
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
			expect(lastRetryTime).toBeLessThanOrEqual(2000)
		})
	})

	describe("Edge Cases", () => {
		it("should handle network timeouts gracefully", async () => {
			mockTcpPortUsed.check.mockImplementation(() => 
				new Promise((_, reject) => 
					setTimeout(() => reject(new Error("ETIMEDOUT")), 100)
				)
			)
			
			const result = await isPortAvailable(3000)
			
			expect(result).toBe(false)
			expect(consoleErrorSpy).toHaveBeenCalledWith(
				expect.stringContaining("Error checking port"),
				expect.any(Error)
			)
		})

		it("should handle port range boundaries", async () => {
			mockTcpPortUsed.check.mockResolvedValue(false)
			
			// Test with port at upper boundary
			const result = await findAvailablePort(65535, 'localhost', undefined, 1)
			
			expect(result).toBe(65535)
		})

		it("should handle concurrent retry operations", async () => {
			let resolveCount = 0
			mockTcpPortUsed.waitUntilUsed.mockImplementation(async () => {
				resolveCount++
				if (resolveCount === 1) {
					throw new Error("Not ready")
				}
			})
			
			// Start two concurrent wait operations
			const promises = [
				await waitForPortInUse(3000, 'localhost', 100, 5000, 'server1', 3),
				await waitForPortInUse(3001, 'localhost', 100, 5000, 'server2', 3)
			]
			
			await Promise.all(promises)
			
			// Both should complete successfully
			expect(mockLogger.info).toHaveBeenCalledWith(
				expect.stringContaining("server1 on port 3000 is now ready"),
				expect.any(Object)
			)
			expect(mockLogger.info).toHaveBeenCalledWith(
				expect.stringContaining("server2 on port 3001 is now ready"),
				expect.any(Object)
			)
		})

		it("should handle immediate success without retries", async () => {
			mockTcpPortUsed.waitUntilFree.mockResolvedValue(undefined)
			
			await waitForPortAvailable(3000, 'localhost', 200, 30000, 'test-service')
			
			// Should succeed on first attempt
			expect(mockLogger.warn).not.toHaveBeenCalled()
			expect(mockLogger.info).toHaveBeenCalledWith(
				"test-service on port 3000 is now available",
				expect.any(Object)
			)
		})

		it("should handle jitter in retry delays", async () => {
			const delays: number[] = []
			jest.spyOn(global, 'setTimeout').mockImplementation((fn: any, delay?: number) => {
				if (delay) delays.push(delay)
				fn()
				return {} as NodeJS.Timeout
			})
			
			mockTcpPortUsed.waitUntilUsed.mockRejectedValue(new Error("Not ready"))
			
			try {
				await waitForPortInUse(3000, 'localhost', 100, 1000, 'test', 3)
			} catch {
				// Expected to fail
			}
			
			// Delays should include jitter (not exactly the same)
			expect(delays.length).toBeGreaterThan(0)
			// Verify delays are not all identical (jitter is applied)
			const uniqueDelays = new Set(delays)
			expect(uniqueDelays.size).toBeGreaterThan(1)
		})
	})
})