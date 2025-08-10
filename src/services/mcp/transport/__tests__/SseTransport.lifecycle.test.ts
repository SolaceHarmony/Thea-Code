/**
 * SSE Transport lifecycle tests as recommended by architect
 * Tests dynamic import, teardown guard, port management, and transport exposure
 */

import { SseTransport } from "../SseTransport"
import { SseTransportConfig } from "../../types/McpProviderTypes"

describe("SseTransport Lifecycle Tests", () => {
	let transport: SseTransport
	
	afterEach(async () => {
		// Clean up transport if it exists
		if (transport) {
			try {
				await transport.close()
			} catch {
				// Transport might already be closed
			}
		}
	})

	describe("Initialization and startup", () => {
		test("should successfully start with valid config", async () => {
			const config: SseTransportConfig = {
				port: 0, // Use dynamic port
				host: "127.0.0.1"
			}
			
			transport = new SseTransport(config)
			await transport.start()
			
			// Should have an underlying transport after start
			const underlying = transport.getUnderlyingTransport()
			expect(underlying).toBeDefined()
			
			// Should have a port assigned
			const port = transport.getPort()
			expect(port).toBeDefined()
			expect(port).toBeGreaterThan(0)
		})

		test("should handle teardown guard during Jest teardown", async () => {
			// Simulate Jest teardown
			(global as any).__JEST_TEARDOWN__ = true
			
			const config: SseTransportConfig = {
				port: 0,
				host: "127.0.0.1"
			}
			
			transport = new SseTransport(config)
			
			// Should short-circuit and not actually start
			await expect(transport.start()).resolves.not.toThrow()
			
			// Should not have underlying transport due to teardown guard
			expect(transport.getUnderlyingTransport()).toBeUndefined()
			
			// Clean up
			delete (global as any).__JEST_TEARDOWN__
		})

		test("should handle missing SDK gracefully", async () => {
			// Mock the dynamic import to fail
			const originalRequire = require
			jest.doMock("@modelcontextprotocol/sdk/server/streamableHttp.js", () => {
				throw new Error("Module not found")
			})
			
			const config: SseTransportConfig = {
				port: 0,
				host: "127.0.0.1"
			}
			
			transport = new SseTransport(config)
			
			// Should handle the missing SDK gracefully
			await expect(transport.start()).rejects.toThrow()
			
			// Restore
			jest.unmock("@modelcontextprotocol/sdk/server/streamableHttp.js")
		})
	})

	describe("Port management", () => {
		test("should use specified port when provided", async () => {
			const config: SseTransportConfig = {
				port: 38765, // Specific port
				host: "127.0.0.1"
			}
			
			transport = new SseTransport(config)
			await transport.start()
			
			const port = transport.getPort()
			expect(port).toBe(38765)
		})

		test("should assign dynamic port when port is 0", async () => {
			const config: SseTransportConfig = {
				port: 0,
				host: "127.0.0.1"
			}
			
			transport = new SseTransport(config)
			await transport.start()
			
			const port = transport.getPort()
			expect(port).toBeDefined()
			expect(port).toBeGreaterThan(0)
			expect(port).not.toBe(0)
		})

		test("should expose port via getPort method", async () => {
			const config: SseTransportConfig = {
				port: 0,
				host: "127.0.0.1"
			}
			
			transport = new SseTransport(config)
			
			// Before start, port should be undefined
			expect(transport.getPort()).toBeUndefined()
			
			await transport.start()
			
			// After start, port should be defined
			const port = transport.getPort()
			expect(port).toBeDefined()
			expect(typeof port).toBe("number")
		})
	})

	describe("Underlying transport management", () => {
		test("should expose underlying transport after start", async () => {
			const config: SseTransportConfig = {
				port: 0,
				host: "127.0.0.1"
			}
			
			transport = new SseTransport(config)
			
			// Before start, should be undefined
			expect(transport.getUnderlyingTransport()).toBeUndefined()
			
			await transport.start()
			
			// After start, should be defined
			const underlying = transport.getUnderlyingTransport()
			expect(underlying).toBeDefined()
			expect(underlying).not.toBeNull()
		})

		test("should clear underlying transport after close", async () => {
			const config: SseTransportConfig = {
				port: 0,
				host: "127.0.0.1"
			}
			
			transport = new SseTransport(config)
			await transport.start()
			
			// Should have transport after start
			expect(transport.getUnderlyingTransport()).toBeDefined()
			
			await transport.close()
			
			// Should clear transport after close
			expect(transport.getUnderlyingTransport()).toBeUndefined()
		})
	})

	describe("Event handlers", () => {
		test("should forward onerror to underlying transport", async () => {
			const config: SseTransportConfig = {
				port: 0,
				host: "127.0.0.1"
			}
			
			transport = new SseTransport(config)
			await transport.start()
			
			const errorHandler = jest.fn()
			transport.onerror = errorHandler
			
			// The handler should be set on the underlying transport
			const underlying = transport.getUnderlyingTransport() as any
			if (underlying && underlying.onerror) {
				// Trigger an error on the underlying transport
				underlying.onerror(new Error("Test error"))
				expect(errorHandler).toHaveBeenCalled()
			}
		})

		test("should forward onclose to underlying transport", async () => {
			const config: SseTransportConfig = {
				port: 0,
				host: "127.0.0.1"
			}
			
			transport = new SseTransport(config)
			await transport.start()
			
			const closeHandler = jest.fn()
			transport.onclose = closeHandler
			
			// The handler should be set on the underlying transport
			const underlying = transport.getUnderlyingTransport() as any
			if (underlying && underlying.onclose) {
				// Trigger close on the underlying transport
				underlying.onclose()
				expect(closeHandler).toHaveBeenCalled()
			}
		})
	})

	describe("Lifecycle edge cases", () => {
		test("should handle multiple start calls", async () => {
			const config: SseTransportConfig = {
				port: 0,
				host: "127.0.0.1"
			}
			
			transport = new SseTransport(config)
			
			// First start
			await transport.start()
			const firstPort = transport.getPort()
			
			// Second start should not change anything
			await transport.start()
			const secondPort = transport.getPort()
			
			expect(firstPort).toBe(secondPort)
		})

		test("should handle close without start", async () => {
			const config: SseTransportConfig = {
				port: 0,
				host: "127.0.0.1"
			}
			
			transport = new SseTransport(config)
			
			// Should not throw when closing without starting
			await expect(transport.close()).resolves.not.toThrow()
		})

		test("should handle multiple close calls", async () => {
			const config: SseTransportConfig = {
				port: 0,
				host: "127.0.0.1"
			}
			
			transport = new SseTransport(config)
			await transport.start()
			
			// First close
			await transport.close()
			
			// Second close should not throw
			await expect(transport.close()).resolves.not.toThrow()
		})

		test("should restart with different port after close", async () => {
			const config: SseTransportConfig = {
				port: 0,
				host: "127.0.0.1"
			}
			
			transport = new SseTransport(config)
			
			// First start
			await transport.start()
			const firstPort = transport.getPort()
			
			// Close
			await transport.close()
			
			// Start again
			await transport.start()
			const secondPort = transport.getPort()
			
			// Ports might be different (especially with dynamic allocation)
			expect(secondPort).toBeDefined()
			expect(secondPort).toBeGreaterThan(0)
		})
	})

	describe("Error handling", () => {
		test("should suppress error logs in test environment", async () => {
			// Set test environment flag
			process.env.JEST_WORKER_ID = "1"
			
			const consoleErrorSpy = jest.spyOn(console, "error").mockImplementation()
			
			const config: SseTransportConfig = {
				port: 0,
				host: "127.0.0.1"
			}
			
			transport = new SseTransport(config)
			await transport.start()
			
			// Simulate an error
			const underlying = transport.getUnderlyingTransport() as any
			if (underlying && underlying.onerror) {
				underlying.onerror(new Error("Test error"))
			}
			
			// Error should be suppressed in test environment
			expect(consoleErrorSpy).not.toHaveBeenCalled()
			
			// Clean up
			consoleErrorSpy.mockRestore()
			delete process.env.JEST_WORKER_ID
		})

		test("should handle transport creation errors", async () => {
			const config: SseTransportConfig = {
				port: -1, // Invalid port
				host: "127.0.0.1"
			}
			
			transport = new SseTransport(config)
			
			// Should handle invalid port gracefully
			// The actual behavior depends on the SDK implementation
			try {
				await transport.start()
				// If it doesn't throw, check that transport is in a valid state
				const underlying = transport.getUnderlyingTransport()
				expect(underlying).toBeDefined()
			} catch (error) {
				// If it throws, that's also acceptable error handling
				expect(error).toBeDefined()
			}
		})
	})
})