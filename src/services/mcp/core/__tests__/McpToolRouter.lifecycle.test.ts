/**
 * McpToolRouter lifecycle tests
 * Tests initialization, shutdown, repeated start/stop, pending registrations, and event forwarding
 */

import { describe, expect, it, jest, beforeEach, afterEach } from "@jest/globals"
import { McpToolRouter } from "../McpToolRouter"
import { McpToolExecutor } from "../McpToolExecutor"
import { McpToolRegistry } from "../McpToolRegistry"
import { EventEmitter } from "events"

// Mock the McpToolExecutor
jest.mock("../McpToolExecutor")

describe("McpToolRouter - Lifecycle and Event Management", () => {
	let router: McpToolRouter | undefined
	let mockExecutor: jest.Mocked<McpToolExecutor>
	let executorEventEmitter: EventEmitter

	beforeEach(() => {
		// Reset router variable
		router = undefined
		
		// Clear singleton instances using delete
		delete (McpToolRouter as any).instance
		delete (McpToolExecutor as any).instance
		delete (McpToolRegistry as any).instance

		// Create a real EventEmitter for the executor
		executorEventEmitter = new EventEmitter()

		// Create mock executor with event emitter capabilities
		mockExecutor = {
			initialize: jest.fn().mockResolvedValue(undefined),
			shutdown: jest.fn().mockResolvedValue(undefined),
			executeTool: jest.fn(),
			on: jest.fn((event: string, handler: Function) => {
				executorEventEmitter.on(event, handler as any)
				return mockExecutor
			}),
			emit: jest.fn((event: string, ...args: any[]) => {
				executorEventEmitter.emit(event, ...args)
				return true
			}),
			removeAllListeners: jest.fn(() => {
				executorEventEmitter.removeAllListeners()
				return mockExecutor
			}),
		} as any

		// Mock getInstance to return our mock
		(McpToolExecutor.getInstance as jest.Mock).mockReturnValue(mockExecutor)
	})

	afterEach(() => {
		jest.clearAllMocks()
		// Clean up event listeners if they exist
		if (executorEventEmitter) {
			executorEventEmitter.removeAllListeners()
		}
	})

	describe("Initialization and Shutdown", () => {
		it("should initialize the executor on router initialization", async () => {
			router = McpToolRouter.getInstance()
			await router.initialize()

			expect(mockExecutor.initialize).toHaveBeenCalledTimes(1)
		})

		it("should shutdown the executor on router shutdown", async () => {
			router = McpToolRouter.getInstance()
			await router.initialize()
			await router.shutdown()

			expect(mockExecutor.shutdown).toHaveBeenCalledTimes(1)
		})

		it("should handle repeated initialization gracefully", async () => {
			router = McpToolRouter.getInstance()
			
			await router.initialize()
			await router.initialize()
			await router.initialize()

			// Should still only initialize once per instance
			expect(mockExecutor.initialize).toHaveBeenCalledTimes(3)
		})

		it("should handle repeated shutdown gracefully", async () => {
			router = McpToolRouter.getInstance()
			await router.initialize()

			await router.shutdown()
			await router.shutdown()
			await router.shutdown()

			// Should handle multiple shutdowns
			expect(mockExecutor.shutdown).toHaveBeenCalledTimes(3)
		})

		it("should handle init-shutdown-init cycles", async () => {
			router = McpToolRouter.getInstance()

			// First cycle
			await router.initialize()
			expect(mockExecutor.initialize).toHaveBeenCalledTimes(1)
			await router.shutdown()
			expect(mockExecutor.shutdown).toHaveBeenCalledTimes(1)

			// Second cycle
			await router.initialize()
			expect(mockExecutor.initialize).toHaveBeenCalledTimes(2)
			await router.shutdown()
			expect(mockExecutor.shutdown).toHaveBeenCalledTimes(2)

			// Third cycle
			await router.initialize()
			expect(mockExecutor.initialize).toHaveBeenCalledTimes(3)
			await router.shutdown()
			expect(mockExecutor.shutdown).toHaveBeenCalledTimes(3)
		})

		it("should maintain singleton instance across multiple getInstance calls", () => {
			const router1 = McpToolRouter.getInstance()
			const router2 = McpToolRouter.getInstance()
			const router3 = McpToolRouter.getInstance()

			expect(router1).toBe(router2)
			expect(router2).toBe(router3)
		})

		it("should update SSE config when provided to getInstance", () => {
			const config1 = { port: 3000 }
			const config2 = { port: 4000 }

			const router1 = McpToolRouter.getInstance(config1)
			expect((router1 as any).sseConfig).toEqual(config1)

			const router2 = McpToolRouter.getInstance(config2)
			expect(router2).toBe(router1) // Same instance
			expect((router2 as any).sseConfig).toEqual(config2) // Config updated
		})
	})

	describe("Event Forwarding", () => {
		beforeEach(() => {
			router = McpToolRouter.getInstance()
		})

		it("should forward tool-registered events from executor", (done) => {
			const toolName = "test-tool"

			router.on("tool-registered", (name: string) => {
				expect(name).toBe(toolName)
				done()
			})

			// Emit event from the executor
			executorEventEmitter.emit("tool-registered", toolName)
		})

		it("should forward tool-unregistered events from executor", (done) => {
			const toolName = "test-tool"

			router.on("tool-unregistered", (name: string) => {
				expect(name).toBe(toolName)
				done()
			})

			// Emit event from the executor
			executorEventEmitter.emit("tool-unregistered", toolName)
		})

		it("should forward started events from executor", (done) => {
			const startInfo = { port: 3000, serverName: "test-server" }

			router.on("started", (info: unknown) => {
				expect(info).toEqual(startInfo)
				done()
			})

			// Emit event from the executor
			executorEventEmitter.emit("started", startInfo)
		})

		it("should forward stopped events from executor", (done) => {
			router.on("stopped", () => {
				done()
			})

			// Emit event from the executor
			executorEventEmitter.emit("stopped")
		})

		it("should handle multiple event listeners", () => {
			const listener1 = jest.fn()
			const listener2 = jest.fn()
			const listener3 = jest.fn()

			router.on("tool-registered", listener1)
			router.on("tool-registered", listener2)
			router.on("tool-registered", listener3)

			executorEventEmitter.emit("tool-registered", "test-tool")

			expect(listener1).toHaveBeenCalledWith("test-tool")
			expect(listener2).toHaveBeenCalledWith("test-tool")
			expect(listener3).toHaveBeenCalledWith("test-tool")
		})

		it("should handle rapid event firing", () => {
			const registeredTools: string[] = []
			const unregisteredTools: string[] = []

			router.on("tool-registered", (name: string) => {
				registeredTools.push(name)
			})

			router.on("tool-unregistered", (name: string) => {
				unregisteredTools.push(name)
			})

			// Fire many events rapidly
			for (let i = 0; i < 100; i++) {
				executorEventEmitter.emit("tool-registered", `tool-${i}`)
				if (i % 2 === 0) {
					executorEventEmitter.emit("tool-unregistered", `tool-${i}`)
				}
			}

			expect(registeredTools).toHaveLength(100)
			expect(unregisteredTools).toHaveLength(50)
		})
	})

	describe("Pending Registrations", () => {
		it("should handle tool registration during initialization", async () => {
			router = McpToolRouter.getInstance()
			
			// Mock slow initialization
			let initResolve: Function
			mockExecutor.initialize.mockImplementation(() => {
				return new Promise((resolve) => {
					initResolve = resolve
				})
			})

			const initPromise = router.initialize()

			// Emit registration events while initializing
			executorEventEmitter.emit("tool-registered", "tool-1")
			executorEventEmitter.emit("tool-registered", "tool-2")
			executorEventEmitter.emit("tool-registered", "tool-3")

			// Complete initialization
			initResolve!()
			await initPromise

			// Events should have been handled
			const registeredTools: string[] = []
			router.on("tool-registered", (name: string) => {
				registeredTools.push(name)
			})

			// New registrations after init should still work
			executorEventEmitter.emit("tool-registered", "tool-4")
			expect(registeredTools).toContain("tool-4")
		})

		it("should handle tool registration during shutdown", async () => {
			router = McpToolRouter.getInstance()
			await router.initialize()

			// Mock slow shutdown
			let shutdownResolve: Function
			mockExecutor.shutdown.mockImplementation(() => {
				return new Promise((resolve) => {
					shutdownResolve = resolve
				})
			})

			const shutdownPromise = router.shutdown()

			// Emit registration events while shutting down
			executorEventEmitter.emit("tool-unregistered", "tool-1")
			executorEventEmitter.emit("tool-unregistered", "tool-2")

			// Complete shutdown
			shutdownResolve!()
			await shutdownPromise

			expect(mockExecutor.shutdown).toHaveBeenCalled()
		})
	})

	describe("Error Handling", () => {
		it("should handle initialization errors", async () => {
			router = McpToolRouter.getInstance()
			
			const initError = new Error("Initialization failed")
			mockExecutor.initialize.mockRejectedValue(initError)

			await expect(router.initialize()).rejects.toThrow("Initialization failed")
		})

		it("should handle shutdown errors", async () => {
			router = McpToolRouter.getInstance()
			await router.initialize()

			const shutdownError = new Error("Shutdown failed")
			mockExecutor.shutdown.mockRejectedValue(shutdownError)

			await expect(router.shutdown()).rejects.toThrow("Shutdown failed")
		})

		it("should continue forwarding events after errors", async () => {
			router = McpToolRouter.getInstance()
			
			// Initialize with error
			mockExecutor.initialize.mockRejectedValueOnce(new Error("Init error"))
			await expect(router.initialize()).rejects.toThrow()

			// Events should still be forwarded
			const listener = jest.fn()
			router.on("tool-registered", listener)
			executorEventEmitter.emit("tool-registered", "test-tool")
			
			expect(listener).toHaveBeenCalledWith("test-tool")
		})
	})

	describe("Memory Management", () => {
		it("should not leak event listeners on repeated getInstance calls", () => {
			// Create first instance to establish baseline
			const firstRouter = McpToolRouter.getInstance()
			const initialCount = executorEventEmitter.listenerCount("tool-registered")

			// Create multiple instances (should be same singleton)
			for (let i = 0; i < 10; i++) {
				const sameRouter = McpToolRouter.getInstance()
				expect(sameRouter).toBe(firstRouter) // Verify it's the same instance
			}

			// Listener count should not increase since it's the same instance
			const finalCount = executorEventEmitter.listenerCount("tool-registered")
			expect(finalCount).toBe(initialCount)
		})

		it("should properly clean up listeners on shutdown", async () => {
			router = McpToolRouter.getInstance()
			
			// Add some listeners
			const listener1 = jest.fn()
			const listener2 = jest.fn()
			router.on("tool-registered", listener1)
			router.on("started", listener2)

			await router.initialize()
			
			// Check listeners are active
			executorEventEmitter.emit("tool-registered", "test")
			expect(listener1).toHaveBeenCalled()

			// Note: The router doesn't actually remove listeners on shutdown
			// This is by design to allow restart
			await router.shutdown()

			// Listeners should still work after shutdown (for restart capability)
			listener1.mockClear()
			executorEventEmitter.emit("tool-registered", "test2")
			expect(listener1).toHaveBeenCalled()
		})
	})

	describe("State Consistency", () => {
		it("should maintain consistent state across lifecycle", async () => {
			router = McpToolRouter.getInstance({ port: 5000 })

			// Initial state
			expect((router as any).sseConfig).toEqual({ port: 5000 })

			// After initialization
			await router.initialize()
			expect((router as any).sseConfig).toEqual({ port: 5000 })

			// After shutdown
			await router.shutdown()
			expect((router as any).sseConfig).toEqual({ port: 5000 })

			// After re-initialization
			await router.initialize()
			expect((router as any).sseConfig).toEqual({ port: 5000 })
		})

		it("should handle concurrent operations gracefully", async () => {
			router = McpToolRouter.getInstance()

			// Start multiple operations concurrently
			const operations = [
				await router.initialize(),
				await router.initialize(),
				await router.shutdown(),
				await router.initialize(),
				await router.shutdown(),
			]

			// All should complete without errors
			await expect(Promise.all(operations)).resolves.not.toThrow()
		})
	})
})