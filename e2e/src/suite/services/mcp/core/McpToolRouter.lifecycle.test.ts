import * as assert from 'assert'
import * as sinon from 'sinon'/**
 * McpToolRouter lifecycle tests
 * Tests initialization, shutdown, repeated start/stop, pending registrations, and event forwarding
 */

import { McpToolRouter } from "../McpToolRouter"
import { McpToolExecutor } from "../McpToolExecutor"
import { McpToolRegistry } from "../McpToolRegistry"
import { EventEmitter } from "events"

// Mock the McpToolExecutor
// TODO: Use proxyquire for module mocking - "../McpToolExecutor")

suite("McpToolRouter - Lifecycle and Event Management", () => {
	let router: McpToolRouter | undefined
	let mockExecutor: sinon.SinonStubbedInstance<McpToolExecutor>
	let executorEventEmitter: EventEmitter

	setup(() => {
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
			initialize: sinon.stub().resolves(undefined),
			shutdown: sinon.stub().resolves(undefined),
			executeTool: sinon.stub(),
			on: sinon.stub((event: string, handler: Function) => {
				executorEventEmitter.on(event, handler as any)
				return mockExecutor
			}),
			emit: sinon.stub((event: string, ...args: any[]) => {
				executorEventEmitter.emit(event, ...args)
				return true
			}),
			removeAllListeners: sinon.stub(() => {
				executorEventEmitter.removeAllListeners()
				return mockExecutor
			}),
		} as any

		// Mock getInstance to return our mock
		(McpToolExecutor.getInstance as sinon.SinonStub).returns(mockExecutor)
	})

	teardown(() => {
		sinon.restore()
		// Clean up event listeners if they exist
		if (executorEventEmitter) {
			executorEventEmitter.removeAllListeners()
		}
	})

	suite("Initialization and Shutdown", () => {
		test("should initialize the executor on router initialization", async () => {
			router = McpToolRouter.getInstance()
			await router.initialize()

			assert.strictEqual(mockExecutor.initialize.callCount, 1)
		})

		test("should shutdown the executor on router shutdown", async () => {
			router = McpToolRouter.getInstance()
			await router.initialize()
			await router.shutdown()

			assert.strictEqual(mockExecutor.shutdown.callCount, 1)
		})

		test("should handle repeated initialization gracefully", async () => {
			router = McpToolRouter.getInstance()
			
			await router.initialize()
			await router.initialize()
			await router.initialize()

			// Should still only initialize once per instance
			assert.strictEqual(mockExecutor.initialize.callCount, 3)
		})

		test("should handle repeated shutdown gracefully", async () => {
			router = McpToolRouter.getInstance()
			await router.initialize()

			await router.shutdown()
			await router.shutdown()
			await router.shutdown()

			// Should handle multiple shutdowns
			assert.strictEqual(mockExecutor.shutdown.callCount, 3)
		})

		test("should handle init-shutdown-init cycles", async () => {
			router = McpToolRouter.getInstance()

			// First cycle
			await router.initialize()
			assert.strictEqual(mockExecutor.initialize.callCount, 1)
			await router.shutdown()
			assert.strictEqual(mockExecutor.shutdown.callCount, 1)

			// Second cycle
			await router.initialize()
			assert.strictEqual(mockExecutor.initialize.callCount, 2)
			await router.shutdown()
			assert.strictEqual(mockExecutor.shutdown.callCount, 2)

			// Third cycle
			await router.initialize()
			assert.strictEqual(mockExecutor.initialize.callCount, 3)
			await router.shutdown()
			assert.strictEqual(mockExecutor.shutdown.callCount, 3)
		})

		test("should maintain singleton instance across multiple getInstance calls", () => {
			const router1 = McpToolRouter.getInstance()
			const router2 = McpToolRouter.getInstance()
			const router3 = McpToolRouter.getInstance()

			assert.strictEqual(router1, router2)
			assert.strictEqual(router2, router3)
		})

		test("should update SSE config when provided to getInstance", () => {
			const config1 = { port: 3000 }
			const config2 = { port: 4000 }

			const router1 = McpToolRouter.getInstance(config1)
			expect((router1 as any).sseConfig).toEqual(config1)

			const router2 = McpToolRouter.getInstance(config2)
			assert.strictEqual(router2, router1) // Same instance
			expect((router2 as any).sseConfig).toEqual(config2) // Config updated
		})
	})

	suite("Event Forwarding", () => {
		setup(() => {
			router = McpToolRouter.getInstance()
		})

		test("should forward tool-registered events from executor", (done) => {
			const toolName = "test-tool"

			router.on("tool-registered", (name: string) => {
				assert.strictEqual(name, toolName)
				done()
			})

			// Emit event from the executor
			executorEventEmitter.emit("tool-registered", toolName)
		})

		test("should forward tool-unregistered events from executor", (done) => {
			const toolName = "test-tool"

			router.on("tool-unregistered", (name: string) => {
				assert.strictEqual(name, toolName)
				done()
			})

			// Emit event from the executor
			executorEventEmitter.emit("tool-unregistered", toolName)
		})

		test("should forward started events from executor", (done) => {
			const startInfo = { port: 3000, serverName: "test-server" }

			router.on("started", (info: unknown) => {
				assert.deepStrictEqual(info, startInfo)
				done()
			})

			// Emit event from the executor
			executorEventEmitter.emit("started", startInfo)
		})

		test("should forward stopped events from executor", (done) => {
			router.on("stopped", () => {
				done()
			})

			// Emit event from the executor
			executorEventEmitter.emit("stopped")
		})

		test("should handle multiple event listeners", () => {
			const listener1 = sinon.stub()
			const listener2 = sinon.stub()
			const listener3 = sinon.stub()

			router.on("tool-registered", listener1)
			router.on("tool-registered", listener2)
			router.on("tool-registered", listener3)

			executorEventEmitter.emit("tool-registered", "test-tool")

			assert.ok(listener1.calledWith("test-tool"))
			assert.ok(listener2.calledWith("test-tool"))
			assert.ok(listener3.calledWith("test-tool"))
		})

		test("should handle rapid event firing", () => {
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

			assert.strictEqual(registeredTools.length, 100)
			assert.strictEqual(unregisteredTools.length, 50)
		})
	})

	suite("Pending Registrations", () => {
		test("should handle tool registration during initialization", async () => {
			router = McpToolRouter.getInstance()
			
			// Mock slow initialization
			let initResolve: Function
			mockExecutor.initialize.callsFake(() => {
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
			assert.ok(registeredTools.includes("tool-4"))
		})

		test("should handle tool registration during shutdown", async () => {
			router = McpToolRouter.getInstance()
			await router.initialize()

			// Mock slow shutdown
			let shutdownResolve: Function
			mockExecutor.shutdown.callsFake(() => {
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

			assert.ok(mockExecutor.shutdown.called)
		})
	})

	suite("Error Handling", () => {
		test("should handle initialization errors", async () => {
			router = McpToolRouter.getInstance()
			
			const initError = new Error("Initialization failed")
			mockExecutor.initialize.rejects(initError)

			await expect(router.initialize()).rejects.toThrow("Initialization failed")
		})

		test("should handle shutdown errors", async () => {
			router = McpToolRouter.getInstance()
			await router.initialize()

			const shutdownError = new Error("Shutdown failed")
			mockExecutor.shutdown.rejects(shutdownError)

			await expect(router.shutdown()).rejects.toThrow("Shutdown failed")
		})

		test("should continue forwarding events after errors", async () => {
			router = McpToolRouter.getInstance()
			
			// Initialize with error
			mockExecutor.initialize.mockRejectedValueOnce(new Error("Init error"))
			await expect(router.initialize()).rejects.toThrow()

			// Events should still be forwarded
			const listener = sinon.stub()
			router.on("tool-registered", listener)
			executorEventEmitter.emit("tool-registered", "test-tool")
			
			assert.ok(listener.calledWith("test-tool"))
		})
	})

	suite("Memory Management", () => {
		test("should not leak event listeners on repeated getInstance calls", () => {
			// Create first instance to establish baseline
			const firstRouter = McpToolRouter.getInstance()
			const initialCount = executorEventEmitter.listenerCount("tool-registered")

			// Create multiple instances (should be same singleton)
			for (let i = 0; i < 10; i++) {
				const sameRouter = McpToolRouter.getInstance()
				assert.strictEqual(sameRouter, firstRouter) // Verify it's the same instance
			}

			// Listener count should not increase since it's the same instance
			const finalCount = executorEventEmitter.listenerCount("tool-registered")
			assert.strictEqual(finalCount, initialCount)
		})

		test("should properly clean up listeners on shutdown", async () => {
			router = McpToolRouter.getInstance()
			
			// Add some listeners
			const listener1 = sinon.stub()
			const listener2 = sinon.stub()
			router.on("tool-registered", listener1)
			router.on("started", listener2)

			await router.initialize()
			
			// Check listeners are active
			executorEventEmitter.emit("tool-registered", "test")
			assert.ok(listener1.called)

			// Note: The router doesn't actually remove listeners on shutdown
			// This is by design to allow restart
			await router.shutdown()

			// Listeners should still work after shutdown (for restart capability)
			listener1.resetHistory()
			executorEventEmitter.emit("tool-registered", "test2")
			assert.ok(listener1.called)
		})
	})

	suite("State Consistency", () => {
		test("should maintain consistent state across lifecycle", async () => {
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

		test("should handle concurrent operations gracefully", async () => {
			router = McpToolRouter.getInstance()

			// Start multiple operations concurrently
			const operations = [
				router.initialize(),
				router.initialize(),
				router.shutdown(),
				router.initialize(),
				router.shutdown(),
			]

			// All should complete without errors
			await expect(Promise.all(operations)).resolves.not.toThrow()
		})
	})
})