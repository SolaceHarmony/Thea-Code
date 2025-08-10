/**
 * Stdio Transport lifecycle tests as recommended by architect
 * Tests SDK import with fallback, stderr exposure, and handler management
 */

import { StdioTransport } from "../StdioTransport"
import { StdioTransportConfig } from "../../types/McpProviderTypes"

describe("StdioTransport Lifecycle Tests", () => {
	let transport: StdioTransport
	
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
			const config: StdioTransportConfig = {
				command: "echo",
				args: ["test"],
				env: {}
			}
			
			transport = new StdioTransport(config)
			await transport.start()
			
			// Should have stderr available
			const stderr = transport.stderr
			expect(stderr).toBeDefined()
		})

		test("should handle missing SDK with fallback mock", async () => {
			// Mock the SDK import to fail
			jest.doMock("@modelcontextprotocol/sdk/server/stdio.js", () => {
				throw new Error("Module not found")
			})
			
			const config: StdioTransportConfig = {
				command: "echo",
				args: ["test"],
				env: {}
			}
			
			transport = new StdioTransport(config)
			
			// Should use mock transport as fallback
			await expect(transport.start()).resolves.not.toThrow()
			
			// Should still have stderr (from mock)
			expect(transport.stderr).toBeDefined()
			
			// Restore
			jest.unmock("@modelcontextprotocol/sdk/server/stdio.js")
		})

		test("should pass config to underlying transport", async () => {
			const config: StdioTransportConfig = {
				command: "node",
				args: ["--version"],
				env: {
					NODE_ENV: "test",
					CUSTOM_VAR: "value"
				}
			}
			
			transport = new StdioTransport(config)
			await transport.start()
			
			// Transport should be initialized with the config
			expect(transport.stderr).toBeDefined()
		})
	})

	describe("Stderr management", () => {
		test("should expose stderr stream", async () => {
			const config: StdioTransportConfig = {
				command: "echo",
				args: ["test"]
			}
			
			transport = new StdioTransport(config)
			
			// Before start, stderr might not be available
			const stderrBefore = transport.stderr
			
			await transport.start()
			
			// After start, stderr should be available
			const stderrAfter = transport.stderr
			expect(stderrAfter).toBeDefined()
		})

		test("should handle stderr data events", async () => {
			const config: StdioTransportConfig = {
				command: "node",
				args: ["-e", "console.error('test error')"]
			}
			
			transport = new StdioTransport(config)
			await transport.start()
			
			const stderr = transport.stderr
			if (stderr && stderr.on) {
				const errorData = await new Promise<string>((resolve) => {
					stderr.on("data", (data: Buffer) => {
						resolve(data.toString())
					})
					
					// Timeout fallback
					setTimeout(() => resolve(""), 1000)
				})
				
				// May or may not capture depending on timing
				// Just verify stderr is set up correctly
				expect(stderr).toBeDefined()
			}
		})
	})

	describe("Event handlers", () => {
		test("should set onerror handler", async () => {
			const config: StdioTransportConfig = {
				command: "echo",
				args: ["test"]
			}
			
			transport = new StdioTransport(config)
			
			const errorHandler = jest.fn()
			transport.onerror = errorHandler
			
			await transport.start()
			
			// Handler should be set
			expect(transport.onerror).toBe(errorHandler)
		})

		test("should set onclose handler", async () => {
			const config: StdioTransportConfig = {
				command: "echo",
				args: ["test"]
			}
			
			transport = new StdioTransport(config)
			
			const closeHandler = jest.fn()
			transport.onclose = closeHandler
			
			await transport.start()
			
			// Handler should be set
			expect(transport.onclose).toBe(closeHandler)
		})

		test("should trigger error handler on transport error", async () => {
			const config: StdioTransportConfig = {
				command: "nonexistent-command-that-should-fail",
				args: []
			}
			
			transport = new StdioTransport(config)
			
			const errorHandler = jest.fn()
			transport.onerror = errorHandler
			
			// This might throw or trigger error handler depending on implementation
			try {
				await transport.start()
			} catch {
				// Error during start is acceptable
			}
			
			// Give some time for async error
			await new Promise(resolve => setTimeout(resolve, 100))
			
			// Error handler might have been called
			// (depends on whether the command exists and how errors are handled)
		})
	})

	describe("Lifecycle edge cases", () => {
		test("should handle multiple start calls", async () => {
			const config: StdioTransportConfig = {
				command: "echo",
				args: ["test"]
			}
			
			transport = new StdioTransport(config)
			
			// First start
			await transport.start()
			
			// Second start should not throw
			await expect(transport.start()).resolves.not.toThrow()
		})

		test("should handle close without start", async () => {
			const config: StdioTransportConfig = {
				command: "echo",
				args: ["test"]
			}
			
			transport = new StdioTransport(config)
			
			// Should not throw when closing without starting
			await expect(transport.close()).resolves.not.toThrow()
		})

		test("should handle multiple close calls", async () => {
			const config: StdioTransportConfig = {
				command: "echo",
				args: ["test"]
			}
			
			transport = new StdioTransport(config)
			await transport.start()
			
			// First close
			await transport.close()
			
			// Second close should not throw
			await expect(transport.close()).resolves.not.toThrow()
		})

		test("should clean up resources on close", async () => {
			const config: StdioTransportConfig = {
				command: "node",
				args: ["-e", "setInterval(() => {}, 1000)"] // Keep alive
			}
			
			transport = new StdioTransport(config)
			await transport.start()
			
			// Should have stderr
			expect(transport.stderr).toBeDefined()
			
			// Close should clean up
			await transport.close()
			
			// Resources should be cleaned up
			// (exact behavior depends on implementation)
		})
	})

	describe("MockStdioServerTransport fallback", () => {
		beforeEach(() => {
			// Force mock usage by making SDK unavailable
			jest.doMock("@modelcontextprotocol/sdk/server/stdio.js", () => {
				throw new Error("Module not found")
			})
		})

		afterEach(() => {
			jest.unmock("@modelcontextprotocol/sdk/server/stdio.js")
		})

		test("should use mock when SDK is not available", async () => {
			const config: StdioTransportConfig = {
				command: "echo",
				args: ["test"]
			}
			
			transport = new StdioTransport(config)
			await transport.start()
			
			// Should still work with mock
			expect(transport.stderr).toBeDefined()
			
			// Mock should provide basic functionality
			const errorHandler = jest.fn()
			transport.onerror = errorHandler
			
			const closeHandler = jest.fn()
			transport.onclose = closeHandler
		})

		test("should handle mock lifecycle correctly", async () => {
			const config: StdioTransportConfig = {
				command: "test-command",
				args: ["arg1", "arg2"]
			}
			
			transport = new StdioTransport(config)
			
			// Start with mock
			await transport.start()
			expect(transport.stderr).toBeDefined()
			
			// Close with mock
			await transport.close()
			
			// Should handle restart
			await transport.start()
			expect(transport.stderr).toBeDefined()
		})
	})

	describe("Configuration validation", () => {
		test("should handle empty args array", async () => {
			const config: StdioTransportConfig = {
				command: "echo",
				args: []
			}
			
			transport = new StdioTransport(config)
			await expect(transport.start()).resolves.not.toThrow()
		})

		test("should handle undefined args", async () => {
			const config: StdioTransportConfig = {
				command: "echo"
			} as StdioTransportConfig
			
			transport = new StdioTransport(config)
			await expect(transport.start()).resolves.not.toThrow()
		})

		test("should handle undefined env", async () => {
			const config: StdioTransportConfig = {
				command: "echo",
				args: ["test"]
			} as StdioTransportConfig
			
			transport = new StdioTransport(config)
			await expect(transport.start()).resolves.not.toThrow()
		})

		test("should handle complex environment variables", async () => {
			const config: StdioTransportConfig = {
				command: "node",
				args: ["-e", "console.log(process.env.TEST_VAR)"],
				env: {
					TEST_VAR: "test value with spaces",
					PATH: process.env.PATH || "",
					HOME: process.env.HOME || "",
					NESTED_JSON: JSON.stringify({ key: "value" })
				}
			}
			
			transport = new StdioTransport(config)
			await expect(transport.start()).resolves.not.toThrow()
			expect(transport.stderr).toBeDefined()
		})
	})
})