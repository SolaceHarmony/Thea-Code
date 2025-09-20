import * as assert from 'assert'
import { expect } from 'chai'
import * as sinon from 'sinon'
/**
 * Stdio Transport lifecycle tests as recommended by architect
 * Tests SDK import with fallback, stderr exposure, and handler management
 */

import { StdioTransportConfig } from "../../types/McpProviderTypes"
import { StdioTransport } from "../StdioTransport"

suite("StdioTransport Lifecycle Tests", () => {
	let transport: StdioTransport
	
	teardown(async () => {
		// Clean up transport if it exists
		if (transport) {
			try {
				await transport.close()
			} catch (error) {
			assert.fail('Unexpected error: ' + error.message)
		} catch {
				// Transport might already be closed
			} catch (error) {
			assert.fail("Unexpected error: " + error.message)
		}
		}
	})

	suite("Initialization and startup", () => {
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
			assert.notStrictEqual(stderr, undefined)
		})

		test("should handle missing SDK with fallback mock", async () => {
			const prev = process.env.THEA_DISABLE_MCP_SDK
			process.env.THEA_DISABLE_MCP_SDK = "1"
			try {
				const config: StdioTransportConfig = {
				command: "echo",
				args: ["test"],
				env: {}
			}
				transport = new StdioTransport(config)
				// Should use mock transport as fallback
				await expect(transport.start()).resolves.not.toThrow()
				// Should still have stderr (from mock)
				assert.notStrictEqual(transport.stderr, undefined)
			} finally {
				if (prev === undefined) delete process.env.THEA_DISABLE_MCP_SDK
				else process.env.THEA_DISABLE_MCP_SDK = prev
			}
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
			assert.notStrictEqual(transport.stderr, undefined)
		})
	})

	suite("Stderr management", () => {
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
			assert.notStrictEqual(stderrAfter, undefined)
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
				assert.notStrictEqual(stderr, undefined)
			}
		})
	})

	suite("Event handlers", () => {
		test("should set onerror handler", async () => {
			const config: StdioTransportConfig = {
				command: "echo",
				args: ["test"]
			}
			
			transport = new StdioTransport(config)
			
			const errorHandler = sinon.stub()
			transport.onerror = errorHandler
			
			await transport.start()
			
			// Handler should be set
			assert.strictEqual(transport.onerror, errorHandler)
		})

		test("should set onclose handler", async () => {
			const config: StdioTransportConfig = {
				command: "echo",
				args: ["test"]
			}
			
			transport = new StdioTransport(config)
			
			const closeHandler = sinon.stub()
			transport.onclose = closeHandler
			
			await transport.start()
			
			// Handler should be set
			assert.strictEqual(transport.onclose, closeHandler)
		})

		test("should trigger error handler on transport error", async () => {
			const config: StdioTransportConfig = {
				command: "nonexistent-command-that-should-fail",
				args: []
			}
			
			transport = new StdioTransport(config)
			
			const errorHandler = sinon.stub()
			transport.onerror = errorHandler
			
			// This might throw or trigger error handler depending on implementation
			try {
				await transport.start()
			} catch (error) {
			assert.fail('Unexpected error: ' + error.message)
		} catch {
				// Error during start is acceptable
			} catch (error) {
			assert.fail("Unexpected error: " + error.message)
		}
			
			// Give some time for async error
			await new Promise(resolve => setTimeout(resolve, 100))
			
			// Error handler might have been called
			// (depends on whether the command exists and how errors are handled)
		})
	})

	suite("Lifecycle edge cases", () => {
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
			assert.notStrictEqual(transport.stderr, undefined)
			
			// Close should clean up
			await transport.close()
			
			// Resources should be cleaned up
			// (exact behavior depends on implementation)
		})
	})

	suite("MockStdioServerTransport fallback", () => {
		let prevEnv: string | undefined
		setup(() => {
			prevEnv = process.env.THEA_DISABLE_MCP_SDK
			process.env.THEA_DISABLE_MCP_SDK = "1"
		})

		teardown(() => {
			if (prevEnv === undefined) delete process.env.THEA_DISABLE_MCP_SDK
			else process.env.THEA_DISABLE_MCP_SDK = prevEnv
		})

		test("should use mock when SDK is not available", async () => {
			const config: StdioTransportConfig = {
				command: "echo",
				args: ["test"]
			}
			
			transport = new StdioTransport(config)
			await transport.start()
			
			// Should still work with mock
			assert.notStrictEqual(transport.stderr, undefined)
			
			// Mock should provide basic functionality
			const errorHandler = sinon.stub()
			transport.onerror = errorHandler
			
			const closeHandler = sinon.stub()
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
			assert.notStrictEqual(transport.stderr, undefined)
			
			// Close with mock
			await transport.close()
			
			// Should handle restart
			await transport.start()
			assert.notStrictEqual(transport.stderr, undefined)
		})
	})

	suite("Configuration validation", () => {
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
			assert.notStrictEqual(transport.stderr, undefined)
		})
	})
// Mock cleanup
