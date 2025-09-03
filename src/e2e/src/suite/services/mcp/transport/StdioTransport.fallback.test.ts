import * as assert from 'assert'
import * as sinon from 'sinon'
/**
 * StdioTransport fallback and mock stderr tests
 * Tests fallback behavior when SDK is not available, mock stderr handling, and error handlers
 */

import { StdioTransportConfig } from "../../types/McpTransportTypes"
import { StdioTransport } from "../StdioTransport"
import { EventEmitter } from "events"

// Create a mock stderr stream
class MockStderrStream extends EventEmitter {
	pipe(destination: any): any {
		return destination
	}
	
	write(data: string): boolean {
		this.emit('data', data)
		return true
	}
	
	end(): void {
		this.emit('end')
	}
}

suite("StdioTransport - Fallback and Mock Behavior", () => {
	let transport: StdioTransport | undefined
	let originalConsoleWarn: typeof console.warn
	let consoleWarnSpy: sinon.SinonStub

	setup(() => {
		transport = undefined
		originalConsoleWarn = console.warn
		consoleWarnSpy = sinon.spy(console, 'warn').callsFake()
	})

	teardown(async () => {
		// Clean up transport
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
		
		// Restore console.warn
		consoleWarnSpy.restore()
		console.warn = originalConsoleWarn
		
		// Clear all mocks
		sinon.restore()
	})

	suite("MockStdioServerTransport Fallback", () => {
		test("should use mock transport when SDK module is not available", async () => {
			// Create transport - it will try to import SDK and fall back to mock
			const config: StdioTransportConfig = {
				command: "test-command",
				args: ["arg1", "arg2"],
				env: { TEST_ENV: "value" }
			}

			transport = new StdioTransport(config)
			
			// Start should work even without SDK
			await expect(transport.start()).resolves.not.toThrow()
			
			// Should have logged warning about missing SDK
			assert.ok(consoleWarnSpy.calledWith("MCP SDK not found, using mock implementation"))
			
			// Mock transport returns undefined for stderr
			assert.strictEqual(transport.stderr, undefined)
		})

		test("should handle start/close lifecycle with mock transport", async () => {
			const config: StdioTransportConfig = {
				command: "mock-test",
				args: [],
				env: {}
			}

			transport = new StdioTransport(config)
			
			// Start multiple times should be safe
			await transport.start()
			await transport.start() // Should not re-initialize
			
			// Close should work
			await expect(transport.close()).resolves.not.toThrow()
			
			// Close multiple times should be safe
			await expect(transport.close()).resolves.not.toThrow()
		})

		test("should return undefined for getPort with mock transport", async () => {
			const config: StdioTransportConfig = {
				command: "test",
				args: [],
				env: {}
			}

			transport = new StdioTransport(config)
			await transport.start()
			
			// StdioTransport doesn't use ports
			expect(transport.getPort()).toBeUndefined()
		})
	})

	suite("Stderr Handling", () => {
		test("should expose stderr when real transport is available", async () => {
			// Mock a successful SDK import
			const mockStderr = new MockStderrStream()
			const MockTransportClass = class {
				stderr = mockStderr
				async start(): Promise<void> {}
				async close(): Promise<void> {}
			}

			// Mock the dynamic import to return our mock transport
			// TODO: Use proxyquire for module mocking
		// Mock for "@modelcontextprotocol/sdk/server/stdio.js" needed here
				StdioServerTransport: MockTransportClass
			}))

			const config: StdioTransportConfig = {
				command: "test",
				args: [],
				env: {}
			}

			transport = new StdioTransport(config)
			await transport.start()
			
			// Should expose the stderr stream
			assert.strictEqual(transport.stderr, mockStderr)
			
			// Cleanup mock
			// TODO: Use actual module - "@modelcontextprotocol/sdk/server/stdio.js")
		})

		test("should handle stderr data events", async () => {
			const mockStderr = new MockStderrStream()
			const stderrData: string[] = []
			
			// Capture stderr data
			mockStderr.on('data', (data: string) => {
				stderrData.push(data)
			})
			
			// Simulate stderr output
			mockStderr.write("Error: something went wrong\n")
			mockStderr.write("Warning: deprecated feature\n")
			
			assert.strictEqual(stderrData.length, 2)
			assert.strictEqual(stderrData[0], "Error: something went wrong\n")
			assert.strictEqual(stderrData[1], "Warning: deprecated feature\n")
		})

		test("should handle stderr stream closure", (done) => {
			const mockStderr = new MockStderrStream()
			
			mockStderr.on('end', () => {
				done()
			})
			
			// Simulate stream ending
			mockStderr.end()
		})
	})

	suite("Error and Close Handlers", () => {
		test("should set error handler on mock transport", async () => {
			const config: StdioTransportConfig = {
				command: "test",
				args: [],
				env: {}
			}

			transport = new StdioTransport(config)
			await transport.start()
			
			const errorHandler = sinon.stub()
			transport.onerror = errorHandler
			
			// Handler should be set (even if mock doesn't trigger it)
			// We're mainly testing that the setter doesn't throw
			assert.doesNotThrow(() => transport!.onerror = errorHandler)
		})

		test("should set close handler on mock transport", async () => {
			const config: StdioTransportConfig = {
				command: "test",
				args: [],
				env: {}
			}

			transport = new StdioTransport(config)
			await transport.start()
			
			const closeHandler = sinon.stub()
			transport.onclose = closeHandler
			
			// Handler should be set (even if mock doesn't trigger it)
			assert.doesNotThrow(() => transport!.onclose = closeHandler)
		})

		test("should forward handlers to real transport when available", async () => {
			const errorHandler = sinon.stub()
			const closeHandler = sinon.stub()
			
			// Mock transport with handler support
			const MockTransportClass = class {
				stderr = undefined
				onerror?: (error: Error) => void
				onclose?: () => void
				
				async start(): Promise<void> {}
				async close(): Promise<void> {}
				
				triggerError(error: Error): void {
					if (this.onerror) {
						this.onerror(error)
					}
				}
				
				triggerClose(): void {
					if (this.onclose) {
						this.onclose()
					}
				}
			}

			const mockInstance = new MockTransportClass()
			
			// TODO: Use proxyquire for module mocking
		// Mock for "@modelcontextprotocol/sdk/server/stdio.js" needed here
				StdioServerTransport: class {
					constructor() {
						return mockInstance
					}
				}
			})

			const config: StdioTransportConfig = {
				command: "test",
				args: [],
				env: {}
			}

			transport = new StdioTransport(config)
			await transport.start()
			
			// Set handlers
			transport.onerror = errorHandler
			transport.onclose = closeHandler
			
			// Trigger events through mock
			const testError = new Error("Test error")
			mockInstance.triggerError(testError)
			mockInstance.triggerClose()
			
			// Handlers should have been called
			assert.ok(errorHandler.calledWith(testError))
			assert.ok(closeHandler.called)
			
			// Cleanup
			// TODO: Use actual module - "@modelcontextprotocol/sdk/server/stdio.js")
		})

		test("should handle setting handlers before transport initialization", () => {
			const config: StdioTransportConfig = {
				command: "test",
				args: [],
				env: {}
			}

			transport = new StdioTransport(config)
			
			// Set handlers before start
			const errorHandler = sinon.stub()
			const closeHandler = sinon.stub()
			
			// Should not throw even without transport initialized
			assert.doesNotThrow(() => {
				transport!.onerror = errorHandler
				transport!.onclose = closeHandler
			})
		})
	})

	suite("Environment Variable Handling", () => {
		test("should merge PATH from process.env with provided env", async () => {
			const originalPath = process.env.PATH
			process.env.PATH = "/usr/bin:/usr/local/bin"
			
			let capturedConfig: any
			
			const MockTransportClass = class {
				constructor(config: any) {
					capturedConfig = config
				}
				stderr = undefined
				async start(): Promise<void> {}
				async close(): Promise<void> {}
			}

			// TODO: Use proxyquire for module mocking
		// Mock for "@modelcontextprotocol/sdk/server/stdio.js" needed here
				StdioServerTransport: MockTransportClass
			}))

			const config: StdioTransportConfig = {
				command: "test",
				args: [],
				env: {
					CUSTOM_VAR: "custom_value"
				}
			}

			transport = new StdioTransport(config)
			await transport.start()
			
			// Should have both custom env and PATH
			assert.strictEqual(capturedConfig.env.CUSTOM_VAR, "custom_value")
			assert.strictEqual(capturedConfig.env.PATH, "/usr/bin:/usr/local/bin")
			
			// Restore PATH
			if (originalPath !== undefined) {
				process.env.PATH = originalPath
} else {
				delete process.env.PATH
			}
			
			// Cleanup
			// TODO: Use actual module - "@modelcontextprotocol/sdk/server/stdio.js")
		})

		test("should handle missing PATH in process.env", async () => {
			const originalPath = process.env.PATH
			delete process.env.PATH
			
			let capturedConfig: any
			
			const MockTransportClass = class {
				constructor(config: any) {
					capturedConfig = config
				}
				stderr = undefined
				async start(): Promise<void> {}
				async close(): Promise<void> {}
			}

			// TODO: Use proxyquire for module mocking
		// Mock for "@modelcontextprotocol/sdk/server/stdio.js" needed here
				StdioServerTransport: MockTransportClass
			}))

			const config: StdioTransportConfig = {
				command: "test",
				args: [],
				env: {
					CUSTOM_VAR: "value"
				}
			}

			transport = new StdioTransport(config)
			await transport.start()
			
			// Should only have custom env, no PATH
			assert.strictEqual(capturedConfig.env.CUSTOM_VAR, "value")
			assert.strictEqual(capturedConfig.env.PATH, undefined)
			
			// Restore PATH
			if (originalPath !== undefined) {
				process.env.PATH = originalPath
			}
			
			// Cleanup
			// TODO: Use actual module - "@modelcontextprotocol/sdk/server/stdio.js")
		})
	})

	suite("Configuration Validation", () => {
		test("should pass command and args to underlying transport", async () => {
			let capturedConfig: any
			
			const MockTransportClass = class {
				constructor(config: any) {
					capturedConfig = config
				}
				stderr = undefined
				async start(): Promise<void> {}
				async close(): Promise<void> {}
			}

			// TODO: Use proxyquire for module mocking
		// Mock for "@modelcontextprotocol/sdk/server/stdio.js" needed here
				StdioServerTransport: MockTransportClass
			}))

			const config: StdioTransportConfig = {
				command: "/usr/bin/my-tool",
				args: ["--option", "value", "--flag"],
				env: {}
			}

			transport = new StdioTransport(config)
			await transport.start()
			
			assert.strictEqual(capturedConfig.command, "/usr/bin/my-tool")
			assert.deepStrictEqual(capturedConfig.args, ["--option", "value", "--flag"])
			assert.strictEqual(capturedConfig.stderr, "pipe") // Always pipes stderr
			
			// Cleanup
			// TODO: Use actual module - "@modelcontextprotocol/sdk/server/stdio.js")
		})

		test("should handle empty args array", async () => {
			const config: StdioTransportConfig = {
				command: "simple-command",
				args: [],
				env: {}
			}

			transport = new StdioTransport(config)
			await expect(transport.start()).resolves.not.toThrow()
		})

		test("should handle complex environment variables", async () => {
			let capturedConfig: any
			
			const MockTransportClass = class {
				constructor(config: any) {
					capturedConfig = config
				}
				stderr = undefined
				async start(): Promise<void> {}
				async close(): Promise<void> {}
			}

			// TODO: Use proxyquire for module mocking
		// Mock for "@modelcontextprotocol/sdk/server/stdio.js" needed here
				StdioServerTransport: MockTransportClass
			}))

			const config: StdioTransportConfig = {
				command: "test",
				args: [],
				env: {
					MULTI_LINE: "line1\nline2\nline3",
					WITH_SPACES: "value with spaces",
					WITH_QUOTES: 'value with "quotes"',
					EMPTY: "",
					NUMBER: "12345"
				}
			}

			transport = new StdioTransport(config)
			await transport.start()
			
			assert.strictEqual(capturedConfig.env.MULTI_LINE, "line1\nline2\nline3")
			assert.strictEqual(capturedConfig.env.WITH_SPACES, "value with spaces")
			assert.strictEqual(capturedConfig.env.WITH_QUOTES, 'value with "quotes"')
			assert.strictEqual(capturedConfig.env.EMPTY, "")
			assert.strictEqual(capturedConfig.env.NUMBER, "12345")
			
			// Cleanup
			// TODO: Use actual module - "@modelcontextprotocol/sdk/server/stdio.js")
		})
	})

	suite("Edge Cases", () => {
		test("should handle rapid start/close cycles", async () => {
			const config: StdioTransportConfig = {
				command: "test",
				args: [],
				env: {}
			}

			transport = new StdioTransport(config)
			
			// Rapid cycles
			for (let i = 0; i < 10; i++) {
				await transport.start()
				await transport.close()
			}
			
			// Should not throw (warning is only shown once when falling back to mock)
			assert.strictEqual(consoleWarnSpy.callCount, 1)
		})

		test("should maintain state consistency after errors", async () => {
			const MockTransportClass = class {
				stderr = undefined
				async start(): Promise<void> {
					throw new Error("Start failed")
				}
				async close(): Promise<void> {}
			}

			// TODO: Use proxyquire for module mocking
		// Mock for "@modelcontextprotocol/sdk/server/stdio.js" needed here
				StdioServerTransport: MockTransportClass
			}))

			const config: StdioTransportConfig = {
				command: "test",
				args: [],
				env: {}
			}

			transport = new StdioTransport(config)
			
			// Start should fail
			await expect(transport.start()).rejects.toThrow("Start failed")
			
			// But close should still work
			await expect(transport.close()).resolves.not.toThrow()
			
			// Cleanup
			// TODO: Use actual module - "@modelcontextprotocol/sdk/server/stdio.js")
		})

		test("should only initialize transport once", async () => {
			let constructorCallCount = 0
			
			const MockTransportClass = class {
				constructor() {
					constructorCallCount++
				}
				stderr = undefined
				async start(): Promise<void> {}
				async close(): Promise<void> {}
			}

			// TODO: Use proxyquire for module mocking
		// Mock for "@modelcontextprotocol/sdk/server/stdio.js" needed here
				StdioServerTransport: MockTransportClass
			}))

			const config: StdioTransportConfig = {
				command: "test",
				args: [],
				env: {}
			}

			transport = new StdioTransport(config)
			
			// Multiple starts should only create one transport instance
			await transport.start()
			await transport.start()
			await transport.start()
			
			assert.strictEqual(constructorCallCount, 1)
			
			// Cleanup
			// TODO: Use actual module - "@modelcontextprotocol/sdk/server/stdio.js")
		})
	})
// Mock cleanup
