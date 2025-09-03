/**
 * StdioTransport fallback and mock stderr tests
 * Tests fallback behavior when SDK is not available, mock stderr handling, and error handlers
 */

import { describe, expect, it, jest, beforeEach, afterEach } from "@jest/globals"
import { StdioTransport } from "../StdioTransport"
import { StdioTransportConfig } from "../../types/McpTransportTypes"
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

describe("StdioTransport - Fallback and Mock Behavior", () => {
	let transport: StdioTransport | undefined
	let originalConsoleWarn: typeof console.warn
	let consoleWarnSpy: jest.SpyInstance

	beforeEach(() => {
		transport = undefined
		originalConsoleWarn = console.warn
		consoleWarnSpy = jest.spyOn(console, 'warn').mockImplementation()
	})

	afterEach(async () => {
		// Clean up transport
		if (transport) {
			try {
				await transport.close()
			} catch {
				// Transport might already be closed
			}
		}
		
		// Restore console.warn
		consoleWarnSpy.mockRestore()
		console.warn = originalConsoleWarn
		
		// Clear all mocks
		jest.clearAllMocks()
	})

	describe("MockStdioServerTransport Fallback", () => {
		it("should use mock transport when SDK module is not available", async () => {
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
			expect(consoleWarnSpy).toHaveBeenCalledWith("MCP SDK not found, using mock implementation")
			
			// Mock transport returns undefined for stderr
			expect(transport.stderr).toBeUndefined()
		})

		it("should handle start/close lifecycle with mock transport", async () => {
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

		it("should return undefined for getPort with mock transport", async () => {
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

	describe("Stderr Handling", () => {
		it("should expose stderr when real transport is available", async () => {
			// Mock a successful SDK import
			const mockStderr = new MockStderrStream()
			const MockTransportClass = class {
				stderr = mockStderr
				async start(): Promise<void> {}
				async close(): Promise<void> {}
			}

			// Mock the dynamic import to return our mock transport
			jest.doMock("@modelcontextprotocol/sdk/server/stdio.js", () => ({
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
			expect(transport.stderr).toBe(mockStderr)
			
			// Cleanup mock
			jest.dontMock("@modelcontextprotocol/sdk/server/stdio.js")
		})

		it("should handle stderr data events", async () => {
			const mockStderr = new MockStderrStream()
			const stderrData: string[] = []
			
			// Capture stderr data
			mockStderr.on('data', (data: string) => {
				stderrData.push(data)
			})
			
			// Simulate stderr output
			mockStderr.write("Error: something went wrong\n")
			mockStderr.write("Warning: deprecated feature\n")
			
			expect(stderrData).toHaveLength(2)
			expect(stderrData[0]).toBe("Error: something went wrong\n")
			expect(stderrData[1]).toBe("Warning: deprecated feature\n")
		})

		it("should handle stderr stream closure", (done) => {
			const mockStderr = new MockStderrStream()
			
			mockStderr.on('end', () => {
				done()
			})
			
			// Simulate stream ending
			mockStderr.end()
		})
	})

	describe("Error and Close Handlers", () => {
		it("should set error handler on mock transport", async () => {
			const config: StdioTransportConfig = {
				command: "test",
				args: [],
				env: {}
			}

			transport = new StdioTransport(config)
			await transport.start()
			
			const errorHandler = jest.fn()
			transport.onerror = errorHandler
			
			// Handler should be set (even if mock doesn't trigger it)
			// We're mainly testing that the setter doesn't throw
			expect(() => transport!.onerror = errorHandler).not.toThrow()
		})

		it("should set close handler on mock transport", async () => {
			const config: StdioTransportConfig = {
				command: "test",
				args: [],
				env: {}
			}

			transport = new StdioTransport(config)
			await transport.start()
			
			const closeHandler = jest.fn()
			transport.onclose = closeHandler
			
			// Handler should be set (even if mock doesn't trigger it)
			expect(() => transport!.onclose = closeHandler).not.toThrow()
		})

		it("should forward handlers to real transport when available", async () => {
			const errorHandler = jest.fn()
			const closeHandler = jest.fn()
			
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
			
			jest.doMock("@modelcontextprotocol/sdk/server/stdio.js", () => ({
				StdioServerTransport: class {
					constructor() {
						return mockInstance
					}
				}
			}))

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
			expect(errorHandler).toHaveBeenCalledWith(testError)
			expect(closeHandler).toHaveBeenCalled()
			
			// Cleanup
			jest.dontMock("@modelcontextprotocol/sdk/server/stdio.js")
		})

		it("should handle setting handlers before transport initialization", () => {
			const config: StdioTransportConfig = {
				command: "test",
				args: [],
				env: {}
			}

			transport = new StdioTransport(config)
			
			// Set handlers before start
			const errorHandler = jest.fn()
			const closeHandler = jest.fn()
			
			// Should not throw even without transport initialized
			expect(() => {
				transport!.onerror = errorHandler
				transport!.onclose = closeHandler
			}).not.toThrow()
		})
	})

	describe("Environment Variable Handling", () => {
		it("should merge PATH from process.env with provided env", async () => {
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

			jest.doMock("@modelcontextprotocol/sdk/server/stdio.js", () => ({
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
			expect(capturedConfig.env.CUSTOM_VAR).toBe("custom_value")
			expect(capturedConfig.env.PATH).toBe("/usr/bin:/usr/local/bin")
			
			// Restore PATH
			if (originalPath !== undefined) {
				process.env.PATH = originalPath
			} else {
				delete process.env.PATH
			}
			
			// Cleanup
			jest.dontMock("@modelcontextprotocol/sdk/server/stdio.js")
		})

		it("should handle missing PATH in process.env", async () => {
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

			jest.doMock("@modelcontextprotocol/sdk/server/stdio.js", () => ({
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
			expect(capturedConfig.env.CUSTOM_VAR).toBe("value")
			expect(capturedConfig.env.PATH).toBeUndefined()
			
			// Restore PATH
			if (originalPath !== undefined) {
				process.env.PATH = originalPath
			}
			
			// Cleanup
			jest.dontMock("@modelcontextprotocol/sdk/server/stdio.js")
		})
	})

	describe("Configuration Validation", () => {
		it("should pass command and args to underlying transport", async () => {
			let capturedConfig: any
			
			const MockTransportClass = class {
				constructor(config: any) {
					capturedConfig = config
				}
				stderr = undefined
				async start(): Promise<void> {}
				async close(): Promise<void> {}
			}

			jest.doMock("@modelcontextprotocol/sdk/server/stdio.js", () => ({
				StdioServerTransport: MockTransportClass
			}))

			const config: StdioTransportConfig = {
				command: "/usr/bin/my-tool",
				args: ["--option", "value", "--flag"],
				env: {}
			}

			transport = new StdioTransport(config)
			await transport.start()
			
			expect(capturedConfig.command).toBe("/usr/bin/my-tool")
			expect(capturedConfig.args).toEqual(["--option", "value", "--flag"])
			expect(capturedConfig.stderr).toBe("pipe") // Always pipes stderr
			
			// Cleanup
			jest.dontMock("@modelcontextprotocol/sdk/server/stdio.js")
		})

		it("should handle empty args array", async () => {
			const config: StdioTransportConfig = {
				command: "simple-command",
				args: [],
				env: {}
			}

			transport = new StdioTransport(config)
			await expect(transport.start()).resolves.not.toThrow()
		})

		it("should handle complex environment variables", async () => {
			let capturedConfig: any
			
			const MockTransportClass = class {
				constructor(config: any) {
					capturedConfig = config
				}
				stderr = undefined
				async start(): Promise<void> {}
				async close(): Promise<void> {}
			}

			jest.doMock("@modelcontextprotocol/sdk/server/stdio.js", () => ({
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
			
			expect(capturedConfig.env.MULTI_LINE).toBe("line1\nline2\nline3")
			expect(capturedConfig.env.WITH_SPACES).toBe("value with spaces")
			expect(capturedConfig.env.WITH_QUOTES).toBe('value with "quotes"')
			expect(capturedConfig.env.EMPTY).toBe("")
			expect(capturedConfig.env.NUMBER).toBe("12345")
			
			// Cleanup
			jest.dontMock("@modelcontextprotocol/sdk/server/stdio.js")
		})
	})

	describe("Edge Cases", () => {
		it("should handle rapid start/close cycles", async () => {
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
			expect(consoleWarnSpy).toHaveBeenCalledTimes(1)
		})

		it("should maintain state consistency after errors", async () => {
			const MockTransportClass = class {
				stderr = undefined
				async start(): Promise<void> {
					throw new Error("Start failed")
				}
				async close(): Promise<void> {}
			}

			jest.doMock("@modelcontextprotocol/sdk/server/stdio.js", () => ({
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
			jest.dontMock("@modelcontextprotocol/sdk/server/stdio.js")
		})

		it("should only initialize transport once", async () => {
			let constructorCallCount = 0
			
			const MockTransportClass = class {
				constructor() {
					constructorCallCount++
				}
				stderr = undefined
				async start(): Promise<void> {}
				async close(): Promise<void> {}
			}

			jest.doMock("@modelcontextprotocol/sdk/server/stdio.js", () => ({
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
			
			expect(constructorCallCount).toBe(1)
			
			// Cleanup
			jest.dontMock("@modelcontextprotocol/sdk/server/stdio.js")
		})
	})
})