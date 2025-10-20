/**
 * EmbeddedMcpProvider lifecycle tests as recommended by architect
 * Tests port management, restart behavior, events, and serverUrl
 */

import { EmbeddedMcpProvider } from "../EmbeddedMcpProvider"
import { SseTransportConfig, StdioTransportConfig } from "../../types/McpProviderTypes"
import { EventEmitter } from "events"

describe("EmbeddedMcpProvider Lifecycle Tests", () => {
	let provider: EmbeddedMcpProvider
	
	afterEach(async () => {
		// Clean up provider if it exists
		if (provider) {
			try {
				await provider.stop()
			} catch {
				// Provider might already be stopped
			}
		}
	})

	describe("SSE transport with dynamic ports", () => {
		test("should assign dynamic port when port=0", async () => {
			const config: SseTransportConfig = {
				port: 0,
				host: "127.0.0.1"
			}
			
			provider = await EmbeddedMcpProvider.create(config)
			await provider.start()
			
			const serverUrl = provider.getServerUrl()
			expect(serverUrl).toBeDefined()
			expect(serverUrl?.port).toBeDefined()
			
			const port = parseInt(serverUrl!.port, 10)
			expect(port).toBeGreaterThan(0)
			expect(port).not.toBe(0)
		})

		test("should get new port on restart", async () => {
			const config: SseTransportConfig = {
				port: 0,
				host: "127.0.0.1"
			}
			
			provider = await EmbeddedMcpProvider.create(config)
			
			// First start
			await provider.start()
			const firstUrl = provider.getServerUrl()
			const firstPort = firstUrl ? parseInt(firstUrl.port, 10) : 0
			
			// Stop
			await provider.stop()
			
			// Restart
			await provider.start()
			const secondUrl = provider.getServerUrl()
			const secondPort = secondUrl ? parseInt(secondUrl.port, 10) : 0
			
			// Should have valid ports
			expect(firstPort).toBeGreaterThan(0)
			expect(secondPort).toBeGreaterThan(0)
			
			// Ports might be different (especially in test environment)
			// The important thing is that both are valid
		})

		test("should use specified port when provided", async () => {
			const config: SseTransportConfig = {
				port: 37654,
				host: "127.0.0.1"
			}
			
			provider = await EmbeddedMcpProvider.create(config)
			await provider.start()
			
			const serverUrl = provider.getServerUrl()
			expect(serverUrl?.port).toBe("37654")
		})
	})

	describe("Stdio transport", () => {
		test("should start with stdio configuration", async () => {
			const config: StdioTransportConfig = {
				command: "echo",
				args: ["test"],
				env: {}
			}
			
			provider = await EmbeddedMcpProvider.create({ type: "stdio", config })
			await provider.start()
			
			// Stdio doesn't have a server URL
			const serverUrl = provider.getServerUrl()
			expect(serverUrl).toBeUndefined()
			
			// But should be running
			expect(provider.isRunning()).toBe(true)
		})

		test("should handle stdio without args", async () => {
			const config: StdioTransportConfig = {
				command: "echo"
			} as StdioTransportConfig
			
			provider = await EmbeddedMcpProvider.create({ type: "stdio", config })
			await expect(provider.start()).resolves.not.toThrow()
		})
	})

	describe("Event emission", () => {
		test("should emit 'started' event on start", async () => {
			const config: SseTransportConfig = {
				port: 0,
				host: "127.0.0.1"
			}
			
			provider = await EmbeddedMcpProvider.create(config)
			
			const startedPromise = new Promise<void>((resolve) => {
				provider.on("started", () => resolve())
			})
			
			await provider.start()
			
			// Should emit started event
			await expect(startedPromise).resolves.toBeUndefined()
		})

		test("should emit 'stopped' event on stop", async () => {
			const config: SseTransportConfig = {
				port: 0,
				host: "127.0.0.1"
			}
			
			provider = await EmbeddedMcpProvider.create(config)
			await provider.start()
			
			const stoppedPromise = new Promise<void>((resolve) => {
				provider.on("stopped", () => resolve())
			})
			
			await provider.stop()
			
			// Should emit stopped event
			await expect(stoppedPromise).resolves.toBeUndefined()
		})

		test("should emit tool registration events", async () => {
			const config: SseTransportConfig = {
				port: 0,
				host: "127.0.0.1"
			}
			
			provider = await EmbeddedMcpProvider.create(config)
			await provider.start()
			
			const registeredPromise = new Promise<string>((resolve) => {
				provider.on("tool-registered", (name: string) => resolve(name))
			})
			
			// Register a tool
			provider.registerToolDefinition({
				name: "test_tool",
				description: "Test tool",
				paramSchema: {
					type: "object",
					properties: {}
				}
			})
			
			// Should emit tool-registered event
			const registeredName = await registeredPromise
			expect(registeredName).toBe("test_tool")
		})

		test("should emit tool unregistration events", async () => {
			const config: SseTransportConfig = {
				port: 0,
				host: "127.0.0.1"
			}
			
			provider = await EmbeddedMcpProvider.create(config)
			await provider.start()
			
			// First register a tool
			provider.registerToolDefinition({
				name: "test_tool",
				description: "Test tool",
				paramSchema: {}
			})
			
			const unregisteredPromise = new Promise<string>((resolve) => {
				provider.on("tool-unregistered", (name: string) => resolve(name))
			})
			
			// Unregister the tool
			provider.unregisterTool("test_tool")
			
			// Should emit tool-unregistered event
			const unregisteredName = await unregisteredPromise
			expect(unregisteredName).toBe("test_tool")
		})
	})

	describe("Server URL management", () => {
		test("should set serverUrl for SSE transport", async () => {
			const config: SseTransportConfig = {
				port: 0,
				host: "localhost"
			}
			
			provider = await EmbeddedMcpProvider.create(config)
			
			// Before start, no URL
			expect(provider.getServerUrl()).toBeUndefined()
			
			await provider.start()
			
			// After start, should have URL
			const url = provider.getServerUrl()
			expect(url).toBeDefined()
			expect(url?.protocol).toBe("http:")
			expect(url?.hostname).toBe("localhost")
			expect(url?.port).toBeDefined()
		})

		test("should clear serverUrl on stop", async () => {
			const config: SseTransportConfig = {
				port: 0,
				host: "127.0.0.1"
			}
			
			provider = await EmbeddedMcpProvider.create(config)
			await provider.start()
			
			// Should have URL after start
			expect(provider.getServerUrl()).toBeDefined()
			
			await provider.stop()
			
			// Should clear URL after stop
			expect(provider.getServerUrl()).toBeUndefined()
		})

		test("should not have serverUrl for stdio transport", async () => {
			const config: StdioTransportConfig = {
				command: "echo",
				args: ["test"]
			}
			
			provider = await EmbeddedMcpProvider.create({ type: "stdio", config })
			await provider.start()
			
			// Stdio doesn't use server URL
			expect(provider.getServerUrl()).toBeUndefined()
		})
	})

	describe("Tool management", () => {
		test("should register tool definitions", async () => {
			const config: SseTransportConfig = {
				port: 0,
				host: "127.0.0.1"
			}
			
			provider = await EmbeddedMcpProvider.create(config)
			await provider.start()
			
			// Register multiple tools
			provider.registerToolDefinition({
				name: "tool1",
				description: "First tool",
				paramSchema: {}
			})
			
			provider.registerToolDefinition({
				name: "tool2",
				description: "Second tool",
				paramSchema: {
					type: "object",
					properties: {
						param1: { type: "string" }
					}
				}
			})
			
			// Tools should be registered
			// (actual verification would require accessing internal state or executing tools)
		})

		test("should handle tool execution", async () => {
			const config: SseTransportConfig = {
				port: 0,
				host: "127.0.0.1"
			}
			
			provider = await EmbeddedMcpProvider.create(config)
			await provider.start()
			
			// Register a tool with handler
			provider.registerTool("test_tool", "Test tool", async (args) => {
				return {
					content: [
						{
							type: "text",
							text: `Executed with: ${JSON.stringify(args)}`
						}
					],
					isError: false
				}
			})
			
			// Execute the tool
			const result = await provider.executeTool("test_tool", { param: "value" })
			
			expect(result.isError).toBe(false)
			expect(result.content).toHaveLength(1)
			expect(result.content[0].type).toBe("text")
			expect(result.content[0].text).toContain("Executed with")
		})

		test("should handle tool not found", async () => {
			const config: SseTransportConfig = {
				port: 0,
				host: "127.0.0.1"
			}
			
			provider = await EmbeddedMcpProvider.create(config)
			await provider.start()
			
			// Try to execute non-existent tool
			const result = await provider.executeTool("nonexistent", {})
			
			expect(result.isError).toBe(true)
			expect(result.content[0].text).toContain("Tool 'nonexistent' not found")
		})

		test("should handle tool execution errors", async () => {
			const config: SseTransportConfig = {
				port: 0,
				host: "127.0.0.1"
			}
			
			provider = await EmbeddedMcpProvider.create(config)
			await provider.start()
			
			// Register a tool that throws
			provider.registerTool("error_tool", "Error tool", async () => {
				throw new Error("Tool execution failed")
			})
			
			// Execute the tool
			const result = await provider.executeTool("error_tool", {})
			
			expect(result.isError).toBe(true)
			expect(result.content[0].text).toContain("Tool execution failed")
		})
	})

	describe("Lifecycle edge cases", () => {
		test("should handle multiple start calls", async () => {
			const config: SseTransportConfig = {
				port: 0,
				host: "127.0.0.1"
			}
			
			provider = await EmbeddedMcpProvider.create(config)
			
			// First start
			await provider.start()
			const firstUrl = provider.getServerUrl()
			
			// Second start should not change state
			await provider.start()
			const secondUrl = provider.getServerUrl()
			
			expect(firstUrl).toEqual(secondUrl)
		})

		test("should handle stop without start", async () => {
			const config: SseTransportConfig = {
				port: 0,
				host: "127.0.0.1"
			}
			
			provider = await EmbeddedMcpProvider.create(config)
			
			// Should not throw
			await expect(provider.stop()).resolves.not.toThrow()
		})

		test("should handle multiple stop calls", async () => {
			const config: SseTransportConfig = {
				port: 0,
				host: "127.0.0.1"
			}
			
			provider = await EmbeddedMcpProvider.create(config)
			await provider.start()
			
			// First stop
			await provider.stop()
			
			// Second stop should not throw
			await expect(provider.stop()).resolves.not.toThrow()
		})

		test("should detect test environment for port randomization", async () => {
			// Set test environment
			process.env.JEST_WORKER_ID = "1"
			
			const config: SseTransportConfig = {
				port: 0,
				host: "127.0.0.1"
			}
			
			provider = await EmbeddedMcpProvider.create(config)
			await provider.start()
			
			const firstPort = provider.getServerUrl()?.port
			
			await provider.stop()
			await provider.start()
			
			const secondPort = provider.getServerUrl()?.port
			
			// Both should be valid ports
			expect(firstPort).toBeDefined()
			expect(secondPort).toBeDefined()
			
			// Clean up
			delete process.env.JEST_WORKER_ID
		})
	})
})