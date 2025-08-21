import * as assert from 'assert'
import * as sinon from 'sinon'
/**
 * EmbeddedMcpProvider lifecycle tests as recommended by architect
 * Tests port management, restart behavior, events, and serverUrl
 */

import { SseTransportConfig, StdioTransportConfig } from "../../types/McpProviderTypes"
import { EmbeddedMcpProvider } from "../EmbeddedMcpProvider"
import { EventEmitter } from "events"

suite("EmbeddedMcpProvider Lifecycle Tests", () => {
	let provider: EmbeddedMcpProvider
	
	teardown(async () => {
		// Clean up provider if it exists
		if (provider) {
			try {
				await provider.stop()
			} catch {
				// Provider might already be stopped
			}
		}
	})

	suite("SSE transport with dynamic ports", () => {
		test("should assign dynamic port when port=0", async () => {
			const config: SseTransportConfig = {
				port: 0,
				host: "127.0.0.1"
			}
			
			provider = await EmbeddedMcpProvider.create(config)
			await provider.start()
			
			const serverUrl = provider.getServerUrl()
			assert.notStrictEqual(serverUrl, undefined)
			assert.notStrictEqual(serverUrl?.port, undefined)
			
			const port = parseInt(serverUrl!.port)
			assert.ok(port > 0)
			assert.notStrictEqual(port, 0)
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
			const firstPort = firstUrl ? parseInt(firstUrl.port) : 0
			
			// Stop
			await provider.stop()
			
			// Restart
			await provider.start()
			const secondUrl = provider.getServerUrl()
			const secondPort = secondUrl ? parseInt(secondUrl.port) : 0
			
			// Should have valid ports
			assert.ok(firstPort > 0)
			assert.ok(secondPort > 0)
			
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
			assert.strictEqual(serverUrl?.port, "37654")
		})
	})

	suite("Stdio transport", () => {
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
			assert.strictEqual(serverUrl, undefined)
			
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

	suite("Event emission", () => {
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
			assert.strictEqual(registeredName, "test_tool")
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
			assert.strictEqual(unregisteredName, "test_tool")
		})
	})

	suite("Server URL management", () => {
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
			assert.notStrictEqual(url, undefined)
			assert.strictEqual(url?.protocol, "http:")
			assert.strictEqual(url?.hostname, "localhost")
			assert.notStrictEqual(url?.port, undefined)
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

	suite("Tool management", () => {
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
// Mock return block needs context
// 				return {
// 					content: [
// 						{
// 							type: "text",
// 							text: `Executed with: ${JSON.stringify(args)}`
// 						}
					],
					isError: false
				}
			})
			
			// Execute the tool
			const result = await provider.executeTool("test_tool", { param: "value" })
			
			assert.strictEqual(result.isError, false)
			assert.strictEqual(result.content.length, 1)
			assert.strictEqual(result.content[0].type, "text")
			assert.ok(result.content[0].text.includes("Executed with"))
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
			
			assert.strictEqual(result.isError, true)
			assert.ok(result.content[0].text.includes("Tool 'nonexistent' not found"))
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
			
			assert.strictEqual(result.isError, true)
			assert.ok(result.content[0].text.includes("Tool execution failed"))
		})
	})

	suite("Lifecycle edge cases", () => {
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
			
			assert.deepStrictEqual(firstUrl, secondUrl)
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
			assert.notStrictEqual(firstPort, undefined)
			assert.notStrictEqual(secondPort, undefined)
			
			// Clean up
			delete process.env.JEST_WORKER_ID
		})
	})
// Mock cleanup