import * as assert from 'assert'
import * as sinon from 'sinon'/* eslint-disable @typescript-eslint/no-unused-vars, @typescript-eslint/require-await, @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-explicit-any */
import { MockMcpProvider } from "../MockMcpProvider"
import { ToolDefinition, ToolCallResult } from "../../types/McpProviderTypes"

suite("MockMcpProvider", () => {
	let mockProvider: MockMcpProvider

	setup(() => {
		mockProvider = new MockMcpProvider()
	})

	teardown(() => {
		// Clean up event listeners
		mockProvider.removeAllListeners()
	})

	suite("lifecycle", () => {
		test("should initialize as not started", () => {
			expect(mockProvider.isRunning()).toBe(false)
			expect(mockProvider.getServerUrl()).toBeUndefined()
		})

		test("should start successfully", async () => {
			const startedSpy = sinon.stub()
			mockProvider.on("started", startedSpy)

			await mockProvider.start()

			expect(mockProvider.isRunning()).toBe(true)
			expect(mockProvider.getServerUrl()).toBeInstanceOf(URL)
			assert.ok(startedSpy.calledWith({ url: mockProvider.getServerUrl())?.toString() })
		})

		test("should not start twice", async () => {
			const startedSpy = sinon.stub()
			mockProvider.on("started", startedSpy)

			await mockProvider.start()
			await mockProvider.start()

			assert.strictEqual(startedSpy.callCount, 1)
		})

		test("should stop successfully", async () => {
			const stoppedSpy = sinon.stub()
			mockProvider.on("stopped", stoppedSpy)

			await mockProvider.start()
			await mockProvider.stop()

			expect(mockProvider.isRunning()).toBe(false)
			expect(mockProvider.getServerUrl()).toBeUndefined()
			assert.ok(stoppedSpy.called)
		})

		test("should not stop if not started", async () => {
			const stoppedSpy = sinon.stub()
			mockProvider.on("stopped", stoppedSpy)

			await mockProvider.stop()

			assert.ok(!stoppedSpy.called)
		})
	})

	suite("tool registration", () => {
		const sampleTool: ToolDefinition = {
			name: "test_tool",
			description: "A test tool",
			paramSchema: { type: "object" },
			handler: async () => ({
				content: [{ type: "text", text: "Test result" }],
				isError: false,
			}),
		}

		test("should register a tool", () => {
			const registeredSpy = sinon.stub()
			mockProvider.on("tool-registered", registeredSpy)

			// registerToolDefinition doesn't return a promise, so no await needed
			mockProvider.registerToolDefinition(sampleTool)

			assert.ok(registeredSpy.calledWith("test_tool"))
		})

		test("should unregister a tool", () => {
			const unregisteredSpy = sinon.stub()
			mockProvider.on("tool-unregistered", unregisteredSpy)

			mockProvider.registerToolDefinition(sampleTool)
			const result = mockProvider.unregisterTool("test_tool")

			assert.strictEqual(result, true)
			assert.ok(unregisteredSpy.calledWith("test_tool"))
		})

		test("should return false when unregistering non-existent tool", () => {
			const unregisteredSpy = sinon.stub()
			mockProvider.on("tool-unregistered", unregisteredSpy)

			const result = mockProvider.unregisterTool("non_existent")

			assert.strictEqual(result, false)
			assert.ok(!unregisteredSpy.called)
		})
	})

	suite("tool execution", () => {
		const successTool: ToolDefinition = {
			name: "success_tool",
			description: "A successful tool",
			paramSchema: { type: "object" },
			handler: async (args) => ({
				content: [{ type: "text", text: `Success with args: ${JSON.stringify(args)}` }],
				isError: false,
			}),
		}

		const errorTool: ToolDefinition = {
			name: "error_tool",
			description: "A tool that returns an error",
			paramSchema: { type: "object" },
			handler: async () => ({
				content: [{ type: "text", text: "Tool failed" }],
				isError: true,
			}),
		}

		const throwTool: ToolDefinition = {
			name: "throw_tool",
			description: "A tool that throws an exception",
			paramSchema: { type: "object" },
			handler: async () => {
				throw new Error("Tool execution failed")
			},
		}

		setup(() => {
			mockProvider.registerToolDefinition(successTool)
			mockProvider.registerToolDefinition(errorTool)
			mockProvider.registerToolDefinition(throwTool)
		})

		test("should execute a tool successfully", async () => {
			const args = { param1: "value1" }
			const result = await mockProvider.executeTool("success_tool", args)

			assert.deepStrictEqual(result, {
				content: [{ type: "text", text: 'Success with args: {"param1":"value1"}' }],
				isError: false,
			})
		})

		test("should handle tool returning error", async () => {
			const result = await mockProvider.executeTool("error_tool", {})

			assert.deepStrictEqual(result, {
				content: [{ type: "text", text: "Tool failed" }],
				isError: true,
			})
		})

		test("should handle tool throwing exception", async () => {
			const result = await mockProvider.executeTool("throw_tool", {})

			assert.deepStrictEqual(result, {
				content: [{ type: "text", text: "Error executing tool 'throw_tool': Tool execution failed" }],
				isError: true,
			})
		})

		test("should handle non-existent tool", async () => {
			const result = await mockProvider.executeTool("non_existent", {})

			assert.deepStrictEqual(result, {
				content: [{ type: "text", text: "Tool 'non_existent' not found" }],
				isError: true,
			})
		})

		test("should handle null/undefined args", async () => {
			const result = await mockProvider.executeTool("success_tool", null as any)

			assert.deepStrictEqual(result, {
				content: [{ type: "text", text: "Success with args: {}" }],
				isError: false,
			})
		})
	})
})
