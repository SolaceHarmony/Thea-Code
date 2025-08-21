import * as assert from 'assert'
import * as sinon from 'sinon'
import { ToolDefinition, ToolCallResult } from "../../types/McpProviderTypes"
import { McpToolRegistry } from "../McpToolRegistry"

suite("McpToolRegistry", () => {
	// Reset the singleton instance before each test
	setup(() => {
		// Access private static instance property using a typed assertion
		;(McpToolRegistry as unknown as { instance?: McpToolRegistry }).instance = undefined
	})

	suite("Singleton Pattern", () => {
		test("should return the same instance when getInstance is called multiple times", () => {
			const instance1 = McpToolRegistry.getInstance()
			const instance2 = McpToolRegistry.getInstance()

			assert.strictEqual(instance1, instance2)
		})
	})

	suite("Tool Registration", () => {
		test("should register a tool successfully", () => {
			const registry = McpToolRegistry.getInstance()
			const mockTool = createMockTool("test-tool")

			registry.registerTool(mockTool)

			assert.strictEqual(registry.hasTool("test-tool"), true)
			assert.strictEqual(registry.getTool("test-tool"), mockTool)
		})

		test("should emit an event when a tool is registered", () => {
			const registry = McpToolRegistry.getInstance()
			const mockTool = createMockTool("event-test-tool")

			// Set up event listener
			const eventHandler = sinon.stub()
			registry.on("tool-registered", eventHandler)

			registry.registerTool(mockTool)

			assert.ok(eventHandler.calledWith("event-test-tool"))
		})

		test("should override an existing tool with the same name", () => {
			const registry = McpToolRegistry.getInstance()
			const mockTool1 = createMockTool("duplicate-tool", "Description 1")
			const mockTool2 = createMockTool("duplicate-tool", "Description 2")

			registry.registerTool(mockTool1)
			registry.registerTool(mockTool2)

			const retrievedTool = registry.getTool("duplicate-tool")
			assert.strictEqual(retrievedTool, mockTool2)
			assert.strictEqual(retrievedTool?.description, "Description 2")
		})
	})

	suite("Tool Retrieval", () => {
		test("should return undefined when getting a non-existent tool", () => {
			const registry = McpToolRegistry.getInstance()

			const tool = registry.getTool("non-existent-tool")

			assert.strictEqual(tool, undefined)
		})

		test("should retrieve all registered tools", () => {
			const registry = McpToolRegistry.getInstance()
			const mockTool1 = createMockTool("tool1")
			const mockTool2 = createMockTool("tool2")

			registry.registerTool(mockTool1)
			registry.registerTool(mockTool2)

			const allTools = registry.getAllTools()

			assert.strictEqual(allTools.size, 2)
			assert.strictEqual(allTools.get("tool1"), mockTool1)
			assert.strictEqual(allTools.get("tool2"), mockTool2)
		})

		test("should return a copy of the tools map, not the original", () => {
			const registry = McpToolRegistry.getInstance()
			const mockTool = createMockTool("tool1")

			registry.registerTool(mockTool)

			const allTools = registry.getAllTools()

			// Modify the returned map
			allTools.delete("tool1")

			// Original registry should still have the tool
			assert.strictEqual(registry.hasTool("tool1"), true)
		})
	})

	suite("Tool Unregistration", () => {
		test("should unregister a tool successfully", () => {
			const registry = McpToolRegistry.getInstance()
			const mockTool = createMockTool("tool-to-remove")

			registry.registerTool(mockTool)
			assert.strictEqual(registry.hasTool("tool-to-remove"), true)

			const result = registry.unregisterTool("tool-to-remove")

			assert.strictEqual(result, true)
			assert.strictEqual(registry.hasTool("tool-to-remove"), false)
		})

		test("should return false when trying to unregister a non-existent tool", () => {
			const registry = McpToolRegistry.getInstance()

			const result = registry.unregisterTool("non-existent-tool")

			assert.strictEqual(result, false)
		})

		test("should emit an event when a tool is unregistered", () => {
			const registry = McpToolRegistry.getInstance()
			const mockTool = createMockTool("event-unregister-tool")

			registry.registerTool(mockTool)

			// Set up event listener
			const eventHandler = sinon.stub()
			registry.on("tool-unregistered", eventHandler)

			registry.unregisterTool("event-unregister-tool")

			assert.ok(eventHandler.calledWith("event-unregister-tool"))
		})

		test("should not emit an event when trying to unregister a non-existent tool", () => {
			const registry = McpToolRegistry.getInstance()

			// Set up event listener
			const eventHandler = sinon.stub()
			registry.on("tool-unregistered", eventHandler)

			registry.unregisterTool("non-existent-tool")

			assert.ok(!eventHandler.called)
		})
	})

	suite("Tool Execution", () => {
		test("should execute a registered tool successfully", async () => {
			const registry = McpToolRegistry.getInstance()
			const mockTool = createMockTool("executable-tool")

			registry.registerTool(mockTool)

			const result = await registry.executeTool("executable-tool", { param: "value" })

			assert.deepStrictEqual(result, {
				content: [{ type: "text", text: "Success" }],
			})
			assert.ok(mockTool.handler.calledWith({ param: "value" }))
		})

		test("should throw an error when executing a non-existent tool", async () => {
			const registry = McpToolRegistry.getInstance()

			await assert.rejects(
				() => registry.executeTool("non-existent-tool"),
				/Tool 'non-existent-tool' not found/
			)
		})

		test("should propagate errors from tool handlers", async () => {
			const registry = McpToolRegistry.getInstance()
			const mockTool = createMockToolWithError("error-tool", "Test error")

			registry.registerTool(mockTool)

			await assert.rejects(
				() => registry.executeTool("error-tool"),
				/Error executing tool 'error-tool': Test error/
			)
		})
	})
// Mock cleanup

// Helper function to create a mock tool definition
function createMockTool(name: string, description: string = "Test tool"): ToolDefinition {
	const mockHandler = sinon.stub(() => {
			return Promise.resolve({
				content: [{ type: "text", text: "Success" }],
			} as ToolCallResult)
		})
// Mock return block needs context
// 
// 	return {
// 		name,
// 		description,
// 		paramSchema: {
// 			type: "object",
// 			properties: {
// 				param: { type: "string" },
// 			},
// 		},
// 		handler: mockHandler,
// 	}
}

// Helper function to create a mock tool that throws an error
function createMockToolWithError(name: string, errorMessage: string): ToolDefinition {
	const mockHandler = sinon.stub(() => {
		return Promise.reject(new Error(errorMessage))
	})
// Mock return block needs context
// 
// 	return {
// 		name,
// 		description: "Error tool",
// 		handler: mockHandler,
// 	}
}
