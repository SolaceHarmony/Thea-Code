import { McpToolRegistry } from "../core/McpToolRegistry"
import { ToolDefinition } from "../types/McpProviderTypes"
import * as assert from 'assert'
import * as sinon from 'sinon'

suite("McpToolRegistry", () => {
	let registry: McpToolRegistry

	setup(() => {
		// Get a fresh instance of the registry for each test
		registry = McpToolRegistry.getInstance()

		// Clear any existing tools
		const tools = registry.getAllTools()
		for (const name of tools.keys()) {
			registry.unregisterTool(name)
		}
	})

	test("should register a tool", () => {
		// Create a test tool definition
				const toolDefinition: ToolDefinition = {
			name: "test_tool",
			description: "A test tool",
			paramSchema: {
				param: { type: "string", description: "A test parameter" },
			},
					handler: (args: Record<string, unknown>) =>
						Promise.resolve({
							content: [{ type: "text", text: `Executed test_tool with param: ${String(args.param)}` }],
						}),
		}

		// Register the tool
		registry.registerTool(toolDefinition)

		// Verify that the tool was registered
		expect(registry.hasTool("test_tool")).toBe(true)
		expect(registry.getTool("test_tool")).toEqual(toolDefinition)
	})

	test("should unregister a tool", () => {
		// Create and register a test tool
				const toolDefinition: ToolDefinition = {
			name: "test_tool",
			description: "A test tool",
					handler: () => Promise.resolve({ content: [] }),
		}

		registry.registerTool(toolDefinition)
		expect(registry.hasTool("test_tool")).toBe(true)

		// Unregister the tool
		const result = registry.unregisterTool("test_tool")

		// Verify that the tool was unregistered
		assert.strictEqual(result, true)
		expect(registry.hasTool("test_tool")).toBe(false)
		expect(registry.getTool("test_tool")).toBeUndefined()
	})

	test("should return false when unregistering a non-existent tool", () => {
		// Attempt to unregister a tool that doesn't exist
		const result = registry.unregisterTool("non_existent_tool")

		// Verify that the result is false
		assert.strictEqual(result, false)
	})

	test("should get all registered tools", () => {
		// Create and register multiple tools
				const tool1: ToolDefinition = {
			name: "tool1",
			description: "Tool 1",
					handler: () => Promise.resolve({ content: [] }),
		}

				const tool2: ToolDefinition = {
			name: "tool2",
			description: "Tool 2",
					handler: () => Promise.resolve({ content: [] }),
		}

		registry.registerTool(tool1)
		registry.registerTool(tool2)

		// Get all tools
		const tools = registry.getAllTools()

		// Verify that all tools are returned
		assert.strictEqual(tools.size, 2)
		expect(tools.get("tool1")).toEqual(tool1)
		expect(tools.get("tool2")).toEqual(tool2)
	})

	test("should execute a tool", async () => {
		// Create and register a test tool
				const toolDefinition: ToolDefinition = {
			name: "test_tool",
			description: "A test tool",
					handler: (args: Record<string, unknown>) =>
						Promise.resolve({
							content: [{ type: "text", text: `Executed with param: ${String(args.param)}` }],
						}),
		}

		registry.registerTool(toolDefinition)

		// Execute the tool
		const result = await registry.executeTool("test_tool", { param: "test value" })

		// Verify the result
		assert.strictEqual(result.content.length, 1)
		assert.strictEqual(result.content[0].type, "text")
		assert.strictEqual(result.content[0].text, "Executed with param: test value")
	})

	test("should throw an error when executing a non-existent tool", async () => {
		// Attempt to execute a tool that doesn't exist
		await expect(registry.executeTool("non_existent_tool")).rejects.toThrow("Tool 'non_existent_tool' not found")
	})

	test("should handle errors in tool execution", async () => {
		// Create and register a tool that throws an error
	const toolDefinition: ToolDefinition = {
			name: "error_tool",
			description: "A tool that throws an error",
			handler: () => {
				throw new Error("Test error")
			},
		}

		registry.registerTool(toolDefinition)

		// Attempt to execute the tool
		await expect(registry.executeTool("error_tool")).rejects.toThrow(
			"Error executing tool 'error_tool': Test error",
		)
	})

	test("should emit events when registering and unregistering tools", () => {
		// Create event listeners
		const registerListener = sinon.stub()
		const unregisterListener = sinon.stub()

		registry.on("tool-registered", registerListener)
		registry.on("tool-unregistered", unregisterListener)

		// Create a test tool
				const toolDefinition: ToolDefinition = {
			name: "test_tool",
			description: "A test tool",
					handler: () => Promise.resolve({ content: [] }),
		}

		// Register the tool
		registry.registerTool(toolDefinition)

		// Verify that the register event was emitted
		assert.ok(registerListener.calledWith("test_tool"))

		// Unregister the tool
		registry.unregisterTool("test_tool")

		// Verify that the unregister event was emitted
		assert.ok(unregisterListener.calledWith("test_tool"))

		// Clean up event listeners
		registry.removeAllListeners()
	})
})
