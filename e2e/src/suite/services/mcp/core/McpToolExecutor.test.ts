import * as assert from 'assert'
import * as sinon from 'sinon'
import { McpToolRegistry } from "../McpToolRegistry"
import { McpToolExecutor } from "../McpToolExecutor"
import { ToolDefinition } from "../../types/McpProviderTypes"
import { NeutralToolUseRequest } from "../../types/McpToolTypes"

// Define interface for the mock EmbeddedMcpProvider
interface MockEmbeddedMcpProviderInstance {
	start: sinon.SinonStub
	stop: sinon.SinonStub
	registerToolDefinition: sinon.SinonStub
	unregisterTool: sinon.SinonStub
	executeTool: sinon.SinonStub
	getServerUrl: sinon.SinonStub
	on: sinon.SinonStub
	off: sinon.SinonStub
	emit: sinon.SinonStub
	removeAllListeners: sinon.SinonStub
}

// Define interface for the mock constructor
interface MockEmbeddedMcpProviderConstructor {
	new (): MockEmbeddedMcpProviderInstance
	create: sinon.SinonStub
}

// Mock the EmbeddedMcpProvider module
const mockEmbeddedMcpProviderInstance = {
	start: sinon.stub().callsFake(() => Promise.resolve()),
	stop: sinon.stub().callsFake(() => Promise.resolve()),
	registerToolDefinition: sinon.stub(),
	unregisterTool: sinon.stub().returns(true),
	executeTool: sinon.stub().callsFake((...args: unknown[]) => {
		const name = args[0] as string
		if (name === "error-tool") {
			return Promise.resolve({
				content: [{ type: "text", text: "Error executing tool" }],
				isError: true,
			})
		}

		if (name === "throw-error-tool") {
			return Promise.reject(new Error("Tool execution failed"))
		}

		return Promise.resolve({
			content: [{ type: "text", text: "Success" }],
		})
	}),
	getServerUrl: sinon.stub().returns(new URL("http://localhost:3000")),
	on: sinon.stub(),
	off: sinon.stub(),
	emit: sinon.stub(),
	removeAllListeners: sinon.stub(),
} as MockEmbeddedMcpProviderInstance

// TODO: Mock setup needs manual migration for "../../providers/EmbeddedMcpProvider"
// 	const MockEmbeddedMcpProvider = sinon.stub().callsFake(() => mockEmbeddedMcpProviderInstance)

	const mockConstructor = MockEmbeddedMcpProvider as unknown as MockEmbeddedMcpProviderConstructor
	mockConstructor.create = sinon.stub().callsFake(() => {
		return Promise.resolve(mockEmbeddedMcpProviderInstance)
	})
// Mock return block needs context
// 
// 	return {
// 		EmbeddedMcpProvider: mockConstructor,
// 	}
// Mock cleanup

// Mock the McpToolRegistry
// TODO: Mock setup needs manual migration for "../McpToolRegistry"
// 	const mockRegistry = {
		getInstance: sinon.stub(),
		registerTool: sinon.stub(),
		unregisterTool: sinon.stub().returns(true),
		getTool: sinon.stub(),
		getAllTools: sinon.stub(),
		hasTool: sinon.stub(),
		executeTool: sinon.stub(),
	}
// Mock return block needs context
// 
// 	return {
// 		McpToolRegistry: {
// 			getInstance: sinon.stub().returns(mockRegistry),
// 		},
// 	}
// Mock cleanup

suite("McpToolExecutor", () => {
	// Reset the singleton instance before each test
	setup(() => {
		// Access private static instance property using type assertion
		// eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
		;(McpToolExecutor as any).instance = undefined

		// Reset all mocks
		sinon.restore()
	})

	teardown(() => {
		// Clean up any event listeners
		const instance = McpToolExecutor.getInstance()
		instance.removeAllListeners()
	})

	suite("Singleton Pattern", () => {
		test("should return the same instance when getInstance is called multiple times", () => {
			const instance1 = McpToolExecutor.getInstance()
			const instance2 = McpToolExecutor.getInstance()

			assert.strictEqual(instance1, instance2)
		})

		test("should return the same instance when getInstance is called multiple times", () => {
			const instance1 = McpToolExecutor.getInstance()
			const instance2 = McpToolExecutor.getInstance()

			assert.strictEqual(instance1, instance2)
		})
	})

	suite("Initialization and Shutdown", () => {
		test("should initialize the MCP provider when initialize is called", async () => {
			const executor = McpToolExecutor.getInstance()

			await executor.initialize()

			// Check that the mocked create method was called
			assert.ok(mockEmbeddedMcpProviderInstance.start.called)
		})

		test("should not initialize the MCP provider if already initialized", async () => {
			const executor = McpToolExecutor.getInstance()

			await executor.initialize()
			await executor.initialize() // Call initialize again

			// start should only be called once
			assert.strictEqual(mockEmbeddedMcpProviderInstance.start.callCount, 1)
		})

		test("should shutdown the MCP provider when shutdown is called", async () => {
			const executor = McpToolExecutor.getInstance()

			await executor.initialize()
			await executor.shutdown()

			assert.ok(mockEmbeddedMcpProviderInstance.stop.called)
		})

		test("should not shutdown the MCP provider if not initialized", async () => {
			const executor = McpToolExecutor.getInstance()

			await executor.shutdown()

			// Since executor was not initialized, mcpProvider should be undefined
			// eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-member-access
			expect((executor as any).mcpProvider).toBeUndefined()
		})
	})

	suite("Tool Registration", () => {
		test("should register a tool with both the MCP provider and the tool registry", async () => {
			const executor = McpToolExecutor.getInstance()
			await executor.initialize()

			const toolDefinition: ToolDefinition = {
				name: "test-tool",
				description: "Test tool",
				handler: () => Promise.resolve({ content: [] }),
			}

			executor.registerTool(toolDefinition)

			assert.ok(mockEmbeddedMcpProviderInstance.registerToolDefinition.calledWith(toolDefinition))
			// Check the mock registry was called correctly - using the mocked getInstance
			const mockRegistry = McpToolRegistry.getInstance()
			// eslint-disable-next-line @typescript-eslint/unbound-method
			assert.ok(mockRegistry.registerTool.calledWith(toolDefinition))
		})

		test("should unregister a tool from both the MCP provider and the tool registry", async () => {
			const executor = McpToolExecutor.getInstance()
			await executor.initialize()

			const result = executor.unregisterTool("test-tool")

			assert.ok(mockEmbeddedMcpProviderInstance.unregisterTool.calledWith("test-tool"))
			const mockRegistry = McpToolRegistry.getInstance()
			// eslint-disable-next-line @typescript-eslint/unbound-method
			assert.ok(mockRegistry.unregisterTool.calledWith("test-tool"))
			assert.strictEqual(result, true)
		})
	})

	suite("Tool Execution", () => {
		test("should execute a tool with valid input in neutral format", async () => {
			const executor = McpToolExecutor.getInstance()
			await executor.initialize()

			const request: NeutralToolUseRequest = {
				type: "tool_use",
				id: "test-id",
				name: "test-tool",
				input: { param: "value" },
			}

			const result = await executor.executeToolFromNeutralFormat(request)

			assert.ok(mockEmbeddedMcpProviderInstance.executeTool.calledWith("test-tool", { param: "value" }))
			assert.deepStrictEqual(result, {
				type: "tool_result",
				tool_use_id: "test-id",
				content: [{ type: "text", text: "Success" }],
				status: "success",
				error: undefined,
			})
		})

		test("should handle errors when executing a tool", async () => {
			const executor = McpToolExecutor.getInstance()
			await executor.initialize()

			const request: NeutralToolUseRequest = {
				type: "tool_use",
				id: "error-id",
				name: "error-tool",
				input: {},
			}

			const result = await executor.executeToolFromNeutralFormat(request)

			assert.ok(mockEmbeddedMcpProviderInstance.executeTool.calledWith("error-tool", {}))
			assert.deepStrictEqual(result, {
				type: "tool_result",
				tool_use_id: "error-id",
				content: [{ type: "text", text: "Error executing tool" }],
				status: "error",
				error: {
					message: "Error executing tool",
				},
			})
		})

		test("should handle exceptions when executing a tool", async () => {
			const executor = McpToolExecutor.getInstance()
			await executor.initialize()

			const request: NeutralToolUseRequest = {
				type: "tool_use",
				id: "exception-id",
				name: "throw-error-tool",
				input: {},
			}

			const result = await executor.executeToolFromNeutralFormat(request)

			assert.ok(mockEmbeddedMcpProviderInstance.executeTool.calledWith("throw-error-tool", {}))
			assert.deepStrictEqual(result, {
				type: "tool_result",
				tool_use_id: "exception-id",
				content: [
					{
						type: "text",
						text: "Error executing tool 'throw-error-tool': Tool execution failed",
					},
				],
				status: "error",
				error: {
					message: "Tool execution failed",
				},
			})
		})
	})

	suite("Event Handling", () => {
		test("should forward tool-registered events from the MCP provider", async () => {
			const executor = McpToolExecutor.getInstance()
			await executor.initialize()

			const eventHandler = sinon.stub()
			executor.on("tool-registered", eventHandler)

			// Emit the event from the provider using our mock instance
			mockEmbeddedMcpProviderInstance.emit("tool-registered", "test-tool")

			assert.ok(eventHandler.calledWith("test-tool"))
		})

		test("should forward tool-unregistered events from the MCP provider", async () => {
			const executor = McpToolExecutor.getInstance()
			await executor.initialize()

			const eventHandler = sinon.stub()
			executor.on("tool-unregistered", eventHandler)

			// Emit the event from the provider using our mock instance
			mockEmbeddedMcpProviderInstance.emit("tool-unregistered", "test-tool")

			assert.ok(eventHandler.calledWith("test-tool"))
		})

		test("should forward started events from the MCP provider", async () => {
			const executor = McpToolExecutor.getInstance()
			await executor.initialize()

			const eventHandler = sinon.stub()
			executor.on("started", eventHandler)

			// Emit the event from the provider using our mock instance
			const info = { url: "http://localhost:3000" }
			mockEmbeddedMcpProviderInstance.emit("started", info)

			assert.ok(eventHandler.calledWith(info))
		})

		test("should forward stopped events from the MCP provider", async () => {
			const executor = McpToolExecutor.getInstance()
			await executor.initialize()

			const eventHandler = sinon.stub()
			executor.on("stopped", eventHandler)

			// Emit the event from the provider using our mock instance
			mockEmbeddedMcpProviderInstance.emit("stopped")

			assert.ok(eventHandler.called)
		})
	})

	suite("Accessors", () => {
		test("should return the tool registry", () => {
			const executor = McpToolExecutor.getInstance()
			const registry = executor.getToolRegistry()

			assert.strictEqual(registry, McpToolRegistry.getInstance())
		})

		test("should return the server URL", async () => {
			const executor = McpToolExecutor.getInstance()
			await executor.initialize()

			const url = executor.getServerUrl()

			assert.deepStrictEqual(url, mockEmbeddedMcpProviderInstance.getServerUrl())
		})
	})
// Mock cleanup
