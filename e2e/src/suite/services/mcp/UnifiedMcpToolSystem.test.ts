import { NeutralToolResult, ToolUseFormat, NeutralToolUseRequest } from "../types/McpToolTypes"
import { McpToolExecutor } from "../core/McpToolExecutor"
import { McpToolRouter } from "../core/McpToolRouter"
import { McpConverters } from "../core/McpConverters"
import { EventEmitter } from "events"
import { IMcpProvider, ToolCallResult, ToolDefinition } from "../types/McpProviderTypes"
import * as assert from 'assert'
import * as sinon from 'sinon'

// Mock EmbeddedMcpProvider
// Mock needs manual implementation
	const mockImplementation = sinon.stub().callsFake(() => {
		const instance = new EventEmitter() as EventEmitter & Partial<IMcpProvider>
		
		// Define mock methods with proper return types
		instance.start = sinon.stub().callsFake(() => Promise.resolve())
		instance.stop = sinon.stub().callsFake(() => Promise.resolve())
		instance.registerToolDefinition = sinon.stub()
		instance.unregisterTool = sinon.stub().returns(true)
		instance.executeTool = sinon.stub().callsFake(() => 
			Promise.resolve({
				content: [{ type: "text", text: "Success" }],
				status: "success",
				tool_use_id: "mock-id",
			} as ToolCallResult)
		)
		instance.getServerUrl = sinon.stub().returns(new URL("http://localhost:3000"))
		instance.isRunning = sinon.stub().returns(true)
		
		return instance as EventEmitter & IMcpProvider
	})

	// Create the static create method
	mockImplementation.create = sinon.stub().callsFake(() => 
		Promise.resolve(mockImplementation())
	)
// Mock removed - needs manual implementation
// Mock cleanup
// Mock McpToolRegistry
// Mock needs manual implementation
		registerTool: sinon.stub(),
		unregisterTool: sinon.stub().returns(true),
		getTool: sinon.stub(),
		getAllTools: sinon.stub(),
		hasTool: sinon.stub(),
		executeTool: sinon.stub(),
	}
// Mock removed - needs manual implementation,
// 	}
// Mock cleanup
// Types for accessing private fields in tests
type McpToolExecutorInternal = {
	mcpProvider: EventEmitter & IMcpProvider
	toolRegistry: {
		registerTool: sinon.SinonStub<void, [ToolDefinition]>
		unregisterTool: sinon.SinonStub<boolean, [string]>
		getTool: sinon.SinonStub
		getAllTools: sinon.SinonStub
		hasTool: sinon.SinonStub<boolean, [string]>
		executeTool: sinon.SinonStub<Promise<ToolCallResult>, [string, Record<string, unknown>]>
	}
}

// Type for accessing private fields in McpToolRouter
type McpToolRouterInternal = {
	mcpToolSystem: {
		executeToolFromNeutralFormat: sinon.SinonStub<Promise<NeutralToolResult>, [NeutralToolUseRequest]>
	}
}

suite("McpToolExecutor", () => {
	let mcpToolSystem: McpToolExecutor

	setup(() => {
		// Clear all mocks
		sinon.restore()

		// Get a fresh instance for each test
		;(McpToolExecutor as unknown as { instance: McpToolExecutor | undefined }).instance = undefined
		mcpToolSystem = McpToolExecutor.getInstance()
	})

	suite("initialization", () => {
		test("should initialize the MCP server", async () => {
			await mcpToolSystem.initialize()

			const { mcpProvider } = mcpToolSystem as unknown as McpToolExecutorInternal
			// Use mockFn.mock.calls to avoid unbound method reference
			expect((mcpProvider.start as sinon.SinonStub).mock.calls.length).toBeGreaterThan(0)
		})

		test("should not initialize the MCP server if already initialized", async () => {
			// Initialize once
			await mcpToolSystem.initialize()

			// Clear the mock
			const { mcpProvider } = mcpToolSystem as unknown as McpToolExecutorInternal
			;(mcpProvider.start as sinon.SinonStub).resetHistory()

			// Initialize again
			await mcpToolSystem.initialize()

			// Should not call start again
			// eslint-disable-next-line @typescript-eslint/unbound-method
			assert.ok(!mcpProvider.start.called)
		})
	})

	suite("tool registration", () => {
		test("should register a tool with both the MCP server and the tool registry", async () => {
			await mcpToolSystem.initialize()

			const toolDefinition: ToolDefinition = {
				name: "test_tool",
				description: "A test tool",
				paramSchema: { type: "object" },
				handler: async () => {
					await Promise.resolve()
// Mock removed - needs manual implementation,
// 			}
// 
			mcpToolSystem.registerTool(toolDefinition)

			const { mcpProvider, toolRegistry } = mcpToolSystem as unknown as McpToolExecutorInternal

			// eslint-disable-next-line @typescript-eslint/unbound-method
			assert.ok(mcpProvider.registerToolDefinition.calledWith(toolDefinition))
			assert.ok(toolRegistry.registerTool.calledWith(toolDefinition))
		})

		test("should unregister a tool from both the MCP server and the tool registry", async () => {
			await mcpToolSystem.initialize()

			const result = mcpToolSystem.unregisterTool("test_tool")

			const { mcpProvider, toolRegistry } = mcpToolSystem as unknown as McpToolExecutorInternal

			// eslint-disable-next-line @typescript-eslint/unbound-method
			assert.ok(mcpProvider.unregisterTool.calledWith("test_tool"))
			assert.ok(toolRegistry.unregisterTool.calledWith("test_tool"))
			assert.strictEqual(result, true)
		})
	})
// Mock cleanup
suite("McpConverters", () => {
	suite("XML conversion", () => {
		test("should convert XML to MCP format", () => {
			const xmlContent = "<test_tool>\n<param1>value1</param1>\n<param2>value2</param2>\n</test_tool>"

			const result = McpConverters.xmlToMcp(xmlContent)

			assert.strictEqual(result.type, "tool_use")
			assert.strictEqual(result.name, "test_tool")
			assert.strictEqual(result.input.param1, "value1")
			assert.strictEqual(result.input.param2, "value2")
		})

		test("should convert MCP format to XML", () => {
			const mcpResult: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "test-123",
				content: [{ type: "text", text: "Test result" }],
				status: "success",
			}

			const result = McpConverters.mcpToXml(mcpResult)

			assert.ok(result.includes('tool_use_id="test-123"'))
			assert.ok(result.includes('status="success"'))
			assert.ok(result.includes("Test result"))
		})
	})

	suite("JSON conversion", () => {
		test("should convert JSON to MCP format", () => {
			const jsonContent = {
				type: "tool_use",
				id: "test-123",
				name: "test_tool",
				input: {
					param1: "value1",
					param2: "value2",
				},
			}

			const result = McpConverters.jsonToMcp(jsonContent) as {
				type: string
				id: string
				name: string
				input: { [key: string]: string }
			}

			assert.strictEqual(result.type, "tool_use")
			assert.strictEqual(result.id, "test-123")
			assert.strictEqual(result.name, "test_tool")
			assert.strictEqual(result.input.param1, "value1")
			assert.strictEqual(result.input.param2, "value2")
		})

		test("should convert MCP format to JSON", () => {
			const mcpResult: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "test-123",
				content: [{ type: "text", text: "Test result" }],
				status: "success",
			}

			const result = McpConverters.mcpToJson(mcpResult)
			const parsed: { type: string; tool_use_id: string; status: string; content: Array<{ text: string }> } =
				JSON.parse(result) as {
					type: string
					tool_use_id: string
					status: string
					content: Array<{ text: string }>
				}

			assert.strictEqual(parsed.type, "tool_result")
			assert.strictEqual(parsed.tool_use_id, "test-123")
			assert.strictEqual(parsed.status, "success")
			assert.strictEqual(parsed.content[0].text, "Test result")
		})
	})

	suite("OpenAI conversion", () => {
		test("should convert OpenAI function call to MCP format", () => {
			const functionCall = {
				function_call: {
					name: "test_tool",
					arguments: '{"param1":"value1","param2":"value2"}',
					id: "call_abc123",
				},
			}

			const result = McpConverters.openAiToMcp(functionCall) as {
				type: string
				id: string
				name: string
				input: Record<string, unknown>
			}

			assert.strictEqual(result.type, "tool_use")
			assert.strictEqual(result.id, "call_abc123")
			assert.strictEqual(result.name, "test_tool")
			assert.strictEqual(result.input.param1, "value1")
			assert.strictEqual(result.input.param2, "value2")
		})

		test("should convert MCP format to OpenAI tool result", () => {
			const mcpResult: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "call_abc123",
				content: [{ type: "text", text: "Test result" }],
				status: "success",
			}

			const result = McpConverters.mcpToOpenAi(mcpResult) as {
				role: string
				tool_call_id: string
				content: string
			}

			assert.strictEqual(result.role, "tool")
			assert.strictEqual(result.tool_call_id, "call_abc123")
			assert.strictEqual(result.content, "Test result")
		})
	})
// Mock cleanup
suite("McpToolRouter", () => {
	let mcpToolRouter: McpToolRouter

	setup(() => {
		// Clear all mocks
		sinon.restore()

		// Get a fresh instance for each test
		// @ts-expect-error - Reset the singleton instance for testing
		McpToolRouter["instance"] = undefined
		mcpToolRouter = McpToolRouter.getInstance()
	})

	suite("format detection", () => {
		test("should detect XML format", () => {
			const content = "<test_tool>\n<param1>value1</param1>\n</test_tool>"

			const format = mcpToolRouter.detectFormat(content)

			assert.strictEqual(format, ToolUseFormat.XML)
		})

		test("should detect JSON format", () => {
			const content = '{"type":"tool_use","name":"test_tool","input":{"param1":"value1"}}'

			const format = mcpToolRouter.detectFormat(content)

			assert.strictEqual(format, ToolUseFormat.NEUTRAL)
		})

		test("should detect OpenAI format", () => {
			const content = '{"function_call":{"name":"test_tool","arguments":"{\\"param1\\":\\"value1\\"}"}}'

			const format = mcpToolRouter.detectFormat(content)

			assert.strictEqual(format, ToolUseFormat.OPENAI)
		})
	})

	suite("tool routing", () => {
		setup(async () => {
			await mcpToolRouter.initialize()
		})

		test("should route XML tool use requests", async () => {
			// Mock the McpToolExecutor's executeToolFromNeutralFormat method
			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			const mockExecute = sinon.stub().resolves({
				type: "tool_result",
				tool_use_id: "test-123",
				content: [{ type: "text", text: "Test result" }],
				status: "success",
			}) as sinon.SinonStub<Promise<NeutralToolResult>, [NeutralToolUseRequest], any>

			;(
				mcpToolRouter as unknown as McpToolRouterInternal
			).mcpToolSystem.executeToolFromNeutralFormat = mockExecute

			const request = {
				format: ToolUseFormat.XML,
				content: "<test_tool>\n<param1>value1</param1>\n</test_tool>",
			}

			const result = await mcpToolRouter.routeToolUse(request)

			assert.strictEqual(result.format, ToolUseFormat.XML)
			assert.ok(result.content.includes('tool_use_id="test-123"'))
			assert.ok(result.content.includes('status="success"'))
			assert.ok(result.content.includes("Test result"))
			assert.ok(mockExecute.called)
		})

		test("should route JSON tool use requests", async () => {
			// Mock the McpToolExecutor's executeToolFromNeutralFormat method
			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			const mockExecute = sinon.stub().resolves({
				type: "tool_result",
				tool_use_id: "test-123",
				content: [{ type: "text", text: "Test result" }],
				status: "success",
			}) as sinon.SinonStub<Promise<NeutralToolResult>, [NeutralToolUseRequest], any>

			;(
				mcpToolRouter as unknown as McpToolRouterInternal
			).mcpToolSystem.executeToolFromNeutralFormat = mockExecute

			const request = {
				format: ToolUseFormat.JSON,
				content: {
					type: "tool_use",
					id: "test-123",
					name: "test_tool",
					input: { param1: "value1" },
				},
			}

			const result = await mcpToolRouter.routeToolUse(request)

			assert.strictEqual(result.format, ToolUseFormat.JSON)
			const parsed: { type: string; tool_use_id: string; status: string; content: Array<{ text: string }> } =
				JSON.parse(result.content as string) as {
					type: string
					tool_use_id: string
					status: string
					content: Array<{ text: string }>
				}
			assert.strictEqual(parsed.type, "tool_result")
			assert.strictEqual(parsed.tool_use_id, "test-123")
			assert.strictEqual(parsed.status, "success")
			assert.strictEqual(parsed.content[0].text, "Test result")
			assert.ok(mockExecute.called)
		})

		test("should handle errors in tool routing", async () => {
			// Mock the McpToolExecutor's executeToolFromNeutralFormat method to throw an error
			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			const mockExecute = sinon.stub().rejects(new Error("Test error")) as sinon.SinonStub<Promise<NeutralToolResult>, [NeutralToolUseRequest], any>

			;(
				mcpToolRouter as unknown as McpToolRouterInternal
			).mcpToolSystem.executeToolFromNeutralFormat = mockExecute

			const request = {
				format: ToolUseFormat.XML,
				content: "<test_tool>\n<param1>value1</param1>\n</test_tool>",
			}

			const result = await mcpToolRouter.routeToolUse(request)

			assert.strictEqual(result.format, ToolUseFormat.XML)
			assert.ok(result.content.includes('status="error"'))
			assert.ok(result.content.includes("Test error"))
			assert.ok(mockExecute.called)
		})
	})
// Mock cleanup
