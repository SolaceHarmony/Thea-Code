import * as assert from 'assert'
import { McpIntegration, handleToolUse } from "../integration/McpIntegration"
import * as sinon from 'sinon'

// Mock the McpToolRouter
// TODO: Mock setup needs manual migration for "../core/McpToolRouter"
// 	const mockInstance = {
		on: sinon.stub(),
		initialize: sinon.stub().resolves(undefined),
		shutdown: sinon.stub().resolves(undefined),
		detectFormat: sinon.stub().returns("xml"),
		routeToolUse: sinon.stub().callsFake((request: { format: string }) => ({
			format: request.format,
			content: `Routed ${request.format} request`,
		})),
	}
// Mock return block needs context
// 
// 	return {
// 		ToolUseFormat: {
// 			XML: "xml",
// 			JSON: "json",
// 			OPENAI: "openai",
// 			NEUTRAL: "neutral",
// 		},
// 		McpToolRouter: {
// 			getInstance: sinon.stub().returns(mockInstance),
// 		},
// 	}
// Mock cleanup

// Mock the McpToolExecutor
// TODO: Mock setup needs manual migration for "../core/McpToolExecutor"
// 	const mockInstance = {
		registerTool: sinon.stub(),
		unregisterTool: sinon.stub().returns(true),
		processXmlToolUse: sinon.stub().callsFake((content) => `Processed XML: ${content}`),
		processJsonToolUse: sinon.stub
			.fn()
			.callsFake(
				(content) => `Processed JSON: ${typeof content === "string" ? content : JSON.stringify(content)}`,
			),
		processOpenAiFunctionCall: sinon.stub().callsFake((content) => ({
			role: "tool",
			content: `Processed OpenAI: ${JSON.stringify(content)}`,
		})),
	}
// Mock return block needs context
// 
// 	return {
// 		McpToolExecutor: {
// 			getInstance: sinon.stub().returns(mockInstance),
// 		},
// 	}
// Mock cleanup

suite("McpIntegration", () => {
	let mcpIntegration: McpIntegration

	setup(() => {
		// Clear all mocks
		sinon.restore()

		// Get a fresh instance for each test
		// @ts-expect-error accessing private singleton for reset

		McpIntegration["instance"] = undefined
		mcpIntegration = McpIntegration.getInstance()
	})

	suite("initialization", () => {
		test("should initialize the MCP tool router", async () => {
			await mcpIntegration.initialize()

			const mcpToolRouter = (mcpIntegration as unknown as { mcpToolRouter: { initialize: sinon.SinonStub } })
				.mcpToolRouter

			assert.ok(mcpToolRouter.initialize.called)
		})

		test("should not initialize the MCP tool router if already initialized", async () => {
			// Initialize once
			await mcpIntegration.initialize()

			// Clear the mock
			const mcpToolRouter = (mcpIntegration as unknown as { mcpToolRouter: { initialize: sinon.SinonStub } })
				.mcpToolRouter

			mcpToolRouter.initialize.resetHistory()

			// Initialize again
			await mcpIntegration.initialize()

			// Should not call initialize again
			assert.ok(!mcpToolRouter.initialize.called)
		})
	})

	suite("tool registration", () => {
		test("should register a tool with the MCP tool system", () => {
			const toolDefinition = {
				name: "test_tool",
				description: "A test tool",
				paramSchema: { type: "object" },
				handler: async () => Promise.resolve({ content: [], isError: false }),
			}

			mcpIntegration.registerTool(toolDefinition)

			const mcpToolSystem = (mcpIntegration as unknown as { mcpToolSystem: { registerTool: sinon.SinonStub } })
				.mcpToolSystem
			assert.ok(mcpToolSystem.registerTool.calledWith(toolDefinition))
		})

		test("should unregister a tool from the MCP tool system", () => {
			const result = mcpIntegration.unregisterTool("test_tool")

			const mcpToolSystem = (mcpIntegration as unknown as { mcpToolSystem: { unregisterTool: sinon.SinonStub } })
				.mcpToolSystem
			assert.ok(mcpToolSystem.unregisterTool.calledWith("test_tool"))
			assert.strictEqual(result, true)
		})
	})

	suite("tool processing", () => {
		setup(async () => {
			await mcpIntegration.initialize()
		})

		test("should process XML tool use requests", async () => {
			const xmlContent = "<test_tool>\n<param1>value1</param1>\n</test_tool>"

			const result = await mcpIntegration.processXmlToolUse(xmlContent)

			const mcpToolRouter = (mcpIntegration as unknown as { mcpToolRouter: { routeToolUse: sinon.SinonStub } })
				.mcpToolRouter
			assert.ok(mcpToolRouter.routeToolUse.calledWith({
				format: "xml",
				content: xmlContent,
			}))
			assert.strictEqual(result, "Routed xml request")
		})

		test("should process JSON tool use requests", async () => {
			const jsonContent = {
				type: "tool_use",
				id: "test-123",
				name: "test_tool",
				input: {
					param1: "value1",
				},
			}

			const result = await mcpIntegration.processJsonToolUse(jsonContent)

			const mcpToolRouter = (mcpIntegration as unknown as { mcpToolRouter: { routeToolUse: sinon.SinonStub } })
				.mcpToolRouter
			assert.ok(mcpToolRouter.routeToolUse.calledWith({
				format: "json",
				content: jsonContent,
			}))
			assert.strictEqual(result, "Routed json request")
		})

		test("should process OpenAI function call requests", async () => {
			const functionCall = {
				function_call: {
					name: "test_tool",
					arguments: '{"param1":"value1"}',
					id: "call_abc123",
				},
			}

			const result = await mcpIntegration.processOpenAiFunctionCall(functionCall)

			const mcpToolRouter = (mcpIntegration as unknown as { mcpToolRouter: { routeToolUse: sinon.SinonStub } })
				.mcpToolRouter

			assert.ok(mcpToolRouter.routeToolUse.calledWith({
				format: "openai",
				content: functionCall,
			}))
			assert.deepStrictEqual(result, "Routed openai request")
		})

		test("should route tool use requests based on format", async () => {
			const content = "<test_tool>\n<param1>value1</param1>\n</test_tool>"

			const result = await mcpIntegration.routeToolUse(content)

			const mcpToolRouter = (
				mcpIntegration as unknown as { mcpToolRouter: { detectFormat: sinon.SinonStub; routeToolUse: sinon.SinonStub } }
			).mcpToolRouter
			assert.ok(mcpToolRouter.detectFormat.calledWith(content))
			assert.ok(mcpToolRouter.routeToolUse.calledWith({
				format: "xml",
				content,
			}))
			assert.strictEqual(result, "Routed xml request")
		})
	})
// Mock cleanup

suite("handleToolUse", () => {
	setup(() => {
		// Clear all mocks
		sinon.restore()

		// Reset the singleton instance
		// @ts-expect-error accessing private singleton for reset

		McpIntegration["instance"] = undefined
	})

	test("should initialize the MCP integration if not already initialized", async () => {
		const content = "<test_tool>\n<param1>value1</param1>\n</test_tool>"

		await handleToolUse(content)

		// Get the instance that was created
		const mcpIntegration = McpIntegration.getInstance()

		// Check that initialize was called
		const mcpToolRouter = (
			mcpIntegration as unknown as { mcpToolRouter: { initialize: sinon.SinonStub; routeToolUse: sinon.SinonStub } }
		).mcpToolRouter

		assert.ok(mcpToolRouter.initialize.called)
	})

	test("should route the tool use request", async () => {
		const content = "<test_tool>\n<param1>value1</param1>\n</test_tool>"

		const result = await handleToolUse(content)

		// Get the instance that was created
		const mcpIntegration = McpIntegration.getInstance()

		// Check that routeToolUse was called
		const mcpToolRouter = (mcpIntegration as unknown as { mcpToolRouter: { routeToolUse: sinon.SinonStub } })
			.mcpToolRouter

		assert.ok(mcpToolRouter.routeToolUse.calledWith({
			format: "xml",
			content,
		}))
		assert.strictEqual(result, "Routed xml request")
	})
// Mock cleanup
