import * as assert from 'assert'
import * as sinon from 'sinon'
/**
 * End-to-end tests for complete tool use flows
 * Tests the full pipeline from tool request to execution and response
 */
import { McpIntegration } from "../../integration/McpIntegration"
/* eslint-disable @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-return, @typescript-eslint/no-require-imports, @typescript-eslint/require-await, @typescript-eslint/no-explicit-any, @typescript-eslint/restrict-template-expressions */
import { McpToolRouter } from "../../core/McpToolRouter"
import { McpToolExecutor } from "../../core/McpToolExecutor"
import { ToolDefinition } from "../../types/McpProviderTypes"
import { McpConverters } from "../../core/McpConverters"
import { ToolUseFormat, NeutralToolUseRequest } from "../../types/McpToolTypes"

// Mock the EmbeddedMcpProvider for E2E tests
// TODO: Mock setup needs manual migration
// TODO: Fix mock - needs proxyquire
/*
=> {
	const { EventEmitter } = require("events")

	const MockEmbeddedMcpProvider = sinon.stub().callsFake(() => {
		const instance = new EventEmitter()
		const tools = new Map()

		instance.start = sinon.stub().callsFake(() => Promise.resolve())
		instance.stop = sinon.stub().callsFake(() => Promise.resolve())
		instance.getServerUrl = sinon.stub().returns(new URL("http://localhost:3000"))
		instance.isRunning = sinon.stub().returns(true)

		instance.registerToolDefinition = sinon.stub().callsFake((tool) => {
			tools.set(tool.name, tool)
			instance.emit("tool-registered", tool.name)

		instance.unregisterTool = sinon.stub().callsFake((name) => {
			const result = tools.delete(name)
			if (result) {
				instance.emit("tool-unregistered", name)

			return result

		instance.executeTool = sinon.stub().callsFake(async (name, args) => {
			const tool = tools.get(name)
			if (!tool) {
				return {
					content: [{ type: "text", text: `Tool '${name}' not found` }],
					isError: true,

			try {
				return await tool.handler(args || {})
			} catch (error) {
				return {
					content: [{ type: "text", text: `Error: ${error.message}` }],
					isError: true,

		return instance

	const MockedProviderClass = MockEmbeddedMcpProvider as any
	MockedProviderClass.create = sinon.stub().callsFake(async () => {
		return new MockEmbeddedMcpProvider()

	return {
		EmbeddedMcpProvider: MockEmbeddedMcpProvider,

})*/

// Mock McpToolRegistry
// TODO: Mock setup needs manual migration
// TODO: Fix mock - needs proxyquire
/*
=> {
	const mockRegistry = {
		registerTool: sinon.stub(),
		unregisterTool: sinon.stub().returns(true),
		getTool: sinon.stub(),
		getAllTools: sinon.stub(),
		hasTool: sinon.stub(),
		executeTool: sinon.stub(),

	return {
		McpToolRegistry: {
			getInstance: sinon.stub().returns(mockRegistry),
		},

})*/

suite("MCP End-to-End Tool Use Flows", () => {
	let mcpIntegration: McpIntegration
	let mcpToolExecutor: McpToolExecutor
	let mcpToolRouter: McpToolRouter

	setup(async () => {
		// Reset singletons
		;(McpIntegration as any).instance = undefined
		;(McpToolExecutor as any).instance = undefined
		;(McpToolRouter as any).instance = undefined

		mcpIntegration = McpIntegration.getInstance()
		mcpToolExecutor = McpToolExecutor.getInstance()
		mcpToolRouter = McpToolRouter.getInstance()

		await mcpIntegration.initialize()

	teardown(async () => {
		if (mcpToolExecutor) {
			await mcpToolExecutor.shutdown()

	suite("XML Tool Use Flow", () => {
		const testTool: ToolDefinition = {
			name: "read_file",
			description: "Read file contents",
			paramSchema: {
				type: "object",
				properties: {
					path: { type: "string" },
					start_line: { type: "number" },
					end_line: { type: "number" },
				},
				required: ["path"],
			},
			handler: async (args) => ({
				content: [
					{
						type: "text",
						text: `File content from ${args.path}, lines ${args.start_line}-${args.end_line}`,
					},
				],
				isError: false,
			}),

		setup(() => {
			mcpIntegration.registerTool(testTool)

		test("should complete full XML tool use flow", async () => {
			const xmlRequest = `
        <read_file>
          <path>src/main.ts</path>
          <start_line>10</start_line>
          <end_line>20</end_line>
        </read_file>
      `

			// Step 1: Detect format
			const format = mcpToolRouter.detectFormat(xmlRequest)
			assert.strictEqual(format, ToolUseFormat.XML)

			// Step 2: Convert to neutral format
			const neutralRequest = McpConverters.xmlToMcp(xmlRequest)
			assert.strictEqual(neutralRequest.type, "tool_use")
			assert.strictEqual(neutralRequest.name, "read_file")
			assert.strictEqual(neutralRequest.input.path, "src/main.ts")
			assert.strictEqual(neutralRequest.input.start_line, 10)
			assert.strictEqual(neutralRequest.input.end_line, 20)

			// Step 3: Execute tool
			const result = await mcpToolExecutor.executeToolFromNeutralFormat(neutralRequest)
			assert.strictEqual(result.type, "tool_result")
			assert.strictEqual(result.status, "success")
			assert.ok(result.content[0].text.includes("File content from src/main.ts, lines 10-20"))

			// Step 4: Convert result back to XML
			const xmlResult = McpConverters.mcpToXml(result)
			assert.ok(xmlResult.includes("<tool_result"))
			assert.ok(xmlResult.includes('status="success"'))
			assert.ok(xmlResult.includes("File content from src/main.ts, lines 10-20"))

		test("should handle XML tool use errors gracefully", async () => {
			const xmlRequest = `
        <non_existent_tool>
          <param>value</param>
        </non_existent_tool>
      `

			const neutralRequest = McpConverters.xmlToMcp(xmlRequest)
			const result = await mcpToolExecutor.executeToolFromNeutralFormat(neutralRequest)

			assert.strictEqual(result.type, "tool_result")
			assert.strictEqual(result.status, "error")
			assert.ok(result.content[0].text.includes("not found"))

	suite("JSON Tool Use Flow", () => {
		const calculatorTool: ToolDefinition = {
			name: "calculator",
			description: "Perform calculations",
			paramSchema: {
				type: "object",
				properties: {
					operation: { type: "string" },
					operands: { type: "array", items: { type: "number" } },
				},
				required: ["operation", "operands"],
			},
			handler: async (args) => {
				const { operation, operands } = args
				let result: number

				switch (operation) {
					case "add":
						result = (operands as number[]).reduce((sum: number, num: number) => sum + num, 0)
						break
					case "multiply":
						result = (operands as number[]).reduce((product: number, num: number) => product * num, 1)
						break
					default:
						throw new Error(`Unsupported operation: ${operation}`)

				return {
					content: [{ type: "text", text: `Result: ${result}` }],
					isError: false,

			},

		setup(() => {
			mcpIntegration.registerTool(calculatorTool)

		test("should complete full JSON tool use flow", async () => {
			const jsonRequest = {
				type: "tool_use",
				id: "calc-001",
				name: "calculator",
				input: {
					operation: "add",
					operands: [10, 20, 30],
				},

			// Step 1: Detect format
			const format = mcpToolRouter.detectFormat(JSON.stringify(jsonRequest))
			assert.strictEqual(format, ToolUseFormat.NEUTRAL)

			// Step 2: Convert to neutral format (already neutral)
			const neutralRequest = McpConverters.jsonToMcp(jsonRequest)
			assert.strictEqual(neutralRequest.type, "tool_use")
			assert.strictEqual(neutralRequest.name, "calculator")

			// Step 3: Execute tool
			const result = await mcpToolExecutor.executeToolFromNeutralFormat(neutralRequest)
			assert.strictEqual(result.type, "tool_result")
			assert.strictEqual(result.status, "success")
			assert.strictEqual(result.content[0].text, "Result: 60")

			// Step 4: Convert result to JSON
			const jsonResult = McpConverters.mcpToJson(result)
			const parsedResult = JSON.parse(jsonResult)
			assert.strictEqual(parsedResult.type, "tool_result")
			assert.strictEqual(parsedResult.status, "success")

	suite("OpenAI Function Call Flow", () => {
		const weatherTool: ToolDefinition = {
			name: "get_weather",
			description: "Get weather information",
			paramSchema: {
				type: "object",
				properties: {
					location: { type: "string" },
					unit: { type: "string", enum: ["celsius", "fahrenheit"] },
				},
				required: ["location"],
			},
			handler: async (args) => ({
				content: [
					{
						type: "text",
						text: `Weather in ${args.location}: 22°${args.unit === "fahrenheit" ? "F" : "C"}, sunny`,
					},
				],
				isError: false,
			}),

		setup(() => {
			mcpIntegration.registerTool(weatherTool)

		test("should complete OpenAI function call flow", async () => {
			const openAiRequest = {
				function_call: {
					name: "get_weather",
					arguments: JSON.stringify({
						location: "New York",
						unit: "celsius",
					}),
				},

			// Step 1: Detect format
			const format = mcpToolRouter.detectFormat(JSON.stringify(openAiRequest))
			assert.strictEqual(format, ToolUseFormat.OPENAI)

			// Step 2: Convert to neutral format
			const neutralRequest = McpConverters.openAiToMcp(openAiRequest)
			assert.strictEqual(neutralRequest.type, "tool_use")
			assert.strictEqual(neutralRequest.name, "get_weather")

			// Step 3: Execute tool
			const result = await mcpToolExecutor.executeToolFromNeutralFormat(neutralRequest)
			assert.strictEqual(result.type, "tool_result")
			assert.strictEqual(result.status, "success")
			assert.ok(result.content[0].text.includes("Weather in New York: 22°C, sunny"))

			// Step 4: Convert result to OpenAI format
			const openAiResult = McpConverters.mcpToOpenAi(result)
			assert.strictEqual(openAiResult.role, "tool")
			assert.ok(openAiResult.content.includes("Weather in New York"))

	suite("Multiple Tool Execution Flow", () => {
		const tools: ToolDefinition[] = [
			{
				name: "list_files",
				description: "List files in directory",
				handler: async (args) => ({
					content: [{ type: "text", text: `Files in ${args.directory}: file1.ts, file2.ts` }],
				}),
			},
			{
				name: "count_lines",
				description: "Count lines in file",
				handler: async (args) => ({
					content: [{ type: "text", text: `${args.file} has 42 lines` }],
				}),
			},

		setup(() => {
			tools.forEach((tool) => mcpIntegration.registerTool(tool))

		test("should handle multiple tool executions in sequence", async () => {
			// Execute first tool
			const listRequest: NeutralToolUseRequest = {
				type: "tool_use",
				id: "list-001",
				name: "list_files",
				input: { directory: "src" },

			const listResult = await mcpToolExecutor.executeToolFromNeutralFormat(listRequest)
			assert.strictEqual(listResult.status, "success")
			assert.ok(listResult.content[0].text.includes("Files in src"))

			// Execute second tool
			const countRequest: NeutralToolUseRequest = {
				type: "tool_use",
				id: "count-001",
				name: "count_lines",
				input: { file: "file1.ts" },

			const countResult = await mcpToolExecutor.executeToolFromNeutralFormat(countRequest)
			assert.strictEqual(countResult.status, "success")
			assert.ok(countResult.content[0].text.includes("42 lines"))

		test("should handle tool registration and unregistration during execution", async () => {
			// Verify tool is available
			const request: NeutralToolUseRequest = {
				type: "tool_use",
				id: "test-001",
				name: "list_files",
				input: { directory: "test" },

			const result1 = await mcpToolExecutor.executeToolFromNeutralFormat(request)
			assert.strictEqual(result1.status, "success")

			// Unregister tool
			mcpIntegration.unregisterTool("list_files")

			// Tool should no longer be available
			const result2 = await mcpToolExecutor.executeToolFromNeutralFormat(request)
			assert.strictEqual(result2.status, "error")
			assert.ok(result2.content[0].text.includes("not found"))

	suite("Error Handling and Edge Cases", () => {
		test("should handle malformed XML gracefully", async () => {
			const malformedXml = "not xml at all"

			// The XML converter should try to convert and then fail during validation
			expect(() => {
				McpConverters.xmlToMcp(malformedXml)
			}).toThrow("Failed to convert XML to MCP format")

		test("should handle malformed JSON gracefully", async () => {
			const malformedJson = '{"type": "tool_use", "invalid": }'

			expect(() => {
				McpConverters.jsonToMcp(malformedJson)
			}).toThrow()

		test("should handle tool execution timeouts", async () => {
			const timeoutTool: ToolDefinition = {
				name: "timeout_tool",
				description: "Tool that simulates timeout",
				handler: async () => {
					// Simulate a long-running operation
					return new Promise((resolve) => {
						setTimeout(() => {
							resolve({
								content: [{ type: "text", text: "Finally completed" }],
								isError: false,
						}, 100) // Short timeout for testing
				},

			mcpIntegration.registerTool(timeoutTool)

			const request: NeutralToolUseRequest = {
				type: "tool_use",
				id: "timeout-001",
				name: "timeout_tool",
				input: {},

			const result = await mcpToolExecutor.executeToolFromNeutralFormat(request)
			assert.strictEqual(result.content[0].text, "Finally completed")
