import { ToolDefinition } from "../types/McpProviderTypes"
import { McpConverters } from "../core/McpConverters"
import * as assert from 'assert'
import { NeutralToolResult } from "../types/McpToolTypes"
import * as sinon from 'sinon'

// Mock the json-xml-bridge utilities
// TODO: Use proxyquire for module mocking
		// Mock for "../../../utils/json-xml-bridge" needed here
	jsonToolUseToXml: sinon.stub((json) => `<mock_xml>${json}</mock_xml>`),
	xmlToolUseToJson: sinon.stub(() => '{"type":"tool_use","id":"test","name":"test_tool","input":{"param":"test"}}'),
	openAiFunctionCallToNeutralToolUse: sinon.stub(() => ({
		type: "tool_use",
		id: "test",
		name: "test_tool",
		input: { param: "test" },
	})),
	neutralToolUseToOpenAiFunctionCall: sinon.stub(),
// Mock cleanup needed

suite("McpConverters", () => {
	suite("toolDefinitionsToOpenAiFunctions", () => {
		test("should convert tool definitions to OpenAI function definitions", () => {
			// Create a map of tool definitions
			const tools = new Map<string, ToolDefinition>()

			tools.set("test_tool", {
				name: "test_tool",
				description: "A test tool",
				paramSchema: {
					type: "object",
					properties: {
						param: {
							type: "string",
							description: "A test parameter",
						},
					},
					required: ["param"],
				},
				handler: async () => Promise.resolve({ content: [] }),
			})

			tools.set("another_tool", {
				name: "another_tool",
				description: "Another test tool",
				paramSchema: {
					type: "object",
					properties: {
						option: {
							type: "boolean",
							description: "A boolean option",
						},
						count: {
							type: "number",
							description: "A number parameter",
						},
					},
					required: ["option"],
				},
				handler: async () => Promise.resolve({ content: [] }),
			})

			// Convert to OpenAI functions
			const functions = McpConverters.toolDefinitionsToOpenAiFunctions(tools)

			// Verify the conversion
			assert.strictEqual(functions.length, 2)

			// Check the first function
			assert.deepStrictEqual(functions[0], {
				name: "test_tool",
				description: "A test tool",
				parameters: {
					type: "object",
					properties: {
						param: {
							type: "string",
							description: "A test parameter",
						},
					},
					required: ["param"],
				},
			})

			// Check the second function
			assert.deepStrictEqual(functions[1], {
				name: "another_tool",
				description: "Another test tool",
				parameters: {
					type: "object",
					properties: {
						option: {
							type: "boolean",
							description: "A boolean option",
						},
						count: {
							type: "number",
							description: "A number parameter",
						},
					},
					required: ["option"],
				},
			})
		})

		test("should handle tool definitions without schemas", () => {
			// Create a map of tool definitions without schemas
			const tools = new Map<string, ToolDefinition>()

			tools.set("simple_tool", {
				name: "simple_tool",
				description: "A simple tool without schema",
				handler: async () => Promise.resolve({ content: [] }),
			})

			// Convert to OpenAI functions
			const functions = McpConverters.toolDefinitionsToOpenAiFunctions(tools)

			// Verify the conversion
			assert.strictEqual(functions.length, 1)
			assert.deepStrictEqual(functions[0], {
				name: "simple_tool",
				description: "A simple tool without schema",
				parameters: {
					type: "object",
					properties: {},
					required: [],
				},
			})
		})

		test("should handle tool definitions without descriptions", () => {
			// Create a map of tool definitions without descriptions
			const tools = new Map<string, ToolDefinition>()

			tools.set("no_description", {
				name: "no_description",
				handler: async () => Promise.resolve({ content: [] }),
			})

			// Convert to OpenAI functions
			const functions = McpConverters.toolDefinitionsToOpenAiFunctions(tools)

			// Verify the conversion
			assert.strictEqual(functions.length, 1)
			assert.deepStrictEqual(functions[0], {
				name: "no_description",
				description: "",
				parameters: {
					type: "object",
					properties: {},
					required: [],
				},
			})
		})

		test("should handle empty tool map", () => {
			// Create an empty map of tool definitions
			const tools = new Map<string, ToolDefinition>()

			// Convert to OpenAI functions
			const functions = McpConverters.toolDefinitionsToOpenAiFunctions(tools)

			// Verify the conversion
			assert.strictEqual(functions.length, 0)
			assert.deepStrictEqual(functions, [])
		})
	})

	suite("format conversion", () => {
		test("should convert XML to MCP format", () => {
			const xmlContent = '<tool_use id="test" name="test_tool"><param>test</param></tool_use>'
			const result = McpConverters.xmlToMcp(xmlContent)

			assert.deepStrictEqual(result, {
				type: "tool_use",
				id: "test",
				name: "test_tool",
				input: { param: "test" },
			})
		})

		test("should convert JSON to MCP format", () => {
			const jsonContent = {
				type: "tool_use",
				id: "test",
				name: "test_tool",
				input: { param: "test" },
			}

			const result = McpConverters.jsonToMcp(jsonContent)

			assert.deepStrictEqual(result, {
				type: "tool_use",
				id: "test",
				name: "test_tool",
				input: { param: "test" },
			})
		})

		test("should convert OpenAI function call to MCP format", () => {
			const functionCall = {
				function_call: {
					name: "test_tool",
					arguments: '{"param":"test"}',
				},
			}

			const result = McpConverters.openAiToMcp(functionCall)

			assert.deepStrictEqual(result, {
				type: "tool_use",
				id: "test",
				name: "test_tool",
				input: { param: "test" },
			})
		})

		test("should convert basic text content to XML", () => {
			const mcpResult: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "test",
				content: [{ type: "text", text: "Test result" }],
				status: "success",
			}

			const result = McpConverters.mcpToXml(mcpResult)

			assert.ok(result.includes('tool_use_id="test"'))
			assert.ok(result.includes('status="success"'))
			assert.ok(result.includes("Test result"))
		})

		test("should properly escape XML special characters", () => {
			const mcpResult: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "test-123",
				content: [{ type: "text", text: "Text with <special> & \"characters\"" }],
				status: "success",
			}

			const result = McpConverters.mcpToXml(mcpResult)

			assert.ok(result.includes('tool_use_id="test-123"'))
			assert.ok(result.includes("Text with &lt;special&gt; &amp; &quot;characters&quot;"))
		})

		test("should handle image content with base64 data", () => {
			const mcpResult: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "test-123",
				content: [{ 
					type: "image", 
					source: {
						type: "base64",
						media_type: "image/png",
						data: "base64data"
					}
				}],
				status: "success",
			}

			const result = McpConverters.mcpToXml(mcpResult)

			assert.ok(result.includes('<image type="image/png" data="base64data" />'))
		})

		test("should handle image content with URL", () => {
			const mcpResult: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "test-123",
				content: [{ 
					type: "image_url", 
					source: {
						type: "image_url",
						url: "https://example.com/image.png"
					}
				}],
				status: "success",
			}

			const result = McpConverters.mcpToXml(mcpResult)

			assert.ok(result.includes('<image url="https://example.com/image.png" />'))
		})

		test("should handle mixed content types", () => {
			const mcpResult: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "test-123",
				content: [
					{ type: "text", text: "Text result" },
					{ 
						type: "image", 
						source: {
							type: "base64",
							media_type: "image/png",
							data: "base64data"
						}
					}
				],
				status: "success",
			}

			const result = McpConverters.mcpToXml(mcpResult)

			assert.ok(result.includes("Text result"))
			assert.ok(result.includes('<image type="image/png" data="base64data" />'))
		})

		test("should handle error details", () => {
			const mcpResult: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "test-123",
				content: [{ type: "text", text: "Error occurred" }],
				status: "error",
				error: {
					message: "Something went wrong",
					details: { code: 500, reason: "Internal error" }
				}
			}

			const result = McpConverters.mcpToXml(mcpResult)

			assert.ok(result.includes('status="error"'))
			assert.ok(result.includes('<error message="Something went wrong"'))
			assert.ok(result.includes('details="{&quot;code&quot;:500,&quot;reason&quot;:&quot;Internal error&quot;}"'))
		})

		test("should handle tool_use content type", () => {
			const mcpResult: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "test-123",
				content: [{ 
					type: "tool_use", 
					name: "test_tool",
					input: { param: "value" }
				}],
				status: "success",
			}

			const result = McpConverters.mcpToXml(mcpResult)

			assert.ok(result.includes('<tool_use name="test_tool" input="{&quot;param&quot;:&quot;value&quot;}" />'))
		})

		test("should handle nested tool_result content type", () => {
			const mcpResult: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "parent-123",
				content: [{ 
					type: "tool_result", 
					tool_use_id: "child-456",
					content: [{ type: "text", text: "Nested result" }],
					status: "success"
				}],
				status: "success",
			}

			const result = McpConverters.mcpToXml(mcpResult)

			assert.ok(result.includes('<nested_tool_result tool_use_id="child-456">Nested result</nested_tool_result>'))
		})

		test("should handle unrecognized content types", () => {
			const mcpResult: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "test-123",
				content: [{ type: "unknown_type", someProperty: "value" }],
				status: "success",
			}

			const result = McpConverters.mcpToXml(mcpResult)

			assert.ok(result.includes('<unknown type="unknown_type" />'))
		})

		test("should convert MCP format to JSON", () => {
			const mcpResult: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "test",
				content: [{ type: "text", text: "Test result" }],
				status: "success",
			}

			const result = McpConverters.mcpToJson(mcpResult)
			const parsed = JSON.parse(result) as unknown as NeutralToolResult

			assert.deepStrictEqual(parsed, mcpResult)
		})

		test("should convert MCP format to OpenAI", () => {
			const mcpResult: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "test",
				content: [{ type: "text", text: "Test result" }],
				status: "success",
			}

			const result = McpConverters.mcpToOpenAi(mcpResult)

			assert.deepStrictEqual(result, {
				role: "tool",
				tool_call_id: "test",
				content: "Test result",
			})
		})
	})
// Mock cleanup
