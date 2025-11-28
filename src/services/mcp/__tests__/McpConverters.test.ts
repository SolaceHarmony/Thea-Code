import { McpConverters } from "../core/McpConverters"
import { ToolDefinition } from "../types/McpProviderTypes"
import { NeutralToolResult } from "../types/McpToolTypes"
import { expect } from "chai"
import * as sinon from "sinon"
import proxyquire from "proxyquire"

// Mock the json-xml-bridge utilities
const jsonXmlBridgeMock = {
	jsonToolUseToXml: sinon.stub().callsFake((json) => `<mock_xml>${json}</mock_xml>`),
	xmlToolUseToJson: sinon.stub().returns('{"type":"tool_use","id":"test","name":"test_tool","input":{"param":"test"}}'),
	openAiFunctionCallToNeutralToolUse: sinon.stub().returns({
		type: "tool_use",
		id: "test",
		name: "test_tool",
		input: { param: "test" },
	}),
	neutralToolUseToOpenAiFunctionCall: sinon.stub(),
}

// Use proxyquire to inject mocks if needed, but McpConverters seems to import directly.
// If McpConverters imports json-xml-bridge, we need proxyquire.
// Assuming McpConverters imports from "../../../utils/json-xml-bridge" based on the original jest.mock

describe("McpConverters", () => {
	// Since we can't easily proxyquire a static class if it's already imported, 
	// we might need to rely on the fact that we are in a test environment where we can maybe overwrite properties if they were exported functions.
	// However, the original code used jest.mock which works at module level.
	// Let's try to use proxyquire to load McpConverters with the mocked dependency.
	
	let McpConverters: any

	before(() => {
		const module = proxyquire("../core/McpConverters", {
			"../../../utils/json-xml-bridge": jsonXmlBridgeMock
		})
		McpConverters = module.McpConverters
	})

	describe("toolDefinitionsToOpenAiFunctions", () => {
		it("should convert tool definitions to OpenAI function definitions", () => {
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
			expect(functions).to.have.lengthOf(2)

			// Check the first function
			expect(functions[0]).to.deep.equal({
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
			expect(functions[1]).to.deep.equal({
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

		it("should handle tool definitions without schemas", () => {
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
			expect(functions).to.have.lengthOf(1)
			expect(functions[0]).to.deep.equal({
				name: "simple_tool",
				description: "A simple tool without schema",
				parameters: {
					type: "object",
					properties: {},
					required: [],
				},
			})
		})

		it("should handle tool definitions without descriptions", () => {
			// Create a map of tool definitions without descriptions
			const tools = new Map<string, ToolDefinition>()

			tools.set("no_description", {
				name: "no_description",
				handler: async () => Promise.resolve({ content: [] }),
			})

			// Convert to OpenAI functions
			const functions = McpConverters.toolDefinitionsToOpenAiFunctions(tools)

			// Verify the conversion
			expect(functions).to.have.lengthOf(1)
			expect(functions[0]).to.deep.equal({
				name: "no_description",
				description: "",
				parameters: {
					type: "object",
					properties: {},
					required: [],
				},
			})
		})

		it("should handle empty tool map", () => {
			// Create an empty map of tool definitions
			const tools = new Map<string, ToolDefinition>()

			// Convert to OpenAI functions
			const functions = McpConverters.toolDefinitionsToOpenAiFunctions(tools)

			// Verify the conversion
			expect(functions).to.have.lengthOf(0)
			expect(functions).to.deep.equal([])
		})
	})

	describe("format conversion", () => {
		it("should convert XML to MCP format", () => {
			const xmlContent = '<tool_use id="test" name="test_tool"><param>test</param></tool_use>'
			const result = McpConverters.xmlToMcp(xmlContent)

			expect(result).to.deep.equal({
				type: "tool_use",
				id: "test",
				name: "test_tool",
				input: { param: "test" },
			})
		})

		it("should convert JSON to MCP format", () => {
			const jsonContent = {
				type: "tool_use",
				id: "test",
				name: "test_tool",
				input: { param: "test" },
			}

			const result = McpConverters.jsonToMcp(jsonContent)

			expect(result).to.deep.equal({
				type: "tool_use",
				id: "test",
				name: "test_tool",
				input: { param: "test" },
			})
		})

		it("should convert OpenAI function call to MCP format", () => {
			const functionCall = {
				function_call: {
					name: "test_tool",
					arguments: '{"param":"test"}',
				},
			}

			const result = McpConverters.openAiToMcp(functionCall)

			expect(result).to.deep.equal({
				type: "tool_use",
				id: "test",
				name: "test_tool",
				input: { param: "test" },
			})
		})

		it("should convert basic text content to XML", () => {
			const mcpResult: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "test",
				content: [{ type: "text", text: "Test result" }],
				status: "success",
			}

			const result = McpConverters.mcpToXml(mcpResult)

			expect(result).to.include('tool_use_id="test"')
			expect(result).to.include('status="success"')
			expect(result).to.include("Test result")
		})

		it("should properly escape XML special characters", () => {
			const mcpResult: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "test-123",
				content: [{ type: "text", text: "Text with <special> & \"characters\"" }],
				status: "success",
			}

			const result = McpConverters.mcpToXml(mcpResult)

			expect(result).to.include('tool_use_id="test-123"')
			expect(result).to.include("Text with &lt;special&gt; &amp; &quot;characters&quot;")
		})

		it("should handle image content with base64 data", () => {
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

			expect(result).to.include('<image type="image/png" data="base64data" />')
		})

		it("should handle image content with URL", () => {
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

			expect(result).to.include('<image url="https://example.com/image.png" />')
		})

		it("should handle mixed content types", () => {
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

			expect(result).to.include("Text result")
			expect(result).to.include('<image type="image/png" data="base64data" />')
		})

		it("should handle error details", () => {
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

			expect(result).to.include('status="error"')
			expect(result).to.include('<error message="Something went wrong"')
			expect(result).to.include('details="{&quot;code&quot;:500,&quot;reason&quot;:&quot;Internal error&quot;}"')
		})

		it("should handle tool_use content type", () => {
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

			expect(result).to.include('<tool_use name="test_tool" input="{&quot;param&quot;:&quot;value&quot;}" />')
		})

		it("should handle nested tool_result content type", () => {
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

			expect(result).to.include('<nested_tool_result tool_use_id="child-456">Nested result</nested_tool_result>')
		})

		it("should handle unrecognized content types", () => {
			const mcpResult: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "test-123",
				content: [{ type: "unknown_type", someProperty: "value" }],
				status: "success",
			}

			const result = McpConverters.mcpToXml(mcpResult)

			expect(result).to.include('<unknown type="unknown_type" />')
		})

		it("should convert MCP format to JSON", () => {
			const mcpResult: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "test",
				content: [{ type: "text", text: "Test result" }],
				status: "success",
			}

			const result = McpConverters.mcpToJson(mcpResult)
			const parsed = JSON.parse(result) as unknown as NeutralToolResult

			expect(parsed).to.deep.equal(mcpResult)
		})

		it("should convert MCP format to OpenAI", () => {
			const mcpResult: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "test",
				content: [{ type: "text", text: "Test result" }],
				status: "success",
			}

			const result = McpConverters.mcpToOpenAi(mcpResult)

			expect(result).to.deep.equal({
				role: "tool",
				tool_call_id: "test",
				content: "Test result",
			})
		})
	})
})
