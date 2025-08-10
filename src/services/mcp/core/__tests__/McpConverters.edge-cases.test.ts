/**
 * Edge case tests for McpConverters as recommended by architect
 * Tests XML/JSON/OpenAI → neutral conversions with edge cases
 */

import { McpConverters } from "../McpConverters"
import { NeutralToolUseRequest, NeutralToolResult } from "../../types/McpToolTypes"

describe("McpConverters Edge Cases", () => {
	describe("xmlToMcp", () => {
		test("should handle valid tool use XML", () => {
			const xml = `<test_tool>
<arg1>value1</arg1>
<arg2>123</arg2>
</test_tool>`
			
			const result = McpConverters.xmlToMcp(xml)
			
			expect(result.type).toBe("tool_use")
			expect(result.name).toBe("test_tool")
			expect(result.input).toEqual({ arg1: "value1", arg2: 123 }) // Numbers are parsed correctly
			expect(result.id).toBeDefined()
		})

		test("should handle missing tool name", () => {
			const xml = `Plain text without tool tags`
			
			expect(() => McpConverters.xmlToMcp(xml)).toThrow()
		})

		test("should handle tool with no parameters", () => {
			const xml = `<test_tool>
</test_tool>`
			
			const result = McpConverters.xmlToMcp(xml)
			
			expect(result.type).toBe("tool_use")
			expect(result.name).toBe("test_tool")
			expect(result.input).toEqual({}) // Empty parameters object
			expect(result.id).toBeDefined()
		})

		test("should handle nested parameter values as JSON", () => {
			const xml = `<nested_tool>
<nested>{"deep": {"value": "test"}}</nested>
</nested_tool>`
			
			const result = McpConverters.xmlToMcp(xml)
			
			expect(result.input).toEqual({
				nested: { deep: { value: "test" } }
			})
		})

		test("should handle invalid JSON in parameter values", () => {
			const xml = `<test_tool>
<param>{invalid json}</param>
</test_tool>`
			
			const result = McpConverters.xmlToMcp(xml)
			// Invalid JSON is kept as a string
			expect(result.input).toEqual({ param: "{invalid json}" })
		})

		test("should handle tool names with underscores", () => {
			const xml = `<test_tool_v2>
<param>value</param>
</test_tool_v2>`
			
			const result = McpConverters.xmlToMcp(xml)
			expect(result.name).toBe("test_tool_v2")
			expect(result.input).toEqual({ param: "value" })
		})
	})

	describe("jsonToMcp", () => {
		test("should handle string input", () => {
			const json = JSON.stringify({
				type: "tool_use",
				id: "test-123",
				name: "test_tool",
				input: { arg1: "value1" }
			})
			
			const result = McpConverters.jsonToMcp(json)
			
			expect(result.type).toBe("tool_use")
			expect(result.id).toBe("test-123")
			expect(result.name).toBe("test_tool")
			expect(result.input).toEqual({ arg1: "value1" })
		})

		test("should handle object input", () => {
			const obj = {
				type: "tool_use",
				id: "test-456",
				name: "test_tool",
				input: { arg1: "value1" }
			}
			
			const result = McpConverters.jsonToMcp(obj)
			
			expect(result.id).toBe("test-456")
			expect(result.name).toBe("test_tool")
		})

		test("should enforce required properties", () => {
			const missingType = { id: "123", name: "test", input: {} }
			const missingName = { type: "tool_use", id: "123", input: {} }
			const missingId = { type: "tool_use", name: "test", input: {} }
			const missingInput = { type: "tool_use", id: "123", name: "test" }
			
			expect(() => McpConverters.jsonToMcp(missingType)).toThrow(/Invalid tool use request format/)
			expect(() => McpConverters.jsonToMcp(missingName)).toThrow(/Invalid tool use request format/)
			expect(() => McpConverters.jsonToMcp(missingId)).toThrow(/Invalid tool use request format/)
			expect(() => McpConverters.jsonToMcp(missingInput)).toThrow(/Invalid tool use request format/)
		})

		test("should handle wrong type value", () => {
			const wrongType = {
				type: "not_tool_use",
				id: "123",
				name: "test",
				input: {}
			}
			
			expect(() => McpConverters.jsonToMcp(wrongType)).toThrow(/Invalid tool use request format/)
		})

		test("should handle deeply nested input", () => {
			const nested = {
				type: "tool_use",
				id: "nested-123",
				name: "nested_tool",
				input: {
					level1: {
						level2: {
							level3: {
								value: "deep"
							}
						}
					}
				}
			}
			
			const result = McpConverters.jsonToMcp(nested)
			expect(result.input).toEqual(nested.input)
		})
	})

	describe("openAiToMcp", () => {
		test("should handle function_call format", () => {
			const functionCall = {
				id: "call_123",
				function_call: {
					name: "test_function",
					arguments: '{"arg1": "value1"}'
				}
			}
			
			const result = McpConverters.openAiToMcp(functionCall)
			
			expect(result.type).toBe("tool_use")
			expect(result.id).toBeDefined() // ID may be generated
			expect(result.name).toBe("test_function")
			expect(result.input).toEqual({ arg1: "value1" })
		})

		test("should handle tool_calls format", () => {
			const toolCall = {
				tool_calls: [{
					id: "tool_123",
					type: "function",
					function: {
						name: "test_tool",
						arguments: '{"param": "value"}'
					}
				}]
			}
			
			const result = McpConverters.openAiToMcp(toolCall)
			
			expect(result.id).toBe("tool_123")
			expect(result.name).toBe("test_tool")
			expect(result.input).toEqual({ param: "value" })
		})

		test("should handle invalid function call", () => {
			const invalid = { id: "123" } // Missing function property
			
			expect(() => McpConverters.openAiToMcp(invalid)).toThrow(/Invalid function call format/)
		})

		test("should handle malformed arguments JSON", () => {
			const malformed = {
				id: "call_123",
				function: {
					name: "test",
					arguments: "{broken json"
				}
			}
			
			expect(() => McpConverters.openAiToMcp(malformed)).toThrow()
		})
	})

	describe("mcpToXml", () => {
		test("should escape XML special characters", () => {
			const result: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "test<>&\"'123",
				content: [{
					type: "text",
					text: "Text with <special> & \"characters\" 'here'"
				}],
				status: "success"
			}
			
			const xml = McpConverters.mcpToXml(result)
			
			expect(xml).toContain("test&lt;&gt;&amp;&quot;&apos;123")
			expect(xml).toContain("Text with &lt;special&gt; &amp; &quot;characters&quot; &apos;here&apos;")
		})

		test("should handle image with base64 data", () => {
			const result: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "img-123",
				content: [{
					type: "image",
					source: {
						type: "base64",
						media_type: "image/png",
						data: "base64encodeddata=="
					}
				}],
				status: "success"
			}
			
			const xml = McpConverters.mcpToXml(result)
			
			expect(xml).toContain('<image type="image/png" data="base64encodeddata==" />')
		})

		test("should handle image with URL", () => {
			const result: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "img-url-123",
				content: [{
					type: "image_url",
					source: {
						type: "image_url",
						url: "https://example.com/image.png?param=value&other=test"
					}
				}],
				status: "success"
			}
			
			const xml = McpConverters.mcpToXml(result)
			
			expect(xml).toContain('<image url="https://example.com/image.png?param=value&amp;other=test" />')
		})

		test("should handle tool_use content", () => {
			const result: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "nested-123",
				content: [{
					type: "tool_use",
					id: "inner-tool",
					name: "nested_tool",
					input: { key: "value with \"quotes\"" }
				} as any],
				status: "success"
			}
			
			const xml = McpConverters.mcpToXml(result)
			
			expect(xml).toContain('<tool_use name="nested_tool"')
			// Check that quotes are properly escaped in the JSON
			expect(xml).toContain('\\&quot;quotes\\&quot;')
		})

		test("should handle nested tool_result", () => {
			const result: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "outer-123",
				content: [{
					type: "tool_result",
					tool_use_id: "inner-123",
					content: [{ type: "text", text: "Nested result" }]
				} as any],
				status: "success"
			}
			
			const xml = McpConverters.mcpToXml(result)
			
			expect(xml).toContain('<nested_tool_result tool_use_id="inner-123">')
			expect(xml).toContain("Nested result")
			expect(xml).toContain("</nested_tool_result>")
		})

		test("should handle error with details", () => {
			const result: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "error-123",
				content: [{ type: "text", text: "Error occurred" }],
				status: "error",
				error: {
					message: "Something went wrong & failed",
					details: { code: 500, reason: "Internal <error>" }
				}
			}
			
			const xml = McpConverters.mcpToXml(result)
			
			expect(xml).toContain('status="error"')
			expect(xml).toContain('message="Something went wrong &amp; failed"')
			expect(xml).toContain('&quot;Internal &lt;error&gt;&quot;')
		})

		test("should handle unknown content types", () => {
			const result: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "unknown-123",
				content: [{
					type: "unknown_type",
					someProperty: "value"
				} as any],
				status: "success"
			}
			
			// Mock console.warn
			const warnSpy = jest.spyOn(console, "warn").mockImplementation()
			
			const xml = McpConverters.mcpToXml(result)
			
			expect(xml).toContain('<unknown type="unknown_type" />')
			expect(warnSpy).toHaveBeenCalledWith("Unhandled content type in mcpToXml: unknown_type")
			
			warnSpy.mockRestore()
		})

		test("should handle mixed content types", () => {
			const result: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "mixed-123",
				content: [
					{ type: "text", text: "First text" },
					{
						type: "image",
						source: {
							type: "base64",
							media_type: "image/jpeg",
							data: "data123"
						}
					},
					{ type: "text", text: "Second text" },
					{
						type: "tool_use",
						id: "tool-1",
						name: "some_tool",
						input: {}
					} as any
				],
				status: "success"
			}
			
			const xml = McpConverters.mcpToXml(result)
			
			expect(xml).toContain("First text")
			expect(xml).toContain('<image type="image/jpeg" data="data123" />')
			expect(xml).toContain("Second text")
			expect(xml).toContain('<tool_use name="some_tool"')
		})

		test("should handle content with embedded quotes and newlines in JSON", () => {
			const result: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "quotes-123",
				content: [{
					type: "tool_use",
					id: "tool-1",
					name: "test_tool",
					input: {
						text: 'String with "quotes" and\nnewlines',
						nested: { value: "Test's value" }
					}
				} as any],
				status: "success"
			}
			
			const xml = McpConverters.mcpToXml(result)
			
			// The JSON should be properly escaped
			expect(xml).toContain('&quot;String with \\&quot;quotes\\&quot; and\\nnewlines&quot;')
		})
	})

	describe("mcpToJson", () => {
		test("should convert to JSON string", () => {
			const result: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "json-123",
				content: [{ type: "text", text: "Test result" }],
				status: "success"
			}
			
			const json = McpConverters.mcpToJson(result)
			const parsed = JSON.parse(json)
			
			expect(parsed.type).toBe("tool_result")
			expect(parsed.tool_use_id).toBe("json-123")
			expect(parsed.content[0].text).toBe("Test result")
		})
	})

	describe("mcpToOpenAi", () => {
		test("should convert to OpenAI format", () => {
			const result: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "openai-123",
				content: [
					{ type: "text", text: "First part" },
					{ type: "text", text: "Second part" }
				],
				status: "success"
			}
			
			const openAiResult = McpConverters.mcpToOpenAi(result)
			
			expect(openAiResult.role).toBe("tool")
			expect(openAiResult.tool_call_id).toBe("openai-123")
			expect(openAiResult.content).toBe("First part\nSecond part")
		})

		test("should handle non-text content", () => {
			const result: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "mixed-123",
				content: [
					{ type: "text", text: "Text" },
					{ type: "image", source: { type: "base64", data: "..." } } as any,
					{ type: "text", text: "More text" }
				],
				status: "success"
			}
			
			const openAiResult = McpConverters.mcpToOpenAi(result)
			
			// Only text content should be included
			expect(openAiResult.content).toBe("Text\n\nMore text")
		})
	})

	describe("toolDefinitionsToOpenAiFunctions", () => {
		test("should convert tool definitions to OpenAI functions", () => {
			const tools = new Map([
				["tool1", {
					name: "tool1",
					description: "First tool",
					paramSchema: {
						type: "object",
						properties: {
							param1: { type: "string" }
						},
						required: ["param1"]
					}
				}],
				["tool2", {
					name: "tool2",
					description: "Second tool",
					paramSchema: {
						type: "object",
						properties: {
							param2: { type: "number" }
						}
					}
				}]
			])
			
			const functions = McpConverters.toolDefinitionsToOpenAiFunctions(tools)
			
			expect(functions).toHaveLength(2)
			expect(functions[0].name).toBe("tool1")
			expect(functions[0].description).toBe("First tool")
			expect(functions[0].parameters).toEqual(tools.get("tool1")!.paramSchema)
			expect(functions[1].name).toBe("tool2")
		})

		test("should handle missing description and paramSchema", () => {
			const tools = new Map([
				["minimal", {
					name: "minimal"
				}]
			])
			
			const functions = McpConverters.toolDefinitionsToOpenAiFunctions(tools)
			
			expect(functions[0].description).toBe("")
			expect(functions[0].parameters).toEqual({
				type: "object",
				properties: {},
				required: []
			})
		})
	})
})