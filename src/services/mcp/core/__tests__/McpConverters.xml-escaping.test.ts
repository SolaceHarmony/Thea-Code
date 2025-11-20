/**
 * Comprehensive XML escaping and mixed content tests for McpConverters
 * Tests proper XML entity escaping, mixed content types, embedded quotes/newlines
 */

import { McpConverters } from "../McpConverters"
import { NeutralToolResult } from "../../types/McpToolTypes"

describe("McpConverters - XML Escaping and Mixed Content", () => {
	let consoleWarnSpy: jest.SpyInstance

	beforeEach(() => {
		consoleWarnSpy = jest.spyOn(console, "warn").mockImplementation()
	})

	afterEach(() => {
		consoleWarnSpy.mockRestore()
	})

	describe("XML Special Character Escaping", () => {
		it("should properly escape all XML special characters in text content", () => {
			const result: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "test<>&\"'123",
				content: [{ 
					type: "text", 
					text: "Text with <special> & \"characters\" and 'quotes'" 
				}],
				status: "success",
			}

			const xml = McpConverters.mcpToXml(result)

			// Check tool_use_id escaping
			expect(xml).toContain('tool_use_id="test&lt;&gt;&amp;&quot;&apos;123"')
			// Check text content escaping
			expect(xml).toContain("Text with &lt;special&gt; &amp; &quot;characters&quot; and &apos;quotes&apos;")
		})

		it("should escape XML characters in error messages and details", () => {
			const result: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "error-test",
				content: [],
				status: "error",
				error: {
					message: "Error with <tag> & \"quotes\"",
					details: { 
						code: 500, 
						reason: "Server <error> & \"failure\"" 
					}
				}
			}

			const xml = McpConverters.mcpToXml(result)

			expect(xml).toContain('message="Error with &lt;tag&gt; &amp; &quot;quotes&quot;"')
			// JSON stringified details should have nested escaping
			expect(xml).toContain('details="{&quot;code&quot;:500,&quot;reason&quot;:&quot;Server &lt;error&gt; &amp; \\&quot;failure\\&quot;&quot;}"')
		})

		it("should handle newlines and tabs in text content", () => {
			const result: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "newline-test",
				content: [{ 
					type: "text", 
					text: "Line 1\nLine 2\tTabbed\r\nWindows line" 
				}],
				status: "success",
			}

			const xml = McpConverters.mcpToXml(result)

			// Newlines and tabs should be preserved in the output
			expect(xml).toContain("Line 1\nLine 2\tTabbed\r\nWindows line")
		})

		it("should handle empty strings and null/undefined values safely", () => {
			const result: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "",
				content: [
					{ type: "text", text: "" },
					{ type: "text", text: undefined as any },
					{ type: "text", text: null as any }
				],
				status: "success",
			}

			const xml = McpConverters.mcpToXml(result)

			expect(xml).toContain('tool_use_id=""')
			expect(xml).not.toContain("undefined")
			expect(xml).not.toContain("null")
		})
	})

	describe("Mixed Content Types", () => {
		it("should handle mixed text and image content", () => {
			const result: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "mixed-content",
				content: [
					{ type: "text", text: "Here is the screenshot:" },
					{ 
						type: "image",
						source: {
							type: "base64",
							media_type: "image/png",
							data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
						}
					},
					{ type: "text", text: "Screenshot captured successfully" }
				],
				status: "success",
			}

			const xml = McpConverters.mcpToXml(result)

			expect(xml).toContain("Here is the screenshot:")
			expect(xml).toContain('<image type="image/png" data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==" />')
			expect(xml).toContain("Screenshot captured successfully")
		})

		it("should handle image with URL source", () => {
			const result: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "image-url",
				content: [
					{ 
						type: "image_url",
						source: {
							type: "image_url",
							url: "https://example.com/image.png?param=value&other=\"quoted\""
						}
					}
				],
				status: "success",
			}

			const xml = McpConverters.mcpToXml(result)

			expect(xml).toContain('<image url="https://example.com/image.png?param=value&amp;other=&quot;quoted&quot;" />')
		})

		it("should handle embedded tool_use content", () => {
			const result: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "parent-tool",
				content: [
					{ type: "text", text: "Executing nested tool:" },
					{
						type: "tool_use",
						name: "nested_tool",
						input: {
							param1: "value with <special> & \"chars\"",
							param2: { nested: true }
						}
					} as any
				],
				status: "success",
			}

			const xml = McpConverters.mcpToXml(result)

			expect(xml).toContain("Executing nested tool:")
			expect(xml).toContain('<tool_use name="nested_tool"')
			// The input should be JSON stringified and escaped
			expect(xml).toContain('input="{&quot;param1&quot;:&quot;value with &lt;special&gt; &amp; \\&quot;chars\\&quot;&quot;,&quot;param2&quot;:{&quot;nested&quot;:true}}"')
		})

		it("should handle nested tool_result content", () => {
			const result: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "parent-result",
				content: [
					{
						type: "tool_result",
						tool_use_id: "nested-result",
						content: [
							{ type: "text", text: "Nested result text with <xml>" },
							{ type: "text", text: "Second nested line" }
						]
					} as any
				],
				status: "success",
			}

			const xml = McpConverters.mcpToXml(result)

			expect(xml).toContain('<nested_tool_result tool_use_id="nested-result">')
			expect(xml).toContain("Nested result text with &lt;xml&gt;")
			expect(xml).toContain("Second nested line")
			expect(xml).toContain("</nested_tool_result>")
		})

		it("should handle unknown content types gracefully", () => {
			const result: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "unknown-types",
				content: [
					{ type: "text", text: "Known type" },
					{ type: "video", url: "video.mp4" } as any,
					{ type: "audio", data: "audio-data" } as any,
					{ type: "custom_type", someField: "value" } as any
				],
				status: "success",
			}

			const xml = McpConverters.mcpToXml(result)

			expect(xml).toContain("Known type")
			expect(xml).toContain('<unknown type="video" />')
			expect(xml).toContain('<unknown type="audio" />')
			expect(xml).toContain('<unknown type="custom_type" />')
			
			// Check that warnings were logged
			expect(consoleWarnSpy).toHaveBeenCalledWith("Unhandled content type in mcpToXml: video")
			expect(consoleWarnSpy).toHaveBeenCalledWith("Unhandled content type in mcpToXml: audio")
			expect(consoleWarnSpy).toHaveBeenCalledWith("Unhandled content type in mcpToXml: custom_type")
		})
	})

	describe("Complex Nested Structures", () => {
		it("should handle deeply nested mixed content with special characters", () => {
			const result: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "complex<test>",
				content: [
					{ type: "text", text: "Start: <root> & \"begin\"" },
					{
						type: "tool_use",
						name: "tool_with_<special>",
						input: {
							command: "echo 'test' && ls -la",
							json: JSON.stringify({ key: "value with \"quotes\"" })
						}
					} as any,
					{
						type: "tool_result",
						tool_use_id: "nested<>&",
						content: [
							{ type: "text", text: "Nested: <tag> & </tag>" }
						]
					} as any,
					{
						type: "image",
						source: {
							type: "base64",
							media_type: "image/svg+xml",
							data: "PHN2Zz48L3N2Zz4=" // <svg></svg> in base64
						}
					},
					{ type: "text", text: "End: </root> & \"finish\"" }
				],
				status: "success",
				error: {
					message: "Warning: <partial> & \"error\"",
					details: {
						xml: "<nested/>",
						special: "& < > \" '"
					}
				}
			}

			const xml = McpConverters.mcpToXml(result)

			// Verify all special characters are properly escaped
			expect(xml).toContain('tool_use_id="complex&lt;test&gt;"')
			expect(xml).toContain("Start: &lt;root&gt; &amp; &quot;begin&quot;")
			expect(xml).toContain('name="tool_with_&lt;special&gt;"')
			expect(xml).toContain('tool_use_id="nested&lt;&gt;&amp;"')
			expect(xml).toContain("Nested: &lt;tag&gt; &amp; &lt;/tag&gt;")
			expect(xml).toContain('<image type="image/svg+xml" data="PHN2Zz48L3N2Zz4=" />')
			expect(xml).toContain("End: &lt;/root&gt; &amp; &quot;finish&quot;")
			expect(xml).toContain('message="Warning: &lt;partial&gt; &amp; &quot;error&quot;"')
		})

		it("should handle content with mixed ordering", () => {
			const result: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "mixed-order",
				content: [
					{
						type: "image",
						source: { type: "base64", media_type: "image/jpeg", data: "data1" }
					},
					{ type: "text", text: "Text 1" },
					{
						type: "tool_result",
						tool_use_id: "nested1",
						content: [{ type: "text", text: "Nested 1" }]
					} as any,
					{
						type: "image_url",
						source: { type: "image_url", url: "http://example.com/img.png" }
					},
					{ type: "unknown_type", data: "unknown" } as any,
					{ type: "text", text: "Text 2" },
					{
						type: "tool_use",
						name: "tool1",
						input: { key: "value" }
					} as any,
					{ type: "text", text: "Text 3" }
				],
				status: "success",
			}

			const xml = McpConverters.mcpToXml(result)

			// Verify content is present (order preserved but might have different line breaks)
			expect(xml).toContain('<image type="image/jpeg" data="data1"')
			expect(xml).toContain("Text 1")
			expect(xml).toContain('<nested_tool_result tool_use_id="nested1">')
			expect(xml).toContain("Nested 1")
			expect(xml).toContain("</nested_tool_result>")
			expect(xml).toContain('<image url="http://example.com/img.png"')
			expect(xml).toContain('<unknown type="unknown_type"')
			expect(xml).toContain("Text 2")
			expect(xml).toContain('<tool_use name="tool1"')
			expect(xml).toContain("Text 3")
			
			// Verify ordering by checking the index positions
			const text1Index = xml.indexOf("Text 1")
			const text2Index = xml.indexOf("Text 2")
			const text3Index = xml.indexOf("Text 3")
			const nestedIndex = xml.indexOf("Nested 1")
			
			expect(text1Index).toBeLessThan(nestedIndex)
			expect(nestedIndex).toBeLessThan(text2Index)
			expect(text2Index).toBeLessThan(text3Index)
		})
	})

	describe("Edge Cases", () => {
		it("should handle empty content array", () => {
			const result: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "empty",
				content: [],
				status: "success",
			}

			const xml = McpConverters.mcpToXml(result)

			// The implementation includes formatting/whitespace
			expect(xml).toContain('<tool_result tool_use_id="empty" status="success">')
			expect(xml).toContain('</tool_result>')
			// Check that there's no content between the tags (just whitespace)
			const contentMatch = xml.match(/>([^<]*)</s)
			if (contentMatch) {
				expect(contentMatch[1].trim()).toBe('')
			}
		})

		it("should handle malformed image source objects", () => {
			const result: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "malformed",
				content: [
					{ type: "image", source: {} } as any,
					{ type: "image", source: { type: "unknown" } } as any,
					{ type: "image_url", source: { no_url: true } } as any,
					{ type: "image_base64", source: null } as any
				],
				status: "success",
			}

			const xml = McpConverters.mcpToXml(result)

			// Should log warnings and output unknown tags
			expect(consoleWarnSpy).toHaveBeenCalledTimes(4)
			expect(xml.match(/<unknown type="image"/g)).toHaveLength(2)
			expect(xml.match(/<unknown type="image_url"/g)).toHaveLength(1)
			expect(xml.match(/<unknown type="image_base64"/g)).toHaveLength(1)
		})

		it("should handle non-string text content safely", () => {
			const result: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "non-string",
				content: [
					{ type: "text", text: 123 } as any,
					{ type: "text", text: true } as any,
					{ type: "text", text: { object: true } } as any,
					{ type: "text", text: ["array"] } as any
				],
				status: "success",
			}

			const xml = McpConverters.mcpToXml(result)

			// Non-string text should be converted to empty strings
			const lines = xml.split('\n').filter(line => line.trim() && !line.includes('tool_result'))
			expect(lines.every(line => line === "")).toBe(true)
		})

		it("should handle tool_use with non-string name", () => {
			const result: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "tool-test",
				content: [
					{
						type: "tool_use",
						name: 123,
						input: { key: "value" }
					} as any
				],
				status: "success",
			}

			const xml = McpConverters.mcpToXml(result)

			// The escapeXml function returns empty string for non-strings
			// So the name will be empty
			expect(xml).toContain('<tool_use name=""')
			expect(xml).toContain('input="{&quot;key&quot;:&quot;value&quot;}"')
		})

		it("should handle very long content without truncation", () => {
			const longText = "x".repeat(100000)
			const result: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "long-content",
				content: [{ type: "text", text: longText }],
				status: "success",
			}

			const xml = McpConverters.mcpToXml(result)

			expect(xml).toContain(longText)
			expect(xml.length).toBeGreaterThan(100000)
		})
	})

	describe("JSON Input Escaping in tool_use", () => {
		it("should properly escape JSON stringified input with embedded quotes and newlines", () => {
			const result: NeutralToolResult = {
				type: "tool_result",
				tool_use_id: "json-escape",
				content: [
					{
						type: "tool_use",
						name: "complex_tool",
						input: {
							query: 'SELECT * FROM users WHERE name = "John\'s"',
							script: 'echo "Hello\nWorld" && cat file.txt',
							json: { 
								nested: { 
									value: "Has \"quotes\" and 'apostrophes'" 
								} 
							},
							xml: "<tag>content</tag>"
						}
					} as any
				],
				status: "success",
			}

			const xml = McpConverters.mcpToXml(result)

			// The JSON stringification and XML escaping should handle all special chars
			expect(xml).toContain('input=')
			expect(xml).toContain('&quot;query&quot;')
			expect(xml).toContain('&quot;script&quot;')
			expect(xml).toContain('&lt;tag&gt;content&lt;/tag&gt;')
			
			// Verify it's valid XML by checking structure
			expect(xml).toContain('<tool_use')
			expect(xml).toContain('name="complex_tool"')
		})
	})
})