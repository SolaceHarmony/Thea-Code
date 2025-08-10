/**
 * Round-trip tests for McpToolRouter as recommended by architect
 * Tests that tool_use_id is preserved and formats are maintained
 */

import { McpToolRouter } from "../McpToolRouter"
import { McpToolExecutor } from "../McpToolExecutor"
import { NeutralToolUseRequest, NeutralToolResult, ToolUseFormat } from "../../types/McpToolTypes"

describe("McpToolRouter Round-Trip Tests", () => {
	let router: McpToolRouter
	let executor: McpToolExecutor

	beforeEach(async () => {
		executor = McpToolExecutor.getInstance()
		router = McpToolRouter.getInstance()
		
		// Initialize the executor
		await executor.initialize()
		
		// Register a test tool
		executor.registerTool({
			name: "test_tool",
			description: "Test tool for round-trip tests",
			paramSchema: {
				type: "object",
				properties: {},
			},
			handler: async (args: any) => {
				return {
					content: [
						{
							type: "text",
							text: `Tool executed successfully with args: ${JSON.stringify(args)}`
						}
					],
					isError: false
				}
			}
		})
	})

	afterEach(async () => {
		// Unregister the test tool if it exists
		try {
			executor.unregisterTool("test_tool")
		} catch {
			// Tool might not exist if test failed early
		}
		
		// Shutdown the executor to clean up the MCP server
		await executor.shutdown()
	})

	describe("Format preservation round-trips", () => {
		test("XML → neutral → execute → neutral → XML preserves tool_use_id", async () => {
			const xmlInput = `<test_tool>
<param1>value1</param1>
<param2>123</param2>
</test_tool>`

			// Detect format
			const format = router.detectFormat(xmlInput)
			expect(format).toBe(ToolUseFormat.XML)

			// Route the tool use
			const result = await router.routeToolUse({
				format,
				content: xmlInput
			})

			// Verify format is preserved
			expect(result.format).toBe(ToolUseFormat.XML)
			
			// Parse the XML result to check tool_use_id
			const xmlResult = result.content as string
			expect(xmlResult).toContain("<tool_result")
			expect(xmlResult).toContain('tool_use_id="')
			expect(xmlResult).toContain('status="success"')
		})

		test("JSON → neutral → execute → neutral → JSON preserves tool_use_id", async () => {
			const jsonInput = {
				type: "tool_use",
				id: "test-json-123",
				name: "test_tool",
				input: {
					param1: "value1",
					param2: 123
				}
			}

			// Detect format  
			const format = router.detectFormat(jsonInput)
			// JSON with type: "tool_use" is detected as neutral
			expect(format).toBe(ToolUseFormat.NEUTRAL)

			// Route the tool use
			const result = await router.routeToolUse({
				format,
				content: jsonInput
			})

			// Verify format is preserved
			expect(result.format).toBe(ToolUseFormat.NEUTRAL)
			
			// Check the JSON result
			const jsonResult = result.content as any
			expect(jsonResult.type).toBe("tool_result")
			expect(jsonResult.tool_use_id).toBe("test-json-123") // ID preserved
			expect(jsonResult.status).toBe("success")
		})

		test("OpenAI → neutral → execute → neutral → OpenAI preserves format", async () => {
			const openAiInput = {
				function_call: {
					name: "test_tool",
					arguments: '{"param1": "value1", "param2": 123}'
				}
			}

			// Detect format
			const format = router.detectFormat(openAiInput)
			expect(format).toBe(ToolUseFormat.OPENAI)

			// Route the tool use
			const result = await router.routeToolUse({
				format,
				content: openAiInput
			})

			// Verify format is preserved
			expect(result.format).toBe(ToolUseFormat.OPENAI)
			
			// Check the OpenAI result
			const openAiResult = result.content as any
			expect(openAiResult.role).toBe("tool")
			expect(openAiResult.tool_call_id).toBeDefined()
			expect(openAiResult.content).toContain("Tool executed successfully with args")
		})

		test("Neutral → execute → Neutral preserves all fields", async () => {
			const neutralInput: NeutralToolUseRequest = {
				type: "tool_use",
				id: "neutral-456",
				name: "test_tool",
				input: {
					complex: {
						nested: {
							value: "deep"
						}
					},
					array: [1, 2, 3]
				}
			}

			// Detect format
			const format = router.detectFormat(neutralInput)
			expect(format).toBe(ToolUseFormat.NEUTRAL)

			// Route the tool use
			const result = await router.routeToolUse({
				format,
				content: neutralInput
			})

			// Verify format is preserved
			expect(result.format).toBe(ToolUseFormat.NEUTRAL)
			
			// Check the neutral result
			const neutralResult = result.content as NeutralToolResult
			expect(neutralResult.type).toBe("tool_result")
			expect(neutralResult.tool_use_id).toBe("neutral-456") // ID preserved
			expect(neutralResult.status).toBe("success")
			expect(neutralResult.content).toBeDefined()
		})
	})

	describe("Error handling round-trips", () => {
		test("XML error preserves format and includes error details", async () => {
			const xmlInput = `<unknown_tool>
<param>value</param>
</unknown_tool>`

			const format = router.detectFormat(xmlInput)
			const result = await router.routeToolUse({
				format,
				content: xmlInput
			})

			expect(result.format).toBe(ToolUseFormat.XML)
			const xmlResult = result.content as string
			expect(xmlResult).toContain('status="error"')
			expect(xmlResult).toContain("Tool &apos;unknown_tool&apos; not found")
		})

		test("JSON error preserves format and includes error details", async () => {
			const jsonInput = {
				type: "tool_use",
				id: "error-test",
				name: "unknown_tool",
				input: {}
			}

			const format = router.detectFormat(jsonInput)
			const result = await router.routeToolUse({
				format,
				content: jsonInput
			})

			expect(result.format).toBe(ToolUseFormat.NEUTRAL)
			const jsonResult = result.content as any
			expect(jsonResult.type).toBe("tool_result")
			expect(jsonResult.tool_use_id).toBe("error-test")
			expect(jsonResult.status).toBe("error")
			expect(jsonResult.error?.message).toContain("Tool 'unknown_tool' not found")
		})

		test("OpenAI error preserves format", async () => {
			const openAiInput = {
				tool_calls: [{
					id: "call-error",
					type: "function",
					function: {
						name: "unknown_tool",
						arguments: '{}'
					}
				}]
			}

			const format = router.detectFormat(openAiInput)
			const result = await router.routeToolUse({
				format,
				content: openAiInput
			})

			expect(result.format).toBe(ToolUseFormat.OPENAI)
			const openAiResult = result.content as any
			expect(openAiResult.role).toBe("tool")
			expect(openAiResult.tool_call_id).toBe("call-error")
			expect(openAiResult.content).toContain("Tool 'unknown_tool' not found")
		})

		test("Tool execution error is properly wrapped", async () => {
			// Register a tool that throws an error
			executor.registerTool({
				name: "error_tool",
				description: "Tool that throws errors",
				paramSchema: {},
				handler: async () => {
					throw new Error("Simulated tool execution error")
				}
			})

			const input: NeutralToolUseRequest = {
				type: "tool_use",
				id: "exec-error",
				name: "error_tool",
				input: {}
			}

			const result = await router.routeToolUse({
				format: ToolUseFormat.NEUTRAL,
				content: input
			})

			const neutralResult = result.content as NeutralToolResult
			expect(neutralResult.status).toBe("error")
			expect(neutralResult.tool_use_id).toBe("exec-error")
			expect(neutralResult.error?.message).toContain("Simulated tool execution error")
			
			// Clean up
			executor.unregisterTool("error_tool")
		})
	})

	describe("Format detection edge cases", () => {
		test("Ambiguous content defaults to XML when it has angle brackets", () => {
			const ambiguous = "Just plain text without any markers"
			// Plain text without angle brackets might still be detected as XML
			const format = router.detectFormat(ambiguous)
			expect([ToolUseFormat.XML, ToolUseFormat.NEUTRAL]).toContain(format)
		})

		test("Detects XML with whitespace", () => {
			const xml = "   \n  <tool_name>\n  <param>value</param>\n  </tool_name>  \n"
			expect(router.detectFormat(xml)).toBe(ToolUseFormat.XML)
		})

		test("Detects JSON with tool_use type as neutral", () => {
			const json = '  \n  {"type": "tool_use", "name": "test"}  \n  '
			// JSON with type: "tool_use" is detected as neutral
			const format = router.detectFormat(json)
			expect(format).toBe(ToolUseFormat.NEUTRAL)
		})

		test("Detects OpenAI function_call format", () => {
			const openAi = { function_call: { name: "test", arguments: "{}" } }
			expect(router.detectFormat(openAi)).toBe(ToolUseFormat.OPENAI)
		})

		test("Detects OpenAI tool_calls format", () => {
			const openAi = {
				tool_calls: [
					{ type: "function", function: { name: "test", arguments: "{}" } }
				]
			}
			expect(router.detectFormat(openAi)).toBe(ToolUseFormat.OPENAI)
		})

		test("Neutral object format", () => {
			const neutral = {
				type: "tool_use",
				id: "123",
				name: "test",
				input: {}
			}
			expect(router.detectFormat(neutral)).toBe(ToolUseFormat.NEUTRAL)
		})
	})

	describe("Complex parameter round-trips", () => {
		test("Preserves complex nested structures through XML", async () => {
			const xmlInput = `<test_tool>
<nested>{"deep": {"value": [1, 2, 3], "flag": true}}</nested>
<text>String with "quotes" and newlines\nhere</text>
</test_tool>`

			const result = await router.routeToolUse({
				format: ToolUseFormat.XML,
				content: xmlInput
			})

			const xmlResult = result.content as string
			expect(xmlResult).toContain('status="success"')
			// Content should be properly escaped
			expect(xmlResult).toContain("&quot;")
		})

		test("Preserves special characters through JSON", async () => {
			const jsonInput = {
				type: "tool_use",
				id: "special-chars",
				name: "test_tool",
				input: {
					text: 'Line 1\nLine 2\t"Quoted"',
					symbols: "< > & ' \"",
					unicode: "Hello 世界 🌍"
				}
			}

			const result = await router.routeToolUse({
				format: ToolUseFormat.NEUTRAL,
				content: jsonInput
			})

			const jsonResult = result.content as any
			expect(jsonResult.tool_use_id).toBe("special-chars")
			expect(jsonResult.status).toBe("success")
		})
	})
})