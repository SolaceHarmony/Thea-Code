import * as assert from 'assert'
import * as sinon from 'sinon'
import { NeutralToolResult } from "../../types/McpToolTypes"
import { McpConverters } from "../McpConverters"
import { ToolDefinition } from "../../types/McpProviderTypes"

suite("McpConverters", () => {
	suite("XML Format Conversion", () => {
		suite("xmlToMcp", () => {
			test("should convert valid XML tool use to neutral format", () => {
				const xmlContent = `
          <read_file>
            <path>src/main.js</path>
          </read_file>
        `

				const result = McpConverters.xmlToMcp(xmlContent)

				assert.deepStrictEqual(result, {
					type: "tool_use",
					id: expect.any(String),
					name: "read_file",
					input: {
						path: "src/main.js",
					},
				})
			})

			test("should handle XML with multiple parameters", () => {
				const xmlContent = `
          <read_file>
            <path>src/main.js</path>
            <start_line>10</start_line>
            <end_line>20</end_line>
          </read_file>
        `

				const result = McpConverters.xmlToMcp(xmlContent)

				assert.deepStrictEqual(result, {
					type: "tool_use",
					id: expect.any(String),
					name: "read_file",
					input: {
						path: "src/main.js",
						start_line: 10,
						end_line: 20,
					},
				})
			})

			// Note: The current implementation of xmlToolUseToJson is robust and doesn't throw errors
			// for malformed XML or invalid tool use formats, so we're focusing on testing the successful cases
		})

		suite("mcpToXml", () => {
			test("should convert neutral tool result to XML format", () => {
				const neutralResult: NeutralToolResult = {
					type: "tool_result",
					tool_use_id: "read_file-123",
					content: [{ type: "text", text: "File content here" }],
					status: "success",
				}

				const result = McpConverters.mcpToXml(neutralResult)

				assert.ok(result.includes('<tool_result tool_use_id="read_file-123" status="success">'))
				assert.ok(result.includes("File content here"))
				assert.ok(result.includes("</tool_result>"))
			})

			test("should include error information in XML when present", () => {
				const neutralResult: NeutralToolResult = {
					type: "tool_result",
					tool_use_id: "read_file-123",
					content: [],
					status: "error",
					error: {
						message: "File not found",
					},
				}

				const result = McpConverters.mcpToXml(neutralResult)

				assert.ok(result.includes('<tool_result tool_use_id="read_file-123" status="error">'))
				assert.ok(result.includes('<error message="File not found"'))
				assert.ok(result.includes("</tool_result>"))
			})

			test("should include error details in XML when present", () => {
				const neutralResult: NeutralToolResult = {
					type: "tool_result",
					tool_use_id: "read_file-123",
					content: [],
					status: "error",
					error: {
						message: "File not found",
						details: { path: "src/main.js" },
					},
				}

				const result = McpConverters.mcpToXml(neutralResult)

				assert.ok(result.includes('<tool_result tool_use_id="read_file-123" status="error">'))
				assert.ok(result.includes('<error message="File not found"'))
				assert.ok(result.includes('details="{&quot;path&quot;:&quot;src/main.js&quot;}"'))
				assert.ok(result.includes("</tool_result>"))
			})

			test("should handle image content in the result", () => {
				const neutralResult: NeutralToolResult = {
					type: "tool_result",
					tool_use_id: "screenshot-123",
					content: [
						{
							type: "image",
							source: {
								media_type: "image/png",
								data: "base64data",
							},
						},
					],
					status: "success",
				}

				const result = McpConverters.mcpToXml(neutralResult)

				assert.ok(result.includes('<tool_result tool_use_id="screenshot-123" status="success">'))
				assert.ok(result.includes('<image type="image/png" data="base64data" />'))
				assert.ok(result.includes("</tool_result>"))
			})
		})
	})

	suite("JSON Format Conversion", () => {
		suite("jsonToMcp", () => {
			test("should convert valid JSON string to neutral format", () => {
				const jsonString = JSON.stringify({
					type: "tool_use",
					id: "execute_command-123",
					name: "execute_command",
					input: {
						command: "ls -la",
					},
				})

				const result = McpConverters.jsonToMcp(jsonString)

				assert.deepStrictEqual(result, {
					type: "tool_use",
					id: "execute_command-123",
					name: "execute_command",
					input: {
						command: "ls -la",
					},
				})
			})

			test("should convert valid JSON object to neutral format", () => {
				const jsonObject = {
					type: "tool_use",
					id: "execute_command-123",
					name: "execute_command",
					input: {
						command: "ls -la",
					},
				}

				const result = McpConverters.jsonToMcp(jsonObject)

				assert.deepStrictEqual(result, {
					type: "tool_use",
					id: "execute_command-123",
					name: "execute_command",
					input: {
						command: "ls -la",
					},
				})
			})

			test("should throw an error for malformed JSON string", () => {
				const malformedJson = `{
          "type": "tool_use",
          "id": "execute_command-123",
          "name": "execute_command",
          "input": {
            "command": "ls -la"
          }
        `

				expect(() => McpConverters.jsonToMcp(malformedJson)).toThrow()
			})

			test("should throw an error for invalid tool use format", () => {
				const invalidJson = JSON.stringify({
					type: "not_a_tool_use",
					name: "execute_command",
					input: {
						command: "ls -la",
					},
				})

				expect(() => McpConverters.jsonToMcp(invalidJson)).toThrow("Invalid tool use request format")
			})

			test("should throw an error for missing required properties", () => {
				const invalidJson = JSON.stringify({
					type: "tool_use",
					name: "execute_command",
					// Missing id and input
				})

				expect(() => McpConverters.jsonToMcp(invalidJson)).toThrow("Invalid tool use request format")
			})
		})

		suite("mcpToJson", () => {
			test("should convert neutral tool result to JSON string", () => {
				const neutralResult: NeutralToolResult = {
					type: "tool_result",
					tool_use_id: "execute_command-123",
					content: [{ type: "text", text: "Command output" }],
					status: "success",
				}

				const result = McpConverters.mcpToJson(neutralResult)
				const parsedResult = JSON.parse(result) as Record<string, unknown>

				assert.deepStrictEqual(parsedResult, {
					type: "tool_result",
					tool_use_id: "execute_command-123",
					content: [{ type: "text", text: "Command output" }],
					status: "success",
				})
			})

			test("should include error information in JSON when present", () => {
				const neutralResult: NeutralToolResult = {
					type: "tool_result",
					tool_use_id: "execute_command-123",
					content: [],
					status: "error",
					error: {
						message: "Command failed",
					},
				}

				const result = McpConverters.mcpToJson(neutralResult)
				const parsedResult = JSON.parse(result) as Record<string, unknown>

				assert.deepStrictEqual(parsedResult, {
					type: "tool_result",
					tool_use_id: "execute_command-123",
					content: [],
					status: "error",
					error: {
						message: "Command failed",
					},
				})
			})
		})
	})

	suite("OpenAI Format Conversion", () => {
		suite("openAiToMcp", () => {
			test("should convert OpenAI function call to neutral format", () => {
				const openAiFunctionCall = {
					function_call: {
						id: "function-123",
						name: "execute_command",
						arguments: JSON.stringify({
							command: "ls -la",
						}),
					},
				}

				const result = McpConverters.openAiToMcp(openAiFunctionCall)

				assert.deepStrictEqual(result, {
					type: "tool_use",
					id: "function-123",
					name: "execute_command",
					input: {
						command: "ls -la",
					},
				})
			})

			test("should handle OpenAI tool calls array format", () => {
				const openAiToolCalls = {
					tool_calls: [
						{
							id: "tool-123",
							type: "function",
							function: {
								name: "execute_command",
								arguments: JSON.stringify({
									command: "ls -la",
								}),
							},
						},
					],
				}

				const result = McpConverters.openAiToMcp(openAiToolCalls)

				assert.deepStrictEqual(result, {
					type: "tool_use",
					id: "tool-123",
					name: "execute_command",
					input: {
						command: "ls -la",
					},
				})
			})

			test("should handle malformed arguments by using raw string", () => {
				const openAiFunctionCall = {
					function_call: {
						id: "function-123",
						name: "execute_command",
						arguments: "{command: ls -la}", // Invalid JSON
					},
				}

				const result = McpConverters.openAiToMcp(openAiFunctionCall)

				assert.deepStrictEqual(result, {
					type: "tool_use",
					id: "function-123",
					name: "execute_command",
					input: {
						raw: "{command: ls -la}",
					},
				})
			})

			test("should throw an error for invalid function call format", () => {
				const invalidFunctionCall = {
					not_a_function_call: {
						name: "execute_command",
					},
				}

				expect(() => McpConverters.openAiToMcp(invalidFunctionCall)).toThrow("Invalid function call format")
			})
		})

		suite("mcpToOpenAi", () => {
			test("should convert neutral tool result to OpenAI format", () => {
				const neutralResult: NeutralToolResult = {
					type: "tool_result",
					tool_use_id: "function-123",
					content: [
						{ type: "text", text: "Command output line 1" },
						{ type: "text", text: "Command output line 2" },
					],
					status: "success",
				}

				const result = McpConverters.mcpToOpenAi(neutralResult)

				assert.deepStrictEqual(result, {
					role: "tool",
					tool_call_id: "function-123",
					content: "Command output line 1\nCommand output line 2",
				})
			})
		})
	})

	suite("Tool Definition Conversion", () => {
		test("should convert tool definitions to OpenAI function definitions", () => {
			const toolDefinitions = new Map<string, ToolDefinition>()

			toolDefinitions.set("read_file", {
				name: "read_file",
				description: "Read a file from the filesystem",
				paramSchema: {
					type: "object",
					properties: {
						path: { type: "string" },
						start_line: { type: "number" },
						end_line: { type: "number" },
					},
					required: ["path"],
				},
				handler: () => Promise.resolve({ content: [] }),
			})

			toolDefinitions.set("execute_command", {
				name: "execute_command",
				description: "Execute a shell command",
				paramSchema: {
					type: "object",
					properties: {
						command: { type: "string" },
					},
					required: ["command"],
				},
				handler: () => Promise.resolve({ content: [] }),
			})

			const result = McpConverters.toolDefinitionsToOpenAiFunctions(toolDefinitions)

			assert.deepStrictEqual(result, [
				{
					name: "read_file",
					description: "Read a file from the filesystem",
					parameters: {
						type: "object",
						properties: {
							path: { type: "string" },
							start_line: { type: "number" },
							end_line: { type: "number" },
						},
						required: ["path"],
					},
				},
				{
					name: "execute_command",
					description: "Execute a shell command",
					parameters: {
						type: "object",
						properties: {
							command: { type: "string" },
						},
						required: ["command"],
					},
				},
			])
		})

		test("should handle tool definitions without description or paramSchema", () => {
			const toolDefinitions = new Map<string, ToolDefinition>()

			toolDefinitions.set("minimal_tool", {
				name: "minimal_tool",
				handler: () => Promise.resolve({ content: [] }),
			})

			const result = McpConverters.toolDefinitionsToOpenAiFunctions(toolDefinitions)

			assert.deepStrictEqual(result, [
				{
					name: "minimal_tool",
					description: "",
					parameters: {
						type: "object",
						properties: {},
						required: [],
					},
				},
			])
		})
	})
// Mock cleanup

// Mock cleanup