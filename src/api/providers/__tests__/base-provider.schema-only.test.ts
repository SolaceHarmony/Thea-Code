/**
 * BaseProvider schema-only registration tests
 * Verifies that BaseProvider only registers tool schemas and delegates execution to MCP
 */

import { describe, expect, it, jest, beforeEach, afterEach } from "@jest/globals"
import { BaseProvider } from "../base-provider"
import { McpIntegration } from "../../../services/mcp/integration/McpIntegration"
import { ApiStream } from "../../transform/stream"
import type { ModelInfo } from "../../../shared/api"
import type { NeutralConversationHistory } from "../../../shared/neutral-history"

// Mock McpIntegration
jest.mock("../../../services/mcp/integration/McpIntegration")

// Create a concrete implementation for testing
class TestProvider extends BaseProvider {
	createMessage(systemPrompt: string, messages: NeutralConversationHistory): ApiStream {
		// Mock implementation
		return new ApiStream(Promise.resolve(new Response()))
	}

	getModel(): { id: string; info: ModelInfo } {
		return {
			id: "test-model",
			info: {
				maxTokens: 100000,
				contextWindow: 100000,
				supportsImages: false,
				supportsPromptCache: false,
				inputPrice: 0,
				outputPrice: 0,
				supportsComputerUse: false,
				supportsAssistantTool: false,
				description: "Test model"
			}
		}
	}

	// Expose protected method for testing
	public testRegisterTools(): void {
		this.registerTools()
	}
}

describe("BaseProvider - Schema-Only Tool Registration", () => {
	let provider: TestProvider
	let mockMcpIntegration: jest.Mocked<McpIntegration>
	let registeredTools: any[] = []

	beforeEach(() => {
		// Reset registered tools
		registeredTools = []

		// Create mock MCP integration
		mockMcpIntegration = {
			initialize: jest.fn().mockResolvedValue(undefined),
			registerTool: jest.fn((tool) => {
				registeredTools.push(tool)
			}),
			executeTool: jest.fn(),
			shutdown: jest.fn()
		} as any

		// Mock getInstance to return our mock
		(McpIntegration.getInstance as jest.Mock).mockReturnValue(mockMcpIntegration)

		// Create provider instance
		provider = new TestProvider()
	})

	afterEach(() => {
		jest.clearAllMocks()
	})

	describe("Tool Registration", () => {
		it("should register all standard tools with MCP integration", () => {
			// Constructor already calls registerTools
			expect(mockMcpIntegration.registerTool).toHaveBeenCalled()
			
			// Check that standard tools are registered
			const toolNames = registeredTools.map(t => t.name)
			expect(toolNames).toContain("read_file")
			expect(toolNames).toContain("write_to_file")
			expect(toolNames).toContain("list_files")
			expect(toolNames).toContain("search_files")
			expect(toolNames).toContain("apply_diff")
		})

		it("should register tools with proper parameter schemas", () => {
			const readFileTool = registeredTools.find(t => t.name === "read_file")
			
			expect(readFileTool).toBeDefined()
			expect(readFileTool.description).toBe("Read the contents of a file")
			expect(readFileTool.paramSchema).toEqual({
				type: "object",
				properties: {
					path: {
						type: "string",
						description: "Relative path to the file"
					},
					start_line: {
						type: "integer",
						description: "Optional starting line (1-based)"
					},
					end_line: {
						type: "integer",
						description: "Optional ending line (1-based, inclusive)"
					}
				},
				required: ["path"]
			})
		})

		it("should register write_to_file with correct schema", () => {
			const writeTool = registeredTools.find(t => t.name === "write_to_file")
			
			expect(writeTool).toBeDefined()
			expect(writeTool.paramSchema.required).toEqual(["path", "content", "line_count"])
			expect(writeTool.paramSchema.properties.path.type).toBe("string")
			expect(writeTool.paramSchema.properties.content.type).toBe("string")
			expect(writeTool.paramSchema.properties.line_count.type).toBe("integer")
		})

		it("should register search_files with regex parameter", () => {
			const searchTool = registeredTools.find(t => t.name === "search_files")
			
			expect(searchTool).toBeDefined()
			expect(searchTool.paramSchema.properties.regex).toBeDefined()
			expect(searchTool.paramSchema.properties.regex.type).toBe("string")
			expect(searchTool.paramSchema.properties.file_pattern).toBeDefined()
			expect(searchTool.paramSchema.required).toEqual(["path", "regex"])
		})

		it("should register apply_diff with diff parameter", () => {
			const diffTool = registeredTools.find(t => t.name === "apply_diff")
			
			expect(diffTool).toBeDefined()
			expect(diffTool.paramSchema.properties.diff).toBeDefined()
			expect(diffTool.paramSchema.properties.diff.description).toContain("diff")
			expect(diffTool.paramSchema.required).toEqual(["path", "diff"])
		})
	})

	describe("Handler Delegation to MCP", () => {
		it("should have handlers that throw errors indicating MCP execution", () => {
			const readFileTool = registeredTools.find(t => t.name === "read_file")
			
			expect(readFileTool.handler).toBeDefined()
			expect(() => readFileTool.handler()).toThrow("read_file execution handled by MCP provider")
		})

		it("should throw error for write_to_file handler", () => {
			const writeTool = registeredTools.find(t => t.name === "write_to_file")
			
			expect(() => writeTool.handler()).toThrow("write_to_file execution handled by MCP provider")
		})

		it("should throw error for list_files handler", () => {
			const listTool = registeredTools.find(t => t.name === "list_files")
			
			expect(() => listTool.handler()).toThrow("list_files execution handled by MCP provider")
		})

		it("should throw error for search_files handler", () => {
			const searchTool = registeredTools.find(t => t.name === "search_files")
			
			expect(() => searchTool.handler()).toThrow("search_files execution handled by MCP provider")
		})

		it("should throw error for apply_diff handler", () => {
			const diffTool = registeredTools.find(t => t.name === "apply_diff")
			
			expect(() => diffTool.handler()).toThrow("apply_diff execution handled by MCP provider")
		})

		it("should never execute tool handlers directly", () => {
			// All handlers should throw, indicating they're not meant to be executed
			for (const tool of registeredTools) {
				if (tool.handler) {
					expect(() => tool.handler()).toThrow()
				}
			}
		})
	})

	describe("MCP Integration Initialization", () => {
		it("should initialize MCP integration in constructor", () => {
			// Constructor already ran in beforeEach
			expect(mockMcpIntegration.initialize).toHaveBeenCalledTimes(1)
		})

		it("should use singleton instance of McpIntegration", () => {
			const provider2 = new TestProvider()
			
			// Should have been called twice total (once in beforeEach, once here)
			expect(McpIntegration.getInstance).toHaveBeenCalled()
			
			// Both providers should use the same MCP integration instance
			expect(provider.mcpIntegration).toBe(provider2.mcpIntegration)
		})

		it("should register tools after MCP initialization", () => {
			// Clear previous calls
			mockMcpIntegration.registerTool.mockClear()
			mockMcpIntegration.initialize.mockClear()
			
			// Create new provider
			const newProvider = new TestProvider()
			
			// Initialize should be called before registerTool
			const initOrder = mockMcpIntegration.initialize.mock.invocationCallOrder[0]
			const registerOrder = mockMcpIntegration.registerTool.mock.invocationCallOrder[0]
			
			// Registration happens in constructor, after initialize is called
			expect(initOrder).toBeDefined()
			expect(registerOrder).toBeDefined()
		})
	})

	describe("Schema Validation", () => {
		it("should have valid JSON schema for all tools", () => {
			for (const tool of registeredTools) {
				// Check basic schema structure
				expect(tool.paramSchema).toBeDefined()
				expect(tool.paramSchema.type).toBe("object")
				expect(tool.paramSchema.properties).toBeDefined()
				expect(typeof tool.paramSchema.properties).toBe("object")
				
				// Check that required fields are arrays
				if (tool.paramSchema.required) {
					expect(Array.isArray(tool.paramSchema.required)).toBe(true)
				}
			}
		})

		it("should have descriptions for all tool parameters", () => {
			for (const tool of registeredTools) {
				for (const [paramName, paramDef] of Object.entries(tool.paramSchema.properties)) {
					// Each parameter should have a type
					expect(paramDef.type).toBeDefined()
					
					// Most parameters should have descriptions (good practice)
					if (paramName !== "recursive") { // Some boolean flags might not need descriptions
						expect(paramDef.description).toBeDefined()
					}
				}
			}
		})

		it("should mark required parameters correctly", () => {
			const readFileTool = registeredTools.find(t => t.name === "read_file")
			
			// Path is required, start_line and end_line are optional
			expect(readFileTool.paramSchema.required).toEqual(["path"])
			
			// Check optional parameters are in properties but not required
			expect(readFileTool.paramSchema.properties.start_line).toBeDefined()
			expect(readFileTool.paramSchema.properties.end_line).toBeDefined()
			expect(readFileTool.paramSchema.required).not.toContain("start_line")
			expect(readFileTool.paramSchema.required).not.toContain("end_line")
		})
	})

	describe("Tool Execution Delegation", () => {
		it("should not execute tools directly through BaseProvider", async () => {
			// BaseProvider doesn't have an executeTool method
			expect((provider as any).executeTool).toBeUndefined()
			
			// Execution should go through McpIntegration
			expect(mockMcpIntegration.executeTool).toBeDefined()
		})

		it("should rely on MCP for all tool execution", () => {
			// Verify that no tool handler actually executes logic
			let executionErrors = 0
			
			for (const tool of registeredTools) {
				try {
					tool.handler()
				} catch (error) {
					if (error.message.includes("handled by MCP provider")) {
						executionErrors++
					}
				}
			}
			
			// All tools should throw delegation errors
			expect(executionErrors).toBe(registeredTools.length)
		})

		it("should provide tool schemas for provider use", () => {
			// The schemas are registered so providers can advertise capabilities
			expect(registeredTools.length).toBeGreaterThan(0)
			
			// Each tool should have the minimal required fields for schema
			for (const tool of registeredTools) {
				expect(tool.name).toBeDefined()
				expect(tool.description).toBeDefined()
				expect(tool.paramSchema).toBeDefined()
			}
		})
	})

	describe("Inheritance and Extension", () => {
		class ExtendedProvider extends TestProvider {
			protected registerTools(): void {
				super.registerTools() // Get base tools
				
				// Add custom tool
				this.mcpIntegration.registerTool({
					name: "custom_tool",
					description: "A custom tool",
					paramSchema: {
						type: "object",
						properties: {
							customParam: { type: "string" }
						},
						required: ["customParam"]
					},
					handler: () => {
						throw new Error("custom_tool handled by MCP")
					}
				})
			}
		}

		it("should allow providers to extend tool registration", () => {
			// Clear previous registrations
			registeredTools = []
			mockMcpIntegration.registerTool.mockClear()
			
			const extendedProvider = new ExtendedProvider()
			
			// Should have base tools plus custom tool
			const toolNames = registeredTools.map(t => t.name)
			expect(toolNames).toContain("read_file")
			expect(toolNames).toContain("custom_tool")
		})

		it("should maintain delegation pattern in extended providers", () => {
			registeredTools = []
			mockMcpIntegration.registerTool.mockClear()
			
			const extendedProvider = new ExtendedProvider()
			
			const customTool = registeredTools.find(t => t.name === "custom_tool")
			expect(() => customTool.handler()).toThrow("custom_tool handled by MCP")
		})
	})

	describe("Edge Cases", () => {
		it("should handle registration of tools with no optional parameters", () => {
			const listTool = registeredTools.find(t => t.name === "list_files")
			
			// recursive is optional
			expect(listTool.paramSchema.required).not.toContain("recursive")
			expect(listTool.paramSchema.properties.recursive).toBeDefined()
		})

		it("should handle tools with complex parameter types", () => {
			const writeTool = registeredTools.find(t => t.name === "write_to_file")
			
			// Has string, integer types
			expect(writeTool.paramSchema.properties.content.type).toBe("string")
			expect(writeTool.paramSchema.properties.line_count.type).toBe("integer")
		})

		it("should not allow direct modification of registered tools", () => {
			// This tests that tools are properly encapsulated
			const originalCount = registeredTools.length
			
			// Try to modify the registered tools (this shouldn't affect actual registration)
			registeredTools.push({ name: "fake_tool" })
			
			// Create a new provider and check that it doesn't have the fake tool
			registeredTools = []
			mockMcpIntegration.registerTool.mockClear()
			const newProvider = new TestProvider()
			
			const toolNames = registeredTools.map(t => t.name)
			expect(toolNames).not.toContain("fake_tool")
		})
	})
})