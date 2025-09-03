import * as assert from 'assert'
import * as sinon from 'sinon'
/**
 * Integration tests for provider-transport interactions
 * Tests MockMcpProvider scenarios and provider interface compliance
 */
/* eslint-disable @typescript-eslint/no-unsafe-assignment, @typescript-eslint/require-await, @typescript-eslint/restrict-template-expressions, @typescript-eslint/unbound-method */
import { ToolDefinition } from "../../types/McpProviderTypes"
import { MockMcpProvider } from "../../providers/MockMcpProvider"

suite("Provider-Transport Integration", () => {
	suite("MockMcpProvider Integration Scenarios", () => {
		let provider: MockMcpProvider

		setup(() => {
			provider = new MockMcpProvider()

		teardown(async () => {
			if (provider && provider.isRunning()) {
				await provider.stop()

			provider.removeAllListeners()

		test("should start and provide connection info", async () => {
			const startedSpy = sinon.stub()
			provider.on("started", startedSpy)

			await provider.start()

			expect(provider.isRunning()).toBe(true)
			expect(provider.getServerUrl()).toBeInstanceOf(URL)
			assert.ok(startedSpy.calledWith(
				sinon.match({
					url: sinon.match.string.and(sinon.match("http://localhost:"))),
				}),

		test("should stop cleanly", async () => {
			const stoppedSpy = sinon.stub()
			provider.on("stopped", stoppedSpy)

			await provider.start()
			await provider.stop()

			expect(provider.isRunning()).toBe(false)
			assert.ok(stoppedSpy.called)

		test("should register and execute tools", async () => {
			const tool: ToolDefinition = {
				name: "integration_tool",
				description: "Test tool for integration",
				paramSchema: { type: "object", properties: { input: { type: "string" } } },
				handler: async (args) => ({
					content: [{ type: "text", text: `Integration result: ${args.input}` }],
					isError: false,
				}),

			await provider.start()
			provider.registerToolDefinition(tool)

			const result = await provider.executeTool("integration_tool", { input: "test data" })

			assert.deepStrictEqual(result, {
				content: [{ type: "text", text: "Integration result: test data" }],
				isError: false,

		test("should handle multiple tool registrations", async () => {
			const tools: ToolDefinition[] = [
				{
					name: "tool1",
					description: "First tool",
					handler: async () => ({ content: [{ type: "text", text: "Tool 1" }] }),
				},
				{
					name: "tool2",
					description: "Second tool",
					handler: async () => ({ content: [{ type: "text", text: "Tool 2" }] }),
				},

			await provider.start()

			tools.forEach((tool) => provider.registerToolDefinition(tool))

			const result1 = await provider.executeTool("tool1", {})
			const result2 = await provider.executeTool("tool2", {})

			assert.deepStrictEqual(result1.content[0], { type: "text", text: "Tool 1" })
			assert.deepStrictEqual(result2.content[0], { type: "text", text: "Tool 2" })

		test("should emit events for tool lifecycle", async () => {
			const registeredSpy = sinon.stub()
			const unregisteredSpy = sinon.stub()

			provider.on("tool-registered", registeredSpy)
			provider.on("tool-unregistered", unregisteredSpy)

			// Properly await the start operation
			await provider.start()

			const tool: ToolDefinition = {
				name: "event_tool",
				description: "Tool for event testing",
				handler: async () => ({ content: [] }),

			provider.registerToolDefinition(tool)
			assert.ok(registeredSpy.calledWith("event_tool"))

			provider.unregisterTool("event_tool")
			assert.ok(unregisteredSpy.calledWith("event_tool"))

		test("should handle concurrent tool executions", async () => {
			const tool: ToolDefinition = {
				name: "concurrent_tool",
				description: "Tool for concurrency testing",
				handler: async (args) => {
					// Simulate async work
					await new Promise((resolve) => setTimeout(resolve, 10))
					return {
						content: [{ type: "text", text: `Concurrent result: ${args.id}` }],
						isError: false,

				},

			await provider.start()
			provider.registerToolDefinition(tool)

			// Execute multiple tools concurrently
			const promises = Array.from({ length: 5 }, (_, i) => provider.executeTool("concurrent_tool", { id: i }))

			const results = await Promise.all(promises)

			results.forEach((result, index) => {
				assert.strictEqual(result.content[0].text, `Concurrent result: ${index}`)
				assert.strictEqual(result.isError, false)

		test("should maintain tool state across provider operations", async () => {
			const tool: ToolDefinition = {
				name: "persistent_tool",
				description: "Tool that persists",
				handler: async () => ({ content: [{ type: "text", text: "persistent" }] }),

			// Register tool before starting
			provider.registerToolDefinition(tool)

			await provider.start()
			const result1 = await provider.executeTool("persistent_tool", {})

			await provider.stop()
			await provider.start()

			// Tool should still be available after restart
			const result2 = await provider.executeTool("persistent_tool", {})

			assert.deepStrictEqual(result1, result2)
			assert.deepStrictEqual(result1.content[0], { type: "text", text: "persistent" })

			await provider.stop()

		test("should handle provider error scenarios", async () => {
			const errorTool: ToolDefinition = {
				name: "error_tool",
				description: "Tool that fails",
				handler: async () => {
					throw new Error("Provider error test")
				},

			await provider.start()
			provider.registerToolDefinition(errorTool)

			const result = await provider.executeTool("error_tool", {})

			assert.strictEqual(result.isError, true)
			assert.ok(result.content[0].text.includes("Provider error test"))

		test("should handle graceful shutdown with active tools", async () => {
			const longRunningTool: ToolDefinition = {
				name: "long_tool",
				description: "Long running tool",
				handler: async () => {
					await new Promise((resolve) => setTimeout(resolve, 50))
					return { content: [{ type: "text", text: "completed" }] }
				},

			await provider.start()
			provider.registerToolDefinition(longRunningTool)

			// Start long-running operation
			const executionPromise = provider.executeTool("long_tool", {})

			// Stop provider while tool is running
			await provider.stop()

			// Tool should still complete
			const result = await executionPromise
			assert.strictEqual(result.content[0].text, "completed")

	suite("Provider Interface Compliance", () => {
		let provider: MockMcpProvider

		setup(() => {
			provider = new MockMcpProvider()

		teardown(() => {
			provider.removeAllListeners()

		test("should implement all required IMcpProvider methods", () => {
			assert.strictEqual(typeof provider.start, "function")
			assert.strictEqual(typeof provider.stop, "function")
			assert.strictEqual(typeof provider.registerToolDefinition, "function")
			assert.strictEqual(typeof provider.unregisterTool, "function")
			assert.strictEqual(typeof provider.executeTool, "function")
			assert.strictEqual(typeof provider.getServerUrl, "function")
			assert.strictEqual(typeof provider.isRunning, "function")

		test("should extend EventEmitter for event handling", () => {
			assert.ok(provider.on !== undefined)
			assert.ok(provider.emit !== undefined)
			assert.ok(provider.removeAllListeners !== undefined)

		test("should return correct types from methods", async () => {
			await provider.start()

			const tool: ToolDefinition = {
				name: "type_test_tool",
				description: "Tool for type testing",
				handler: async () => ({ content: [{ type: "text", text: "test" }] }),

			// Method return types
			expect(typeof provider.isRunning()).toBe("boolean")
			expect(provider.getServerUrl()).toBeInstanceOf(URL)

			provider.registerToolDefinition(tool)
			expect(typeof provider.unregisterTool("type_test_tool")).toBe("boolean")

			const result = await provider.executeTool("type_test_tool", {})
			assert.ok(result.hasOwnProperty('content'))
			expect(Array.isArray(result.content)).toBe(true)
			assert.strictEqual(typeof result.isError, "boolean")
