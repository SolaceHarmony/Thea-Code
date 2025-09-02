import { EventEmitter } from "events"
import { McpToolRouter } from "../McpToolRouter"
import { McpToolExecutor } from "../McpToolExecutor"
import { McpConverters } from "../McpConverters"
import {
import * as assert from 'assert'
import * as sinon from 'sinon'
	NeutralToolUseRequest,
	NeutralToolResult,
	ToolUseFormat,
	ToolUseRequestWithFormat,
} from "../../types/McpToolTypes"
import { SseTransportConfig } from "../../types/McpTransportTypes"

// Mock McpToolExecutor
// TODO: Mock setup needs manual migration
// TODO: Fix mock - needs proxyquire
/*
=> {
	return {
		McpToolExecutor: {
			getInstance: sinon.stub(),
		},

})*/

// Mock the McpConverters
// TODO: Mock setup needs manual migration
// TODO: Fix mock - needs proxyquire
/*
=> {
	return {
		McpConverters: {
			xmlToMcp: sinon.stub(),
			jsonToMcp: sinon.stub(),
			openAiToMcp: sinon.stub(),
			mcpToXml: sinon.stub(),
			mcpToJson: sinon.stub(),
			mcpToOpenAi: sinon.stub(),
		},

})*/

// Create a mock class with proper typing
interface MockExecutor extends EventEmitter {
	initialize: sinon.SinonStub Promise<void>>
	shutdown: sinon.SinonStub Promise<void>>
	executeToolFromNeutralFormat: sinon.SinonStub Promise<NeutralToolResult>>

function createMockExecutor(): MockExecutor {
	const emitter = new EventEmitter()
	const mockObj = {
		initialize: sinon.stub(),
		shutdown: sinon.stub(),
		executeToolFromNeutralFormat: sinon.stub(),

	return Object.assign(emitter, mockObj) as unknown as MockExecutor

suite("McpToolRouter", () => {
	let mockExecutor: MockExecutor

	setup(() => {
		// Create a proper mock executor
		mockExecutor = createMockExecutor()

		// Set up default implementations
		mockExecutor.initialize.mockResolvedValue()
		mockExecutor.shutdown.mockResolvedValue()
		mockExecutor.executeToolFromNeutralFormat.callsFake(
			(request: NeutralToolUseRequest): Promise<NeutralToolResult> => {
				// Handle error cases for testing
				const testRequest = request as NeutralToolUseRequest & {
					error?: string
					errorType?: string

				if (testRequest.error) {
					const error = new Error(testRequest.error)
					error.name = testRequest.errorType || "Error"
					return Promise.reject(error)

				return Promise.resolve({
					type: "tool_result",
					tool_use_id: request.id,
					content: [{ type: "text", text: "Success" }],
					status: "success",
			},

		// Mock the static getInstance method
		sinon.spy(McpToolExecutor, "getInstance").returns(mockExecutor as unknown as McpToolExecutor)

		// Reset router instance - access private property carefully
		interface RouterWithInstance {
			instance?: McpToolRouter

		;(McpToolRouter as unknown as RouterWithInstance).instance = undefined

		// Clear all mocks
		sinon.restore()

		// Set up default mock implementations for McpConverters
		;(McpConverters.xmlToMcp as sinon.SinonStub).callsFake(() => ({
			type: "tool_use",
			id: "xml-123",
			name: "read_file",
			input: { path: "test.txt" },

		;(McpConverters.jsonToMcp as sinon.SinonStub).callsFake(() => ({
			type: "tool_use",
			id: "json-123",
			name: "execute_command",
			input: { command: "ls -la" },

		;(McpConverters.openAiToMcp as sinon.SinonStub).callsFake(() => ({
			type: "tool_use",
			id: "openai-123",
			name: "search_files",
			input: { path: ".", regex: "test" },

		;(McpConverters.mcpToXml as sinon.SinonStub).callsFake((result: unknown) => {
			const typedResult = result as NeutralToolResult
			return `<tool_result tool_use_id="${typedResult.tool_use_id}" status="${typedResult.status}">Success</tool_result>`

		;(McpConverters.mcpToJson as sinon.SinonStub).callsFake((result: unknown) => {
			const typedResult = result as NeutralToolResult
			return JSON.stringify(typedResult)

		;(McpConverters.mcpToOpenAi as sinon.SinonStub).callsFake((result: unknown) => {
			const typedResult = result as NeutralToolResult
			return {
				role: "tool",
				tool_call_id: typedResult.tool_use_id,
				content: "Success",

		// Explicitly create the McpToolRouter instance AFTER mock setup
		McpToolRouter.getInstance()

	teardown(() => {
		// Clean up any event listeners
		const instance = McpToolRouter.getInstance()
		instance.removeAllListeners()
		// Restore the original implementation of getInstance
		sinon.restore()

	suite("Singleton Pattern", () => {
		test("should return the same instance when getInstance is called multiple times", () => {
			const instance1 = McpToolRouter.getInstance()
			const instance2 = McpToolRouter.getInstance()

			assert.strictEqual(instance1, instance2)

		test("should update the SSE config when provided in getInstance", () => {
			const instance1 = McpToolRouter.getInstance()
			const config: SseTransportConfig = {
				port: 3001,
				hostname: "localhost",
				allowExternalConnections: true,

			const instance2 = McpToolRouter.getInstance(config)

			assert.strictEqual(instance1, instance2)
			// We can't directly test the private sseConfig property, but we can verify
			// that the same instance is returned

		suite("Initialization and Shutdown", () => {
			test("should initialize the MCP tool executor when initialize is called", async () => {
				const router = McpToolRouter.getInstance()

				await router.initialize()

				assert.ok(mockExecutor.initialize.called)

			test("should shutdown the MCP tool executor when shutdown is called", async () => {
				const router = McpToolRouter.getInstance()

				await router.shutdown()

				assert.ok(mockExecutor.shutdown.called)

		suite("Format Detection", () => {
			test("should detect XML format from string content", () => {
				const router = McpToolRouter.getInstance()
				const content = "<read_file><path>test.txt</path></read_file>"

				const format = router.detectFormat(content)

				assert.strictEqual(format, ToolUseFormat.XML)

			test("should detect JSON format from string content", () => {
				const router = McpToolRouter.getInstance()
				const content = JSON.stringify({
					name: "execute_command",
					input: { command: "ls -la" },

				const format = router.detectFormat(content)

				assert.strictEqual(format, ToolUseFormat.JSON)

			test("should detect OpenAI format from string content", () => {
				const router = McpToolRouter.getInstance()
				const content = JSON.stringify({
					function_call: {
						name: "execute_command",
						arguments: JSON.stringify({ command: "ls -la" }),
					},

				const format = router.detectFormat(content)

				assert.strictEqual(format, ToolUseFormat.OPENAI)

			test("should detect neutral format from string content", async () => {
				const router = McpToolRouter.getInstance()
				const neutralRequest: NeutralToolUseRequest = {
					type: "tool_use",
					id: "neutral-123",
					name: "list_files",
					input: { path: "." },

				const request: ToolUseRequestWithFormat = {
					format: ToolUseFormat.NEUTRAL,
					content: JSON.stringify(neutralRequest),

				const result = await router.routeToolUse(request)

				// Verify execution
				assert.ok(mockExecutor.executeToolFromNeutralFormat.calledWith(neutralRequest))

				// Verify result format
				assert.strictEqual(result.format, ToolUseFormat.NEUTRAL)

			test("should route neutral format tool use requests (object)", async () => {
				const router = McpToolRouter.getInstance()
				const neutralRequest: NeutralToolUseRequest = {
					type: "tool_use",
					id: "neutral-123",
					name: "list_files",
					input: { path: "." },

				const request: ToolUseRequestWithFormat = {
					format: ToolUseFormat.NEUTRAL,
					content: neutralRequest as unknown as Record<string, unknown>,

				const result = await router.routeToolUse(request)

				// Verify execution
				assert.ok(mockExecutor.executeToolFromNeutralFormat.calledWith(neutralRequest))

				// Verify result format
				assert.strictEqual(result.format, ToolUseFormat.NEUTRAL)

			test("should throw an error for invalid neutral format (missing properties)", async () => {
				const router = McpToolRouter.getInstance()
				const invalidRequest: ToolUseRequestWithFormat = {
					format: ToolUseFormat.NEUTRAL,
					content: {
						// Missing required properties
						type: "tool_use",
					},

				await expect(router.routeToolUse(invalidRequest)).resolves.toEqual({
					format: ToolUseFormat.NEUTRAL,
					content: sinon.match({
						type: "tool_result",
						status: "error",
						error: expect.objectContaining({
							message: sinon.match.string.and(sinon.match("Invalid tool use request format")),
						}),
					}),

			test("should throw an error for unsupported format", async () => {
				const router = McpToolRouter.getInstance()

				// Create a request that will trigger an error
				const invalidRequest = {
					format: "invalid-format" as ToolUseFormat,
					content: {
						type: "tool_use",
						id: "test-error",
						name: "test_tool",
						error: "Unsupported format: invalid-format",
						errorType: "FormatError",
					},

				const result = await router.routeToolUse(invalidRequest)

				assert.deepStrictEqual(result, {
					format: "invalid-format",
					content: sinon.match({
						type: "tool_result",
						status: "error",
						error: {
							message: "Unsupported format: invalid-format",
						},
					}),

		suite("Error Handling", () => {
			setup(() => {
				// Override the default success XML template
				;(McpConverters.mcpToXml as sinon.SinonStub).callsFake((result: unknown) => {
					const typedResult = result as NeutralToolResult
					return `<tool_result tool_use_id="${typedResult.tool_use_id}" status="${typedResult.status}">${
						typedResult.error ? typedResult.error.message : "Success"
					}</tool_result>`

			test("should handle conversion errors", async () => {
				const router = McpToolRouter.getInstance()

				// Set up mock to throw an error
				;(McpConverters.xmlToMcp as sinon.SinonStub).onFirstCall().callsFake(() => {
					throw new Error("Conversion error")

				const request: ToolUseRequestWithFormat = {
					format: ToolUseFormat.XML,
					content: "<invalid>xml</invalid>",

				const result = await router.routeToolUse(request)

				assert.strictEqual(result.format, ToolUseFormat.XML)
				assert.ok(result.content.includes("Conversion error"))

			test("should handle execution errors", async () => {
				const router = McpToolRouter.getInstance()
				const mockExecutor = McpToolExecutor.getInstance()

				// Set up mock to throw an error
				;(mockExecutor.executeToolFromNeutralFormat as sinon.SinonStub).onFirstCall().callsFake(() => {
					throw new Error("Execution error")

				const request: ToolUseRequestWithFormat = {
					format: ToolUseFormat.XML,
					content: "<read_file><path>test.txt</path></read_file>",

				const result = await router.routeToolUse(request)

				assert.strictEqual(result.format, ToolUseFormat.XML)
				assert.ok(result.content.includes("Execution error"))

		suite("Event Forwarding", () => {
			setup(() => {
				// Override the default success XML template
				;(McpConverters.mcpToXml as sinon.SinonStub).callsFake((result: unknown) => {
					const typedResult = result as NeutralToolResult
					return `<tool_result tool_use_id="${typedResult.tool_use_id}" status="${typedResult.status}">${
						typedResult.error ? typedResult.error.message : "Success"
					}</tool_result>`

			test("should forward tool-registered events from the MCP tool executor", async () => {
				const router = McpToolRouter.getInstance()
				const eventHandler = sinon.stub()
				const mockExecutor = McpToolExecutor.getInstance()

				// Register the event handler and wait for next tick to ensure registration is complete
				router.on("tool-registered", eventHandler)
				await new Promise((resolve) => setTimeout(resolve, 0))

				// Emit the event
				mockExecutor.emit("tool-registered", "test-tool")
				await new Promise((resolve) => setTimeout(resolve, 0))

				assert.ok(eventHandler.calledWith("test-tool"))

			test("should forward tool-unregistered events from the MCP tool executor", async () => {
				const router = McpToolRouter.getInstance()
				const eventHandler = sinon.stub()
				const mockExecutor = McpToolExecutor.getInstance()

				// Register the event handler and wait for next tick
				router.on("tool-unregistered", eventHandler)
				await new Promise((resolve) => setTimeout(resolve, 0))

				// Emit the event
				mockExecutor.emit("tool-unregistered", "test-tool")
				await new Promise((resolve) => setTimeout(resolve, 0))

				assert.ok(eventHandler.calledWith("test-tool"))

			test("should forward started events from the MCP tool executor", async () => {
				const router = McpToolRouter.getInstance()
				const eventHandler = sinon.stub()
				const mockExecutor = McpToolExecutor.getInstance()

				// Register the event handler and wait for next tick
				router.on("started", eventHandler)
				await new Promise((resolve) => setTimeout(resolve, 0))

				// Emit the event
				const info = { url: "http://localhost:3000" }
				mockExecutor.emit("started", info)
				await new Promise((resolve) => setTimeout(resolve, 0))

				assert.ok(eventHandler.calledWith(info))

			test("should forward stopped events from the MCP tool executor", async () => {
				const router = McpToolRouter.getInstance()
				const eventHandler = sinon.stub()
				const mockExecutor = McpToolExecutor.getInstance()

				// Register the event handler and wait for next tick
				router.on("stopped", eventHandler)
				await new Promise((resolve) => setTimeout(resolve, 0))

				// Emit the event
				mockExecutor.emit("stopped")
				await new Promise((resolve) => setTimeout(resolve, 0))

				assert.ok(eventHandler.called)

	// Restore the original implementation of getInstance in afterEach
	teardown(() => {
		sinon.restore()
