import * as assert from 'assert'
import * as sinon from 'sinon'
/* eslint-disable @typescript-eslint/no-unused-vars, @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-return, @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-require-imports, @typescript-eslint/no-explicit-any */
import { McpClient } from "../McpClient"
import { SseClientFactory } from "../SseClientFactory"

// Mock the MCP SDK
// TODO: Mock setup needs manual migration
// TODO: Mock needs manual migration
// TODO: Implement proper mock with proxyquire,

// TODO: Mock setup needs manual migration
// TODO: Mock needs manual migration
// TODO: Implement proper mock with proxyquire,

suite("SseClientFactory", () => {
	const testUrl = new URL("http://localhost:3000")

	setup(() => {
		sinon.restore()

	suite("createClient", () => {
		test("should create an SDK client when SDK is available", async () => {
			const client = await SseClientFactory.createClient(testUrl)

			assert.ok(client instanceof McpClient)

		test("should fall back to mock client when SDK is not available", async () => {
			// Mock require to throw an error
			const originalRequire = require
			const globalWithRequire = global as unknown as { require: sinon.SinonStub }
			globalWithRequire.require = sinon.stub().callsFake((module: string) => {
				if (module.includes("@modelcontextprotocol/sdk")) {
					throw new Error("Module not found")

				return originalRequire(module)

			const client = await SseClientFactory.createClient(testUrl)

			assert.ok(client instanceof McpClient)

			// Restore original require
			globalWithRequire.require = originalRequire as sinon.SinonStub

		test("should create SDK client with correct client info", async () => {
			const MockClient = require("@modelcontextprotocol/sdk/client").Client

			await SseClientFactory.createClient(testUrl)

			assert.ok(MockClient.calledWith({
				name: "TheaCodeMcpClient",
				version: "1.0.0",

	suite("SDK client wrapper", () => {
		let client: McpClient

		setup(async () => {
			client = await SseClientFactory.createClient(testUrl)

		test("should connect using transport", async () => {
			const transport = { url: testUrl.toString() }

			await client.connect(transport)

			const MockClient = require("@modelcontextprotocol/sdk/client").Client
			const mockInstance = MockClient.mock.results[0].value
			assert.ok(mockInstance.connect.calledWith(transport))

		test("should close connection", async () => {
			await client.close()

			const MockClient = require("@modelcontextprotocol/sdk/client").Client
			const mockInstance = MockClient.mock.results[0].value
			assert.ok(mockInstance.close.called)

		test("should list tools", async () => {
			const result = await client.listTools()

			assert.deepStrictEqual(result, { tools: [] })

			const MockClient = require("@modelcontextprotocol/sdk/client").Client
			const mockInstance = MockClient.mock.results[0].value
			assert.ok(mockInstance.listTools.called)

		test("should call tool", async () => {
			const params = { name: "test_tool", arguments: {} }
			const result = await client.callTool(params)

			assert.deepStrictEqual(result, { content: [] })

			const MockClient = require("@modelcontextprotocol/sdk/client").Client
			const mockInstance = MockClient.mock.results[0].value
			assert.ok(mockInstance.callTool.calledWith(params))

	suite("mock client fallback", () => {
		let client: McpClient

		setup(async () => {
			// Mock require to throw an error to trigger fallback
			const originalRequire = require
			const globalWithRequire = global as unknown as { require: sinon.SinonStub }
			globalWithRequire.require = sinon.stub().callsFake((module: string) => {
				if (module.includes("@modelcontextprotocol/sdk")) {
					throw new Error("Module not found")

				return originalRequire(module)

			client = await SseClientFactory.createClient(testUrl)

			// Restore original require
			globalWithRequire.require = originalRequire as sinon.SinonStub

		test("should connect without error", async () => {
			await expect(client.connect({})).resolves.toBeUndefined()

		test("should close without error", async () => {
			await expect(client.close()).resolves.toBeUndefined()

		test("should list tools returning empty array", async () => {
			const result = await client.listTools()
			assert.deepStrictEqual(result, { tools: [] })

		test("should call tool returning empty content", async () => {
			const result = await client.callTool({ name: "test" })
			assert.deepStrictEqual(result, { content: [] })
