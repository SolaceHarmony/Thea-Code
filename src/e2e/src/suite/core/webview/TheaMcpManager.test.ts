import * as vscode from "vscode"
import * as os from "os"
import fs from "fs/promises"
import { TheaMcpManager } from "../mcp/TheaMcpManager"
import { McpHub } from "../../../services/mcp/management/McpHub"
import { EXTENSION_DISPLAY_NAME, EXTENSION_CONFIG_DIR } from "../../../shared/config/thea-config"

// Mock dependencies
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
import * as assert from 'assert'
import * as sinon from 'sinon'
import * as vscode from 'vscode'
// TODO: Fix mock - needs proxyquire
/*
=> {
	const originalPath: Record<string, unknown> = // TODO: requireActual("path")
	return {
		...originalPath,
		// Ensure consistent path separators in tests regardless of platform
		join: (...paths: string[]) => {
			return paths.join("/")
		},

})*/

suite("TheaMcpManager", () => {
	let mcpManager: TheaMcpManager
	let mockContext: vscode.ExtensionContext
	let mockMcpHub: sinon.SinonStubStatic<McpHub>

	// Declare mock functions in outer scope to avoid unbound method issues
	let updateServerTimeoutMock: sinon.SinonStub
	let deleteServerMock: sinon.SinonStub
	let toggleServerDisabledMock: sinon.SinonStub
	let restartConnectionMock: sinon.SinonStub
	let getMcpSettingsFilePathMock: sinon.SinonStub
	let getAllServersMock: sinon.SinonStub

	const TEST_TEMP_DIR = "/tmp/thea-test"
	// Store original platform value
	const originalPlatform = process.platform

	setup(() => {
		// Reset mocks
		sinon.restore()

		// Mock context
		mockContext = {
			extensionPath: "/test/path",
			extensionUri: {} as vscode.Uri,
		} as unknown as vscode.ExtensionContext

		// Mock os functions with temp directory
		;(os.homedir as sinon.SinonStub).returns(TEST_TEMP_DIR)
		;(os.tmpdir as sinon.SinonStub).returns(TEST_TEMP_DIR)

		// Mock fs functions
		;(fs.mkdir as sinon.SinonStub).resolves(undefined)
		;(fs.access as sinon.SinonStub).resolves(undefined)

		// Create individual mock functions to avoid unbound method issues
		updateServerTimeoutMock = sinon.stub().resolves(undefined)
		deleteServerMock = sinon.stub().resolves(undefined)
		toggleServerDisabledMock = sinon.stub().resolves(undefined)
		restartConnectionMock = sinon.stub().resolves(undefined)
		getMcpSettingsFilePathMock = sinon.stub().resolves(`${TEST_TEMP_DIR}/mcp/settings.json`)
		getAllServersMock = sinon.stub().returns([{ name: "server1", host: "localhost", port: 8000 }])

		mockMcpHub = {
			updateServerTimeout: updateServerTimeoutMock,
			deleteServer: deleteServerMock,
			toggleServerDisabled: toggleServerDisabledMock,
			restartConnection: restartConnectionMock,
			getMcpSettingsFilePath: getMcpSettingsFilePathMock,
			getAllServers: getAllServersMock,
		} as unknown as sinon.SinonStubStatic<McpHub>

		// Create instance of ClineMcpManager
		mcpManager = new TheaMcpManager(mockContext)

		// Mock console to prevent test output noise
		sinon.spy(console, "log").callsFake(() => {})
		sinon.spy(console, "warn").callsFake(() => {})
		sinon.spy(console, "error").callsFake(() => {})

	teardown(() => {
		sinon.restore()
		// Restore original process.platform
		Object.defineProperty(process, "platform", { value: originalPlatform })

	test("setMcpHub sets the hub instance", () => {
		// Execute
		mcpManager.setMcpHub(mockMcpHub)

		// Verify
		expect(mcpManager.getMcpHub()).toBe(mockMcpHub)

	test("getMcpHub returns undefined when not set", () => {
		// Execute & Verify
		expect(mcpManager.getMcpHub()).toBeUndefined()

	suite("ensureMcpServersDirectoryExists", () => {
		test("creates and returns correct directory path on macOS", async () => {
			// Setup - override process.platform
			Object.defineProperty(process, "platform", { value: "darwin" })

			// Execute
			const result = await mcpManager.ensureMcpServersDirectoryExists()

			// Verify
			const expectedPath = `${TEST_TEMP_DIR}/Documents/${EXTENSION_DISPLAY_NAME}/MCP`
			assert.strictEqual(result, expectedPath)
			assert.ok(fs.mkdir.calledWith(expectedPath, { recursive: true }))

		test("creates and returns correct directory path on Windows", async () => {
			// Setup - override process.platform
			Object.defineProperty(process, "platform", { value: "win32" })

			// Execute
			const result = await mcpManager.ensureMcpServersDirectoryExists()

			// Verify
			const expectedPath = `${TEST_TEMP_DIR}/AppData/Roaming/${EXTENSION_DISPLAY_NAME}/MCP`
			assert.strictEqual(result, expectedPath)
			assert.ok(fs.mkdir.calledWith(expectedPath, { recursive: true }))

		test("creates and returns correct directory path on Linux", async () => {
			// Setup - override process.platform
			Object.defineProperty(process, "platform", { value: "linux" })

			// Execute
			const result = await mcpManager.ensureMcpServersDirectoryExists()

			// Verify
			const expectedPath = `${TEST_TEMP_DIR}/.local/share/${EXTENSION_DISPLAY_NAME}/MCP`
			assert.strictEqual(result, expectedPath)
			assert.ok(fs.mkdir.calledWith(expectedPath, { recursive: true }))

		test("falls back to alternate path when directory creation fails", async () => {
			// Setup - use macOS for this test
			Object.defineProperty(process, "platform", { value: "darwin" })
			;(fs.mkdir as sinon.SinonStub).rejects(new Error("Permission denied"))

			// Execute
			const result = await mcpManager.ensureMcpServersDirectoryExists()

			// Verify
			const expectedPath = `${TEST_TEMP_DIR}/${EXTENSION_CONFIG_DIR}/mcp`
			assert.strictEqual(result, expectedPath)
			assert.ok(console.error.called)

	suite("McpHub delegation methods", () => {
		setup(() => {
			// Set the McpHub for all delegation tests
			mcpManager.setMcpHub(mockMcpHub)

		test("updateServerTimeout delegates to McpHub", async () => {
			// Execute
			await mcpManager.updateServerTimeout("server1", 60000)

			// Verify
			assert.ok(updateServerTimeoutMock.calledWith("server1", 60000))

		test("deleteServer delegates to McpHub", async () => {
			// Execute
			await mcpManager.deleteServer("server1")

			// Verify
			assert.ok(deleteServerMock.calledWith("server1"))

		test("toggleServerDisabled delegates to McpHub", async () => {
			// Execute
			await mcpManager.toggleServerDisabled("server1", true)

			// Verify
			assert.ok(toggleServerDisabledMock.calledWith("server1", true))

		test("restartConnection delegates to McpHub", async () => {
			// Execute
			await mcpManager.restartConnection("server1")

			// Verify
			assert.ok(restartConnectionMock.calledWith("server1"))

		test("getMcpSettingsFilePath delegates to McpHub", async () => {
			// Execute
			const result = await mcpManager.getMcpSettingsFilePath()

			// Verify
			assert.ok(getMcpSettingsFilePathMock.called)
			assert.strictEqual(result, `${TEST_TEMP_DIR}/mcp/settings.json`)

		test("getAllServers delegates to McpHub", () => {
			// Execute
			const result = mcpManager.getAllServers()

			// Verify
			assert.ok(getAllServersMock.called)
			assert.deepStrictEqual(result, [{ name: "server1", host: "localhost", port: 8000 }])

	suite("McpHub fallback behavior when not set", () => {
		test("updateServerTimeout logs warning when McpHub not set", async () => {
			// Execute
			await mcpManager.updateServerTimeout("server1", 60000)

			// Verify
			assert.ok(console.warn.calledWith(
				sinon.match.string.and(sinon.match("McpHub not available for updateServerTimeout"))),

		test("deleteServer logs warning when McpHub not set", async () => {
			// Execute
			await mcpManager.deleteServer("server1")

			// Verify
			assert.ok(console.warn.calledWith(sinon.match.string.and(sinon.match("McpHub not available for deleteServer"))))

		test("toggleServerDisabled logs warning when McpHub not set", async () => {
			// Execute
			await mcpManager.toggleServerDisabled("server1", true)

			// Verify
			assert.ok(console.warn.calledWith(
				sinon.match.string.and(sinon.match("McpHub not available for toggleServerDisabled"))),

		test("restartConnection logs warning when McpHub not set", async () => {
			// Execute
			await mcpManager.restartConnection("server1")

			// Verify
			assert.ok(console.warn.calledWith(
				sinon.match.string.and(sinon.match("McpHub not available for restartConnection"))),

		test("getMcpSettingsFilePath uses fallback path when McpHub not set", async () => {
			// Setup
			Object.defineProperty(process, "platform", { value: "darwin" })
			const expectedFallbackPath = `${TEST_TEMP_DIR}/Documents/${EXTENSION_DISPLAY_NAME}/MCP/mcp_settings.json`

			// Execute
			const result = await mcpManager.getMcpSettingsFilePath()

			// Verify
			assert.ok(console.warn.calledWith(
				sinon.match.string.and(sinon.match("McpHub not available for getMcpSettingsFilePath"))),

			assert.strictEqual(result, expectedFallbackPath)

		test("getAllServers returns empty array when McpHub not set", () => {
			// Execute
			const result = mcpManager.getAllServers()

			// Verify
			assert.deepStrictEqual(result, [])

	test("dispose clears the McpHub reference", () => {
		// Setup
		mcpManager.setMcpHub(mockMcpHub)
		expect(mcpManager.getMcpHub()).toBe(mockMcpHub)

		// Execute
		mcpManager.dispose()

		// Verify
		expect(mcpManager.getMcpHub()).toBeUndefined()
		assert.ok(console.log.calledWith(sinon.match.string.and(sinon.match("TheaMcpManager disposed"))))
