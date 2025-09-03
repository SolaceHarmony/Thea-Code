// filepath: /Volumes/stuff/Projects/Thea-Code/src/core/webview/__tests__/ClineStateManager.test.ts
import * as assert from 'assert'
import * as sinon from 'sinon'
import * as vscode from 'vscode'
/* eslint-disable @typescript-eslint/unbound-method */
import * as vscode from "vscode"
import * as os from "os"
import { TheaStateManager } from "../thea/TheaStateManager" // Renamed import and path
import { ContextProxy } from "../../config/ContextProxy"
import { ProviderSettingsManager } from "../../config/ProviderSettingsManager"
import { CustomModesManager } from "../../config/CustomModesManager"
import { defaultModeSlug } from "../../../shared/modes"
import { experimentDefault } from "../../../shared/experiments"
import { formatLanguage } from "../../../shared/language"
import { TERMINAL_SHELL_INTEGRATION_TIMEOUT } from "../../../integrations/terminal/Terminal"

// Mock dependencies
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
suite("TheaStateManager", () => {
	// Renamed describe block
	let stateManager: TheaStateManager // Renamed type
	let mockContext: vscode.ExtensionContext
	let mockContextProxy: sinon.SinonStubStatic<ContextProxy>
	let mockProviderSettingsManager: sinon.SinonStubStatic<ProviderSettingsManager>
	let mockCustomModesManager: sinon.SinonStubStatic<CustomModesManager>

	setup(() => {
		// Reset mocks
		sinon.restore()

		// Mock os.platform
		;(os.platform as sinon.SinonStub).returns("darwin")

		// Mock formatLanguage
		;(formatLanguage as sinon.SinonStub).returns("en")

		// Mock context
		mockContext = {
			extensionPath: "/test/path",
			extensionUri: {} as vscode.Uri,
			globalState: {
				get: sinon.stub(),
				update: sinon.stub(),
				keys: sinon.stub(),
			},
			secrets: {
				get: sinon.stub(),
				store: sinon.stub(),
				delete: sinon.stub(),
			},
			subscriptions: [],
			extension: {
				packageJSON: { version: "1.0.0" },
			},
		} as unknown as vscode.ExtensionContext

		// Mock contextProxy
		mockContextProxy = {
			getValues: sinon.stub().returns({}),
			setValue: sinon.stub().callsFake(() => Promise.resolve()),
			getValue: sinon.stub().callsFake(() => Promise.resolve(undefined)),
			setValues: sinon.stub().callsFake(() => Promise.resolve()),
			getProviderSettings: sinon.stub().returns({ apiProvider: "anthropic" }),
			setProviderSettings: sinon.stub().callsFake(() => Promise.resolve()),
		} as unknown as sinon.SinonStubStatic<ContextProxy>

		// Mock ContextProxy constructor
		;(ContextProxy as sinon.SinonStubStaticClass<typeof ContextProxy>).callsFake(() => mockContextProxy)

		// Mock provider settings manager
		mockProviderSettingsManager = {} as sinon.SinonStubStatic<ProviderSettingsManager>

		// Mock custom modes manager
		mockCustomModesManager = {
			getCustomModes: jest
				.fn()
				.resolves([{ slug: "custom", name: "Custom", roleDefinition: "Role", groups: ["read"] }]),
		} as unknown as sinon.SinonStubStatic<CustomModesManager>

		// Create instance of ClineStateManager
		stateManager = new TheaStateManager(mockContext, mockProviderSettingsManager, mockCustomModesManager) // Renamed constructor

	test("getState returns correct default state when no values are provided", async () => {
		// Setup - empty state values from contextProxy
		mockContextProxy.getValues.returns({})
		mockContextProxy.getProviderSettings.returns({ apiProvider: "anthropic" })

		// Execute
		const state = await stateManager.getState()

		// Verify
		assert.deepStrictEqual(state, 
			expect.objectContaining({
				apiConfiguration: { apiProvider: "anthropic" },
				osInfo: "unix",
				alwaysAllowReadOnly: false,
				alwaysAllowReadOnlyOutsideWorkspace: false,
				alwaysAllowWrite: false,
				alwaysAllowWriteOutsideWorkspace: false,
				alwaysAllowExecute: false,
				alwaysAllowBrowser: false,
				alwaysAllowMcp: false,
				alwaysAllowModeSwitch: false,
				alwaysAllowSubtasks: false,
				soundEnabled: false,
				ttsEnabled: false,
				ttsSpeed: 1.0,
				diffEnabled: true,
				enableCheckpoints: true,
				checkpointStorage: "task",
				browserViewportSize: "900x600",
				screenshotQuality: 75,
				remoteBrowserEnabled: false,
				fuzzyMatchThreshold: 1.0,
				writeDelayMs: 1000,
				terminalOutputLineLimit: 500,
				terminalShellIntegrationTimeout: TERMINAL_SHELL_INTEGRATION_TIMEOUT,
				mode: defaultModeSlug,
				language: "en",
				mcpEnabled: true,
				enableMcpServerCreation: true,
				alwaysApproveResubmit: false,
				requestDelaySeconds: 10,
				rateLimitSeconds: 0,
				currentApiConfigName: "default",
				listApiConfigMeta: [],
				pinnedApiConfigs: {},
				modeApiConfigs: {},
				customModePrompts: {},
				customSupportPrompts: {},
				experiments: experimentDefault,
				autoApprovalEnabled: false,
				customModes: [{ slug: "custom", name: "Custom", roleDefinition: "Role", groups: ["read"] }],
				maxOpenTabsContext: 20,
				maxWorkspaceFiles: 200,
				openRouterUseMiddleOutTransform: true,
				browserToolEnabled: true,
				telemetrySetting: "unset",
				showTheaIgnoredFiles: true,
				maxReadFileLine: 500,
			}),

	test("getState correctly integrates stored state values", async () => {
		// Setup - provide specific state values
		mockContextProxy.getValues.returns({
			mode: "architect",
			soundEnabled: true,
			diffEnabled: false,
			browserViewportSize: "1200x800",
			maxWorkspaceFiles: 300,
			customModePrompts: { architect: { customInstructions: "Test instructions" } },
		mockContextProxy.getProviderSettings.returns({
			apiProvider: "openrouter",
			openRouterApiKey: "test-key",

		// Execute
		const state = await stateManager.getState()

		// Verify
		assert.deepStrictEqual(state, 
			expect.objectContaining({
				apiConfiguration: {
					apiProvider: "openrouter",
					openRouterApiKey: "test-key",
				},
				mode: "architect",
				soundEnabled: true,
				diffEnabled: false,
				browserViewportSize: "1200x800",
				maxWorkspaceFiles: 300,
				customModePrompts: { architect: { customInstructions: "Test instructions" } },
			}),

	test("getState uses getCustomModes method if available", async () => {
		// Setup - mock custom getCustomModes method
		const mockGetCustomModes = sinon.stub().resolves([
			{ slug: "custom1", name: "Custom 1", roleDefinition: "Role 1", groups: ["read"] },
			{ slug: "custom2", name: "Custom 2", roleDefinition: "Role 2", groups: ["read", "execute"] },
		])
		stateManager.getCustomModes = mockGetCustomModes

		// Execute
		const state = await stateManager.getState()

		// Verify
		assert.ok(mockGetCustomModes.called)
		assert.ok(!mockCustomModesManager.getCustomModes.called)
		assert.deepStrictEqual(state.customModes, [
			{ slug: "custom1", name: "Custom 1", roleDefinition: "Role 1", groups: ["read"] },
			{ slug: "custom2", name: "Custom 2", roleDefinition: "Role 2", groups: ["read", "execute"] },
		])

	test("getState falls back to customModesManager if getCustomModes is not set", async () => {
		// Setup - ensure getCustomModes is undefined
		stateManager.getCustomModes = undefined

		// Execute
		const state = await stateManager.getState()

		// Verify
		assert.ok(mockCustomModesManager.getCustomModes.called)
		assert.deepStrictEqual(state.customModes, [
			{ slug: "custom", name: "Custom", roleDefinition: "Role", groups: ["read"] },
		])

	test("getState enforces minimum requestDelaySeconds", async () => {
		// Setup - provide state with low requestDelaySeconds
		mockContextProxy.getValues.returns({
			requestDelaySeconds: 2,

		// Execute
		const state = await stateManager.getState()

		// Verify - should enforce minimum of 5 seconds
		assert.strictEqual(state.requestDelaySeconds, 5)

	test("updateGlobalState delegates to contextProxy.setValue", async () => {
		// Execute
		await stateManager.updateGlobalState("diffEnabled", false)

		// Verify
		assert.ok(mockContextProxy.setValue.calledWith("diffEnabled", false))

	test("getGlobalState delegates to contextProxy.getValue", () => {
		// Setup
		mockContextProxy.getValue.callsFake(() => Promise.resolve(true))

		// Execute
		const result = stateManager.getGlobalState("diffEnabled")

		// Verify
		assert.ok(mockContextProxy.getValue.calledWith("diffEnabled"))
		assert.strictEqual(result, true)

	test("setValue delegates to contextProxy.setValue", async () => {
		// Execute
		await stateManager.setValue("diffEnabled", false)

		// Verify
		assert.ok(mockContextProxy.setValue.calledWith("diffEnabled", false))

	test("getValue delegates to contextProxy.getValue", () => {
		// Setup
		mockContextProxy.getValue.callsFake(() => Promise.resolve(true))

		// Execute
		const result = stateManager.getValue("diffEnabled")

		// Verify
		assert.ok(mockContextProxy.getValue.calledWith("diffEnabled"))
		assert.strictEqual(result, true)

	test("getValues delegates to contextProxy.getValues", () => {
		// Setup
		const mockValues = { diffEnabled: true, mode: "code" }
		mockContextProxy.getValues.returns(mockValues)

		// Execute
		const result = stateManager.getValues()

		// Verify
		assert.ok(mockContextProxy.getValues.called)
		assert.strictEqual(result, mockValues)

	test("setValues delegates to contextProxy.setValues", async () => {
		// Setup
		const mockValues = { diffEnabled: true, mode: "code" }

		// Execute
		await stateManager.setValues(mockValues)

		// Verify
		assert.ok(mockContextProxy.setValues.calledWith(mockValues))
