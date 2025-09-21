/* eslint-disable @typescript-eslint/unbound-method */
import * as assert from 'assert'
import * as sinon from 'sinon'
import * as vscode from "vscode"
import axios from "axios"
import { ContextProxy } from "../../config/ContextProxy"
import { TheaApiManager } from "../../webview/api/TheaApiManager"
import { ProviderSettingsManager } from "../../config/ProviderSettingsManager"
	openRouterDefaultModelId,
	openRouterDefaultModelInfo,
	glamaDefaultModelId,
	glamaDefaultModelInfo,
	requestyDefaultModelId,
	requestyDefaultModelInfo,
} from "../../../shared/api"
import { buildApiHandler, ApiHandler } from "../../../api"

// Mock dependencies
// Mock needs manual implementation
// Mock needs manual implementation
// Mock needs manual implementation

suite("ClineApiManager", () => {
	let manager: TheaApiManager
	let mockContext: vscode.ExtensionContext
	let mockOutputChannel: vscode.OutputChannel
	let mockContextProxy: sinon.SinonStubbedInstance<ContextProxy>
	let mockProviderSettingsManager: sinon.SinonStubbedInstance<ProviderSettingsManager>

	setup(() => {
		// Reset mocks
		sinon.restore()

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

		// Mock output channel
		mockOutputChannel = {
			appendLine: sinon.stub(),
			clear: sinon.stub(),
			dispose: sinon.stub(),
		} as unknown as vscode.OutputChannel

		// Mock context proxy
		mockContextProxy = {
			getValue: sinon.stub().callsFake(() => Promise.resolve(undefined)),
			setValue: sinon.stub().callsFake(() => Promise.resolve()),
			getProviderSettings: sinon.stub().callsFake(() => ({})),
			setProviderSettings: sinon.stub().callsFake(() => Promise.resolve()),
		} as unknown as sinon.SinonStubbedInstance<ContextProxy>

		// Mock provider settings manager
		mockProviderSettingsManager = {
			getModeConfigId: sinon.stub(),
			setModeConfig: sinon.stub(),
			listConfig: sinon.stub(),
			loadConfig: sinon.stub(),
			saveConfig: sinon.stub(),
		} as unknown as sinon.SinonStubbedInstance<ProviderSettingsManager>

		// Create instance of ClineApiManager
		manager = new TheaApiManager(mockContext, mockOutputChannel, mockContextProxy, mockProviderSettingsManager)
	})

	test("updateApiConfiguration updates mode config association", async () => {
		// Setup
		const mockApiConfig = { apiProvider: "openrouter" as const }
		mockContextProxy.getValue.callsFake((key) => {
			if (key === "mode") return "code"
			if (key === "currentApiConfigName") return "test-config"
			return undefined
		})
		mockProviderSettingsManager.listConfig.resolves([
			{ name: "test-config", id: "test-id", apiProvider: "openrouter" },
		])

		// Execute
		await manager.updateApiConfiguration(mockApiConfig)

		// Verify
		assert.ok(mockProviderSettingsManager.setModeConfig.calledWith("code", "test-id"))
		assert.ok(mockContextProxy.setProviderSettings.calledWith(mockApiConfig))
	})

	test("handleModeSwitch loads saved config for mode", async () => {
		// Setup
		const mockApiConfig = { apiProvider: "anthropic" as const }
		mockProviderSettingsManager.getModeConfigId.resolves("saved-config-id")
		mockProviderSettingsManager.listConfig.resolves([
			{ name: "saved-config", id: "saved-config-id", apiProvider: "anthropic" },
		])
		mockProviderSettingsManager.loadConfig.resolves(mockApiConfig)

		// Execute
		const result = await manager.handleModeSwitch("architect")

		// Verify
		assert.ok(mockContextProxy.setValue.calledWith("mode", "architect"))
		assert.ok(mockProviderSettingsManager.getModeConfigId.calledWith("architect"))
		assert.ok(mockProviderSettingsManager.loadConfig.calledWith("saved-config"))
		assert.ok(mockContextProxy.setValue.calledWith("currentApiConfigName", "saved-config"))
		assert.deepStrictEqual(result, mockApiConfig)
	})

	test("handleModeSwitch saves current config when no saved config exists", async () => {
		// Setup
		const mockApiConfig = { apiProvider: "anthropic" as const }
		mockProviderSettingsManager.getModeConfigId.resolves(undefined)
		mockProviderSettingsManager.listConfig.resolves([
			{ name: "current-config", id: "current-id", apiProvider: "anthropic" },
		])
		mockContextProxy.getValue.callsFake(() => "current-config")
		mockProviderSettingsManager.loadConfig.resolves(mockApiConfig)

		// Execute
		await manager.handleModeSwitch("architect")

		// Verify
		assert.ok(mockContextProxy.setValue.calledWith("mode", "architect"))
		assert.ok(mockProviderSettingsManager.setModeConfig.calledWith("architect", "current-id"))
	})

	test("upsertApiConfiguration saves config and updates state", async () => {
		// Setup
		const mockApiConfig = { apiProvider: "anthropic" as const }
		mockProviderSettingsManager.listConfig.resolves([
			{ name: "test-config", id: "test-id", apiProvider: "anthropic" },
		])

		// Mock the imported buildApiHandler function
		const mockedBuildApiHandler = buildApiHandler as sinon.SinonStubbedInstanceFunction<typeof buildApiHandler>
		mockedBuildApiHandler.returns({} as ApiHandler)

		// Execute
		await manager.upsertApiConfiguration("test-config", mockApiConfig)

		// Verify
		assert.ok(mockProviderSettingsManager.saveConfig.calledWith("test-config", mockApiConfig))
		assert.ok(mockContextProxy.setValue.calledWith("listApiConfigMeta", [
			{ name: "test-config", id: "test-id", apiProvider: "anthropic" },
		]))
		assert.ok(mockContextProxy.setValue.calledWith("currentApiConfigName", "test-config"))
		assert.ok(mockContextProxy.setProviderSettings.calledWith(mockApiConfig))
		assert.ok(buildApiHandler.calledWith(mockApiConfig))
	})

	test("upsertApiConfiguration handles errors gracefully", async () => {
		// Setup
		const mockApiConfig = { apiProvider: "anthropic" as const }
		mockProviderSettingsManager.saveConfig.rejects(new Error("Failed to save config"))

		// Mock window.showErrorMessage
		(vscode.window.showErrorMessage as any) = sinon.stub()

		// Execute & Verify
		await assert.rejects(() => manager.upsertApiConfiguration("test-config", mockApiConfig))
		assert.ok(mockOutputChannel.appendLine.called)
		assert.ok(vscode.window.showErrorMessage.called)
	})

	test("handleOpenRouterCallback exchanges code for API key and updates config", async () => {
		// Setup
		const mockApiConfig = { apiProvider: "openrouter" as const }
		mockContextProxy.getProviderSettings.callsFake(() => mockApiConfig)
		mockContextProxy.getValue.callsFake(() => Promise.resolve("default"))

		// Mock axios response
		;(axios.post as sinon.SinonStub).resolves({
			data: { key: "test-api-key" },
		})

		// Mock upsertApiConfiguration method
		manager.upsertApiConfiguration = sinon.stub().resolves(undefined)

		// Execute
		await manager.handleOpenRouterCallback("test-code")

		// Verify
		assert.ok(axios.post.calledWith("https://openrouter.ai/api/v1/auth/keys", { code: "test-code" }))
		assert.ok(manager.upsertApiConfiguration.calledWith("default", {
			...mockApiConfig,
			apiProvider: "openrouter",
			openRouterApiKey: "test-api-key",
			openRouterModelId: openRouterDefaultModelId,
			openRouterModelInfo: openRouterDefaultModelInfo,
		}))
	})

	test("handleGlamaCallback exchanges code for API key and updates config", async () => {
		// Setup
		const mockApiConfig = { apiProvider: "glama" as const }
		mockContextProxy.getProviderSettings.callsFake(() => mockApiConfig)
		mockContextProxy.getValue.callsFake(() => Promise.resolve("default"))

		// Mock axios response
		;(axios.post as sinon.SinonStub).resolves({
			data: { apiKey: "test-api-key" },
		})

		// Mock upsertApiConfiguration method
		manager.upsertApiConfiguration = sinon.stub().resolves(undefined)

		// Execute
		await manager.handleGlamaCallback("test-code")

		// Verify
		assert.ok(axios.post.calledWith("https://glama.ai/api/gateway/v1/auth/exchange-code", {
			code: "test-code",
		}))
		assert.ok(manager.upsertApiConfiguration.calledWith("default", {
			...mockApiConfig,
			apiProvider: "glama",
			glamaApiKey: "test-api-key",
			glamaModelId: glamaDefaultModelId,
			glamaModelInfo: glamaDefaultModelInfo,
		}))
	})

	test("handleRequestyCallback updates config with provided code as API key", async () => {
		// Setup
		const mockApiConfig = { apiProvider: "requesty" as const }
		mockContextProxy.getProviderSettings.callsFake(() => mockApiConfig)
		mockContextProxy.getValue.callsFake(() => Promise.resolve("default"))

		// Mock upsertApiConfiguration method
		manager.upsertApiConfiguration = sinon.stub().resolves(undefined)

		// Execute
		await manager.handleRequestyCallback("test-api-key")

		// Verify
		assert.ok(manager.upsertApiConfiguration.calledWith("default", {
			...mockApiConfig,
			apiProvider: "requesty",
			requestyApiKey: "test-api-key",
			requestyModelId: requestyDefaultModelId,
			requestyModelInfo: requestyDefaultModelInfo,
		}))
	})

	test("buildApiHandler calls the global buildApiHandler function", () => {
		// Setup
		const mockApiConfig = { apiProvider: "openrouter" as const }
		const mockApiHandler = { getModel: sinon.stub() }
		const mockedBuildApiHandler = buildApiHandler as sinon.SinonStubbedInstanceFunction<typeof buildApiHandler>
		mockedBuildApiHandler.returns(mockApiHandler as unknown as ApiHandler)

		// Execute
		const result = manager.buildApiHandler(mockApiConfig)

		// Verify
		assert.ok(buildApiHandler.calledWith(mockApiConfig))
		assert.strictEqual(result, mockApiHandler)
	})
// Mock cleanup
})
