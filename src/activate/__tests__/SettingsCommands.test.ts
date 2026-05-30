// npx jest src/activate/__tests__/SettingsCommands.test.ts

import * as vscode from "vscode"
import { TaskManager } from "../../core/TaskManager"
import { ProviderSettingsManager } from "../../core/config/ProviderSettingsManager"
import { ContextProxy } from "../../core/config/ContextProxy"
import { EXTENSION_NAME } from "../../shared/config/thea-config"
import * as registerCommands from "../../activate/registerCommands"

// Mock VSCode APIs
const mockInputBox = jest.fn()
const mockShowQuickPick = jest.fn()
const mockExecuteCommand = jest.fn()
const mockShowInformationMessage = jest.fn()
const mockShowErrorMessage = jest.fn()

jest.mock("vscode", () => ({
	window: {
		showInputBox: mockInputBox,
		showQuickPick: mockShowQuickPick,
		showInformationMessage: mockShowInformationMessage,
		showErrorMessage: mockShowErrorMessage,
	},
	commands: {
		executeCommand: mockExecuteCommand,
	},
	Uri: {
		parse: jest.fn(),
	},
	env: {
		openExternal: jest.fn(),
	},
}))

describe("Settings Commands - API Provider Configuration", () => {
	let taskManager: TaskManager
	let providerSettingsManager: ProviderSettingsManager
	let contextProxy: ContextProxy

	beforeEach(() => {
		jest.clearAllMocks()

		// Create mock context
		const mockExtensionContext = {
			secrets: {
				get: jest.fn(),
				store: jest.fn(),
				delete: jest.fn(),
			},
			globalState: {
				get: jest.fn(),
				update: jest.fn(),
			},
		} as any

		contextProxy = new ContextProxy(mockExtensionContext)
		providerSettingsManager = new ProviderSettingsManager(mockExtensionContext)

		taskManager = {
			providerSettingsManager,
			contextProxy,
		} as any
	})

	describe("API Provider QuickPick", () => {
		let showApiProviderQuickPick: any

		beforeEach(() => {
			showApiProviderQuickPick =
				(registerCommands as any).showApiProviderQuickPick ??
				(registerCommands as any).default?.showApiProviderQuickPick
			if (typeof showApiProviderQuickPick !== "function") {
				throw new Error("showApiProviderQuickPick is not exported as a function from ../../activate/registerCommands")
			}
		})

		it("should show API provider options", async () => {
			mockShowQuickPick.mockResolvedValue(undefined)

			await showApiProviderQuickPick(taskManager)

			expect(mockShowQuickPick).toHaveBeenCalled()
			const quickPickOptions = mockShowQuickPick.mock.calls[0][0]
			expect(Array.isArray(quickPickOptions)).toBe(true)

			// Verify all providers are included
			const providerLabels = quickPickOptions.map((opt: any) => opt.label)
			expect(providerLabels).toContain("$(cloud) Ollama (Local)")
			expect(providerLabels).toContain("$(cloud) Ollama (Remote)")
			expect(providerLabels).toContain("$(key) Anthropic")
			expect(providerLabels).toContain("$(key) OpenAI")
			expect(providerLabels).toContain("$(globe) OpenRouter")
			expect(providerLabels).toContain("$(server) LM Studio")
		})

		it("should handle provider selection cancellation", async () => {
			mockShowQuickPick.mockResolvedValue(undefined)

			await expect(showApiProviderQuickPick(taskManager)).resolves.not.toThrow()

			expect(mockInputBox).not.toHaveBeenCalled()
		})

		it("should configure Ollama provider when selected", async () => {
			// Mock user selecting Ollama (Local)
			mockShowQuickPick.mockResolvedValue({
				label: "$(cloud) Ollama (Local)",
				value: "ollama",
			})

			// Mock user entering model name
			mockInputBox.mockResolvedValue("glm4")

			// Mock saveConfig
			jest.spyOn(providerSettingsManager, "saveConfig").mockResolvedValue()

			// Mock updateGlobalState
			jest.spyOn(contextProxy, "updateGlobalState").mockResolvedValue()

			await showApiProviderQuickPick(taskManager)

			expect(mockShowQuickPick).toHaveBeenCalled()
			expect(mockInputBox).toHaveBeenCalledWith(
				expect.objectContaining({
					prompt: "Enter model name (e.g., llama3.2, glm4, qwen2.5)",
					value: "glm4",
				}),
			)

			expect(providerSettingsManager.saveConfig).toHaveBeenCalledWith(
				"ollama",
				expect.objectContaining({
					apiProvider: "ollama",
					ollamaBaseUrl: "http://localhost:11434",
					ollamaModelId: "glm4",
				}),
			)

			expect(contextProxy.updateGlobalState).toHaveBeenCalledWith("currentApiConfigName", "ollama")
			expect(mockShowInformationMessage).toHaveBeenCalledWith("Configured Ollama with model: glm4")
		})

		it("should configure Anthropic provider with API key when selected", async () => {
			// Mock user selecting Anthropic
			mockShowQuickPick.mockResolvedValue({
				label: "$(key) Anthropic",
				value: "anthropic",
			})

			// Mock user entering API key
			mockInputBox.mockResolvedValue("sk-ant-test123")

			// Mock saveConfig
			jest.spyOn(providerSettingsManager, "saveConfig").mockResolvedValue()
			jest.spyOn(contextProxy, "updateGlobalState").mockResolvedValue()

			await showApiProviderQuickPick(taskManager)

			expect(mockInputBox).toHaveBeenCalledWith(
				expect.objectContaining({
					prompt: "Anthropic API Key",
					password: true,
					placeHolder: "sk-ant-...",
				}),
			)

			expect(providerSettingsManager.saveConfig).toHaveBeenCalledWith(
				"anthropic",
				expect.objectContaining({
					apiProvider: "anthropic",
					anthropicApiKey: "sk-ant-test123",
				}),
			)

			expect(contextProxy.updateGlobalState).toHaveBeenCalledWith("currentApiConfigName", "anthropic")
			expect(mockShowInformationMessage).toHaveBeenCalledWith("Configured anthropic provider")
		})

		it("should handle configuration cancellation", async () => {
			// Mock user selecting Ollama then cancelling input
			mockShowQuickPick.mockResolvedValue({
				label: "$(cloud) Ollama (Local)",
				value: "ollama",
			})

			mockInputBox.mockResolvedValue(undefined) // User cancels

			jest.spyOn(providerSettingsManager, "saveConfig").mockResolvedValue()
			jest.spyOn(contextProxy, "updateGlobalState").mockResolvedValue()

			await showApiProviderQuickPick(taskManager)

			expect(providerSettingsManager.saveConfig).not.toHaveBeenCalled()
			expect(contextProxy.updateGlobalState).not.toHaveBeenCalled()
		})

		it("should handle configuration errors gracefully", async () => {
			mockShowQuickPick.mockResolvedValue({
				label: "$(key) OpenAI",
				value: "openai",
			})

			mockInputBox.mockResolvedValue("sk-test123")

			// Mock saveConfig to throw error
			jest.spyOn(providerSettingsManager, "saveConfig").mockRejectedValue(new Error("Failed to save config"))

			jest.spyOn(contextProxy, "updateGlobalState").mockResolvedValue()

			await showApiProviderQuickPick(taskManager)

			expect(mockShowErrorMessage).toHaveBeenCalledWith("Failed to configure openai: Failed to save config")
		})
	})
})
