/**
 * Command registration using native VS Code APIs.
 * Uses TaskManager instead of TheaProvider for webview-free operation.
 */
import * as vscode from "vscode"
import fs from "fs/promises"
import * as path from "path"

import { TaskManager } from "../core/TaskManager"
import { EXTENSION_NAME, HOMEPAGE_URL, COMMANDS, EXTENSION_CONFIG_DIR } from "../shared/config/thea-config"
import { supportPrompt } from "../shared/support-prompt"
import { importSettings, exportSettings } from "../core/config/importExport"
import type { GlobalState } from "../schemas"

import { handleNewTask } from "./handleTask"

export type RegisterCommandOptions = {
	context: vscode.ExtensionContext
	outputChannel: vscode.OutputChannel
	taskManager: TaskManager
}

export const registerCommands = ({ context, taskManager }: RegisterCommandOptions) => {
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	const commandHandlers: Record<string, (...args: any[]) => Promise<void> | void> = {
		[`${EXTENSION_NAME}.activationCompleted`]: () => {},
		[COMMANDS.PLUS_BUTTON]: async () => {
			await vscode.commands.executeCommand("thea-code.newTask")
		},
		[COMMANDS.NEW_TASK]: handleNewTask,
		[COMMANDS.MCP_BUTTON]: async () => {
			await showMcpServerQuickPick(taskManager)
		},
		[COMMANDS.PROMPTS_BUTTON]: async () => {
			await showPromptQuickPick(taskManager)
		},
		[COMMANDS.HISTORY_BUTTON]: async () => {
			await showHistoryQuickPick(taskManager)
		},
		[COMMANDS.SETTINGS_BUTTON]: async () => {
			await showSettingsQuickPick(taskManager)
		},
		[COMMANDS.POPOUT_BUTTON]: async () => {
			await vscode.commands.executeCommand("workbench.action.openChat.view")
		},
		[COMMANDS.OPEN_NEW_TAB]: async () => {
			await vscode.commands.executeCommand("workbench.action.openChat.view")
		},
		[COMMANDS.HELP_BUTTON]: async () => {
			await vscode.env.openExternal(vscode.Uri.parse(HOMEPAGE_URL))
		},
		[COMMANDS.SET_CUSTOM_STORAGE_PATH]: async () => {
			const { promptForCustomStoragePath } = await import("../shared/storagePathManager")
			await promptForCustomStoragePath()
		},
		[COMMANDS.ADD_TO_CONTEXT]: async () => {
			await vscode.commands.executeCommand(
				"thea-code.chat.respond",
				taskManager.getCurrent()?.taskId || "",
				"Add to context",
			)
		},
		[COMMANDS.EXPLAIN_CODE]: async () => {
			await vscode.commands.executeCommand(
				"thea-code.chat.respond",
				taskManager.getCurrent()?.taskId || "",
				"Explain code",
			)
		},
		[COMMANDS.FIX_CODE]: async () => {
			await vscode.commands.executeCommand(
				"thea-code.chat.respond",
				taskManager.getCurrent()?.taskId || "",
				"Fix code",
			)
		},
		[COMMANDS.IMPROVE_CODE]: async () => {
			await vscode.commands.executeCommand(
				"thea-code.chat.respond",
				taskManager.getCurrent()?.taskId || "",
				"Improve code",
			)
		},
	}

	for (const [command, handler] of Object.entries(commandHandlers)) {
		context.subscriptions.push(vscode.commands.registerCommand(command, handler))
	}

	context.subscriptions.push(
		vscode.commands.registerCommand(COMMANDS.TERMINAL_ADD_TO_CONTEXT, async () => {
			await vscode.commands.executeCommand(
				"thea-code.chat.respond",
				taskManager.getCurrent()?.taskId || "",
				"Add terminal to context",
			)
		}),
	)
	context.subscriptions.push(
		vscode.commands.registerCommand(COMMANDS.TERMINAL_FIX, async () => {
			await vscode.commands.executeCommand(
				"thea-code.chat.respond",
				taskManager.getCurrent()?.taskId || "",
				"Fix terminal command",
			)
		}),
	)
	context.subscriptions.push(
		vscode.commands.registerCommand(COMMANDS.TERMINAL_EXPLAIN, async () => {
			await vscode.commands.executeCommand(
				"thea-code.chat.respond",
				taskManager.getCurrent()?.taskId || "",
				"Explain terminal command",
			)
		}),
	)
	context.subscriptions.push(
		vscode.commands.registerCommand(COMMANDS.TERMINAL_FIX_CURRENT, async () => {
			await vscode.commands.executeCommand(
				"thea-code.chat.respond",
				taskManager.getCurrent()?.taskId || "",
				"Fix terminal command (current task)",
			)
		}),
	)
	context.subscriptions.push(
		vscode.commands.registerCommand(COMMANDS.TERMINAL_EXPLAIN_CURRENT, async () => {
			await vscode.commands.executeCommand(
				"thea-code.chat.respond",
				taskManager.getCurrent()?.taskId || "",
				"Explain terminal command (current task)",
			)
		}),
	)
}

// Legacy panel tracking - no-op during migration
export function setPanel(_panel: vscode.WebviewPanel | vscode.WebviewView | undefined, _type: "sidebar" | "tab"): void {
	// No-op - native VS Code Chat API doesn't use panels
}

interface RunnableQuickPickItem extends vscode.QuickPickItem {
	run: () => Promise<void | boolean>
	keepOpen?: boolean
	isCategory?: boolean
}

type SettingsQuickPickItem = RunnableQuickPickItem

async function showHistoryQuickPick(taskManager: TaskManager) {
	try {
		const history = taskManager.contextProxy.getGlobalState("taskHistory") ?? []

		if (history.length === 0) {
			await vscode.window.showInformationMessage("No previous tasks found.")
			return
		}

		const items = history
			.slice()
			.sort((a, b) => b.ts - a.ts)
			.map((item) => {
				const date = new Date(item.ts)
				return {
					label: item.task || "Untitled task",
					description: date.toLocaleString(),
					detail: `Task #${item.number}`,
					taskId: item.id,
				}
			})

		const selection = await vscode.window.showQuickPick(items, {
			placeHolder: "Select a task to reopen",
		})

		if (!selection?.taskId) {
			return
		}

		// Find the history item and create a task from it
		const historyItem = history.find((h) => h.id === selection.taskId)
		if (historyItem) {
			await taskManager.createTaskFromHistory(historyItem)
			await vscode.window.showInformationMessage(`Reopened task ${selection.label}`)
		}
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error)
		await vscode.window.showErrorMessage(`Unable to show task history: ${message}`)
	}
}

async function showPromptQuickPick(taskManager: TaskManager) {
	try {
		const state = await taskManager.getStateManager().getState()
		const mergedPrompts = {
			...supportPrompt.default,
			...(state.customSupportPrompts ?? {}),
		} as Record<string, string | undefined>

		const items = Object.entries(mergedPrompts)
			.filter(([, template]) => typeof template === "string" && template.trim().length > 0)
			.map(([key, template]) => ({
				label: key,
				template: template!,
				detail: template!.split("\n")[0],
			}))

		if (items.length === 0) {
			await vscode.window.showInformationMessage("No prompts available.")
			return
		}

		const selection = await vscode.window.showQuickPick(items, {
			placeHolder: "Select a prompt to insert",
		})

		if (!selection) {
			return
		}

		await vscode.commands.executeCommand(
			"thea-code.chat.respond",
			taskManager.getCurrent()?.taskId || "",
			selection.template,
		)
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error)
		await vscode.window.showErrorMessage(`Unable to show prompts: ${message}`)
	}
}

async function showMcpServerQuickPick(_taskManager: TaskManager) {
	// MCP server management - simplified for now
	// Full MCP management will be added through McpHub access
	const items: RunnableQuickPickItem[] = [
		{
			label: "$(gear) Open global MCP settings",
			detail: "Open the extension-level MCP configuration file.",
			run: async () => {
				await openGlobalMcpSettings()
			},
			keepOpen: true,
		},
		{
			label: "$(file-code) Open workspace MCP settings",
			detail: "Open or create .thea/mcp.json in the current workspace.",
			run: async () => {
				await openProjectMcpSettings()
			},
			keepOpen: true,
		},
	]

	const selection = await vscode.window.showQuickPick(items, {
		placeHolder: "Manage MCP servers",
		ignoreFocusOut: true,
	})

	if (!selection) {
		return
	}

	try {
		await selection.run()
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error)
		await vscode.window.showErrorMessage(`Unable to perform MCP action: ${message}`)
	}
}

async function showSettingsQuickPick(taskManager: TaskManager) {
	const securitySettings: Array<{ key: keyof GlobalState; label: string; detail?: string }> = [
		{
			key: "autoApprovalEnabled",
			label: "Auto-approve follow-up actions",
			detail: "Automatically approve follow-up questions and actions",
		},
		{
			key: "alwaysApproveResubmit",
			label: "Always approve retry requests",
			detail: "Skip approval for retry attempts",
		},
		{
			key: "alwaysAllowSubtasks",
			label: "Always allow subtasks",
			detail: "Automatically approve spawning of subtasks",
		},
		{
			key: "alwaysAllowModeSwitch",
			label: "Always allow mode switches",
			detail: "Automatically approve mode transitions",
		},
	]

	const permissionSettings: Array<{ key: keyof GlobalState; label: string; detail?: string }> = [
		{
			key: "alwaysAllowReadOnly",
			label: "Always allow read-only operations",
			detail: "Skip approval for read operations",
		},
		{
			key: "alwaysAllowReadOnlyOutsideWorkspace",
			label: "Always allow read-only outside workspace",
			detail: "Read files outside project directory",
		},
		{
			key: "alwaysAllowWrite",
			label: "Always allow write operations",
			detail: "Skip approval for write operations",
		},
		{
			key: "alwaysAllowWriteOutsideWorkspace",
			label: "Always allow write outside workspace",
			detail: "Write files outside project directory",
		},
		{
			key: "alwaysAllowExecute",
			label: "Always allow command execution",
			detail: "Skip approval for terminal commands",
		},
		{
			key: "alwaysAllowBrowser",
			label: "Always allow browser control",
			detail: "Skip approval for browser automation",
		},
		{ key: "alwaysAllowMcp", label: "Always allow MCP access", detail: "Skip approval for MCP tool calls" },
	]

	const featureSettings: Array<{ key: keyof GlobalState; label: string; detail?: string }> = [
		{
			key: "browserToolEnabled",
			label: "Enable browser tool",
			detail: "Enable web scraping and browser automation",
		},
		{ key: "mcpEnabled", label: "Enable MCP support", detail: "Enable MCP (Model Context Protocol) integrations" },
		{
			key: "enableMcpServerCreation",
			label: "Allow MCP server creation",
			detail: "Create new MCP servers during tasks",
		},
		{ key: "enableCheckpoints", label: "Enable checkpoints", detail: "Create automatic task checkpoints" },
		{ key: "diffEnabled", label: "Enable diff viewer", detail: "Show file comparison before changes" },
		{ key: "soundEnabled", label: "Enable sounds", detail: "Play notification sounds" },
		{ key: "ttsEnabled", label: "Enable text-to-speech", detail: "Read responses aloud" },
		{ key: "remoteBrowserEnabled", label: "Enable remote browser", detail: "Use remote browser for automation" },
		{ key: "showTheaIgnoredFiles", label: "Show Thea ignored files", detail: "Display files ignored by Thea" },
	]

	while (true) {
		const items: SettingsQuickPickItem[] = []

		items.push(
			{
				label: "$(plug) Configure API Provider",
				detail: "Set up Ollama, OpenAI, Anthropic, or other AI providers",
				run: async () => {
					await showApiProviderQuickPick(taskManager)
				},
				keepOpen: true,
			},
			{
				label: "$(gear) Open extension settings",
				detail: "Manage settings via VS Code's Settings UI",
				run: async () => {
					await vscode.commands.executeCommand("workbench.action.openSettings", "thea-code")
				},
			},
			{
				label: "$(cloud-download) Import settings from file",
				detail: "Merge settings from a saved JSON export",
				run: async () => {
					const { success } = await importSettings({
						providerSettingsManager: taskManager.providerSettingsManager,
						contextProxy: taskManager.contextProxy,
					})

					if (success) {
						await vscode.window.showInformationMessage("Settings imported successfully.")
					}
				},
				keepOpen: true,
			},
			{
				label: "$(cloud-upload) Export settings to file",
				detail: "Save provider profiles and global settings to disk",
				run: async () => {
					await exportSettings({
						providerSettingsManager: taskManager.providerSettingsManager,
						contextProxy: taskManager.contextProxy,
					})
					await vscode.window.showInformationMessage("Settings exported.")
				},
				keepOpen: true,
			},
			{
				label: "$(trash) Reset Thea state",
				detail: "Clear stored state and custom modes",
				run: async () => {
					await taskManager.contextProxy.resetAllState()
					await vscode.window.showInformationMessage("State reset successfully.")
				},
			},
		)

		// Category: Security Settings
		items.push({
			label: "$(shield) Security Settings",
			detail: "Auto-approval and workflow settings",
			isCategory: true,
			run: async () => {
				await showCategoryQuickPick(taskManager, "security", securitySettings)
			},
			keepOpen: true,
		})

		// Category: Permission Settings
		items.push({
			label: "$(key) Permission Settings",
			detail: "Control what operations Thea can perform",
			isCategory: true,
			run: async () => {
				await showCategoryQuickPick(taskManager, "permissions", permissionSettings)
			},
			keepOpen: true,
		})

		// Category: Feature Settings
		items.push({
			label: "$(tools) Feature Settings",
			detail: "Enable or disable specific features",
			isCategory: true,
			run: async () => {
				await showCategoryQuickPick(taskManager, "features", featureSettings)
			},
			keepOpen: true,
		})

		// VS Code Settings
		items.push({
			label: "$(settings-gear) Open VS Code Settings",
			detail: "Edit Thea configuration in settings.json",
			run: async () => {
				await vscode.commands.executeCommand("workbench.action.openSettings", "thea-code")
			},
		})

		const selection = await vscode.window.showQuickPick<SettingsQuickPickItem>(items, {
			placeHolder: "Thea settings",
			ignoreFocusOut: true,
		})

		if (!selection) {
			return
		}

		try {
			await selection.run()
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error)
			await vscode.window.showErrorMessage(`Unable to apply setting: ${message}`)
		}

		if (!selection.keepOpen) {
			return
		}
	}
}

async function showCategoryQuickPick(
	taskManager: TaskManager,
	categoryName: string,
	settings: Array<{ key: keyof GlobalState; label: string; detail?: string }>,
) {
	// Store the parent state at the time the category was entered
	const stateSnapshot = await taskManager.getStateManager().getState()

	while (true) {
		const items: SettingsQuickPickItem[] = []

		for (const descriptor of settings) {
			const currentValue = Boolean((stateSnapshot as Record<string, unknown>)[descriptor.key])
			const icon = currentValue ? "$(check)" : "$(circle-large-outline)"
			const description = currentValue ? "Enabled" : "Disabled"

			items.push({
				label: `${icon} ${descriptor.label}`,
				description,
				detail: descriptor.detail || "",
				run: async () => {
					const nextValue = !currentValue
					await taskManager.contextProxy.updateGlobalState(descriptor.key, nextValue)
					return nextValue
				},
				keepOpen: true,
			})
		}

		const selection = await vscode.window.showQuickPick<SettingsQuickPickItem>(items, {
			placeHolder: `${categoryName} settings`,
			ignoreFocusOut: true,
		})

		if (!selection) {
			// Back to main menu
			return
		}

		try {
			// NOTE: Run the toggle and get the new value
			const newValue = (await selection.run()) as boolean
			// Update the snapshot for the next iteration if we stay in this view
			;(stateSnapshot as Record<string, unknown>)[settings.find((s) => selection.label.includes(s.label))!.key] =
				newValue

			if (!selection.keepOpen) {
				return
			}
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error)
			await vscode.window.showErrorMessage(`Unable to apply setting: ${message}`)
		}
	}
}

async function openGlobalMcpSettings() {
	// Get global storage path for MCP settings
	const homeDir = process.env.HOME || process.env.USERPROFILE || ""
	const globalMcpPath = path.join(homeDir, ".thea", "mcp.json")

	try {
		await fs.mkdir(path.dirname(globalMcpPath), { recursive: true })
		await openFileInEditor(globalMcpPath, JSON.stringify({ mcpServers: {} }, null, 2))
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error)
		await vscode.window.showErrorMessage(`Unable to open global MCP settings: ${message}`)
	}
}

async function openProjectMcpSettings() {
	if (!vscode.workspace.workspaceFolders?.length) {
		await vscode.window.showErrorMessage("Open a workspace to manage project MCP settings.")
		return
	}

	const workspaceFolder = vscode.workspace.workspaceFolders[0]
	const configDir = path.join(workspaceFolder.uri.fsPath, EXTENSION_CONFIG_DIR)
	const mcpPath = path.join(configDir, "mcp.json")

	try {
		await fs.mkdir(configDir, { recursive: true })
		await openFileInEditor(mcpPath, JSON.stringify({ mcpServers: {} }, null, 2))
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error)
		await vscode.window.showErrorMessage(`Unable to open project MCP settings: ${message}`)
	}
}

async function openFileInEditor(filePath: string, defaultContent: string) {
	try {
		await fs.access(filePath)
	} catch {
		await fs.writeFile(filePath, defaultContent, "utf-8")
	}

	const doc = await vscode.workspace.openTextDocument(filePath)
	await vscode.window.showTextDocument(doc)
}

async function showApiProviderQuickPick(taskManager: TaskManager) {
	const providers: Array<{ label: string; value: string; detail: string }> = [
		{ label: "$(cloud) Ollama (Local)", value: "ollama", detail: "Run models locally with Ollama" },
		{ label: "$(cloud) Ollama (Remote)", value: "ollama-remote", detail: "Connect to a remote Ollama server" },
		{ label: "$(key) Anthropic", value: "anthropic", detail: "Use Claude models via Anthropic API" },
		{ label: "$(key) OpenAI", value: "openai", detail: "Use GPT models via OpenAI API" },
		{ label: "$(globe) OpenRouter", value: "openrouter", detail: "Access multiple providers via OpenRouter" },
		{ label: "$(server) LM Studio", value: "lmstudio", detail: "Use LM Studio local server" },
	]

	const selection = await vscode.window.showQuickPick(providers, {
		placeHolder: "Select an API provider",
	})

	if (!selection) {
		return
	}

	if (selection.value === "ollama" || selection.value === "ollama-remote") {
		await configureOllama(taskManager, selection.value === "ollama-remote")
	} else if (selection.value === "anthropic") {
		await configureWithApiKey(taskManager, "anthropic", "Anthropic API Key", "sk-ant-...")
	} else if (selection.value === "openai") {
		await configureWithApiKey(taskManager, "openai", "OpenAI API Key", "sk-...")
	} else if (selection.value === "openrouter") {
		await configureWithApiKey(taskManager, "openrouter", "OpenRouter API Key", "sk-or-...")
	} else if (selection.value === "lmstudio") {
		await configureLmStudio(taskManager)
	}
}

async function configureOllama(taskManager: TaskManager, isRemote: boolean) {
	// Get base URL
	let baseUrl = "http://localhost:11434"
	if (isRemote) {
		const url = await vscode.window.showInputBox({
			prompt: "Enter Ollama server URL",
			value: "http://localhost:11434",
			placeHolder: "http://your-server:11434",
		})
		if (!url) {
			return
		}
		baseUrl = url
	}

	// Get model name
	const modelId = await vscode.window.showInputBox({
		prompt: "Enter model name (e.g., llama3.2, glm4, qwen2.5)",
		value: "glm4",
		placeHolder: "Model name from 'ollama list'",
	})

	if (!modelId) {
		return
	}

	// Save the configuration
	try {
		await taskManager.providerSettingsManager.saveConfig("ollama", {
			apiProvider: "ollama",
			ollamaBaseUrl: baseUrl,
			ollamaModelId: modelId,
		})

		await taskManager.contextProxy.updateGlobalState("currentApiConfigName", "ollama")
		await vscode.window.showInformationMessage(`Configured Ollama with model: ${modelId}`)
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error)
		await vscode.window.showErrorMessage(`Failed to configure Ollama: ${message}`)
	}
}

async function configureWithApiKey(
	taskManager: TaskManager,
	provider: "anthropic" | "openai" | "openrouter",
	keyPrompt: string,
	placeholder: string,
) {
	const apiKey = await vscode.window.showInputBox({
		prompt: keyPrompt,
		password: true,
		placeHolder: placeholder,
	})

	if (!apiKey) {
		return
	}

	try {
		const config: Record<string, unknown> = { apiProvider: provider }

		if (provider === "anthropic") {
			config.anthropicApiKey = apiKey
		} else if (provider === "openai") {
			config.openAiApiKey = apiKey
		} else if (provider === "openrouter") {
			config.openRouterApiKey = apiKey
		}

		await taskManager.providerSettingsManager.saveConfig(provider, config)
		await taskManager.contextProxy.updateGlobalState("currentApiConfigName", provider)
		await vscode.window.showInformationMessage(`Configured ${provider} provider`)
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error)
		await vscode.window.showErrorMessage(`Failed to configure ${provider}: ${message}`)
	}
}

async function configureLmStudio(taskManager: TaskManager) {
	const baseUrl = await vscode.window.showInputBox({
		prompt: "Enter LM Studio server URL",
		value: "http://localhost:1234",
		placeHolder: "http://localhost:1234",
	})

	if (!baseUrl) {
		return
	}

	const modelId = await vscode.window.showInputBox({
		prompt: "Enter model name",
		placeHolder: "Model loaded in LM Studio",
	})

	if (!modelId) {
		return
	}

	try {
		await taskManager.providerSettingsManager.saveConfig("lmstudio", {
			apiProvider: "lmstudio",
			lmStudioBaseUrl: baseUrl,
			lmStudioModelId: modelId,
		})

		await taskManager.contextProxy.updateGlobalState("currentApiConfigName", "lmstudio")
		await vscode.window.showInformationMessage(`Configured LM Studio with model: ${modelId}`)
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error)
		await vscode.window.showErrorMessage(`Failed to configure LM Studio: ${message}`)
	}
}
