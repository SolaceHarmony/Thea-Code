import * as vscode from "vscode"
import delay from "delay"

import { TheaProvider } from "../core/webview/TheaProvider" // Renamed import
import { EXTENSION_NAME, HOMEPAGE_URL, COMMANDS } from "../shared/config/thea-config"
import { supportPrompt } from "../shared/support-prompt"
import type { McpServer } from "../shared/mcp"
import { importSettings, exportSettings } from "../core/config/importExport"
import type { GlobalState } from "../schemas"

import { handleNewTask } from "./handleTask"

export type RegisterCommandOptions = {
	context: vscode.ExtensionContext
	outputChannel: vscode.OutputChannel
	provider: TheaProvider
}

export const registerCommands = ({ context, outputChannel, provider }: RegisterCommandOptions) => {
const commandHandlers: Record<string, () => Promise<unknown> | unknown> = {
		[`${EXTENSION_NAME}.activationCompleted`]: () => {},
		[COMMANDS.PLUS_BUTTON]: async () => {
			await vscode.commands.executeCommand("thea-code.newTask")
		},
		[COMMANDS.NEW_TASK]: handleNewTask,
		[COMMANDS.MCP_BUTTON]: async () => {
			await showMcpServerQuickPick(provider)
		},
		[COMMANDS.PROMPTS_BUTTON]: async () => {
			await showPromptQuickPick(provider)
		},
		[COMMANDS.HISTORY_BUTTON]: async () => {
			await showHistoryQuickPick(provider)
		},
		[COMMANDS.SETTINGS_BUTTON]: async () => {
			await showSettingsQuickPick(provider)
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
			await vscode.commands.executeCommand("thea-code.chat.respond", provider.getCurrent()?.taskId || "", "Add to context")
		},
		[COMMANDS.EXPLAIN_CODE]: async () => {
			await vscode.commands.executeCommand("thea-code.chat.respond", provider.getCurrent()?.taskId || "", "Explain code")
		},
		[COMMANDS.FIX_CODE]: async () => {
			await vscode.commands.executeCommand("thea-code.chat.respond", provider.getCurrent()?.taskId || "", "Fix code")
		},
		[COMMANDS.IMPROVE_CODE]: async () => {
			await vscode.commands.executeCommand("thea-code.chat.respond", provider.getCurrent()?.taskId || "", "Improve code")
		},
	}

	for (const [command, handler] of Object.entries(commandHandlers)) {
		context.subscriptions.push(vscode.commands.registerCommand(command, handler))
	}

	context.subscriptions.push(
		vscode.commands.registerCommand(COMMANDS.TERMINAL_ADD_TO_CONTEXT, async () => {
			await vscode.commands.executeCommand("thea-code.chat.respond", provider.getCurrent()?.taskId || "", "Add terminal to context")
		}),
	)
	context.subscriptions.push(
		vscode.commands.registerCommand(COMMANDS.TERMINAL_FIX_COMMAND, async () => {
			await vscode.commands.executeCommand("thea-code.chat.respond", provider.getCurrent()?.taskId || "", "Fix terminal command")
		}),
	)
	context.subscriptions.push(
		vscode.commands.registerCommand(COMMANDS.TERMINAL_EXPLAIN_COMMAND, async () => {
			await vscode.commands.executeCommand("thea-code.chat.respond", provider.getCurrent()?.taskId || "", "Explain terminal command")
		}),
	)
	context.subscriptions.push(
		vscode.commands.registerCommand(COMMANDS.TERMINAL_FIX_COMMAND_IN_CURRENT_TASK, async () => {
			await vscode.commands.executeCommand("thea-code.chat.respond", provider.getCurrent()?.taskId || "", "Fix terminal command (current task)")
		}),
	)
	context.subscriptions.push(
		vscode.commands.registerCommand(COMMANDS.TERMINAL_EXPLAIN_COMMAND_IN_CURRENT_TASK, async () => {
			await vscode.commands.executeCommand("thea-code.chat.respond", provider.getCurrent()?.taskId || "", "Explain terminal command (current task)")
		}),
	)
}

interface SettingsQuickPickItem extends vscode.QuickPickItem {
	run: () => Promise<void>
	keepOpen?: boolean
}

async function showHistoryQuickPick(provider: TheaProvider) {
	try {
		const state = await provider.getStateToPostToWebview()
		const history = state.taskHistory ?? []

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
					label: item.title || item.task || "Untitled task",
					description: date.toLocaleString(),
					detail: item.task,
					taskId: item.id,
				}
			})

		const selection = await vscode.window.showQuickPick(items, {
			placeHolder: "Select a task to reopen",
		})

		if (!selection?.taskId) {
			return
		}

		await provider.showTaskWithId(selection.taskId)
		await vscode.window.showInformationMessage(`Reopened task ${selection.label}`)
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error)
		await vscode.window.showErrorMessage(`Unable to show task history: ${message}`)
	}
}

async function showPromptQuickPick(provider: TheaProvider) {
	try {
		const state = await provider.getStateToPostToWebview()
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
			provider.getCurrent()?.taskId || "",
			selection.template,
		)
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error)
		await vscode.window.showErrorMessage(`Unable to show prompts: ${message}`)
	}
}

async function showMcpServerQuickPick(provider: TheaProvider) {
	try {
		const servers: McpServer[] = provider.theaMcpManagerInstance.getAllServers()

		if (servers.length === 0) {
			await vscode.window.showInformationMessage("No MCP servers configured.")
			return
		}

		const items = servers.map((server) => ({
			label: server.name,
			description: server.status === "connected" ? "Connected" : server.status,
			detail: server.config,
			server,
		}))

		const selection = await vscode.window.showQuickPick(items, {
			placeHolder: "Select an MCP server",
		})

		if (!selection) {
			return
		}

		const summary = `MCP Server: ${selection.label}\nStatus: ${selection.description}\nConfig: ${selection.detail}`
		await vscode.commands.executeCommand(
			"thea-code.chat.respond",
			provider.getCurrent()?.taskId || "",
			summary,
		)
	} catch (error) {
		const message = error instanceof Error ? error.message : String(error)
		await vscode.window.showErrorMessage(`Unable to show MCP servers: ${message}`)
	}
}

async function showSettingsQuickPick(provider: TheaProvider) {
	const toggleDescriptors: Array<{ key: keyof GlobalState; label: string; detail?: string }> = [
		{ key: "autoApprovalEnabled", label: "Auto-approve follow-up actions" },
		{ key: "alwaysApproveResubmit", label: "Always approve retry requests" },
		{ key: "alwaysAllowReadOnly", label: "Always allow read-only operations" },
		{ key: "alwaysAllowReadOnlyOutsideWorkspace", label: "Always allow read-only outside workspace" },
		{ key: "alwaysAllowWrite", label: "Always allow write operations" },
		{ key: "alwaysAllowWriteOutsideWorkspace", label: "Always allow write outside workspace" },
		{ key: "alwaysAllowExecute", label: "Always allow command execution" },
		{ key: "alwaysAllowBrowser", label: "Always allow browser control" },
		{ key: "alwaysAllowMcp", label: "Always allow MCP access" },
		{ key: "alwaysAllowModeSwitch", label: "Always allow mode switches" },
		{ key: "alwaysAllowSubtasks", label: "Always allow subtasks" },
		{ key: "browserToolEnabled", label: "Enable browser tool" },
		{ key: "mcpEnabled", label: "Enable MCP support" },
		{ key: "enableMcpServerCreation", label: "Allow MCP server creation" },
		{ key: "enableCheckpoints", label: "Enable checkpoints" },
		{ key: "diffEnabled", label: "Enable diff viewer" },
		{ key: "soundEnabled", label: "Enable sounds" },
		{ key: "ttsEnabled", label: "Enable text-to-speech" },
		{ key: "remoteBrowserEnabled", label: "Enable remote browser" },
		{ key: "showTheaIgnoredFiles", label: "Show Thea ignored files" },
	]

	while (true) {
		let state: Awaited<ReturnType<TheaProvider["getStateToPostToWebview"]>>
		try {
			state = await provider.getStateToPostToWebview()
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error)
			await vscode.window.showErrorMessage(`Unable to load settings: ${message}`)
			return
		}

		const items: SettingsQuickPickItem[] = []

		items.push(
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
						providerSettingsManager: provider.providerSettingsManager,
						contextProxy: provider.contextProxy,
					})

					if (success) {
						provider.settingsImportedAt = Date.now()
						await provider.postStateToWebview()
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
						providerSettingsManager: provider.providerSettingsManager,
						contextProxy: provider.contextProxy,
					})
					await vscode.window.showInformationMessage("Settings exported.")
				},
				keepOpen: true,
			},
			{
				label: "$(trash) Reset Thea state",
				detail: "Clear stored state and custom modes",
				run: async () => {
					await provider.resetState()
				},
			},
		)

		for (const descriptor of toggleDescriptors) {
			const currentValue = Boolean((state as Record<string, unknown>)[descriptor.key])
			const icon = currentValue ? "$(check)" : "$(circle-large-outline)"
			const description = currentValue ? "Enabled" : "Disabled"

			items.push({
				label: `${icon} ${descriptor.label}`,
				description,
				detail: descriptor.detail,
				run: async () => {
					const nextValue = (!currentValue) as GlobalState[typeof descriptor.key]
					await provider.updateGlobalState(descriptor.key, nextValue)
					await provider.postStateToWebview()
					const status = nextValue ? "enabled" : "disabled"
					await vscode.window.showInformationMessage(`${descriptor.label} ${status}.`)
				},
				keepOpen: true,
			})
		}

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
