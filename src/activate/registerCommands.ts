import * as vscode from "vscode"
import fs from "fs/promises"
import * as path from "path"

import { TheaProvider } from "../core/webview/TheaProvider" // Renamed import
import { EXTENSION_NAME, HOMEPAGE_URL, COMMANDS, EXTENSION_CONFIG_DIR } from "../shared/config/thea-config"
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
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	const commandHandlers: Record<string, (...args: any[]) => Promise<unknown> | unknown> = {
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
		vscode.commands.registerCommand(COMMANDS.TERMINAL_FIX, async () => {
			await vscode.commands.executeCommand("thea-code.chat.respond", provider.getCurrent()?.taskId || "", "Fix terminal command")
		}),
	)
	context.subscriptions.push(
		vscode.commands.registerCommand(COMMANDS.TERMINAL_EXPLAIN, async () => {
			await vscode.commands.executeCommand("thea-code.chat.respond", provider.getCurrent()?.taskId || "", "Explain terminal command")
		}),
	)
	context.subscriptions.push(
		vscode.commands.registerCommand(COMMANDS.TERMINAL_FIX_CURRENT, async () => {
			await vscode.commands.executeCommand("thea-code.chat.respond", provider.getCurrent()?.taskId || "", "Fix terminal command (current task)")
		}),
	)
	context.subscriptions.push(
		vscode.commands.registerCommand(COMMANDS.TERMINAL_EXPLAIN_CURRENT, async () => {
			await vscode.commands.executeCommand("thea-code.chat.respond", provider.getCurrent()?.taskId || "", "Explain terminal command (current task)")
		}),
	)
}

// Legacy panel tracking - kept for backward compatibility during migration
// Will be removed once fully migrated to Chat Participant API
export function setPanel(
	_panel: vscode.WebviewPanel | vscode.WebviewView | undefined,
	_type: "sidebar" | "tab",
): void {
	// No-op during migration to native VS Code APIs
}

interface RunnableQuickPickItem extends vscode.QuickPickItem {
	run: () => Promise<void>
	keepOpen?: boolean
}

type SettingsQuickPickItem = RunnableQuickPickItem

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
	while (true) {
		try {
			const servers: McpServer[] = provider.theaMcpManagerInstance.getAllServers()
			const items: RunnableQuickPickItem[] = []

			items.push(
				{
					label: "$(gear) Open global MCP settings",
					detail: "Open the extension-level MCP configuration file.",
					run: async () => {
						await openGlobalMcpSettings(provider)
					},
					keepOpen: true,
				},
				{
					label: "$(file-code) Open workspace MCP settings",
					detail: "Open or create .thea/mcp.json in the current workspace.",
					run: async () => {
						await openProjectMcpSettings(provider)
					},
					keepOpen: true,
				},
			)

			if (servers.length === 0) {
				items.push({
					label: "No MCP servers configured",
					detail: "Use the settings files to add new servers.",
					run: async () => {},
				})
			} else {
				const serverItems = servers.map<RunnableQuickPickItem>((server) => ({
					label: `${server.disabled ? "$(circle-slash)" : "$(plug)"} ${server.name}`,
					description: `${server.status}${server.disabled ? " • disabled" : ""}`,
					detail: server.config,
					run: async () => {
						await showMcpServerActions(provider, server.name)
					},
					keepOpen: true,
				}))
				items.push(...serverItems)
			}

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

			if (!selection.keepOpen) {
				return
			}
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error)
			await vscode.window.showErrorMessage(`Unable to show MCP servers: ${message}`)
			return
		}
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

async function openGlobalMcpSettings(provider: TheaProvider) {
	const pathOrUndefined = await provider.theaMcpManagerInstance.getMcpSettingsFilePath()
	if (!pathOrUndefined) {
		await vscode.window.showWarningMessage("No global MCP settings file is available.")
		return
	}

	await openFileInEditor(pathOrUndefined, JSON.stringify({ mcpServers: {} }, null, 2))
}

async function openProjectMcpSettings(provider: TheaProvider) {
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

async function showMcpServerActions(provider: TheaProvider, serverName: string) {
	while (true) {
		const server = provider.theaMcpManagerInstance.getAllServers().find((s) => s.name === serverName)
		if (!server) {
			await vscode.window.showInformationMessage(`Server ${serverName} is no longer available.`)
			return
		}

		const items: RunnableQuickPickItem[] = []
		const isDisabled = Boolean(server.disabled)

		items.push({
			label: isDisabled ? "$(play) Enable server" : "$(circle-slash) Disable server",
			description: isDisabled ? "Currently disabled" : "Currently enabled",
			run: async () => {
				await provider.theaMcpManagerInstance.toggleServerDisabled(server.name, !isDisabled)
				await provider.postStateToWebview()
				await vscode.window.showInformationMessage(
					`Server ${server.name} ${isDisabled ? "enabled" : "disabled"}.`,
				)
			},
			keepOpen: true,
		})

		items.push({
			label: "$(refresh) Restart connection",
			description: server.status === "connecting" ? "Currently connecting" : undefined,
			run: async () => {
				await provider.theaMcpManagerInstance.restartConnection(server.name)
				await vscode.window.showInformationMessage(`Restarted connection for ${server.name}.`)
			},
			keepOpen: true,
		})

		if (server.tools?.length) {
			items.push({
				label: "$(checklist) Toggle tool approvals",
				detail: "Enable or disable always-allow for server tools.",
				run: async () => {
					await showMcpToolQuickPick(provider, server.name, server.source ?? "global")
				},
				keepOpen: true,
			})
		}

		items.push({
			label: "$(comment-discussion) Send summary to chat",
			detail: "Post server details into the active Thea task.",
			run: async () => {
				const summary = formatMcpServerSummary(server)
				await vscode.commands.executeCommand(
					"thea-code.chat.respond",
					provider.getCurrent()?.taskId || "",
					summary,
				)
			},
			keepOpen: true,
		})

		items.push({
			label: "$(trash) Delete server",
			detail: "Remove this server from configuration.",
			run: async () => {
				const choice = await vscode.window.showWarningMessage(
					`Delete MCP server ${server.name}?`,
					{ modal: true },
					"Delete",
				)
				if (choice !== "Delete") {
					return
				}
				await provider.theaMcpManagerInstance.deleteServer(server.name)
				await provider.postStateToWebview()
				await vscode.window.showInformationMessage(`Deleted MCP server ${server.name}.`)
			},
		})

		items.push({
			label: "$(arrow-left) Back",
			run: async () => {},
		})

		const selection = await vscode.window.showQuickPick(items, {
			placeHolder: `Manage ${server.name}`,
			ignoreFocusOut: true,
		})

		if (!selection) {
			return
		}

		try {
			await selection.run()
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error)
			await vscode.window.showErrorMessage(`Unable to update ${server.name}: ${message}`)
		}

		if (!selection.keepOpen) {
			return
		}
	}
}

async function showMcpToolQuickPick(
	provider: TheaProvider,
	serverName: string,
	source: "global" | "project",
) {
	while (true) {
		const server = provider.theaMcpManagerInstance.getAllServers().find((s) => s.name === serverName)
		if (!server) {
			await vscode.window.showInformationMessage(`Server ${serverName} is no longer available.`)
			return
		}

		const tools = server.tools ?? []
		if (tools.length === 0) {
			await vscode.window.showInformationMessage(`Server ${serverName} has no tools to manage.`)
			return
		}

		const hub = provider.theaMcpManagerInstance.getMcpHub()
		if (!hub) {
			await vscode.window.showErrorMessage("MCP hub is not ready; cannot toggle tools.")
			return
		}

		const items: RunnableQuickPickItem[] = tools.map((tool) => {
			const alwaysAllow = Boolean(tool.alwaysAllow)
			return {
				label: `${alwaysAllow ? "$(check)" : "$(circle-large-outline)"} ${tool.name}`,
				description: alwaysAllow ? "Always allow" : "Requires approval",
				detail: tool.description,
				run: async () => {
					await hub.toggleToolAlwaysAllow(server.name, source, tool.name, !alwaysAllow)
					await provider.postStateToWebview()
					await vscode.window.showInformationMessage(
						`${tool.name} is now ${!alwaysAllow ? "always allowed" : "approval required"}.`,
					)
				},
				keepOpen: true,
			}
		})

		items.push({
			label: "$(arrow-left) Back",
			run: async () => {},
		})

		const selection = await vscode.window.showQuickPick(items, {
			placeHolder: `Toggle tools for ${serverName}`,
			ignoreFocusOut: true,
		})

		if (!selection) {
			return
		}

		try {
			await selection.run()
		} catch (error) {
			const message = error instanceof Error ? error.message : String(error)
			await vscode.window.showErrorMessage(`Unable to toggle tool: ${message}`)
		}

		if (!selection.keepOpen) {
			return
		}
	}
}

function formatMcpServerSummary(server: McpServer): string {
	const lines = [
		`MCP Server: ${server.name}`,
		`Status: ${server.status}${server.disabled ? " (disabled)" : ""}`,
		`Source: ${server.source ?? "global"}`,
		`Config: ${server.config}`,
	]

	if (server.tools?.length) {
		lines.push(`Tools: ${server.tools.length}`)
	}

	if (server.resources?.length || server.resourceTemplates?.length) {
		const count = (server.resources?.length ?? 0) + (server.resourceTemplates?.length ?? 0)
		lines.push(`Resources: ${count}`)
	}

	return lines.join("\n")
}

async function openFileInEditor(filePath: string, defaultContent?: string) {
	const uri = vscode.Uri.file(filePath)
	try {
		await fs.mkdir(path.dirname(filePath), { recursive: true })
	} catch {}

	try {
		await vscode.workspace.fs.stat(uri)
	} catch {
		const content = Buffer.from(defaultContent ?? "", "utf8")
		await vscode.workspace.fs.writeFile(uri, content)
	}

	const document = await vscode.workspace.openTextDocument(uri)
	await vscode.window.showTextDocument(document, { preview: false })
}
