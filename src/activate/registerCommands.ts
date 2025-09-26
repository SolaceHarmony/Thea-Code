import * as vscode from "vscode"
import delay from "delay"

import { TheaProvider } from "../core/webview/TheaProvider" // Renamed import
import { EXTENSION_NAME, HOMEPAGE_URL, COMMANDS } from "../shared/config/thea-config"

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
			await vscode.commands.executeCommand("thea-code.chat.respond", provider.getCurrent()?.taskId || "", "Open MCP Servers")
		},
		[COMMANDS.PROMPTS_BUTTON]: async () => {
			await vscode.commands.executeCommand("thea-code.chat.respond", provider.getCurrent()?.taskId || "", "Show prompts")
		},
		[COMMANDS.HISTORY_BUTTON]: async () => {
			await vscode.commands.executeCommand("thea-code.chat.respond", provider.getCurrent()?.taskId || "", "Show history")
		},
		[COMMANDS.SETTINGS_BUTTON]: async () => {
			await vscode.commands.executeCommand("workbench.action.openSettings", "thea-code")
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
