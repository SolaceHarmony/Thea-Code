import * as vscode from "vscode"

import { TaskManager } from "../core/TaskManager"
import { registerChatParticipant } from "./chat/registerChatParticipant"
import { registerSidebarView } from "./sidebar/TheaSidebarProvider"

export interface RegisterUxOptions {
	context: vscode.ExtensionContext
	taskManager: TaskManager
	outputChannel: vscode.OutputChannel
}

export function registerUx({ context, taskManager, outputChannel }: RegisterUxOptions) {
	outputChannel.appendLine("Registering Chat Participant...")
	const chatParticipant = registerChatParticipant(context, taskManager)

	outputChannel.appendLine("Registering Sidebar TreeView...")
	const sidebarView = registerSidebarView(context, taskManager)

	return {
		chatParticipant,
		sidebarView,
	}
}

export { TaskManager }
export { registerChatParticipant }
export { registerSidebarView }
