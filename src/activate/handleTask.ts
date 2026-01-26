import * as vscode from "vscode"
import { t } from "../i18n"

/**
 * Handles the "New Task" action by prompting for input and opening the chat.
 * Uses native VS Code Chat API instead of webview.
 */
export const handleNewTask = async (params: { prompt?: string } | null | undefined) => {
	let prompt = params?.prompt
	if (!prompt) {
		prompt = await vscode.window.showInputBox({
			prompt: t("common:input.task_prompt"),
			placeHolder: t("common:input.task_placeholder"),
		})
	}
	if (!prompt) {
		// Open chat panel even without a prompt
		await vscode.commands.executeCommand("workbench.action.chat.open")
		return
	}

	// Open chat with the prompt directed to Thea
	// The @thea prefix tells VS Code to route this to our chat participant
	await vscode.commands.executeCommand("workbench.action.chat.open", {
		query: `@thea ${prompt}`,
	})
}
