import * as vscode from "vscode"

import { TheaProvider } from "../core/webview/TheaProvider"
import { registerChatParticipant } from "./chat/registerChatParticipant"

export interface RegisterUxOptions {
	context: vscode.ExtensionContext
	provider: TheaProvider
	outputChannel: vscode.OutputChannel
}

export function registerUx({ context, provider }: RegisterUxOptions) {
	const chatParticipant = registerChatParticipant(context, provider)

	return {
		chatParticipant,
	}
}
