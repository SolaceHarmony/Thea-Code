import * as vscode from "vscode"
import { VsCodeLmHandler } from "../../api/providers/vscode-lm"
import { NeutralConversationHistory, NeutralMessage } from "../../shared/neutral-history"
import { EXTENSION_DISPLAY_NAME } from "../../shared/config/thea-config"

export function registerChatParticipant(context: vscode.ExtensionContext): vscode.Disposable {
    const handler: vscode.ChatRequestHandler = async (
        request: vscode.ChatRequest,
        context: vscode.ChatContext,
        stream: vscode.ChatResponseStream,
        token: vscode.CancellationToken,
    ) => {
        try {
            // 1. Convert history to NeutralConversationHistory
            const messages: NeutralConversationHistory = []

            // Process history
            for (const turn of context.history) {
                if (turn instanceof vscode.ChatRequestTurn) {
                    messages.push({
                        role: "user",
                        content: turn.prompt,
                    })
                } else if (turn instanceof vscode.ChatResponseTurn) {
                    let assistantContent = ""
                    for (const part of turn.response) {
                        if (part instanceof vscode.ChatResponseMarkdownPart) {
                            assistantContent += part.value.value
                        }
                    }
                    if (assistantContent) {
                        messages.push({
                            role: "assistant",
                            content: assistantContent,
                        })
                    }
                }
            }

            // Add current request
            messages.push({
                role: "user",
                content: request.prompt,
            })

            // 2. Initialize Handler
            // We use a default selector or one configured by the user
            const config = vscode.workspace.getConfiguration("thea-code")
            const selector = config.get<vscode.LanguageModelChatSelector>("vsCodeLmModelSelector") || {
                vendor: "copilot",
                family: "gpt-4",
            }

            const lmHandler = new VsCodeLmHandler({
                vsCodeLmModelSelector: selector,
            })

            // 3. Stream Response
            const systemPrompt = `You are ${EXTENSION_DISPLAY_NAME}, a helpful AI assistant for coding.`

            for await (const chunk of lmHandler.createMessage(systemPrompt, messages)) {
                if (token.isCancellationRequested) {
                    break
                }

                if (chunk.type === "text") {
                    stream.markdown(chunk.text)
                }
            }
        } catch (err) {
            if (err instanceof Error) {
                stream.markdown(`\n\n**Error:** ${err.message}`)
            } else {
                stream.markdown(`\n\n**Error:** Unknown error occurred.`)
            }
        }
    }

    const participant = vscode.chat.createChatParticipant("thea-code.thea", handler)
    participant.iconPath = vscode.Uri.joinPath(context.extensionUri, "assets", "icons", "icon.svg")

    return participant
}
