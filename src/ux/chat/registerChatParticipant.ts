/**
 * VS Code Chat Participant for Thea Code.
 * Uses native VS Code Chat API instead of webview.
 */
import * as vscode from "vscode"

import { TaskManager } from "../../core/TaskManager"
import { TheaTask } from "../../core/TheaTask"
import type { TheaMessage } from "../../shared/ExtensionMessage"

const CHAT_PARTICIPANT_ID = "thea-code.thea"
const APPROVE_COMMAND_ID = "thea-code.chat.approve"
const REJECT_COMMAND_ID = "thea-code.chat.reject"
const RESPOND_COMMAND_ID = "thea-code.chat.respond"

const approvalAskTypes = new Set([
	"tool",
	"command",
	"command_output",
	"browser_action_launch",
	"use_mcp_server",
	"mistake_limit_reached",
	"resume_task",
	"resume_completed_task",
	"completion_result",
])

const activeTasks = new Map<string, TheaTask>()

/**
 * Registers the Thea chat participant using native VS Code Chat API.
 * This replaces the webview-based UI with native chat integration.
 */
export function registerChatParticipant(
	context: vscode.ExtensionContext,
	taskManager: TaskManager,
): vscode.ChatParticipant {
	// Register approval command
	context.subscriptions.push(
		vscode.commands.registerCommand(APPROVE_COMMAND_ID, (taskId?: string) => {
			const task = resolveTask(taskId, taskManager)
			if (!task) {
				void vscode.window.showWarningMessage("No active task to approve.")
				return
			}
			task.webviewCommunicator.handleWebviewAskResponse("yesButtonClicked")
		}),
	)

	// Register rejection command
	context.subscriptions.push(
		vscode.commands.registerCommand(REJECT_COMMAND_ID, (taskId?: string) => {
			const task = resolveTask(taskId, taskManager)
			if (!task) {
				void vscode.window.showWarningMessage("No active task to reject.")
				return
			}
			task.webviewCommunicator.handleWebviewAskResponse("noButtonClicked")
		}),
	)

	// Register respond command
	context.subscriptions.push(
		vscode.commands.registerCommand(RESPOND_COMMAND_ID, async (taskId?: string, prompt?: string) => {
			const task = resolveTask(taskId, taskManager)
			if (!task) {
				void vscode.window.showWarningMessage("No active task to respond to.")
				return
			}
			const value = await vscode.window.showInputBox({
				prompt: "Provide input for Thea",
				value: prompt,
			})
			if (value === undefined) {
				return
			}
			task.webviewCommunicator.handleWebviewAskResponse("messageResponse", value)
		}),
	)

	// Create the chat participant
	const participant = vscode.chat.createChatParticipant(
		CHAT_PARTICIPANT_ID,
		async (request, _chatContext, response, token) => {
			const promptCandidate = request.prompt ?? ""
			const prompt = request.command ? `${request.command} ${promptCandidate}`.trim() : promptCandidate.trim()

			if (!prompt) {
				response.markdown("Hi! Ask me something about your workspace or run `/thea` with a question.")
				return { metadata: { preview: true } }
			}

			if (token.isCancellationRequested) {
				return { metadata: { cancelled: true } }
			}

			response.progress("Starting Thea task...")

			let task: TheaTask | undefined
			try {
				// Remove any existing task first
				await taskManager.removeFromStack()
				// Create new task using TaskManager directly
				task = await taskManager.createTask(prompt, undefined)
			} catch (error) {
				const message = error instanceof Error ? error.message : String(error)
				response.markdown(`Failed to start task: ${message}`)
				return { metadata: { error: true } }
			}

			if (!task) {
				response.markdown("Task did not start. Try again.")
				return { metadata: { error: true } }
			}

			response.markdown(`**Task ${task.taskId}** started`)
			activeTasks.set(task.taskId, task)

			const disposables: vscode.Disposable[] = []

			// Handle messages from task
			const handleMessage = ({ message }: { taskId: string; action: "created" | "updated"; message: TheaMessage }) => {
				if (message.partial) {
					return
				}
				if (message.type === "ask") {
					const formatted = formatAskMessage(message)
					if (formatted) {
						response.markdown(formatted)
					}
					addAskButtons(response, task.taskId, message)
					return
				}

				const formatted = formatSayMessage(message)
				if (formatted) {
					response.markdown(formatted)
				}
			}

			// Handle task completion
			const handleTaskCompleted = (_taskId: string, usage?: { totalCost?: number }) => {
				if (usage) {
					const cost = typeof usage.totalCost === "number" ? ` (est. $${usage.totalCost.toFixed(4)})` : ""
					response.markdown(`Task completed${cost}`)
				} else {
					response.markdown("Task completed")
				}
				activeTasks.delete(task.taskId)
				cleanup()
			}

			// Handle task abort
			const handleTaskAborted = () => {
				response.markdown("Task aborted")
				activeTasks.delete(task.taskId)
				cleanup()
			}

			// Cleanup function
			const cleanup = () => {
				disposables.forEach((d) => {
					d.dispose()
				})
				disposables.length = 0
			}

			// Subscribe to task events
			task.on("message", handleMessage)
			task.once("taskCompleted", handleTaskCompleted)
			task.once("taskAborted", handleTaskAborted)

			// Track listeners for cleanup
			disposables.push({
				dispose: () => {
					task.off("message", handleMessage)
					task.off("taskCompleted", handleTaskCompleted)
					task.off("taskAborted", handleTaskAborted)
				},
			})

			// Handle cancellation
			token.onCancellationRequested(() => {
				cleanup()
				void taskManager.cancelTask()
				activeTasks.delete(task.taskId)
				response.markdown("Task cancelled")
			})

			return { metadata: { taskId: task.taskId } }
		},
	)

	context.subscriptions.push(participant)

	return participant
}

function formatSayMessage(message: TheaMessage): string | undefined {
	if (message.type === "say") {
		switch (message.say) {
			case "text":
			case "reasoning":
			case "command":
			case "command_output":
			case "browser_action":
			case "browser_action_result":
			case "tool":
			case "error":
				return message.text ? normalizeMarkdown(message.text) : undefined
			case "api_req_started":
				return "Calling API..."
			case "api_req_finished":
				return "API request finished"
			case "api_req_retry_delayed":
				return message.text ? `Retry delayed: ${normalizeMarkdown(message.text)}` : "Retry delayed"
			case "api_req_retried":
				return "Retrying API request"
			case "task":
				return message.text ? normalizeMarkdown(message.text) : undefined
			case "completion_result":
				return message.text ? normalizeMarkdown(message.text) : undefined
			case "user_feedback":
			case "user_feedback_diff":
				return message.text ? normalizeMarkdown(message.text) : undefined
			case "mcp_server_request_started":
				return "Contacting MCP server..."
			case "mcp_server_response":
				return message.text ? normalizeMarkdown(message.text) : "MCP server responded"
			case "new_task":
			case "new_task_started":
				return message.text ? normalizeMarkdown(message.text) : undefined
			case "checkpoint_saved":
				return message.text ? normalizeMarkdown(message.text) : "Checkpoint saved"
			case "shell_integration_warning":
				return message.text ? normalizeMarkdown(message.text) : "Shell integration warning"
			case "theaignore_error":
				return message.text ? normalizeMarkdown(message.text) : ".theaignore prevented this action"
			default:
				return message.text ? normalizeMarkdown(message.text) : undefined
		}
	}

	if (message.type === "ask") {
		switch (message.ask) {
			case "followup":
				return message.text ? `Follow-up: ${normalizeMarkdown(message.text)}` : "Follow-up question"
			case "command":
			case "command_output":
				return message.text ? normalizeMarkdown(message.text) : "Command action required"
			case "completion_result":
				return message.text ? normalizeMarkdown(message.text) : "Review completion"
			case "tool":
				return message.text ? normalizeMarkdown(message.text) : "Tool approval needed"
			case "resume_task":
			case "resume_completed_task":
				return "Resume task?"
			case "mistake_limit_reached":
				return "Mistake limit reached"
			case "browser_action_launch":
				return "Browser action requested"
			case "use_mcp_server":
				return "MCP server approval needed"
			default:
				return message.text ? normalizeMarkdown(message.text) : undefined
		}
	}

	return message.text ? normalizeMarkdown(message.text) : undefined
}

function formatAskMessage(message: TheaMessage): string | undefined {
	if (message.type !== "ask") {
		return undefined
	}
	switch (message.ask) {
		case "followup":
			return message.text ? `Follow-up: ${normalizeMarkdown(message.text)}` : "Follow-up question"
		case "command":
		case "command_output":
			return message.text ? normalizeMarkdown(message.text) : "Command input requested"
		case "completion_result":
			return message.text ? normalizeMarkdown(message.text) : "Review completion result"
		case "tool":
			return message.text ? normalizeMarkdown(message.text) : "Tool approval needed"
		case "resume_task":
		case "resume_completed_task":
			return "Resume task?"
		case "mistake_limit_reached":
			return "Mistake limit reached"
		case "browser_action_launch":
			return "Browser action requested"
		case "use_mcp_server":
			return "MCP server approval needed"
		case "finishTask":
			return message.text ? normalizeMarkdown(message.text) : "Finish task?"
		default:
			return message.text ? normalizeMarkdown(message.text) : undefined
	}
}

function addAskButtons(response: vscode.ChatResponseStream, taskId: string, message: TheaMessage) {
	if (message.type !== "ask") {
		return
	}
	if (message.ask && approvalAskTypes.has(message.ask)) {
		response.button({
			title: "Approve",
			command: APPROVE_COMMAND_ID,
			arguments: [taskId],
		})
		response.button({
			title: "Reject",
			command: REJECT_COMMAND_ID,
			arguments: [taskId],
		})
	}
	response.button({
		title: "Respond...",
		command: RESPOND_COMMAND_ID,
		arguments: [taskId, message.text ?? ""],
	})
}

function normalizeMarkdown(text: string): string {
	return text.replace(/\r\n/g, "\n").trim()
}

function resolveTask(taskId: string | undefined, taskManager: TaskManager): TheaTask | undefined {
	if (taskId && activeTasks.has(taskId)) {
		const task = activeTasks.get(taskId)
		if (task) {
			return task
		}
	}

	const current = taskManager.getCurrent()
	if (current && (!taskId || current.taskId === taskId)) {
		return current
	}

	return undefined
}
