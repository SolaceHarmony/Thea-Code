/**
 * TaskManager - Manages TheaTask lifecycle without webview dependencies.
 * This is the core task orchestrator for the native VS Code Chat Participant API.
 */
import EventEmitter from "events"
import * as vscode from "vscode"

import { TheaTask, TheaTaskOptions } from "./TheaTask"
import { TheaTaskStack } from "./webview/thea/TheaTaskStack"
import { TheaStateManager } from "./webview/thea/TheaStateManager"
import { ContextProxy } from "./config/ContextProxy"
import { ProviderSettingsManager } from "./config/ProviderSettingsManager"
import { CustomModesManager } from "./config/CustomModesManager"
import { McpHub } from "../services/mcp/management/McpHub"
import { McpServerManager } from "../services/mcp/management/McpServerManager"
import { HistoryItem } from "../shared/HistoryItem"
import { getSettingsDirectoryPath } from "../shared/storagePathManager"
import { EXTENSION_DISPLAY_NAME } from "../shared/config/thea-config"
import { preferLocalConfig, getPreferredMcpServersDir } from "../shared/config/paths"
import * as fs from "fs/promises"
import * as path from "path"
import * as os from "os"
import { TokenUsage } from "../schemas"
import { TheaMessage } from "../shared/ExtensionMessage"
import { PromptComponent } from "../shared/modes"
import { ShadowCheckpointService } from "../services/checkpoints/ShadowCheckpointService"
import { t } from "../i18n"
import { getWorkspacePath } from "../utils/path-vscode"

export type TaskManagerEvents = {
	taskCreated: [task: TheaTask]
	message: [{ taskId: string; action: "created" | "updated"; message: TheaMessage }]
	taskStarted: [taskId: string]
	taskPaused: [taskId: string]
	taskUnpaused: [taskId: string]
	taskAskResponded: [taskId: string]
	taskAborted: [taskId: string]
	taskSpawned: [taskId: string, childTaskId: string]
	taskCompleted: [taskId: string, usage: TokenUsage]
	taskTokenUsageUpdated: [taskId: string, usage: TokenUsage]
}

/**
 * Interface that TheaTask uses to interact with its manager.
 * This replaces the tight coupling to TheaProvider.
 */
export interface TaskContext {
	context: vscode.ExtensionContext
	outputChannel: vscode.OutputChannel
	contextProxy: ContextProxy
	mcpHub?: McpHub
	cwd: string
	postMessageToWebview?: (message: unknown) => Promise<void>
	postStateToWebview?: () => Promise<void>
	getTaskHistory: () => HistoryItem[]
	updateTaskHistory: (history: HistoryItem[]) => Promise<void>
	getState: () => ReturnType<TheaStateManager["getState"]>
	setValue: <K extends string>(key: K, value: unknown) => Promise<void>
}

export class TaskManager extends EventEmitter<TaskManagerEvents> {
	private readonly taskStack: TheaTaskStack
	private readonly stateManager: TheaStateManager
	public readonly contextProxy: ContextProxy
	public readonly providerSettingsManager: ProviderSettingsManager
	public readonly customModesManager: CustomModesManager
	private mcpHub?: McpHub
	private readonly outputChannel: vscode.OutputChannel
	private isInitialized = false

	constructor(
		public readonly context: vscode.ExtensionContext,
		outputChannel: vscode.OutputChannel,
	) {
		super()
		this.outputChannel = outputChannel
		this.contextProxy = new ContextProxy(context)
		this.taskStack = new TheaTaskStack()

		const isTest = process.env.THEA_E2E === "1" || process.env.NODE_ENV === "test"

		this.providerSettingsManager = new ProviderSettingsManager(context, { enableInitialize: !isTest })
		this.customModesManager = new CustomModesManager(
			context,
			async () => {
				// State change callback - no webview to update
			},
			!isTest,
		)

		this.stateManager = new TheaStateManager(context, this.providerSettingsManager, this.customModesManager)
		this.stateManager.getCustomModes = () => this.customModesManager.getCustomModes()
	}

	async initialize(): Promise<void> {
		if (this.isInitialized) {
			return
		}

		await this.contextProxy.initialize()

		// Initialize MCP Hub
		try {
			// Create a minimal provider-like interface for McpServerManager
			// eslint-disable-next-line @typescript-eslint/no-this-alias
			const manager = this
			const minimalProvider = {
				context: this.context,
				outputChannel: this.outputChannel,
				contextProxy: this.contextProxy,
				postMessageToWebview: async () => {},
				postStateToWebview: async () => {},
				ensureSettingsDirectoryExists: async () => {
					const globalStoragePath = manager.context.globalStorageUri.fsPath
					const settingsDir = await getSettingsDirectoryPath(globalStoragePath)
					await fs.mkdir(settingsDir, { recursive: true })
					return settingsDir
				},
			}
			this.mcpHub = await McpServerManager.getInstance(
				this.context,
				minimalProvider as unknown as Parameters<typeof McpServerManager.getInstance>[1],
			)
			this.outputChannel.appendLine("MCP Hub initialized in TaskManager")
		} catch (error) {
			this.outputChannel.appendLine(`Failed to initialize MCP Hub: ${error}`)
		}

		this.isInitialized = true
	}

	get cwd(): string {
		return getWorkspacePath() || ""
	}

	/**
	 * Creates a task context for TheaTask to use.
	 * This provides the minimal interface needed by TheaTask.
	 */
	getTaskContext(): TaskContext {
		return {
			context: this.context,
			outputChannel: this.outputChannel,
			contextProxy: this.contextProxy,
			mcpHub: this.mcpHub,
			cwd: this.cwd,
			postMessageToWebview: async () => {
				// No webview - messages are handled via events
			},
			postStateToWebview: async () => {
				// No webview - state is managed internally
			},
			getTaskHistory: () => this.contextProxy.getGlobalState("taskHistory") || [],
			updateTaskHistory: async (history: HistoryItem[]) => {
				await this.contextProxy.updateGlobalState("taskHistory", history)
			},
			getState: () => this.stateManager.getState(),
			setValue: async <K extends Parameters<typeof this.stateManager.setValue>[0]>(
				key: K,
				value: Parameters<typeof this.stateManager.setValue>[1],
			) => {
				await this.stateManager.setValue(key, value)
			},
		}
	}

	/**
	 * Creates a new task with the given prompt.
	 */
	async createTask(task?: string, images?: string[], parentTask?: TheaTask): Promise<TheaTask> {
		if (!this.isInitialized) {
			await this.initialize()
		}

		const {
			apiConfiguration,
			customModePrompts,
			diffEnabled: enableDiff,
			enableCheckpoints,
			checkpointStorage,
			fuzzyMatchThreshold,
			mode,
			customInstructions: globalInstructions,
			experiments,
		} = await this.stateManager.getState()

		const modePrompt = customModePrompts?.[mode] as PromptComponent
		const effectiveInstructions = [globalInstructions, modePrompt?.customInstructions].filter(Boolean).join("\n\n")

		// Create TheaTask with this manager as the provider
		// Note: TheaTask currently expects TheaProvider - we'll need to update it
		// For now, we create with a minimal provider interface
		const theaTask = new TheaTask({
			provider: this.createProviderShim(),
			apiConfiguration,
			customInstructions: effectiveInstructions,
			enableDiff,
			enableCheckpoints,
			checkpointStorage,
			fuzzyMatchThreshold,
			task,
			images,
			experiments,
			rootTask: this.taskStack.getSize() > 0 ? this.getTaskByIndex(0) : undefined,
			parentTask,
			taskNumber: this.taskStack.getSize() + 1,
			onCreated: (task) => this.emit("taskCreated", task),
		})

		// Forward task events to manager events
		this.wireTaskEvents(theaTask)

		const state = await this.stateManager.getState()
		if (!state || typeof state.mode !== "string") {
			throw new Error(t("common:errors.retrieve_current_mode"))
		}

		await this.taskStack.addTheaTask(theaTask)
		this.outputChannel.appendLine(
			`[TaskManager] ${theaTask.parentTask ? "child" : "parent"} task ${theaTask.taskId}.${theaTask.instanceId} created`,
		)

		return theaTask
	}

	/**
	 * Creates a task from a history item.
	 */
	async createTaskFromHistory(
		historyItem: HistoryItem & { rootTask?: TheaTask; parentTask?: TheaTask },
	): Promise<TheaTask> {
		if (!this.isInitialized) {
			await this.initialize()
		}

		await this.taskStack.removeCurrentTheaTask()

		const {
			apiConfiguration,
			customModePrompts,
			diffEnabled: enableDiff,
			enableCheckpoints,
			checkpointStorage,
			fuzzyMatchThreshold,
			mode,
			customInstructions: globalInstructions,
			experiments,
		} = await this.stateManager.getState()

		const modePrompt = customModePrompts?.[mode] as PromptComponent
		const effectiveInstructions = [globalInstructions, modePrompt?.customInstructions].filter(Boolean).join("\n\n")

		const taskId = historyItem.id
		const globalStorageDir = this.contextProxy.globalStorageUri.fsPath
		const workspaceDir = this.cwd

		const checkpoints: Pick<TheaTaskOptions, "enableCheckpoints" | "checkpointStorage"> = {
			enableCheckpoints,
			checkpointStorage,
		}

		if (enableCheckpoints) {
			try {
				checkpoints.checkpointStorage = await ShadowCheckpointService.getTaskStorage({
					taskId,
					globalStorageDir,
					workspaceDir,
				})
			} catch (error) {
				checkpoints.enableCheckpoints = false
				this.outputChannel.appendLine(
					`[TaskManager] Error getting task storage: ${error instanceof Error ? error.message : String(error)}`,
				)
			}
		}

		const theaTask = new TheaTask({
			provider: this.createProviderShim(),
			apiConfiguration,
			customInstructions: effectiveInstructions,
			enableDiff,
			...checkpoints,
			fuzzyMatchThreshold,
			historyItem,
			experiments,
			rootTask: historyItem.rootTask,
			parentTask: historyItem.parentTask,
			taskNumber: historyItem.number,
			onCreated: (task) => this.emit("taskCreated", task),
		})

		this.wireTaskEvents(theaTask)

		const state = await this.stateManager.getState()
		if (!state || typeof state.mode !== "string") {
			throw new Error(t("common:errors.retrieve_current_mode"))
		}

		await this.taskStack.addTheaTask(theaTask)
		return theaTask
	}

	/**
	 * Wires task events to manager events for forwarding.
	 */
	private wireTaskEvents(task: TheaTask): void {
		task.on("message", (data) => this.emit("message", data))
		task.on("taskStarted", (taskId) => this.emit("taskStarted", taskId))
		task.on("taskPaused", (taskId) => this.emit("taskPaused", taskId))
		task.on("taskUnpaused", (taskId) => this.emit("taskUnpaused", taskId))
		task.on("taskAskResponded", (taskId) => this.emit("taskAskResponded", taskId))
		task.on("taskAborted", (taskId) => this.emit("taskAborted", taskId))
		task.on("taskSpawned", (taskId) => this.emit("taskSpawned", task.taskId, taskId))
		task.on("taskCompleted", (taskId, usage) => this.emit("taskCompleted", taskId, usage))
		task.on("taskTokenUsageUpdated", (taskId, usage) => this.emit("taskTokenUsageUpdated", taskId, usage))
	}

	/**
	 * Creates a shim that provides the interface TheaTask expects from TheaProvider.
	 * This allows gradual migration without changing TheaTask immediately.
	 */
	private createProviderShim(): TheaTaskOptions["provider"] {
		// eslint-disable-next-line @typescript-eslint/no-this-alias
		const manager = this

		// Return a shim object that TheaTask can use
		// This implements the minimal interface needed by TheaTask
		return {
			context: this.context,
			outputChannel: this.outputChannel,
			contextProxy: this.contextProxy,
			mcpHub: this.mcpHub,
			providerSettingsManager: this.providerSettingsManager,
			customModesManager: this.customModesManager,

			get cwd() {
				return manager.cwd
			},

			// Methods TheaTask calls on provider
			postMessageToWebview: async () => {
				// No-op: messages go through events now
			},

			postStateToWebview: async () => {
				// No-op: no webview to update
			},

			getTaskHistory: () => {
				return manager.contextProxy.getGlobalState("taskHistory") || []
			},

			updateTaskHistory: async (history: HistoryItem[]) => {
				await manager.contextProxy.updateGlobalState("taskHistory", history)
			},

			getState: () => manager.stateManager.getState(),

			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			setValue: async (key: string, value: unknown) => {
				// eslint-disable-next-line @typescript-eslint/no-explicit-any
				await manager.stateManager.setValue(key as any, value as any)
			},

			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			getValue: (key: string) => {
				// eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-return
				return manager.stateManager.getValue(key as any)
			},

			getMcpHub: () => manager.mcpHub,

			log: (message: string) => {
				manager.outputChannel.appendLine(message)
			},

			// MCP directory methods needed by McpHub
			ensureMcpServersDirectoryExists: async () => {
				// Prefer sandboxed local config in tests or when explicitly requested
				if (preferLocalConfig()) {
					const workspaceRoot = await import("vscode").then(
						(v) => v.workspace.workspaceFolders?.[0]?.uri.fsPath,
					)
					const localDir = getPreferredMcpServersDir({ context: manager.context, workspaceRoot })
					await fs.mkdir(localDir, { recursive: true })
					return localDir
				}

				// Legacy home-directory locations outside tests
				let mcpServersDir: string
				if (process.platform === "win32") {
					mcpServersDir = path.join(os.homedir(), "AppData", "Roaming", String(EXTENSION_DISPLAY_NAME), "MCP")
				} else if (process.platform === "darwin") {
					mcpServersDir = path.join(os.homedir(), "Documents", String(EXTENSION_DISPLAY_NAME), "MCP")
				} else {
					mcpServersDir = path.join(os.homedir(), ".config", String(EXTENSION_DISPLAY_NAME), "MCP")
				}

				await fs.mkdir(mcpServersDir, { recursive: true })
				return mcpServersDir
			},

			ensureSettingsDirectoryExists: async () => {
				const globalStoragePath = manager.context.globalStorageUri.fsPath
				const settingsDir = await getSettingsDirectoryPath(globalStoragePath)
				await fs.mkdir(settingsDir, { recursive: true })
				return settingsDir
			},
		} as unknown as TheaTaskOptions["provider"]
	}

	/**
	 * Gets the current active task.
	 */
	getCurrent(): TheaTask | undefined {
		return this.taskStack.getCurrentTheaTask()
	}

	/**
	 * Gets a task by its index in the stack.
	 */
	private getTaskByIndex(index: number): TheaTask | undefined {
		const stack = this.taskStack.getTaskStack()
		if (index >= 0 && index < stack.length) {
			// FIXME: Expose a method on taskStack to retrieve a task by its ID or index
			// instead of always returning the current active task.
			return this.taskStack.getCurrentTheaTask()
		}
		return undefined
	}

	/**
	 * Removes the current task from the stack.
	 */
	async removeFromStack(): Promise<TheaTask | undefined> {
		return this.taskStack.removeCurrentTheaTask()
	}

	/**
	 * Cancels the current task.
	 */
	async cancelTask(): Promise<void> {
		const task = this.taskStack.getCurrentTheaTask()
		if (!task) {
			return
		}

		// Signal abort
		await task.abortTask()
	}

	/**
	 * Gets the task stack for debugging/monitoring.
	 */
	getTaskStack(): string[] {
		return this.taskStack.getTaskStack()
	}

	/**
	 * Gets the current task stack size.
	 */
	getStackSize(): number {
		return this.taskStack.getSize()
	}

	/**
	 * Finishes the current subtask and resumes parent.
	 */
	async finishSubTask(lastMessage?: string): Promise<void> {
		await this.taskStack.finishSubTask(lastMessage)
	}

	/**
	 * Gets the state manager for external access.
	 */
	getStateManager(): TheaStateManager {
		return this.stateManager
	}

	/**
	 * Disposes resources.
	 */
	dispose(): void {
		// Clean up any active tasks
		const current = this.taskStack.getCurrentTheaTask()
		if (current) {
			void current.abortTask(true)
		}
	}
}
