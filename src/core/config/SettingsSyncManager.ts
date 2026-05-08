import * as vscode from "vscode"

import { GlobalState, GlobalSettings } from "../../schemas"
import { ContextProxy } from "./ContextProxy"

/**
 * SettingsSyncManager - Manages synchronization between VS Code Settings API and Global State.
 * This allows settings to be configured both through the VS Code Settings UI and through the extension's QuickPick.
 */

type GlobalStateKey = keyof GlobalState & string
type GlobalStateValue = GlobalState[keyof GlobalState]

export class SettingsSyncManager {
	private readonly context: vscode.ExtensionContext
	private readonly contextProxy: ContextProxy
	private disposables: vscode.Disposable[] = []

	constructor(context: vscode.ExtensionContext, contextProxy: ContextProxy) {
		this.context = context
		this.contextProxy = contextProxy
	}

	/**
	 * Initialize settings synchronization
	 */
	async initialize(): Promise<void> {
		// First sync from VS Code settings to global state
		await this.syncFromSettingsToState()

		// Listen for settings changes
		this.disposables.push(
			vscode.workspace.onDidChangeConfiguration(async (event) => {
				if (event.affectsConfiguration("thea-code")) {
					await this.syncFromSettingsToState()
				}
			}),
		)
	}

	/**
	 * Sync all settings from VS Code settings.json to global state
	 */
	async syncFromSettingsToState(): Promise<void> {
		const config = vscode.workspace.getConfiguration("thea-code")
		const promises: Promise<void>[] = []

		// Map VS Code settings to global state keys
		const settingsMap: Record<string, GlobalStateKey> = {
			autoApprovalEnabled: "autoApprovalEnabled",
			alwaysApproveResubmit: "alwaysApproveResubmit",
			alwaysAllowReadOnly: "alwaysAllowReadOnly",
			alwaysAllowReadOnlyOutsideWorkspace: "alwaysAllowReadOnlyOutsideWorkspace",
			alwaysAllowWrite: "alwaysAllowWrite",
			alwaysAllowWriteOutsideWorkspace: "alwaysAllowWriteOutsideWorkspace",
			alwaysAllowExecute: "alwaysAllowExecute",
			alwaysAllowBrowser: "alwaysAllowBrowser",
			alwaysAllowMcp: "alwaysAllowMcp",
			alwaysAllowModeSwitch: "alwaysAllowModeSwitch",
			alwaysAllowSubtasks: "alwaysAllowSubtasks",
			browserToolEnabled: "browserToolEnabled",
			mcpEnabled: "mcpEnabled",
			enableMcpServerCreation: "enableMcpServerCreation",
			enableCheckpoints: "enableCheckpoints",
			diffEnabled: "diffEnabled",
			soundEnabled: "soundEnabled",
			ttsEnabled: "ttsEnabled",
			remoteBrowserEnabled: "remoteBrowserEnabled",
			showTheaIgnoredFiles: "showTheaIgnoredFiles",
			allowedCommands: "allowedCommands",
		}

		const numericSettingsMap: Record<string, GlobalStateKey> = {
			screenshotQuality: "screenshotQuality",
			terminalOutputLineLimit: "terminalOutputLineLimit",
			maxReadFileLine: "maxReadFileLine",
			maxOpenTabsContext: "maxOpenTabsContext",
			maxWorkspaceFiles: "maxWorkspaceFiles",
		}

		const stringSettingsMap: Record<string, GlobalStateKey> = {
			browserViewportSize: "browserViewportSize",
			customInstructions: "customInstructions",
		}

		for (const [configKey, stateKey] of Object.entries(settingsMap)) {
			const value = config.get<any>(configKey)
			if (value !== undefined) {
				// eslint-disable-next-line @typescript-eslint/no-explicit-any
				promises.push(this.contextProxy.updateGlobalState(stateKey as keyof GlobalState, value as any))
			}
		}

		for (const [configKey, stateKey] of Object.entries(numericSettingsMap)) {
			const value = config.get<number>(configKey)
			if (value !== undefined) {
				promises.push(this.contextProxy.updateGlobalState(stateKey as keyof GlobalState, value))
			}
		}

		for (const [configKey, stateKey] of Object.entries(stringSettingsMap)) {
			const value = config.get<string>(configKey)
			if (value !== undefined) {
				promises.push(this.contextProxy.updateGlobalState(stateKey as keyof GlobalState, value))
			}
		}

		await Promise.all<void>(promises as Promise<void>[][] as any)
	}

	/**
	 * Update a setting both in VS Code settings and global state (two-way sync)
	 */
	async updateSetting<T extends keyof GlobalSettings>(
		key: T,
		value: GlobalSettings[T],
		updateTarget: "both" | "state" | "settings" = "both",
	): Promise<void> {
		const config = vscode.workspace.getConfiguration("thea-code")

		if (updateTarget === "both" || updateTarget === "settings") {
			// Update VS Code settings.json
			const updatePromise = config.update(key, value, vscode.ConfigurationTarget.Global)
			// Convert Thenable to Promise
			await new Promise<void>((resolve, reject) => {
				updatePromise.then(
					() => resolve(),
					(err) => reject(err),
				)
			})
		}

		if (updateTarget === "both" || updateTarget === "state") {
			// Update global state
			const stateKey = key as keyof GlobalState
			await this.contextProxy.updateGlobalState(stateKey, value as GlobalStateValue)
		}
	}

	/**
	 * Get current configuration value
	 */
	getConfiguration<T>(key: string): T | undefined {
		const config = vscode.workspace.getConfiguration("thea-code")
		return config.get<T>(key)
	}

	dispose(): void {
		this.disposables.forEach((d) => d.dispose())
		this.disposables = []
	}
}
