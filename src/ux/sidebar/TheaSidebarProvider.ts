/**
 * TreeView-based sidebar for Thea Code.
 * Provides quick access to tasks, MCP servers, and settings without webview.
 */
import * as vscode from "vscode"

import { TaskManager } from "../../core/TaskManager"
import { COMMANDS } from "../../shared/config/thea-config"
import type { McpServer } from "../../shared/mcp"

type SidebarItemType = "action" | "task" | "mcp" | "info"

interface SidebarItemData {
	type: SidebarItemType
	id: string
	label: string
	description?: string
	icon?: string
	command?: string
	children?: SidebarItemData[]
}

class SidebarItem extends vscode.TreeItem {
	constructor(
		public readonly data: SidebarItemData,
		public readonly collapsibleState: vscode.TreeItemCollapsibleState,
	) {
		super(data.label, collapsibleState)

		this.description = data.description
		this.tooltip = data.label + (data.description ? `: ${data.description}` : "")

		if (data.icon) {
			this.iconPath = new vscode.ThemeIcon(data.icon)
		}

		if (data.command) {
			this.command = {
				command: data.command,
				title: data.label,
			}
		}

		this.contextValue = data.type
	}
}

export class TheaSidebarProvider implements vscode.TreeDataProvider<SidebarItem> {
	private _onDidChangeTreeData = new vscode.EventEmitter<SidebarItem | undefined | null | void>()
	readonly onDidChangeTreeData = this._onDidChangeTreeData.event

	constructor(private readonly taskManager: TaskManager) {
		// Refresh when task state changes
		taskManager.on("taskCreated", () => this.refresh())
		taskManager.on("taskCompleted", () => this.refresh())
		taskManager.on("taskAborted", () => this.refresh())
	}

	refresh(): void {
		this._onDidChangeTreeData.fire()
	}

	getTreeItem(element: SidebarItem): vscode.TreeItem {
		return element
	}

	getChildren(element?: SidebarItem): SidebarItem[] {
		if (!element) {
			return this.getRootItems()
		}

		if (element.data.children) {
			return element.data.children.map(
				(child) =>
					new SidebarItem(
						child,
						child.children
							? vscode.TreeItemCollapsibleState.Collapsed
							: vscode.TreeItemCollapsibleState.None,
					),
			)
		}

		return []
	}

	private getRootItems(): SidebarItem[] {
		const items: SidebarItem[] = []

		// Current task section
		const currentTask = this.taskManager.getCurrent()
		if (currentTask) {
			items.push(
				new SidebarItem(
					{
						type: "task",
						id: "current-task",
						label: "Current Task",
						description: currentTask.isStreaming ? "Running..." : "Active",
						icon: currentTask.isStreaming ? "sync~spin" : "play-circle",
					},
					vscode.TreeItemCollapsibleState.None,
				),
			)
		}

		// Quick Actions
		items.push(
			new SidebarItem(
				{
					type: "action",
					id: "new-task",
					label: "New Task",
					description: "Start a new conversation",
					icon: "add",
					command: COMMANDS.PLUS_BUTTON,
				},
				vscode.TreeItemCollapsibleState.None,
			),
		)

		items.push(
			new SidebarItem(
				{
					type: "action",
					id: "open-chat",
					label: "Open Chat",
					description: "Open VS Code Chat panel",
					icon: "comment-discussion",
					command: "workbench.action.chat.open",
				},
				vscode.TreeItemCollapsibleState.None,
			),
		)

		items.push(
			new SidebarItem(
				{
					type: "action",
					id: "history",
					label: "Task History",
					description: "View previous tasks",
					icon: "history",
					command: COMMANDS.HISTORY_BUTTON,
				},
				vscode.TreeItemCollapsibleState.None,
			),
		)

		// MCP Section - show as expandable if we have an MCP hub
		const mcpHub = this.taskManager.getTaskContext().mcpHub
		const mcpServers = mcpHub?.getAllServers() ?? []

		if (mcpServers.length > 0) {
			// Create children for each server
			const serverChildren: SidebarItemData[] = mcpServers.map((server: McpServer) => ({
				type: "mcp" as const,
				id: `mcp-server-${server.name}`,
				label: server.name,
				description: server.disabled ? "Disabled" : server.status === "connected" ? "Connected" : server.status,
				icon: server.disabled
					? "circle-slash"
					: server.status === "connected"
						? "pass-filled"
						: "circle-outline",
			}))

			items.push(
				new SidebarItem(
					{
						type: "mcp",
						id: "mcp-servers",
						label: "MCP Servers",
						description: `${mcpServers.filter((s: McpServer) => !s.disabled && s.status === "connected").length}/${mcpServers.length} connected`,
						icon: "server",
						command: COMMANDS.MCP_BUTTON,
						children: serverChildren,
					},
					vscode.TreeItemCollapsibleState.Collapsed,
				),
			)
		} else {
			items.push(
				new SidebarItem(
					{
						type: "mcp",
						id: "mcp-servers",
						label: "MCP Servers",
						description: "Configure servers",
						icon: "server",
						command: COMMANDS.MCP_BUTTON,
					},
					vscode.TreeItemCollapsibleState.None,
				),
			)
		}

		// Settings Section
		items.push(
			new SidebarItem(
				{
					type: "action",
					id: "prompts",
					label: "Prompts",
					description: "Quick prompts",
					icon: "zap",
					command: COMMANDS.PROMPTS_BUTTON,
				},
				vscode.TreeItemCollapsibleState.None,
			),
		)

		items.push(
			new SidebarItem(
				{
					type: "action",
					id: "settings",
					label: "Settings",
					description: "Configure Thea",
					icon: "gear",
					command: COMMANDS.SETTINGS_BUTTON,
				},
				vscode.TreeItemCollapsibleState.None,
			),
		)

		items.push(
			new SidebarItem(
				{
					type: "action",
					id: "help",
					label: "Help & Docs",
					description: "Open documentation",
					icon: "question",
					command: COMMANDS.HELP_BUTTON,
				},
				vscode.TreeItemCollapsibleState.None,
			),
		)

		return items
	}
}

export function registerSidebarView(
	context: vscode.ExtensionContext,
	taskManager: TaskManager,
): vscode.TreeView<SidebarItem> {
	const sidebarProvider = new TheaSidebarProvider(taskManager)

	const treeView = vscode.window.createTreeView("thea-code.sidebarView", {
		treeDataProvider: sidebarProvider,
		showCollapseAll: false,
	})

	// Register refresh command
	context.subscriptions.push(
		vscode.commands.registerCommand("thea-code.refreshSidebar", () => {
			sidebarProvider.refresh()
		}),
	)

	// Register focus command for the sidebar view
	// VS Code automatically creates {viewId}.focus, but we register an alias for backward compatibility
	context.subscriptions.push(
		vscode.commands.registerCommand("thea-code.SidebarProvider.focus", async () => {
			await vscode.commands.executeCommand("thea-code.sidebarView.focus")
		}),
	)

	context.subscriptions.push(treeView)

	return treeView
}
