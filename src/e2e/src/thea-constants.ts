// Canonical constants for e2e tests, derived from root package.json contributes.
// Keep this file as the single source of truth for IDs referenced in tests.

// Publisher.name => extension ID used by VS Code
export const EXTENSION_ID = "SolaceHarmony.thea-code"

// Views
export const VIEW_CONTAINER_ID = "thea-code-ActivityBar"
export const SIDEBAR_VIEW_ID = "thea-code.SidebarProvider"

// Submenus
export const CONTEXT_SUBMENU_ID = "thea-code.contextMenu"
export const TERMINAL_SUBMENU_ID = "thea-code.terminalMenu"

// Commands (command IDs as declared)
export const CMD = {
	plusButtonClicked: "thea-code.plusButtonClicked",
	mcpButtonClicked: "thea-code.mcpButtonClicked",
	promptsButtonClicked: "thea-code.promptsButtonClicked",
	historyButtonClicked: "thea-code.historyButtonClicked",
	popoutButtonClicked: "thea-code.popoutButtonClicked",
	settingsButtonClicked: "thea-code.settingsButtonClicked",
	helpButtonClicked: "thea-code.helpButtonClicked",
	openInNewTab: "thea-code.openInNewTab",
	explainCode: "thea-code.explainCode",
	fixCode: "thea-code.fixCode",
	improveCode: "thea-code.improveCode",
	addToContext: "thea-code.addToContext",
	newTask: "thea-code.newTask",
	terminalAddToContext: "thea-code.terminalAddToContext",
	terminalFixCommand: "thea-code.terminalFixCommand",
	terminalExplainCommand: "thea-code.terminalExplainCommand",
	terminalFixCommandInCurrentTask: "thea-code.terminalFixCommandInCurrentTask",
	terminalExplainCommandInCurrentTask: "thea-code.terminalExplainCommandInCurrentTask",
	setCustomStoragePath: "thea-code.setCustomStoragePath",
} as const

// Configuration keys
export const CONFIG = {
	allowedCommands: "thea-code.allowedCommands",
	vsCodeLmModelSelector: "thea-code.vsCodeLmModelSelector",
	customStoragePath: "thea-code.customStoragePath",
} as const

export type CommandId = (typeof CMD)[keyof typeof CMD]
