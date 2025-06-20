import type { ToolGroup } from "../schemas"

// Define tool group configuration
export type ToolGroupConfig = {
	tools: readonly string[]
	alwaysAvailable?: boolean // Whether this group is always available and shouldn't show in prompts view
}

// Map of tool slugs to their display names
export const TOOL_DISPLAY_NAMES = {
	execute_command: "run commands",
	read_file: "read files",
	fetch_instructions: "fetch instructions",
	write_to_file: "write files",
	apply_diff: "apply changes",
	search_files: "search files",
	list_files: "list files",
	list_code_definition_names: "list definitions",
	browser_action: "use a browser",
	use_mcp_tool: "use mcp tools",
	access_mcp_resource: "access mcp resources",
	ask_followup_question: "ask questions",
	attempt_completion: "complete tasks",
	switch_mode: "switch modes",
	new_task: "create new task",
} as const

export type { ToolGroup }

// Define available tool groups
export const TOOL_GROUPS: Record<ToolGroup, ToolGroupConfig> = {
	read: {
		tools: ["read_file", "fetch_instructions", "search_files", "list_files", "list_code_definition_names"],
	},
	edit: {
		tools: ["apply_diff", "write_to_file", "insert_content", "search_and_replace"],
	},
	browser: {
		tools: ["browser_action"],
	},
	command: {
		tools: ["execute_command"],
	},
	mcp: {
		tools: ["use_mcp_tool", "access_mcp_resource"],
	},
	modes: {
		tools: ["switch_mode", "new_task"],
		alwaysAvailable: true,
	},
}

// Tools that are always available to all modes
export const ALWAYS_AVAILABLE_TOOLS = ["ask_followup_question", "attempt_completion"] as const

// Tool name types for type safety
export type ToolName = keyof typeof TOOL_DISPLAY_NAMES

// Tool helper functions
export function getToolName(toolConfig: string | readonly [ToolName, ...unknown[]]): ToolName {
	return typeof toolConfig === "string" ? (toolConfig as ToolName) : toolConfig[0]
}
