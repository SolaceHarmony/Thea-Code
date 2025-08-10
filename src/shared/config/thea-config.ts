// Static Thea configuration constants
// This file replaces dynamic branding.json generation. Keep values updated here.
// Do NOT auto-generate; all branding/config must be edited in source and committed.

export const EXTENSION_NAME = "thea-code"
export const EXTENSION_DISPLAY_NAME = "Thea Code"
export const EXTENSION_PUBLISHER = "SolaceHarmony"
export const EXTENSION_VERSION = "0.0.5" // Keep in sync with package.json
export const EXTENSION_ID = `${EXTENSION_PUBLISHER}.${EXTENSION_NAME}`
export const EXTENSION_SECRETS_PREFIX = "thea_code_config_"
export const EXTENSION_CONFIG_DIR = ".thea"
export const CONFIG_DIR_NAME = ".thea"
export const REPOSITORY_URL = "https://github.com/SolaceHarmony/Thea-Code"
export const HOMEPAGE_URL = "https://github.com/SolaceHarmony/Thea-Code"
export const AUTHOR_NAME = "Sydney Renee"
export const AUTHOR_EMAIL = "sydney@solace.ofharmony.ai"
export const AI_IDENTITY_NAME = "Thea"
export const BRANCH_PREFIX = "thea-"

export const COMMANDS = {
  PLUS_BUTTON: `${EXTENSION_NAME}.plusButtonClicked`,
  MCP_BUTTON: `${EXTENSION_NAME}.mcpButtonClicked`,
  PROMPTS_BUTTON: `${EXTENSION_NAME}.promptsButtonClicked`,
  HISTORY_BUTTON: `${EXTENSION_NAME}.historyButtonClicked`,
  POPOUT_BUTTON: `${EXTENSION_NAME}.popoutButtonClicked`,
  SETTINGS_BUTTON: `${EXTENSION_NAME}.settingsButtonClicked`,
  HELP_BUTTON: `${EXTENSION_NAME}.helpButtonClicked`,
  OPEN_NEW_TAB: `${EXTENSION_NAME}.openInNewTab`,
  EXPLAIN_CODE: `${EXTENSION_NAME}.explainCode`,
  FIX_CODE: `${EXTENSION_NAME}.fixCode`,
  IMPROVE_CODE: `${EXTENSION_NAME}.improveCode`,
  ADD_TO_CONTEXT: `${EXTENSION_NAME}.addToContext`,
  TERMINAL_ADD_TO_CONTEXT: `${EXTENSION_NAME}.terminalAddToContext`,
  TERMINAL_FIX: `${EXTENSION_NAME}.terminalFixCommand`,
  TERMINAL_EXPLAIN: `${EXTENSION_NAME}.terminalExplainCommand`,
  TERMINAL_FIX_CURRENT: `${EXTENSION_NAME}.terminalFixCommandInCurrentTask`,
  TERMINAL_EXPLAIN_CURRENT: `${EXTENSION_NAME}.terminalExplainCommandInCurrentTask`,
  NEW_TASK: `${EXTENSION_NAME}.newTask`,
} as const

export const VIEWS = {
  SIDEBAR: `${EXTENSION_NAME}.SidebarProvider`,
  TAB_PANEL: `${EXTENSION_NAME}.TabPanelProvider`,
  ACTIVITY_BAR: `${EXTENSION_NAME}-ActivityBar`,
} as const

export const CONFIG = {
  SECTION: EXTENSION_NAME,
  ALLOWED_COMMANDS: `allowedCommands`,
  VS_CODE_LM_SELECTOR: `vsCodeLmModelSelector`,
  CHECKPOINTS_PREFIX: `${EXTENSION_NAME}-checkpoints`,
} as const

export const MENU_GROUPS = { AI_COMMANDS: `${EXTENSION_DISPLAY_NAME} Commands`, NAVIGATION: "navigation" } as const

export const TEXT_PATTERNS = {
  createRoleDefinition: (role: string): string => {
    return `You are ${AI_IDENTITY_NAME}, ${role}`
  },
  logPrefix: (): string => `${EXTENSION_DISPLAY_NAME} <Language Model API>:`,
}

export const API_REFERENCES = {
  REPO_URL: REPOSITORY_URL,
  HOMEPAGE: HOMEPAGE_URL,
  APP_TITLE: EXTENSION_DISPLAY_NAME,
  DISCORD_URL: "https://discord.gg/EmberHarmony",
  REDDIT_URL: "https://reddit.com/r/SolaceHarmony",
} as const

export const GLOBAL_FILENAMES = {
  IGNORE_FILENAME: ".thea_ignore",
  MODES_FILENAME: ".thea_modes",
} as const

export const SPECIFIC_STRINGS = {
  AI_IDENTITY_NAME_LOWERCASE: "thea",
  IGNORE_ERROR_IDENTIFIER: "theaignore_error",
  IGNORE_CONTENT_VAR_NAME: "theaIgnoreContent",
  IGNORE_PARSED_VAR_NAME: "theaIgnoreParsed",
  IGNORE_CONTROLLER_CLASS_NAME: "TheaIgnoreController",
  SETTINGS_FILE_NAME: "thea-code-settings.json",
  PORTAL_NAME: "thea-portal",
} as const

export const SETTING_KEYS = { SHOW_IGNORED_FILES: "showTheaCodeIgnoredFiles" } as const

export const TYPE_NAMES = { API: "TheaCodeAPI", EVENTS: "TheaCodeEvents" } as const

export const prefixCommand = (command: string): string => `${EXTENSION_NAME}.${command}`
export const brandMessage = (message: string): string => `${EXTENSION_DISPLAY_NAME}: ${message}`
export const configSection = (): string => CONFIG.SECTION

export const GIT_DISABLED_SUFFIX = "_disabled"
export const ANTHROPIC_DEFAULT_MAX_TOKENS = 4096
