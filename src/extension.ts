import * as vscode from "vscode"

// Lightweight imports only - heavy modules will be loaded dynamically
import { EXTENSION_DISPLAY_NAME, EXTENSION_NAME } from "./shared/config/thea-config"

let outputChannel: vscode.OutputChannel
let extensionContext: vscode.ExtensionContext

/**
 * Native VS Code extension using Chat Participant API.
 * Migrated from webview-based UI to native VS Code APIs.
 */

// This method is called when your extension is activated.
export async function activate(context: vscode.ExtensionContext) {
	const isE2E = process.env.THEA_E2E === "1" || process.env.NODE_ENV === "test"

	extensionContext = context
	outputChannel = vscode.window.createOutputChannel(String(EXTENSION_DISPLAY_NAME))
	context.subscriptions.push(outputChannel)
	outputChannel.appendLine(`${EXTENSION_DISPLAY_NAME} extension activated`)
	outputChannel.appendLine(`Activation starting (testMode = ${isE2E})`)

	if (isE2E) {
		return activateE2EMode(context)
	}

	return activateProductionMode(context)
}

/**
 * E2E/Test mode activation - minimal setup for testing.
 */
async function activateE2EMode(context: vscode.ExtensionContext) {
	outputChannel.appendLine("E2E mode detected: Performing lightweight activation")

	// Register minimal commands for tests
	const testCommands = [
		`${EXTENSION_NAME}.plusButtonClicked`,
		`${EXTENSION_NAME}.mcpButtonClicked`,
		`${EXTENSION_NAME}.historyButtonClicked`,
		`${EXTENSION_NAME}.popoutButtonClicked`,
		`${EXTENSION_NAME}.settingsButtonClicked`,
		`${EXTENSION_NAME}.helpButtonClicked`,
		`${EXTENSION_NAME}.openInNewTab`,
		`${EXTENSION_NAME}.explainCode`,
		`${EXTENSION_NAME}.fixCode`,
		`${EXTENSION_NAME}.improveCode`,
		`${EXTENSION_NAME}.promptsButtonClicked`,
	]

	for (const command of testCommands) {
		context.subscriptions.push(
			vscode.commands.registerCommand(command, () => {
				outputChannel.appendLine(`Test stub command executed: ${command}`)
			}),
		)
	}

	// Browser capture command for E2E tests
	context.subscriptions.push(
		vscode.commands.registerCommand(
			`${EXTENSION_NAME}.test.browserCapture`,
			async (payload: {
				url?: string
				urls?: string[]
				format?: "webp" | "png"
				fullPage?: boolean
				clipping?: boolean
				viewport?: string
			}) => {
				try {
					if (
						!payload ||
						((!payload.url || payload.url.length === 0) && (!Array.isArray(payload.urls) || payload.urls.length === 0))
					) {
						throw new Error("Invalid payload: { url: string } or { urls: string[] } required")
					}
					await context.globalState.update("remoteBrowserEnabled", false)
					await context.globalState.update("browserViewportSize", payload.viewport || "900x600")
					await context.globalState.update("useClipping", payload.clipping ?? false)
					await context.globalState.update("captureFullPage", payload.fullPage ?? false)
					if (payload.format) {
						await context.globalState.update("screenshotFormat", payload.format)
					}
					const { BrowserSession } = await import("./services/browser/BrowserSession")
					const session = new BrowserSession(context)
					await session.launchBrowser()
					if (Array.isArray(payload.urls) && payload.urls.length > 0) {
						const steps: unknown[] = []
						for (const u of payload.urls) {
							steps.push(await session.navigateToUrl(u))
						}
						await session.closeBrowser()
						return { steps }
					} else {
						const result = await session.navigateToUrl(payload.url!)
						await session.closeBrowser()
						return result
					}
				} catch (err) {
					outputChannel.appendLine(`[E2E] browserCapture error: ${String(err)}`)
					throw err
				}
			},
		),
	)

	// Initialize TaskManager for E2E
	const { TaskManager } = await import("./core/TaskManager")
	const taskManager = new TaskManager(context, outputChannel)
	await taskManager.initialize()

	// Register chat participant
	const { registerChatParticipant } = await import("./ux/chat/registerChatParticipant")
	context.subscriptions.push(registerChatParticipant(context, taskManager))

	// Register diff view provider
	const { DIFF_VIEW_URI_SCHEME } = await import("./integrations/editor/DiffViewProvider")
	const diffContentProvider = new (class implements vscode.TextDocumentContentProvider {
		provideTextDocumentContent(uri: vscode.Uri): string {
			return Buffer.from(uri.query, "base64").toString("utf-8")
		}
	})()
	context.subscriptions.push(vscode.workspace.registerTextDocumentContentProvider(DIFF_VIEW_URI_SCHEME, diffContentProvider))

	// URI handler
	const { handleUri } = await import("./activate")
	context.subscriptions.push(vscode.window.registerUriHandler({ handleUri }))

	// Initialize terminal
	const { TerminalRegistry } = await import("./integrations/terminal/TerminalRegistry")
	TerminalRegistry.initialize()

	// Return API for tests
	return {
		outputChannel,
		taskManager,
		isTestMode: true,
		version: (context.extension?.packageJSON as { version?: string })?.version ?? "",
	}
}

/**
 * Production mode activation - full feature set.
 */
async function activateProductionMode(context: vscode.ExtensionContext) {
	outputChannel.appendLine("Starting production initialization...")

	try {
		// Load environment variables
		try {
			const dotenvx = await import("@dotenvx/dotenvx")
			const path = await import("path")
			const envPath = path.join(__dirname, "..", ".env")
			dotenvx.config({ path: envPath })
		} catch (e) {
			outputChannel.appendLine(`Failed to load .env file: ${e}`)
		}

		// Load path utilities
		await import("./utils/path")

		// Load configuration
		const { configSection } = await import("./shared/config/thea-config")

		// Migration and settings
		const { migrateSettings } = await import("./utils/migrateSettings")
		try {
			await migrateSettings(context, outputChannel)
		} catch (err) {
			outputChannel.appendLine(`migrateSettings failed: ${String(err)}`)
		}

		// Initialize telemetry
		const { telemetryService } = await import("./services/telemetry/TelemetryService")
		telemetryService.initialize()

		// Initialize i18n
		const { initializeI18n } = await import("./i18n")
		const { formatLanguage } = await import("./shared/language")
		await initializeI18n(context.globalState.get("language") ?? formatLanguage(vscode.env.language))

		// Initialize terminal
		const { TerminalRegistry } = await import("./integrations/terminal/TerminalRegistry")
		TerminalRegistry.initialize()

		// Get default commands from configuration
		const defaultCommands =
			vscode.workspace.getConfiguration((configSection as () => string)()).get<string[]>("allowedCommands") || []
		if (!context.globalState.get("allowedCommands")) {
			context.globalState.update("allowedCommands", defaultCommands)
		}

		// Create TaskManager (replaces TheaProvider for task management)
		outputChannel.appendLine("Creating TaskManager...")
		const { TaskManager } = await import("./core/TaskManager")
		const taskManager = new TaskManager(context, outputChannel)
		await taskManager.initialize()
		outputChannel.appendLine("TaskManager created and initialized")

		// Register chat participant using native VS Code Chat API
		outputChannel.appendLine("Registering Thea chat participant...")
		const { registerUx } = await import("./ux")
		registerUx({ context, taskManager, outputChannel })

		// Register commands using native QuickPick workflows
		const { registerCommands } = await import("./activate")
		registerCommands({ context, outputChannel, taskManager })

		// Register diff view provider
		const { DIFF_VIEW_URI_SCHEME } = await import("./integrations/editor/DiffViewProvider")
		const diffContentProvider = new (class implements vscode.TextDocumentContentProvider {
			provideTextDocumentContent(uri: vscode.Uri): string {
				return Buffer.from(uri.query, "base64").toString("utf-8")
			}
		})()
		context.subscriptions.push(
			vscode.workspace.registerTextDocumentContentProvider(DIFF_VIEW_URI_SCHEME, diffContentProvider),
		)

		// URI handler
		const { handleUri } = await import("./activate")
		context.subscriptions.push(vscode.window.registerUriHandler({ handleUri }))

		// Register code actions provider
		const { CodeActionProvider } = await import("./core/CodeActionProvider")
		context.subscriptions.push(
			vscode.languages.registerCodeActionsProvider({ pattern: "**/*" }, new CodeActionProvider(), {
				providedCodeActionKinds: CodeActionProvider.providedCodeActionKinds,
			}),
		)

		// Register code and terminal actions
		const { registerCodeActions, registerTerminalActions } = await import("./activate")
		registerCodeActions(context)
		registerTerminalActions(context)

		// Signal activation complete
		vscode.commands.executeCommand(`${EXTENSION_NAME}.activationCompleted`)

		// Return API
		return {
			outputChannel,
			taskManager,
			version: (context.extension?.packageJSON as { version?: string })?.version ?? "",
		}
	} catch (error) {
		outputChannel.appendLine(`Failed to initialize extension: ${error}`)
		console.error("[Thea Code] Failed to initialize:", error)

		if (process.env.THEA_E2E === "1" || process.env.NODE_ENV === "test") {
			outputChannel.appendLine("E2E fallback API returned due to initialization error")
			return {
				outputChannel,
				isTestMode: true,
				version: "",
			}
		}

		void vscode.window.showErrorMessage(
			`Thea Code failed to initialize: ${error instanceof Error ? error.message : String(error)}`,
		)
		throw error
	}
}

// This method is called when your extension is deactivated
export async function deactivate() {
	if (outputChannel) {
		outputChannel.appendLine(`${EXTENSION_DISPLAY_NAME} extension deactivated`)
	}

	try {
		const { McpServerManager } = await import("./services/mcp/management/McpServerManager")
		await McpServerManager.cleanup(extensionContext)
	} catch {}

	try {
		const { telemetryService } = await import("./services/telemetry/TelemetryService")
		await telemetryService.shutdown()
	} catch {}

	try {
		const { TerminalRegistry } = await import("./integrations/terminal/TerminalRegistry")
		TerminalRegistry.cleanup()
	} catch {}
}
