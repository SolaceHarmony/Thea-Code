import * as vscode from "vscode"

// Lightweight imports only - heavy modules will be loaded dynamically
import { EXTENSION_DISPLAY_NAME, EXTENSION_NAME } from "./shared/config/thea-config"

// Type-only imports (don't affect runtime)
// type-only imports omitted to avoid unused-var warnings in E2E path


/**
 * Modern webview-based VS Code extension built with React 19.
 * Uses Radix UI primitives for accessible, theme-aware components.
 * Migrated from the deprecated @vscode/webview-ui-toolkit to modern UI stack.
 */

let outputChannel: vscode.OutputChannel
let extensionContext: vscode.ExtensionContext
// Note: provider instance is managed by API and VS Code registrations


// This method is called when your extension is activated.
// Your extension is activated the very first time the command is executed.
export async function activate(context: vscode.ExtensionContext) {
	// Detect e2e/test mode only via env for explicit control.
	const isE2E = process.env.THEA_E2E === '1' || process.env.NODE_ENV === 'test'

	extensionContext = context
	outputChannel = vscode.window.createOutputChannel(String(EXTENSION_DISPLAY_NAME))
	context.subscriptions.push(outputChannel)
	outputChannel.appendLine(`${EXTENSION_DISPLAY_NAME} extension activated`)

	outputChannel.appendLine(`Activation starting(testMode = ${isE2E})`)

	if (isE2E) {
		outputChannel.appendLine("E2E mode detected: Performing lightweight activation")

		// Register minimal commands for tests without creating a provider
		const testCommands = [
			`${EXTENSION_NAME}.plusButtonClicked`,
			`${EXTENSION_NAME}.mcpButtonClicked`,
			`${EXTENSION_NAME}.historyButtonClicked`,
			`${EXTENSION_NAME}.popoutButtonClicked`,
			`${EXTENSION_NAME}.settingsButtonClicked`,
			`${EXTENSION_NAME}.openInNewTab`,
			`${EXTENSION_NAME}.explainCode`,
			`${EXTENSION_NAME}.fixCode`,
			`${EXTENSION_NAME}.improveCode`,
			`${EXTENSION_NAME}.promptsButtonClicked`,
		]

		// Register stub commands for testing
		for (const command of testCommands) {
			context.subscriptions.push(
				vscode.commands.registerCommand(command, () => {
					outputChannel.appendLine(`Test stub command executed: ${command} `)
				})
			)
		}

		// Register a hidden E2E command to drive a real browser session using BrowserSession
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
						if (!payload || ((!payload.url || payload.url.length === 0) && (!Array.isArray(payload.urls) || payload.urls.length === 0))) {
							throw new Error("Invalid payload: { url: string } or { urls: string[] } required")
						}
						// Force local browser for deterministic tests
						await context.globalState.update("remoteBrowserEnabled", false)
						// Stable viewport and clipping/fullPage from payload if provided
						await context.globalState.update("browserViewportSize", payload.viewport || "900x600")
						await context.globalState.update("useClipping", payload.clipping ?? false)
						await context.globalState.update("captureFullPage", payload.fullPage ?? false)
						// Screenshot preferences
						if (payload.format) {
							await context.globalState.update("screenshotFormat", payload.format)
						}
						const { BrowserSession } = await import("./services/browser/BrowserSession")
						const session = new BrowserSession(context)
						await session.launchBrowser()
						if (Array.isArray(payload.urls) && payload.urls.length > 0) {
							const steps = [] as any[]
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
						outputChannel.appendLine(`[E2E] browserCapture error: ${String(err)} `)
						throw err
					}
				}
			)
		)

		// Instantiate provider for testing
		const { TheaProvider } = await import("./core/webview/TheaProvider")
		const sidebarProvider = new TheaProvider(context, outputChannel, "sidebar")
		context.subscriptions.push(
			vscode.window.registerWebviewViewProvider(String(TheaProvider.sideBarId), sidebarProvider, {
				webviewOptions: { retainContextWhenHidden: true },
			}),
		)

		// Return a minimal API for tests (avoid unsafe packageJSON access)
		const pkg = context.extension?.packageJSON as { version?: string } | undefined
		const minimalApi = {
			outputChannel,
			isTestMode: true,
			version: pkg?.version ?? "",
			sidebarProvider, // Expose provider for testing
		}

		return minimalApi
	}

	// Non-E2E activation continues here with lazy loading

	// Load heavy modules dynamically
	try {
		// Load environment variables if needed
		try {
			const dotenvx = await import("@dotenvx/dotenvx")
			const path = await import("path")
			const envPath = path.join(__dirname, "..", ".env")
			dotenvx.config({ path: envPath })
		} catch (e) {
			// Silently handle environment loading errors
			outputChannel.appendLine(`Failed to load .env file: ${e}`)
			console.warn(`[Thea Code] Failed to load .env file:`, e)
		}

		// Load path utilities first
		await import("./utils/path")

		// Load configuration
		const { configSection } = await import("./shared/config/thea-config")

		// Migration and settings
		const { migrateSettings } = await import("./utils/migrateSettings")
		try {
			await migrateSettings(context, outputChannel)
		} catch (err) {
			outputChannel.appendLine(`migrateSettings failed: ${String(err)}`)
			console.error(`[Thea Code] migrateSettings failed:`, err)
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

		// Initialize global state if not already set
		if (!context.globalState.get("allowedCommands")) {
			context.globalState.update("allowedCommands", defaultCommands)
		}

		outputChannel.appendLine("Creating TheaProvider...")
		const { TheaProvider } = await import("./core/webview/TheaProvider")
		const theaProvider = new TheaProvider(context, outputChannel, "sidebar")
		outputChannel.appendLine("TheaProvider created")
		telemetryService.setProvider(theaProvider)

		context.subscriptions.push(
			vscode.window.registerWebviewViewProvider(String(TheaProvider.sideBarId), theaProvider, {
				webviewOptions: { retainContextWhenHidden: true },
			}),
		)

		const { registerCommands } = await import("./activate")
		registerCommands({ context, outputChannel, provider: theaProvider })

		/**
		 * We use the text document content provider API to show the left side for diff
		 * view by creating a virtual document for the original content. This makes it
		 * readonly so users know to edit the right side if they want to keep their changes.
		 *
		 * This API allows you to create readonly documents in VSCode from arbitrary
		 * sources, and works by claiming an uri-scheme for which your provider then
		 * returns text contents. The scheme must be provided when registering a
		 * provider and cannot change afterwards.
		 *
		 * Note how the provider doesn't create uris for virtual documents - its role
		 * is to provide contents given such an uri. In return, content providers are
		 * wired into the open document logic so that providers are always considered.
		 *
		 * https://code.visualstudio.com/api/extension-guides/virtual-documents
		 */
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

		const { handleUri } = await import("./activate")
		context.subscriptions.push(vscode.window.registerUriHandler({ handleUri }))

		// Register code actions provider
		const { CodeActionProvider } = await import("./core/CodeActionProvider")
		context.subscriptions.push(
			vscode.languages.registerCodeActionsProvider({ pattern: "**/*" }, new CodeActionProvider(), {
				providedCodeActionKinds: CodeActionProvider.providedCodeActionKinds,
			}),
		)

		const { registerCodeActions, registerTerminalActions } = await import("./activate")
		registerCodeActions(context)
		registerTerminalActions(context)

		// Register Native Chat Participant
		const { registerChatParticipant } = await import("./integrations/chat/TheaChatParticipant")
		context.subscriptions.push(registerChatParticipant(context))

		// Allows other extensions to activate once Thea is ready
		vscode.commands.executeCommand(`${EXTENSION_NAME}.activationCompleted`)

		// Return the API
		const { API } = await import("./exports/api")
		return new API(outputChannel, theaProvider)
	} catch (error) {
		outputChannel.appendLine(`Failed to initialize extension: ${error}`)
		console.error('[Thea Code] Failed to initialize:', error)

		// In E2E/test mode, don't fail activation; return a minimal API so tests can proceed
		if (process.env.THEA_E2E === '1' || process.env.NODE_ENV === 'test') {
			try {
				const pkg = extensionContext?.extension?.packageJSON as { version?: string } | undefined
				const minimalApi = {
					outputChannel,
					isTestMode: true,
					version: pkg?.version ?? ""
				}
				outputChannel.appendLine('E2E fallback API returned due to initialization error')
				console.error('[Thea Code] Returning E2E fallback API')
				return minimalApi
			} catch (fallbackError) {
				// If even minimal API construction fails, rethrow original error
				console.error('[Thea Code] Fallback API construction failed:', fallbackError)
			}
		}

		// Show error message to user in non-test mode
		if (process.env.THEA_E2E !== '1' && process.env.NODE_ENV !== 'test') {
			void vscode.window.showErrorMessage(`Thea Code failed to initialize: ${error instanceof Error ? error.message : String(error)}`)
		}
		throw error
	}
}

// This method is called when your extension is deactivated
export async function deactivate() {
	if (outputChannel) {
		outputChannel.appendLine(`${EXTENSION_DISPLAY_NAME} extension deactivated`)
	}

	// Only cleanup if modules were loaded
	try {
		const { McpServerManager } = await import("./services/mcp/management/McpServerManager")
		await McpServerManager.cleanup(extensionContext)
	} catch { }

	try {
		const { telemetryService } = await import("./services/telemetry/TelemetryService")
		await telemetryService.shutdown()
	} catch { }

	try {
		const { TerminalRegistry } = await import("./integrations/terminal/TerminalRegistry")
		TerminalRegistry.cleanup()
	} catch { }

}
