import * as vscode from "vscode"
import * as dotenvx from "@dotenvx/dotenvx"
import * as path from "path"

// Load environment variables from .env file
try {
	// Specify path to .env file in the project root directory
	const envPath = path.join(__dirname, "..", ".env")
	dotenvx.config({ path: envPath })
} catch (e) {
	// Silently handle environment loading errors
	console.warn("Failed to load environment variables:", e)
}

import "./utils/path" // Necessary to have access to String.prototype.toPosix.

import { initializeI18n } from "./i18n"
import { TheaProvider } from "./core/webview/TheaProvider" // Renamed import
import { CodeActionProvider } from "./core/CodeActionProvider"
import { DIFF_VIEW_URI_SCHEME } from "./integrations/editor/DiffViewProvider" // Fixed: removed type-only import
import { McpServerManager } from "./services/mcp/management/McpServerManager"
import { telemetryService } from "./services/telemetry/TelemetryService"
import { TerminalRegistry } from "./integrations/terminal/TerminalRegistry"
import { API } from "./exports/api"
import { migrateSettings } from "./utils/migrateSettings"

import { handleUri, registerCommands, registerCodeActions, registerTerminalActions } from "./activate"
import { formatLanguage } from "./shared/language"
import { EXTENSION_DISPLAY_NAME, EXTENSION_NAME, configSection } from "./shared/config/thea-config"

/**
 * Built using https://github.com/microsoft/vscode-webview-ui-toolkit
 *
 * Inspired by:
 *  - https://github.com/microsoft/vscode-webview-ui-toolkit-samples/tree/main/default/weather-webview
 *  - https://github.com/microsoft/vscode-webview-ui-toolkit-samples/tree/main/frameworks/hello-world-react-cra
 */

let outputChannel: vscode.OutputChannel
let extensionContext: vscode.ExtensionContext
let provider: TheaProvider | undefined // <-- single shared provider variable

// This method is called when your extension is activated.
// Your extension is activated the very first time the command is executed.
export async function activate(context: vscode.ExtensionContext) {
	extensionContext = context
	outputChannel = vscode.window.createOutputChannel(String(EXTENSION_DISPLAY_NAME))
	context.subscriptions.push(outputChannel)
	outputChannel.appendLine(`${EXTENSION_DISPLAY_NAME} extension activated`)

	// Detect e2e/test mode to reduce heavy startup during integration tests
	const isE2E = process.env.THEA_E2E === "1" || process.env.NODE_ENV === "test"
	outputChannel.appendLine(`Activation starting (testMode=${isE2E})`)

	if (isE2E) {
		outputChannel.appendLine("E2E mode detected: Performing lightweight activation")

		// Create a minimal provider and register it so commands can interact with it
		provider = new TheaProvider(context, outputChannel, "sidebar")
		context.subscriptions.push(
			vscode.window.registerWebviewViewProvider(String(TheaProvider.sideBarId), provider, {
				webviewOptions: { retainContextWhenHidden: true },
			}),
		)

		// Register the commands expected by the tests
		registerCommands({ context, outputChannel, provider: provider! })

		// Return the API so the e2e setup can obtain it via extension.exports
		return new API(outputChannel, provider!)
	}

	// Non-E2E activation continues here; perform full initialization safely
	try {
		// Migrate old settings to new
		await migrateSettings(context, outputChannel)
	} catch (err) {
		outputChannel.appendLine(`migrateSettings failed in non-E2E activation: ${String(err)}`)
	}

	// Initialize telemetry service after environment variables are loaded.
	telemetryService.initialize()

	// Initialize i18n for internationalization support
	await initializeI18n(context.globalState.get("language") ?? formatLanguage(vscode.env.language))

	// Initialize terminal shell execution handlers.
	TerminalRegistry.initialize()

	// Get default commands from configuration.
	const defaultCommands =
		vscode.workspace.getConfiguration((configSection as () => string)()).get<string[]>("allowedCommands") || []

	// Initialize global state if not already set.
	if (!context.globalState.get("allowedCommands")) {
		context.globalState.update("allowedCommands", defaultCommands)
	}

	outputChannel.appendLine("Creating TheaProvider...")
	provider = new TheaProvider(context, outputChannel, "sidebar") // assign to shared variable
	outputChannel.appendLine("TheaProvider created")
	if (!isE2E) {
		telemetryService.setProvider(provider!)
	}

	context.subscriptions.push(
		vscode.window.registerWebviewViewProvider(String(TheaProvider.sideBarId), provider!, {
			webviewOptions: { retainContextWhenHidden: true },
		}),
	)

	registerCommands({ context, outputChannel, provider: provider! })

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
	const diffContentProvider = new (class implements vscode.TextDocumentContentProvider {
		provideTextDocumentContent(uri: vscode.Uri): string {
			return Buffer.from(uri.query, "base64").toString("utf-8")
		}
	})()

	context.subscriptions.push(
		vscode.workspace.registerTextDocumentContentProvider(DIFF_VIEW_URI_SCHEME, diffContentProvider),
	)

	context.subscriptions.push(vscode.window.registerUriHandler({ handleUri }))

	// Register code actions provider.
	context.subscriptions.push(
		vscode.languages.registerCodeActionsProvider({ pattern: "**/*" }, new CodeActionProvider(), {
			providedCodeActionKinds: CodeActionProvider.providedCodeActionKinds,
		}),
	)

	registerCodeActions(context)
	registerTerminalActions(context)

	// Allows other extensions to activate once Thea is ready.
	vscode.commands.executeCommand(`${EXTENSION_NAME}.activationCompleted`)

	// Implements the `TheaCodeAPI` interface.
	return new API(outputChannel, provider!)
}

// This method is called when your extension is deactivated
export async function deactivate() {
	outputChannel.appendLine(`${EXTENSION_DISPLAY_NAME} extension deactivated`)
	// Clean up MCP server manager
	await McpServerManager.cleanup(extensionContext)
	await telemetryService.shutdown()

	// Clean up terminal handlers
	TerminalRegistry.cleanup()
}
