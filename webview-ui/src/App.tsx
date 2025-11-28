import { useCallback, useEffect, useRef, useState } from "react"
import { useEvent } from "react-use"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"

import { ExtensionMessage } from "../../src/shared/ExtensionMessage"
import TranslationProvider from "./i18n/TranslationContext"

import { vscode } from "./utils/vscode"
import { telemetryClient } from "./utils/TelemetryClient"
import { ExtensionStateContextProvider, useExtensionState } from "./context/ExtensionStateContext"
import ChatView from "./components/chat/ChatView"
import HistoryView from "./components/history/HistoryView"
import SettingsView, { SettingsViewRef } from "./components/settings/SettingsView"
import WelcomeView from "./components/welcome/WelcomeView"
import McpView from "./components/mcp/McpView"
import PromptsView from "./components/prompts/PromptsView"

type Tab = "settings" | "history" | "mcp" | "prompts" | "chat"

const tabsByMessageAction: Partial<Record<NonNullable<ExtensionMessage["action"]>, Tab>> = {
	chatButtonClicked: "chat",
	settingsButtonClicked: "settings",
	promptsButtonClicked: "prompts",
	mcpButtonClicked: "mcp",
	historyButtonClicked: "history",
}

const App = () => {
	const { didHydrateState, showWelcome, shouldShowAnnouncement, telemetrySetting, telemetryKey, machineId } =
		useExtensionState()

	const [showAnnouncement, setShowAnnouncement] = useState(false)
	const [tab, setTab] = useState<Tab>("chat")

	const settingsRef = useRef<SettingsViewRef>(null)

	const switchTab = useCallback((newTab: Tab) => {
		if (settingsRef.current?.checkUnsaveChanges) {
			settingsRef.current.checkUnsaveChanges(() => setTab(newTab))
		} else {
			setTab(newTab)
		}
	}, [])

	const onMessage = useCallback(
		(e: MessageEvent) => {
			const message: ExtensionMessage = e.data

			if (message.type === "action" && message.action) {
				const newTab = tabsByMessageAction[message.action]

				// Don't switch tabs while in WelcomeView - user must configure API first
				if (newTab && !showWelcome) {
					switchTab(newTab)
				}
			}
		},
		[switchTab, showWelcome],
	)

	useEvent("message", onMessage)

	useEffect(() => {
		if (shouldShowAnnouncement) {
			setShowAnnouncement(true)
			vscode.postMessage({ type: "didShowAnnouncement" })
		}
	}, [shouldShowAnnouncement])

	useEffect(() => {
		if (didHydrateState) {
			telemetryClient.updateTelemetryState(telemetrySetting, telemetryKey, machineId)
		}
	}, [telemetrySetting, telemetryKey, machineId, didHydrateState])

	// Tell the extension that we are ready to receive messages.
	useEffect(() => vscode.postMessage({ type: "webviewDidLaunch" }), [])

	if (!didHydrateState) {
		// Show loading indicator while waiting for initial state
		return (
			<div
				style={{
					display: "flex",
					justifyContent: "center",
					alignItems: "center",
					height: "100vh",
					flexDirection: "column",
					gap: "16px",
				}}
			>
				<div className="codicon codicon-loading codicon-modifier-spin" style={{ fontSize: "32px" }} />
				<div style={{ opacity: 0.7 }}>Loading Thea Code...</div>
			</div>
		)
	}

	// Do not conditionally load ChatView, it's expensive and there's state we
	// don't want to lose (user input, disableInput, askResponse promise, etc.)
	// ChatView is always rendered but hidden when showWelcome is true or when another tab is active
	const { renderContext } = useExtensionState()

	return (
		<div
			style={{
				height: "100vh",
				display: "flex",
				flexDirection: "column",
				overflow: "hidden", // Ensure main container doesn't scroll, let children handle it
			}}
		>
			{showWelcome && <WelcomeView />}

			{/* Non-Chat Views: Wrap in a scrollable container with adaptive layout */}
			{!showWelcome && tab !== "chat" && (
				<div
					className={`flex-1 overflow-y-auto ${renderContext === "editor"
							? "w-full max-w-5xl mx-auto p-6" // Editor mode: Centered, constrained width, padded
							: "w-full p-0"                   // Sidebar mode: Full width, no padding
						}`}
				>
					{tab === "prompts" && <PromptsView onDone={() => switchTab("chat")} />}
					{tab === "mcp" && <McpView onDone={() => switchTab("chat")} />}
					{tab === "history" && <HistoryView onDone={() => switchTab("chat")} />}
					{tab === "settings" && <SettingsView ref={settingsRef} onDone={() => switchTab("chat")} />}
				</div>
			)}

			<ChatView
				isHidden={showWelcome || tab !== "chat"}
				showAnnouncement={showAnnouncement}
				hideAnnouncement={() => setShowAnnouncement(false)}
				showHistoryView={() => switchTab("history")}
			/>
		</div>
	)
}

const queryClient = new QueryClient()

const AppWithProviders = () => (
	<ExtensionStateContextProvider>
		<TranslationProvider>
			<QueryClientProvider client={queryClient}>
				<App />
			</QueryClientProvider>
		</TranslationProvider>
	</ExtensionStateContextProvider>
)

export default AppWithProviders
