// npx jest src/__tests__/App.test.tsx

import React from "react"
import { render, screen, act, cleanup } from "@testing-library/react"
import "@testing-library/jest-dom"
import sinon from "sinon"
import proxyquire from "proxyquire"

const proxyquireStrict = proxyquire.noPreserveCache()

describe("App", () => {
	let sandbox: sinon.SinonSandbox
	let AppWithProviders: typeof import("../App").default

	beforeEach(() => {
		sandbox = sinon.createSandbox()
		window.removeEventListener("message", () => {})

		AppWithProviders = proxyquireStrict("../App", {
			"../utils/vscode": {
				vscode: {
					postMessage: sandbox.stub(),
				},
			},
			"../components/chat/ChatView": {
				__esModule: true,
				default: function ChatView({ isHidden }: { isHidden: boolean }) {
					return (
						<div data-testid="chat-view" data-hidden={isHidden}>
							Chat View
						</div>
					)
				},
			},
			"../components/settings/SettingsView": {
				__esModule: true,
				default: function SettingsView({ onDone }: { onDone: () => void }) {
					return (
						<div data-testid="settings-view" onClick={onDone}>
							Settings View
						</div>
					)
				},
			},
			"../components/history/HistoryView": {
				__esModule: true,
				default: function HistoryView({ onDone }: { onDone: () => void }) {
					return (
						<div data-testid="history-view" onClick={onDone}>
							History View
						</div>
					)
				},
			},
			"../components/mcp/McpView": {
				__esModule: true,
				default: function McpView({ onDone }: { onDone: () => void }) {
					return (
						<div data-testid="mcp-view" onClick={onDone}>
							MCP View
						</div>
					)
				},
			},
			"../components/prompts/PromptsView": {
				__esModule: true,
				default: function PromptsView({ onDone }: { onDone: () => void }) {
					return (
						<div data-testid="prompts-view" onClick={onDone}>
							Prompts View
						</div>
					)
				},
			},
			"../context/ExtensionStateContext": {
				useExtensionState: () => ({
					didHydrateState: true,
					showWelcome: false,
					shouldShowAnnouncement: false,
				}),
				ExtensionStateContextProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
			},
		}).default
	})

	afterEach(() => {
		cleanup()
		window.removeEventListener("message", () => {})
		sandbox.restore()
	})

	const triggerMessage = (action: string) => {
		const messageEvent = new MessageEvent("message", {
			data: {
				type: "action",
				action,
			},
		})
		window.dispatchEvent(messageEvent)
	}

	it("shows chat view by default", () => {
		render(<AppWithProviders />)

		const chatView = screen.getByTestId("chat-view")
		expect(chatView).toBeInTheDocument()
		expect(chatView.getAttribute("data-hidden")).toBe("false")
	})

	it("switches to settings view when receiving settingsButtonClicked action", async () => {
		render(<AppWithProviders />)

		act(() => {
			triggerMessage("settingsButtonClicked")
		})

		const settingsView = await screen.findByTestId("settings-view")
		expect(settingsView).toBeInTheDocument()

		const chatView = screen.getByTestId("chat-view")
		expect(chatView.getAttribute("data-hidden")).toBe("true")
	})

	it("switches to history view when receiving historyButtonClicked action", async () => {
		render(<AppWithProviders />)

		act(() => {
			triggerMessage("historyButtonClicked")
		})

		const historyView = await screen.findByTestId("history-view")
		expect(historyView).toBeInTheDocument()

		const chatView = screen.getByTestId("chat-view")
		expect(chatView.getAttribute("data-hidden")).toBe("true")
	})

	it("switches to MCP view when receiving mcpButtonClicked action", async () => {
		render(<AppWithProviders />)

		act(() => {
			triggerMessage("mcpButtonClicked")
		})

		const mcpView = await screen.findByTestId("mcp-view")
		expect(mcpView).toBeInTheDocument()

		const chatView = screen.getByTestId("chat-view")
		expect(chatView.getAttribute("data-hidden")).toBe("true")
	})

	it("switches to prompts view when receiving promptsButtonClicked action", async () => {
		render(<AppWithProviders />)

		act(() => {
			triggerMessage("promptsButtonClicked")
		})

		const promptsView = await screen.findByTestId("prompts-view")
		expect(promptsView).toBeInTheDocument()

		const chatView = screen.getByTestId("chat-view")
		expect(chatView.getAttribute("data-hidden")).toBe("true")
	})

	it("returns to chat view when clicking done in settings view", async () => {
		render(<AppWithProviders />)

		act(() => {
			triggerMessage("settingsButtonClicked")
		})

		const settingsView = await screen.findByTestId("settings-view")

		act(() => {
			settingsView.click()
		})

		const chatView = screen.getByTestId("chat-view")
		expect(chatView.getAttribute("data-hidden")).toBe("false")
		expect(screen.queryByTestId("settings-view")).not.toBeInTheDocument()
	})

	for (const view of ["history", "mcp", "prompts"]) {
		it(`returns to chat view when clicking done in ${view} view`, async () => {
			render(<AppWithProviders />)

			act(() => {
				triggerMessage(`${view}ButtonClicked`)
			})

			const viewElement = await screen.findByTestId(`${view}-view`)

			act(() => {
				viewElement.click()
			})

			const chatView = screen.getByTestId("chat-view")
			expect(chatView.getAttribute("data-hidden")).toBe("false")
			expect(screen.queryByTestId(`${view}-view`)).not.toBeInTheDocument()
		})
	}
})
