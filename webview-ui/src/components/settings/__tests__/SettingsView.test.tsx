// npx jest src/components/settings/__tests__/SettingsView.test.ts

import React from "react"
import { render, screen, fireEvent } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"

import { vscode } from "@/utils/vscode"
import { ExtensionStateContextProvider } from "@/context/ExtensionStateContext"

import SettingsView from "../SettingsView"

// Mock vscode API
jest.mock("../../../utils/vscode", () => ({
	vscode: {
		postMessage: jest.fn(),
	},
}))

// Mock all lucide-react icons with a proxy to handle any icon requested
jest.mock("lucide-react", () => {
	return new Proxy(
		{},
		{
			get: function (obj, prop) {
				// Return a component factory for any icon that's requested
				if (prop === "__esModule") {
					return true
				}
				return () => <div data-testid={`${String(prop)}-icon`}>{String(prop)}</div>
			},
		},
	)
})

// Mock ApiConfigManager component
jest.mock("../ApiConfigManager", () => ({
	__esModule: true,
	default: ({ currentApiConfigName }: { [key: string]: unknown }) => (
		<div data-testid="api-config-management">
			<span>Current config: {currentApiConfigName as React.ReactNode}</span>
		</div>
	),
}))

// Mock VSCode components
jest.mock("@/components/ui/vscode-components", () => ({
	VSCodeButton: ({ children, onClick, appearance, "data-testid": dataTestId }: { [key: string]: unknown }) =>
		appearance === "icon" ? (
			<button
				onClick={onClick as React.MouseEventHandler<HTMLButtonElement>}
				className="codicon codicon-close"
				aria-label="Remove command"
				data-testid={dataTestId as string}>
				<span className="codicon codicon-close" />
			</button>
		) : (
			<button onClick={onClick as React.MouseEventHandler<HTMLButtonElement>} data-appearance={appearance as string} data-testid={dataTestId as string}>
				{children as React.ReactNode}
			</button>
		),
	VSCodeCheckbox: ({ children, onChange, checked, "data-testid": dataTestId }: { [key: string]: unknown }) => (
		<label>
			<input
				type="checkbox"
				checked={checked as boolean}
				onChange={(e) => (onChange as ((event: { target: { checked: boolean } }) => void))?.(({ target: { checked: e.target.checked } }))}
				aria-label={typeof children === "string" ? children : undefined}
				data-testid={dataTestId as string}
			/>
			{children as React.ReactNode}
		</label>
	),
	VSCodeTextField: ({ value, onInput, placeholder, "data-testid": dataTestId }: { [key: string]: unknown }) => (
		<input
			type="text"
			value={value as string}
			onChange={(e) => (onInput as ((event: { target: { value: string } }) => void))?.(({ target: { value: e.target.value } }))}
			placeholder={placeholder as string}
			data-testid={dataTestId as string}
		/>
	),
	VSCodeLink: ({ children, href }: { [key: string]: unknown }) => <a href={(href as string) || "#"}>{children as React.ReactNode}</a>,
	VSCodeRadio: ({ value, checked, onChange }: { [key: string]: unknown }) => (
		<input type="radio" value={value as string} checked={checked as boolean} onChange={onChange as React.ChangeEventHandler<HTMLInputElement>} />
	),
	VSCodeRadioGroup: ({ children, onChange }: { [key: string]: unknown }) => <div onChange={onChange as React.FormEventHandler<HTMLDivElement>}>{children as React.ReactNode}</div>,
}))

// Mock Slider component
jest.mock("@/components/ui", () => ({
	...jest.requireActual("@/components/ui"),
	Slider: ({ value, onValueChange, "data-testid": dataTestId }: { [key: string]: unknown }) => (
		<input
			type="range"
			value={(value as number[])?.[0] || 0}
			onChange={(e) => (onValueChange as ((value: number[]) => void))?.(([parseFloat(e.target.value)]))}
			data-testid={dataTestId as string}
		/>
	),
}))

// Mock window.postMessage to trigger state hydration
const mockPostMessage = (state: unknown) => {
	window.postMessage(
		{
			type: "state",
			state: {
				version: "1.0.0",
				clineMessages: [],
				taskHistory: [],
				shouldShowAnnouncement: false,
				allowedCommands: [],
				alwaysAllowExecute: false,
				ttsEnabled: false,
				ttsSpeed: 1,
				soundEnabled: false,
				soundVolume: 0.5,
				...(state as Record<string, unknown>),
			},
		},
		"*",
	)
}

class MockResizeObserver {
	observe() {}
	unobserve() {}
	disconnect() {}
}

global.ResizeObserver = MockResizeObserver

const renderSettingsView = () => {
	const onDone = jest.fn()
	const queryClient = new QueryClient()

	render(
		<ExtensionStateContextProvider>
			<QueryClientProvider client={queryClient}>
				<SettingsView onDone={onDone} />
			</QueryClientProvider>
		</ExtensionStateContextProvider>,
	)

	// Hydrate initial state.
	mockPostMessage({})

	return { onDone }
}

describe("SettingsView - Sound Settings", () => {
	beforeEach(() => {
		jest.clearAllMocks()
	})

	it("initializes with tts disabled by default", () => {
		renderSettingsView()

		const ttsCheckbox = screen.getByTestId("tts-enabled-checkbox")
		expect(ttsCheckbox).not.toBeChecked()

		// Speed slider should not be visible when tts is disabled
		expect(screen.queryByTestId("tts-speed-slider")).not.toBeInTheDocument()
	})

	it("initializes with sound disabled by default", () => {
		renderSettingsView()

		const soundCheckbox = screen.getByTestId("sound-enabled-checkbox")
		expect(soundCheckbox).not.toBeChecked()

		// Volume slider should not be visible when sound is disabled
		expect(screen.queryByTestId("sound-volume-slider")).not.toBeInTheDocument()
	})

	it("toggles tts setting and sends message to VSCode", () => {
		renderSettingsView()

		const ttsCheckbox = screen.getByTestId("tts-enabled-checkbox")

		// Enable tts
		fireEvent.click(ttsCheckbox)
		expect(ttsCheckbox).toBeChecked()

		// Click Save to save settings
		const saveButton = screen.getByTestId("save-button")
		fireEvent.click(saveButton)

		expect(vscode.postMessage).toHaveBeenCalledWith(
			expect.objectContaining({
				type: "ttsEnabled",
				bool: true,
			}),
		)
	})

	it("toggles sound setting and sends message to VSCode", () => {
		renderSettingsView()

		const soundCheckbox = screen.getByTestId("sound-enabled-checkbox")

		// Enable sound
		fireEvent.click(soundCheckbox)
		expect(soundCheckbox).toBeChecked()

		// Click Save to save settings
		const saveButton = screen.getByTestId("save-button")
		fireEvent.click(saveButton)

		expect(vscode.postMessage).toHaveBeenCalledWith(
			expect.objectContaining({
				type: "soundEnabled",
				bool: true,
			}),
		)
	})

	it("shows tts slider when sound is enabled", () => {
		renderSettingsView()

		// Enable tts
		const ttsCheckbox = screen.getByTestId("tts-enabled-checkbox")
		fireEvent.click(ttsCheckbox)

		// Speed slider should be visible
		const speedSlider = screen.getByTestId("tts-speed-slider")
		expect(speedSlider).toBeInTheDocument()
		expect(speedSlider).toHaveValue("1")
	})

	it("shows volume slider when sound is enabled", () => {
		renderSettingsView()

		// Enable sound
		const soundCheckbox = screen.getByTestId("sound-enabled-checkbox")
		fireEvent.click(soundCheckbox)

		// Volume slider should be visible
		const volumeSlider = screen.getByTestId("sound-volume-slider")
		expect(volumeSlider).toBeInTheDocument()
		expect(volumeSlider).toHaveValue("0.5")
	})

	it("updates speed and sends message to VSCode when slider changes", () => {
		renderSettingsView()

		// Enable tts
		const ttsCheckbox = screen.getByTestId("tts-enabled-checkbox")
		fireEvent.click(ttsCheckbox)

		// Change speed
		const speedSlider = screen.getByTestId("tts-speed-slider")
		fireEvent.change(speedSlider, { target: { value: "0.75" } })

		// Click Save to save settings
		const saveButton = screen.getByTestId("save-button")
		fireEvent.click(saveButton)

		// Verify message sent to VSCode
		expect(vscode.postMessage).toHaveBeenCalledWith({
			type: "ttsSpeed",
			value: 0.75,
		})
	})

	it("updates volume and sends message to VSCode when slider changes", () => {
		renderSettingsView()

		// Enable sound
		const soundCheckbox = screen.getByTestId("sound-enabled-checkbox")
		fireEvent.click(soundCheckbox)

		// Change volume
		const volumeSlider = screen.getByTestId("sound-volume-slider")
		fireEvent.change(volumeSlider, { target: { value: "0.75" } })

		// Click Save to save settings
		const saveButton = screen.getByTestId("save-button")
		fireEvent.click(saveButton)

		// Verify message sent to VSCode
		expect(vscode.postMessage).toHaveBeenCalledWith({
			type: "soundVolume",
			value: 0.75,
		})
	})
})

describe("SettingsView - API Configuration", () => {
	beforeEach(() => {
		jest.clearAllMocks()
	})

	it("renders ApiConfigManagement with correct props", () => {
		renderSettingsView()

		expect(screen.getByTestId("api-config-management")).toBeInTheDocument()
	})
})

describe("SettingsView - Allowed Commands", () => {
	beforeEach(() => {
		jest.clearAllMocks()
	})

	it("shows allowed commands section when alwaysAllowExecute is enabled", () => {
		renderSettingsView()

		// Enable always allow execute
		const executeCheckbox = screen.getByTestId("always-allow-execute-checkbox")
		fireEvent.click(executeCheckbox)
		// Verify allowed commands section appears
		expect(screen.getByTestId("allowed-commands-heading")).toBeInTheDocument()
		expect(screen.getByTestId("command-input")).toBeInTheDocument()
	})

	it("adds new command to the list", () => {
		renderSettingsView()

		// Enable always allow execute
		const executeCheckbox = screen.getByTestId("always-allow-execute-checkbox")
		fireEvent.click(executeCheckbox)

		// Add a new command
		const input = screen.getByTestId("command-input")
		fireEvent.change(input, { target: { value: "npm test" } })

		const addButton = screen.getByTestId("add-command-button")
		fireEvent.click(addButton)

		// Verify command was added
		expect(screen.getByText("npm test")).toBeInTheDocument()

		// Verify VSCode message was sent
		expect(vscode.postMessage).toHaveBeenCalledWith({
			type: "allowedCommands",
			commands: ["npm test"],
		})
	})

	it("removes command from the list", () => {
		renderSettingsView()

		// Enable always allow execute
		const executeCheckbox = screen.getByTestId("always-allow-execute-checkbox")
		fireEvent.click(executeCheckbox)

		// Add a command
		const input = screen.getByTestId("command-input")
		fireEvent.change(input, { target: { value: "npm test" } })
		const addButton = screen.getByTestId("add-command-button")
		fireEvent.click(addButton)

		// Remove the command
		const removeButton = screen.getByTestId("remove-command-0")
		fireEvent.click(removeButton)

		// Verify command was removed
		expect(screen.queryByText("npm test")).not.toBeInTheDocument()

		// Verify VSCode message was sent
		expect(vscode.postMessage).toHaveBeenLastCalledWith({
			type: "allowedCommands",
			commands: [],
		})
	})

	it("prevents duplicate commands", () => {
		renderSettingsView()

		// Enable always allow execute
		const executeCheckbox = screen.getByTestId("always-allow-execute-checkbox")
		fireEvent.click(executeCheckbox)

		// Add a command twice
		const input = screen.getByTestId("command-input")
		const addButton = screen.getByTestId("add-command-button")

		// First addition
		fireEvent.change(input, { target: { value: "npm test" } })
		fireEvent.click(addButton)

		// Second addition attempt
		fireEvent.change(input, { target: { value: "npm test" } })
		fireEvent.click(addButton)

		// Verify command appears only once
		const commands = screen.getAllByText("npm test")
		expect(commands).toHaveLength(1)
	})

	it("saves allowed commands when clicking Save", () => {
		renderSettingsView()

		// Enable always allow execute
		const executeCheckbox = screen.getByTestId("always-allow-execute-checkbox")
		fireEvent.click(executeCheckbox)

		// Add a command
		const input = screen.getByTestId("command-input")
		fireEvent.change(input, { target: { value: "npm test" } })
		const addButton = screen.getByTestId("add-command-button")
		fireEvent.click(addButton)

		// Click Save
		const saveButton = screen.getByTestId("save-button")
		fireEvent.click(saveButton)

		// Verify VSCode messages were sent
		expect(vscode.postMessage).toHaveBeenCalledWith(
			expect.objectContaining({
				type: "allowedCommands",
				commands: ["npm test"],
			}),
		)
	})
})
