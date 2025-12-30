import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import sinon from "sinon"
import proxyquire from "proxyquire"
import { ExtensionStateContext, ExtensionStateContextType } from "../../../context/ExtensionStateContext"

const proxyquireStrict = proxyquire.noPreserveCache()

describe("PromptsView", () => {
	let sandbox: sinon.SinonSandbox
	let PromptsView: typeof import("../PromptsView").default
	let vscode: { postMessage: sinon.SinonStub }

	const buildMockExtensionState = () => ({
		customModePrompts: {},
		listApiConfigMeta: [
			{ id: "config1", name: "Config 1" },
			{ id: "config2", name: "Config 2" },
		],
		enhancementApiConfigId: "",
		setEnhancementApiConfigId: sandbox.stub(),
		mode: "code",
		customInstructions: "Initial instructions",
		setCustomInstructions: sandbox.stub(),
	})

	const renderPromptsView = (props = {}) => {
		const mockOnDone = sandbox.stub()
		const state = { ...buildMockExtensionState(), ...props } as Partial<ExtensionStateContextType>
		return render(
			<ExtensionStateContext.Provider value={state as ExtensionStateContextType}>
				<PromptsView onDone={mockOnDone} />
			</ExtensionStateContext.Provider>,
		)
	}

	beforeEach(() => {
		sandbox = sinon.createSandbox()
		vscode = { postMessage: sandbox.stub() }
		PromptsView = proxyquireStrict("../PromptsView", {
			"../../../utils/vscode": { vscode },
		}).default
	})

	afterEach(() => {
		sandbox.restore()
	})

	it("renders all mode tabs", () => {
		renderPromptsView()
		expect(screen.getByTestId("code-tab")).toBeInTheDocument()
		expect(screen.getByTestId("ask-tab")).toBeInTheDocument()
		expect(screen.getByTestId("architect-tab")).toBeInTheDocument()
	})

	it("defaults to current mode as active tab", () => {
		renderPromptsView({ mode: "ask" })

		const codeTab = screen.getByTestId("code-tab")
		const askTab = screen.getByTestId("ask-tab")
		const architectTab = screen.getByTestId("architect-tab")

		expect(askTab).toHaveAttribute("data-active", "true")
		expect(codeTab).toHaveAttribute("data-active", "false")
		expect(architectTab).toHaveAttribute("data-active", "false")
	})

	it("switches between tabs correctly", async () => {
		const state = buildMockExtensionState()
		const { rerender } = render(
			<ExtensionStateContext.Provider
				value={
					{
						...state,
						mode: "code",
					} as Partial<ExtensionStateContextType> as ExtensionStateContextType
				}>
				<PromptsView onDone={sandbox.stub()} />
			</ExtensionStateContext.Provider>,
		)

		const codeTab = screen.getByTestId("code-tab")
		const askTab = screen.getByTestId("ask-tab")
		const architectTab = screen.getByTestId("architect-tab")

		// Initial state matches current mode (code)
		expect(codeTab).toHaveAttribute("data-active", "true")
		expect(askTab).toHaveAttribute("data-active", "false")
		expect(architectTab).toHaveAttribute("data-active", "false")

		// Click Ask tab and update context
		fireEvent.click(askTab)
		rerender(
			<ExtensionStateContext.Provider
				value={
					{
						...state,
						mode: "ask",
					} as Partial<ExtensionStateContextType> as ExtensionStateContextType
				}>
				<PromptsView onDone={sandbox.stub()} />
			</ExtensionStateContext.Provider>,
		)

		expect(askTab).toHaveAttribute("data-active", "true")
		expect(codeTab).toHaveAttribute("data-active", "false")
		expect(architectTab).toHaveAttribute("data-active", "false")

		// Click Architect tab and update context
		fireEvent.click(architectTab)
		rerender(
			<ExtensionStateContext.Provider
				value={
					{
						...state,
						mode: "architect",
					} as Partial<ExtensionStateContextType> as ExtensionStateContextType
				}>
				<PromptsView onDone={sandbox.stub()} />
			</ExtensionStateContext.Provider>,
		)

		expect(architectTab).toHaveAttribute("data-active", "true")
		expect(askTab).toHaveAttribute("data-active", "false")
		expect(codeTab).toHaveAttribute("data-active", "false")
	})

	it("handles prompt changes correctly", async () => {
		renderPromptsView()

		// Get the textarea
		const textarea = await waitFor(() => screen.getByTestId("code-prompt-textarea"))
		fireEvent.change(textarea, {
			target: { value: "New prompt value" },
		})

		expect(
			vscode.postMessage.calledWith({
				type: "updatePrompt",
				promptMode: "code",
				customPrompt: { roleDefinition: "New prompt value" },
			}),
		).toBe(true)
	})

	it("resets role definition only for built-in modes", async () => {
		const customMode = {
			slug: "custom-mode",
			name: "Custom Mode",
			roleDefinition: "Custom role",
			groups: [],
		}

		// Test with built-in mode (code)
		const { unmount } = render(
			<ExtensionStateContext.Provider
				value={
					{
						...buildMockExtensionState(),
						mode: "code",
						customModes: [customMode],
					} as Partial<ExtensionStateContextType> as ExtensionStateContextType
				}>
				<PromptsView onDone={sandbox.stub()} />
			</ExtensionStateContext.Provider>,
		)

		// Find and click the role definition reset button
		const resetButton = screen.getByTestId("role-definition-reset")
		expect(resetButton).toBeInTheDocument()
		fireEvent.click(resetButton)

		// Verify it only resets role definition
		expect(
			vscode.postMessage.calledWith({
				type: "updatePrompt",
				promptMode: "code",
				customPrompt: { roleDefinition: undefined },
			}),
		).toBe(true)

		// Cleanup before testing custom mode
		unmount()

		// Test with custom mode
		render(
			<ExtensionStateContext.Provider
				value={
					{
						...buildMockExtensionState(),
						mode: "custom-mode",
						customModes: [customMode],
					} as Partial<ExtensionStateContextType> as ExtensionStateContextType
				}>
				<PromptsView onDone={sandbox.stub()} />
			</ExtensionStateContext.Provider>,
		)

		// Verify reset button is not present for custom mode
		expect(screen.queryByTestId("role-definition-reset")).not.toBeInTheDocument()
	})

	it("handles API configuration selection", async () => {
		const state = buildMockExtensionState()
		renderPromptsView(state)

		// Click the ENHANCE tab first to show the API config dropdown
		const enhanceTab = screen.getByTestId("ENHANCE-tab")
		fireEvent.click(enhanceTab)

		// Wait for the ENHANCE tab click to take effect
		const dropdown = await waitFor(() => screen.getByTestId("api-config-dropdown"))
		fireEvent.change(dropdown, {
			target: { value: "config1" },
		})

		expect(state.setEnhancementApiConfigId.calledWith("config1")).toBe(true)
		expect(vscode.postMessage.calledWith({ type: "enhancementApiConfigId", text: "config1" })).toBe(true)
	})

	it("handles clearing custom instructions correctly", async () => {
		const setCustomInstructions = sandbox.stub()
		renderPromptsView({
			...buildMockExtensionState(),
			customInstructions: "Initial instructions",
			setCustomInstructions,
		})

		const textarea = screen.getByTestId("global-custom-instructions-textarea")
		fireEvent.change(textarea, {
			target: { value: "" },
		})

		expect(setCustomInstructions.calledWith(undefined)).toBe(true)
		expect(vscode.postMessage.calledWith({ type: "customInstructions", text: undefined })).toBe(true)
	})
})
