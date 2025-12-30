// npx jest src/components/settings/__tests__/ContextManagementSettings.test.ts

import { render, screen, fireEvent } from "@testing-library/react"
import sinon from "sinon"
import proxyquire from "proxyquire"

const proxyquireStrict = proxyquire.noPreserveCache()

class MockResizeObserver {
	observe() {}
	unobserve() {}
	disconnect() {}
}

global.ResizeObserver = MockResizeObserver

describe("ContextManagementSettings", () => {
	let sandbox: sinon.SinonSandbox
	let ContextManagementSettings: typeof import("../ContextManagementSettings").ContextManagementSettings

	beforeEach(() => {
		sandbox = sinon.createSandbox()
		const actualUi = proxyquireStrict("@/components/ui", {}) as typeof import("@/components/ui")
		ContextManagementSettings = proxyquireStrict("../ContextManagementSettings", {
			"@/components/ui": {
				...actualUi,
				Slider: ({
					value,
					onValueChange,
					"data-testid": dataTestId,
				}: {
					value: number[]
					onValueChange: (value: number[]) => void
					"data-testid"?: string
				}) => (
					<input
						type="range"
						value={value[0]}
						onChange={(e) => onValueChange([parseFloat(e.target.value)])}
						data-testid={dataTestId}
					/>
				),
			},
		}).ContextManagementSettings
	})

	afterEach(() => {
		sandbox.restore()
	})

	const buildDefaultProps = () => ({
		maxOpenTabsContext: 20,
		maxWorkspaceFiles: 200,
		showTheaIgnoredFiles: false,
		setCachedStateField: sandbox.stub(),
	})

	it("renders all controls", () => {
		const defaultProps = buildDefaultProps()
		render(<ContextManagementSettings {...defaultProps} />)

		// Open tabs context limit
		const openTabsSlider = screen.getByTestId("open-tabs-limit-slider")
		expect(openTabsSlider).toBeInTheDocument()

		// Workspace files limit
		const workspaceFilesSlider = screen.getByTestId("workspace-files-limit-slider")
		expect(workspaceFilesSlider).toBeInTheDocument()

		// Show .thea_ignore'd files
		const showTheaIgnoredFilesCheckbox = screen.getByTestId("show-theaignored-files-checkbox")
		expect(showTheaIgnoredFilesCheckbox).toBeInTheDocument()
		expect(screen.getByTestId("show-theaignored-files-checkbox")).not.toBeChecked()
	})

	it("updates open tabs context limit", () => {
		const defaultProps = buildDefaultProps()
		render(<ContextManagementSettings {...defaultProps} />)

		const slider = screen.getByTestId("open-tabs-limit-slider")
		fireEvent.change(slider, { target: { value: "50" } })

		expect(defaultProps.setCachedStateField.calledWith("maxOpenTabsContext", 50)).toBe(true)
	})

	it("updates workspace files contextlimit", () => {
		const defaultProps = buildDefaultProps()
		render(<ContextManagementSettings {...defaultProps} />)

		const slider = screen.getByTestId("workspace-files-limit-slider")
		fireEvent.change(slider, { target: { value: "50" } })

		expect(defaultProps.setCachedStateField.calledWith("maxWorkspaceFiles", 50)).toBe(true)
	})

	it("updates show theaignored files setting", () => {
		const defaultProps = buildDefaultProps()
		render(<ContextManagementSettings {...defaultProps} />)

		const checkbox = screen.getByTestId("show-theaignored-files-checkbox")
		fireEvent.click(checkbox)

		expect(defaultProps.setCachedStateField.calledWith("showTheaIgnoredFiles", true)).toBe(true)
	})
})
