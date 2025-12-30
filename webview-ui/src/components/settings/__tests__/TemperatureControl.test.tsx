// npx jest src/components/settings/__tests__/TemperatureControl.test.ts

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

describe("TemperatureControl", () => {
	let sandbox: sinon.SinonSandbox
	let TemperatureControl: typeof import("../TemperatureControl").TemperatureControl

	beforeEach(() => {
		sandbox = sinon.createSandbox()

		const actualUi = proxyquireStrict("@/components/ui", {}) as typeof import("@/components/ui")
		TemperatureControl = proxyquireStrict("../TemperatureControl", {
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
		}).TemperatureControl
	})

	afterEach(() => {
		sandbox.restore()
	})

	it("renders with default temperature disabled", () => {
		const onChange = sandbox.stub()
		render(<TemperatureControl value={undefined} onChange={onChange} />)

		const checkbox = screen.getByRole("checkbox")
		expect(checkbox).not.toBeChecked()
		expect(screen.queryByRole("textbox")).not.toBeInTheDocument()
	})

	it("renders with custom temperature enabled", () => {
		const onChange = sandbox.stub()
		render(<TemperatureControl value={0.7} onChange={onChange} />)

		const checkbox = screen.getByRole("checkbox")
		expect(checkbox).toBeChecked()

		const input = screen.getByRole("slider")
		expect(input).toBeInTheDocument()
		expect(input).toHaveValue("0.7")
	})

	it("updates when checkbox is toggled", async () => {
		const onChange = sandbox.stub()
		render(<TemperatureControl value={0.7} onChange={onChange} />)

		const checkbox = screen.getByRole("checkbox")

		// Uncheck - should clear temperature.
		fireEvent.click(checkbox)

		// Waiting for debounce.
		await new Promise((x) => setTimeout(x, 100))
		expect(onChange.calledWith(null)).toBe(true)

		// Check - should restore previous temperature.
		fireEvent.click(checkbox)

		// Waiting for debounce.
		await new Promise((x) => setTimeout(x, 100))
		expect(onChange.calledWith(0.7)).toBe(true)
	})

	it("syncs checkbox state when value prop changes", () => {
		const onChange = sandbox.stub()
		const { rerender } = render(<TemperatureControl value={0.7} onChange={onChange} />)

		// Initially checked.
		const checkbox = screen.getByRole("checkbox")
		expect(checkbox).toBeChecked()

		// Update to undefined.
		rerender(<TemperatureControl value={undefined} onChange={onChange} />)
		expect(checkbox).not.toBeChecked()

		// Update back to a value.
		rerender(<TemperatureControl value={0.5} onChange={onChange} />)
		expect(checkbox).toBeChecked()
	})
})
