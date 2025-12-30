import { render, screen, fireEvent } from "@testing-library/react"
import sinon from "sinon"
import proxyquire from "proxyquire"
import { ModelInfo } from "../../../../../src/shared/api"

const proxyquireStrict = proxyquire.noPreserveCache()

describe("ThinkingBudget", () => {
	let sandbox: sinon.SinonSandbox
	let ThinkingBudget: typeof import("../ThinkingBudget").ThinkingBudget

	const mockModelInfo: ModelInfo = {
		thinking: true,
		maxTokens: 16384,
		contextWindow: 200000,
		supportsPromptCache: true,
		supportsImages: true,
	}

	beforeEach(() => {
		sandbox = sinon.createSandbox()

		ThinkingBudget = proxyquireStrict("../ThinkingBudget", {
			"@/components/ui": {
				Slider: ({
					value,
					onValueChange,
					min,
					max,
				}: {
					value: number[]
					onValueChange: (value: number[]) => void
					min: number
					max: number
				}) => (
					<input
						type="range"
						data-testid="slider"
						min={min}
						max={max}
						value={value[0]}
						onChange={(e) => onValueChange([parseInt(e.target.value)])}
					/>
				),
			},
		}).ThinkingBudget
	})

	afterEach(() => {
		sandbox.restore()
	})

	const buildDefaultProps = () => ({
		apiConfiguration: {},
		setApiConfigurationField: sandbox.stub(),
		modelInfo: mockModelInfo,
	})

	it("should render nothing when model doesn't support thinking", () => {
		const defaultProps = buildDefaultProps()
		const { container } = render(
			<ThinkingBudget
				{...defaultProps}
				modelInfo={{
					...mockModelInfo,
					thinking: false,
					maxTokens: 16384,
					contextWindow: 200000,
					supportsPromptCache: true,
					supportsImages: true,
				}}
			/>,
		)

		expect(container.firstChild).toBeNull()
	})

	it("should render sliders when model supports thinking", () => {
		const defaultProps = buildDefaultProps()
		render(<ThinkingBudget {...defaultProps} />)

		expect(screen.getAllByTestId("slider")).toHaveLength(2)
	})

	it("should update modelMaxThinkingTokens", () => {
		const setApiConfigurationField = sandbox.stub()

		render(
			<ThinkingBudget
				{...buildDefaultProps()}
				apiConfiguration={{ modelMaxThinkingTokens: 4096 }}
				setApiConfigurationField={setApiConfigurationField}
			/>,
		)

		const sliders = screen.getAllByTestId("slider")
		fireEvent.change(sliders[1], { target: { value: "5000" } })

		expect(setApiConfigurationField.calledWith("modelMaxThinkingTokens", 5000)).toBe(true)
	})

	it("should cap thinking tokens at 80% of max tokens", () => {
		const setApiConfigurationField = sandbox.stub()

		render(
			<ThinkingBudget
				{...buildDefaultProps()}
				apiConfiguration={{ modelMaxTokens: 10000, modelMaxThinkingTokens: 9000 }}
				setApiConfigurationField={setApiConfigurationField}
			/>,
		)

		// Effect should trigger and cap the value
		expect(setApiConfigurationField.calledWith("modelMaxThinkingTokens", 8000)).toBe(true) // 80% of 10000
	})

	it("should use default thinking tokens if not provided", () => {
		render(<ThinkingBudget {...buildDefaultProps()} apiConfiguration={{ modelMaxTokens: 10000 }} />)

		// Default is 80% of max tokens, capped at 8192
		const sliders = screen.getAllByTestId("slider")
		expect(sliders[1]).toHaveValue("8000") // 80% of 10000
	})

	it("should use min thinking tokens of 1024", () => {
		render(<ThinkingBudget {...buildDefaultProps()} apiConfiguration={{ modelMaxTokens: 1000 }} />)

		const sliders = screen.getAllByTestId("slider")
		expect(sliders[1].getAttribute("min")).toBe("1024")
	})

	it("should update max tokens when slider changes", () => {
		const setApiConfigurationField = sandbox.stub()

		render(
			<ThinkingBudget
				{...buildDefaultProps()}
				apiConfiguration={{ modelMaxTokens: 10000 }}
				setApiConfigurationField={setApiConfigurationField}
			/>,
		)

		const sliders = screen.getAllByTestId("slider")
		fireEvent.change(sliders[0], { target: { value: "12000" } })

		expect(setApiConfigurationField.calledWith("modelMaxTokens", 12000)).toBe(true)
	})
})
