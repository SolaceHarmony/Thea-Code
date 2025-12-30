import React from "react"
import { render, fireEvent, screen } from "@testing-library/react"
import sinon from "sinon"
import proxyquire from "proxyquire"

const proxyquireStrict = proxyquire.noPreserveCache()

describe("McpToolRow", () => {
	let sandbox: sinon.SinonSandbox
	let McpToolRow: typeof import("../McpToolRow").default
	let vscode: { postMessage: sinon.SinonStub }

	const mockTool = {
		name: "test-tool",
		description: "A test tool",
		alwaysAllow: false,
	}

	beforeEach(() => {
		sandbox = sinon.createSandbox()
		vscode = { postMessage: sandbox.stub() }

		McpToolRow = proxyquireStrict("../McpToolRow", {
			"../../../i18n/TranslationContext": {
				useAppTranslation: () => ({
					t: (key: string) => {
						const translations: Record<string, string> = {
							"mcp:tool.alwaysAllow": "Always allow",
							"mcp:tool.parameters": "Parameters",
							"mcp:tool.noDescription": "No description",
						}
						return translations[key] || key
					},
				}),
			},
			"../../../utils/vscode": {
				vscode,
			},
			"@/components/ui/vscode-components": {
				VSCodeCheckbox: function MockVSCodeCheckbox({
					children,
					checked,
					onChange,
				}: {
					children?: React.ReactNode
					checked?: boolean
					onChange?: (e: React.ChangeEvent<HTMLInputElement>) => void
				}) {
					return (
						<label>
							<input type="checkbox" checked={checked} onChange={onChange} />
							{children}
						</label>
					)
				},
			},
		}).default
	})

	afterEach(() => {
		sandbox.restore()
	})

	it("renders tool name and description", () => {
		render(<McpToolRow tool={mockTool} />)

		expect(screen.getByText("test-tool")).toBeInTheDocument()
		expect(screen.getByText("A test tool")).toBeInTheDocument()
	})

	it("does not show always allow checkbox when serverName is not provided", () => {
		render(<McpToolRow tool={mockTool} />)

		expect(screen.queryByText("Always allow")).not.toBeInTheDocument()
	})

	it("shows always allow checkbox when serverName and alwaysAllowMcp are provided", () => {
		render(<McpToolRow tool={mockTool} serverName="test-server" alwaysAllowMcp={true} />)

		expect(screen.getByText("Always allow")).toBeInTheDocument()
	})

	it("sends message to toggle always allow when checkbox is clicked", () => {
		render(<McpToolRow tool={mockTool} serverName="test-server" alwaysAllowMcp={true} />)

		const checkbox = screen.getByRole("checkbox")
		fireEvent.click(checkbox)

		expect(
			vscode.postMessage.calledWith({
				type: "toggleToolAlwaysAllow",
				serverName: "test-server",
				toolName: "test-tool",
				alwaysAllow: true,
				source: "global",
			}),
		).toBe(true)
	})

	it("reflects always allow state in checkbox", () => {
		const alwaysAllowedTool = {
			...mockTool,
			alwaysAllow: true,
		}

		render(<McpToolRow tool={alwaysAllowedTool} serverName="test-server" alwaysAllowMcp={true} />)

		const checkbox = screen.getByRole("checkbox") as HTMLInputElement
		expect(checkbox.checked).toBe(true)
	})

	it("prevents event propagation when clicking the checkbox", () => {
		const mockOnClick = sandbox.stub()
		render(
			<div onClick={mockOnClick}>
				<McpToolRow tool={mockTool} serverName="test-server" alwaysAllowMcp={true} />
			</div>,
		)

		const container = screen.getByTestId("tool-row-container")
		fireEvent.click(container)

		expect(mockOnClick.called).toBe(false)
	})

	it("displays input schema parameters when provided", () => {
		const toolWithSchema = {
			...mockTool,
			inputSchema: {
				type: "object",
				properties: {
					param1: {
						type: "string",
						description: "First parameter",
					},
					param2: {
						type: "number",
						description: "Second parameter",
					},
				},
				required: ["param1"],
			},
		}

		render(<McpToolRow tool={toolWithSchema} serverName="test-server" />)

		expect(screen.getByText("Parameters")).toBeInTheDocument()
		expect(screen.getByText("param1")).toBeInTheDocument()
		expect(screen.getByText("param2")).toBeInTheDocument()
		expect(screen.getByText("First parameter")).toBeInTheDocument()
		expect(screen.getByText("Second parameter")).toBeInTheDocument()
	})
})
