// npx jest src/components/settings/__tests__/ApiConfigManager.test.tsx

import React from "react"
import { render, screen, fireEvent, within } from "@testing-library/react"
import sinon from "sinon"
import proxyquire from "proxyquire"

const proxyquireStrict = proxyquire.noPreserveCache()

describe("ApiConfigManager", () => {
	let sandbox: sinon.SinonSandbox
	let ApiConfigManager: typeof import("../ApiConfigManager").default

	beforeEach(() => {
		sandbox = sinon.createSandbox()
		const actualUi = proxyquireStrict("@/components/ui", {}) as typeof import("@/components/ui")

		ApiConfigManager = proxyquireStrict("../ApiConfigManager", {
			"@/components/ui/vscode-components": {
				VSCodeTextField: ({
					value,
					onInput,
					placeholder,
					onKeyDown,
					"data-testid": dataTestId,
				}: {
					[key: string]: unknown
				}) => (
					<input
						value={value as string}
						onChange={(e) => (onInput as ((e: React.ChangeEvent<HTMLInputElement>) => void))?.(e)}
						placeholder={placeholder as string}
						onKeyDown={onKeyDown as React.KeyboardEventHandler<HTMLInputElement>}
						data-testid={dataTestId as string}
						ref={undefined}
					/>
				),
			},
			"@/components/ui": {
				...actualUi,
				Dialog: ({ children, open }: { [key: string]: unknown }) => (
					<div role="dialog" aria-modal="true" style={{ display: open ? "block" : "none" }} data-testid="dialog">
						{children as React.ReactNode}
					</div>
				),
				DialogContent: ({ children }: { [key: string]: unknown }) => (
					<div data-testid="dialog-content">{children as React.ReactNode}</div>
				),
				DialogTitle: ({ children }: { [key: string]: unknown }) => (
					<div data-testid="dialog-title">{children as React.ReactNode}</div>
				),
				Button: ({ children, onClick, disabled, "data-testid": dataTestId }: { [key: string]: unknown }) => (
					<button
						onClick={onClick as React.MouseEventHandler<HTMLButtonElement>}
						disabled={disabled as boolean}
						data-testid={dataTestId as string}>
						{children as React.ReactNode}
					</button>
				),
				Input: ({ value, onInput, placeholder, onKeyDown, "data-testid": dataTestId }: { [key: string]: unknown }) => (
					<input
						value={value as string}
						onChange={(e) => (onInput as ((e: React.ChangeEvent<HTMLInputElement>) => void))?.(e)}
						placeholder={placeholder as string}
						onKeyDown={onKeyDown as React.KeyboardEventHandler<HTMLInputElement>}
						data-testid={dataTestId as string}
					/>
				),
				Select: ({ value, onValueChange }: { [key: string]: unknown }) => (
					<select
						value={value as string}
						onChange={(e) => {
							if (onValueChange) (onValueChange as ((value: string) => void))(e.target.value)
						}}
						data-testid="select-component">
						<option value="Default Config">Default Config</option>
						<option value="Another Config">Another Config</option>
					</select>
				),
				SelectTrigger: ({ children }: { [key: string]: unknown }) => (
					<div className="select-trigger-mock">{children as React.ReactNode}</div>
				),
				SelectValue: ({ children }: { [key: string]: unknown }) => (
					<div className="select-value-mock">{children as React.ReactNode}</div>
				),
				SelectContent: ({ children }: { [key: string]: unknown }) => (
					<div className="select-content-mock">{children as React.ReactNode}</div>
				),
				SelectItem: ({ children, value }: { [key: string]: unknown }) => (
					<option value={value as string} className="select-item-mock">
						{children as React.ReactNode}
					</option>
				),
			},
		}).default
	})

	afterEach(() => {
		sandbox.restore()
	})

	const buildDefaultProps = () => ({
		currentApiConfigName: "Default Config",
		listApiConfigMeta: [
			{ id: "default", name: "Default Config" },
			{ id: "another", name: "Another Config" },
		],
		onSelectConfig: sandbox.stub(),
		onDeleteConfig: sandbox.stub(),
		onRenameConfig: sandbox.stub(),
		onUpsertConfig: sandbox.stub(),
	})

	const getRenameForm = () => screen.getByTestId("rename-form")
	const getDialogContent = () => screen.getByTestId("dialog-content")

	it("opens new profile dialog when clicking add button", () => {
		const defaultProps = buildDefaultProps()
		render(<ApiConfigManager {...defaultProps} />)

		const addButton = screen.getByTestId("add-profile-button")
		fireEvent.click(addButton)

		expect(screen.getByTestId("dialog")).toBeVisible()
		expect(screen.getByTestId("dialog-title")).toHaveTextContent("settings:providers.newProfile")
	})

	it("creates new profile with entered name", () => {
		const defaultProps = buildDefaultProps()
		render(<ApiConfigManager {...defaultProps} />)

		// Open dialog
		const addButton = screen.getByTestId("add-profile-button")
		fireEvent.click(addButton)

		// Enter new profile name
		const input = screen.getByTestId("new-profile-input")
		fireEvent.input(input, { target: { value: "New Profile" } })

		// Click create button
		const createButton = screen.getByText("settings:providers.createProfile")
		fireEvent.click(createButton)

		expect(defaultProps.onUpsertConfig.calledWith("New Profile")).toBe(true)
	})

	it("shows error when creating profile with existing name", () => {
		const defaultProps = buildDefaultProps()
		render(<ApiConfigManager {...defaultProps} />)

		// Open dialog
		const addButton = screen.getByTestId("add-profile-button")
		fireEvent.click(addButton)

		// Enter existing profile name
		const input = screen.getByTestId("new-profile-input")
		fireEvent.input(input, { target: { value: "Default Config" } })

		// Click create button to trigger validation
		const createButton = screen.getByText("settings:providers.createProfile")
		fireEvent.click(createButton)

		// Verify error message
		const dialogContent = getDialogContent()
		const errorMessage = within(dialogContent).getByTestId("error-message")
		expect(errorMessage).toHaveTextContent("settings:providers.nameExists")
		expect(defaultProps.onUpsertConfig.called).toBe(false)
	})

	it("prevents creating profile with empty name", () => {
		const defaultProps = buildDefaultProps()
		render(<ApiConfigManager {...defaultProps} />)

		// Open dialog
		const addButton = screen.getByTestId("add-profile-button")
		fireEvent.click(addButton)

		// Enter empty name
		const input = screen.getByTestId("new-profile-input")
		fireEvent.input(input, { target: { value: "   " } })

		// Verify create button is disabled
		const createButton = screen.getByText("settings:providers.createProfile")
		expect(createButton).toBeDisabled()
		expect(defaultProps.onUpsertConfig.called).toBe(false)
	})

	it("allows renaming the current config", () => {
		const defaultProps = buildDefaultProps()
		render(<ApiConfigManager {...defaultProps} />)

		// Start rename
		const renameButton = screen.getByTestId("rename-profile-button")
		fireEvent.click(renameButton)

		// Find input and enter new name
		const input = screen.getByDisplayValue("Default Config")
		fireEvent.input(input, { target: { value: "New Name" } })

		// Save
		const saveButton = screen.getByTestId("save-rename-button")
		fireEvent.click(saveButton)

		expect(defaultProps.onRenameConfig.calledWith("Default Config", "New Name")).toBe(true)
	})

	it("shows error when renaming to existing config name", () => {
		const defaultProps = buildDefaultProps()
		render(<ApiConfigManager {...defaultProps} />)

		// Start rename
		const renameButton = screen.getByTestId("rename-profile-button")
		fireEvent.click(renameButton)

		// Find input and enter existing name
		const input = screen.getByDisplayValue("Default Config")
		fireEvent.input(input, { target: { value: "Another Config" } })

		// Save to trigger validation
		const saveButton = screen.getByTestId("save-rename-button")
		fireEvent.click(saveButton)

		// Verify error message
		const renameForm = getRenameForm()
		const errorMessage = within(renameForm).getByTestId("error-message")
		expect(errorMessage).toHaveTextContent("settings:providers.nameExists")
		expect(defaultProps.onRenameConfig.called).toBe(false)
	})

	it("prevents renaming to empty name", () => {
		const defaultProps = buildDefaultProps()
		render(<ApiConfigManager {...defaultProps} />)

		// Start rename
		const renameButton = screen.getByTestId("rename-profile-button")
		fireEvent.click(renameButton)

		// Find input and enter empty name
		const input = screen.getByDisplayValue("Default Config")
		fireEvent.input(input, { target: { value: "   " } })

		// Verify save button is disabled
		const saveButton = screen.getByTestId("save-rename-button")
		expect(saveButton).toBeDisabled()
		expect(defaultProps.onRenameConfig.called).toBe(false)
	})

	it("allows selecting a different config", () => {
		const defaultProps = buildDefaultProps()
		render(<ApiConfigManager {...defaultProps} />)

		const select = screen.getByTestId("select-component")
		fireEvent.change(select, { target: { value: "Another Config" } })

		expect(defaultProps.onSelectConfig.calledWith("Another Config")).toBe(true)
	})

	it("allows deleting the current config when not the only one", () => {
		const defaultProps = buildDefaultProps()
		render(<ApiConfigManager {...defaultProps} />)

		const deleteButton = screen.getByTestId("delete-profile-button")
		expect(deleteButton).not.toBeDisabled()

		fireEvent.click(deleteButton)
		expect(defaultProps.onDeleteConfig.calledWith("Default Config")).toBe(true)
	})

	it("disables delete button when only one config exists", () => {
		const defaultProps = buildDefaultProps()
		render(<ApiConfigManager {...defaultProps} listApiConfigMeta={[{ id: "default", name: "Default Config" }]} />)

		const deleteButton = screen.getByTestId("delete-profile-button")
		expect(deleteButton).toHaveAttribute("disabled")
	})

	it("cancels rename operation when clicking cancel", () => {
		const defaultProps = buildDefaultProps()
		render(<ApiConfigManager {...defaultProps} />)

		// Start rename
		const renameButton = screen.getByTestId("rename-profile-button")
		fireEvent.click(renameButton)

		// Find input and enter new name
		const input = screen.getByDisplayValue("Default Config")
		fireEvent.input(input, { target: { value: "New Name" } })

		// Cancel
		const cancelButton = screen.getByTestId("cancel-rename-button")
		fireEvent.click(cancelButton)

		// Verify rename was not called
		expect(defaultProps.onRenameConfig.called).toBe(false)

		// Verify we're back to normal view
		expect(screen.queryByDisplayValue("New Name")).not.toBeInTheDocument()
	})

	it("handles keyboard events in new profile dialog", () => {
		const defaultProps = buildDefaultProps()
		render(<ApiConfigManager {...defaultProps} />)

		// Open dialog
		const addButton = screen.getByTestId("add-profile-button")
		fireEvent.click(addButton)

		const input = screen.getByTestId("new-profile-input")

		// Test Enter key
		fireEvent.input(input, { target: { value: "New Profile" } })
		fireEvent.keyDown(input, { key: "Enter" })
		expect(defaultProps.onUpsertConfig.calledWith("New Profile")).toBe(true)

		// Test Escape key
		fireEvent.keyDown(input, { key: "Escape" })
		expect(screen.getByTestId("dialog")).not.toBeVisible()
	})

	it("handles keyboard events in rename mode", () => {
		const defaultProps = buildDefaultProps()
		render(<ApiConfigManager {...defaultProps} />)

		// Start rename
		const renameButton = screen.getByTestId("rename-profile-button")
		fireEvent.click(renameButton)

		const input = screen.getByDisplayValue("Default Config")

		// Test Enter key
		fireEvent.input(input, { target: { value: "New Name" } })
		fireEvent.keyDown(input, { key: "Enter" })
		expect(defaultProps.onRenameConfig.calledWith("Default Config", "New Name")).toBe(true)

		// Test Escape key
		fireEvent.keyDown(input, { key: "Escape" })
		expect(screen.queryByDisplayValue("New Name")).not.toBeInTheDocument()
	})
})
