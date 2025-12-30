import { render, fireEvent, screen } from "@testing-library/react"
import sinon from "sinon"
import proxyquire from "proxyquire"
import { defaultModeSlug } from "../../../../../src/shared/modes"

const proxyquireStrict = proxyquire.noPreserveCache()

// Custom query function to get the enhance prompt button
const getEnhancePromptButton = () => {
	return screen.getByRole("button", {
		name: (_, element) => {
			// Find the button with the sparkle icon
			return element.querySelector(".codicon-sparkle") !== null
		},
	})
}

describe("ChatTextArea", () => {
	let sandbox: sinon.SinonSandbox
	let ChatTextArea: typeof import("../ChatTextArea").default
	let useExtensionState: sinon.SinonStub
	let vscode: { postMessage: sinon.SinonStub }
	let convertToMentionPath: sinon.SinonStub

	const buildDefaultProps = () => ({
		inputValue: "",
		setInputValue: sandbox.stub(),
		onSend: sandbox.stub(),
		textAreaDisabled: false,
		onSelectImages: sandbox.stub(),
		shouldDisableImages: false,
		placeholderText: "Type a message...",
		selectedImages: [],
		setSelectedImages: sandbox.stub(),
		onHeightChange: sandbox.stub(),
		mode: defaultModeSlug,
		setMode: sandbox.stub(),
		modeShortcutText: "(⌘. for next mode)",
	})

	beforeEach(() => {
		sandbox = sinon.createSandbox()
		vscode = { postMessage: sandbox.stub() }
		convertToMentionPath = sandbox.stub().callsFake((path: string, cwd: string) => {
			if (!path.startsWith(cwd)) {
				return path
			}

			const relativePath = path.substring(cwd.length)
			const normalized = relativePath.replace(/^[\/]+/, "").replace(/\\/g, "/")
			return `@/${normalized}`
		})
		useExtensionState = sandbox.stub().returns({
			filePaths: [],
			openedTabs: [],
			apiConfiguration: {
				apiProvider: "anthropic",
			},
			osInfo: "unix",
		})

		ChatTextArea = proxyquireStrict("../ChatTextArea", {
			"../../../utils/vscode": { vscode },
			"../../../components/common/CodeBlock": { __esModule: true, default: () => null },
			"../../../components/common/MarkdownBlock": { __esModule: true, default: () => null },
			"../../../utils/path-mentions": { convertToMentionPath },
			"../../../context/ExtensionStateContext": { useExtensionState },
		}).default
	})

	afterEach(() => {
		sandbox.restore()
	})

	describe("enhance prompt button", () => {
		it("should be disabled when textAreaDisabled is true", () => {
			useExtensionState.returns({
				filePaths: [],
				openedTabs: [],
			})
			render(<ChatTextArea {...buildDefaultProps()} textAreaDisabled={true} />)
			const enhanceButton = getEnhancePromptButton()
			expect(enhanceButton).toHaveClass("cursor-not-allowed")
		})
	})

	describe("handleEnhancePrompt", () => {
		it("should send message with correct configuration when clicked", () => {
			const apiConfiguration = {
				apiProvider: "openrouter",
				apiKey: "test-key",
			}

			useExtensionState.returns({
				filePaths: [],
				openedTabs: [],
				apiConfiguration,
			})

			render(<ChatTextArea {...buildDefaultProps()} inputValue="Test prompt" />)

			const enhanceButton = getEnhancePromptButton()
			fireEvent.click(enhanceButton)

			expect(
				vscode.postMessage.calledWith({
					type: "enhancePrompt",
					text: "Test prompt",
				}),
			).toBe(true)
		})

		it("should not send message when input is empty", () => {
			useExtensionState.returns({
				filePaths: [],
				openedTabs: [],
				apiConfiguration: {
					apiProvider: "openrouter",
				},
			})

			render(<ChatTextArea {...buildDefaultProps()} inputValue="" />)

			const enhanceButton = getEnhancePromptButton()
			fireEvent.click(enhanceButton)

			expect(vscode.postMessage.called).toBe(false)
		})

		it("should show loading state while enhancing", () => {
			useExtensionState.returns({
				filePaths: [],
				openedTabs: [],
				apiConfiguration: {
					apiProvider: "openrouter",
				},
			})

			render(<ChatTextArea {...buildDefaultProps()} inputValue="Test prompt" />)

			const enhanceButton = getEnhancePromptButton()
			fireEvent.click(enhanceButton)

			const loadingSpinner = screen.getByText("", { selector: ".codicon-loading" })
			expect(loadingSpinner).toBeInTheDocument()
		})
	})

	describe("effect dependencies", () => {
		it("should update when apiConfiguration changes", () => {
			const { rerender } = render(<ChatTextArea {...buildDefaultProps()} />)

			// Update apiConfiguration
			useExtensionState.returns({
				filePaths: [],
				openedTabs: [],
				apiConfiguration: {
					apiProvider: "openrouter",
					newSetting: "test",
				},
			})

			rerender(<ChatTextArea {...buildDefaultProps()} />)

			// Verify the enhance button appears after apiConfiguration changes
			expect(getEnhancePromptButton()).toBeInTheDocument()
		})
	})

	describe("enhanced prompt response", () => {
		it("should update input value when receiving enhanced prompt", () => {
			const setInputValue = sandbox.stub()

			render(<ChatTextArea {...buildDefaultProps()} setInputValue={setInputValue} />)

			// Simulate receiving enhanced prompt message
			window.dispatchEvent(
				new window.MessageEvent("message", {
					data: {
						type: "enhancedPrompt",
						text: "Enhanced test prompt",
					},
				}),
			)

			expect(setInputValue.calledWith("Enhanced test prompt")).toBe(true)
		})
	})

	describe("multi-file drag and drop", () => {
		const mockCwd = "/Users/test/project"

		beforeEach(() => {
			useExtensionState.returns({
				filePaths: [],
				openedTabs: [],
				cwd: mockCwd,
				osInfo: "unix",
			})
			convertToMentionPath.resetHistory()
		})

		it("should process multiple file paths separated by newlines", () => {
			const setInputValue = sandbox.stub()

			const { container } = render(
				<ChatTextArea {...buildDefaultProps()} setInputValue={setInputValue} inputValue="Initial text" />,
			)

			// Create a mock dataTransfer object with text data containing multiple file paths
			const dataTransfer = {
				getData: sandbox.stub().returns("/Users/test/project/file1.js\n/Users/test/project/file2.js"),
				files: [],
			}

			// Simulate drop event
			fireEvent.drop(container.querySelector(".chat-text-area")!, {
				dataTransfer,
				preventDefault: sandbox.stub(),
			})

			// Verify convertToMentionPath was called for each file path
			expect(convertToMentionPath.calledTwice).toBe(true)
			expect(convertToMentionPath.calledWith("/Users/test/project/file1.js", mockCwd, "unix")).toBe(true)
			expect(convertToMentionPath.calledWith("/Users/test/project/file2.js", mockCwd, "unix")).toBe(true)

			// Verify setInputValue was called with the correct value
			// The mock implementation of convertToMentionPath will convert the paths to @/file1.js and @/file2.js
			expect(setInputValue.calledWith("@/file1.js @/file2.js Initial text")).toBe(true)
		})

		it("should filter out empty lines in the dragged text", () => {
			const setInputValue = sandbox.stub()

			const { container } = render(
				<ChatTextArea {...buildDefaultProps()} setInputValue={setInputValue} inputValue="Initial text" />,
			)

			// Create a mock dataTransfer object with text data containing empty lines
			const dataTransfer = {
				getData: sandbox.stub().returns("/Users/test/project/file1.js\n\n/Users/test/project/file2.js\n\n"),
				files: [],
			}

			// Simulate drop event
			fireEvent.drop(container.querySelector(".chat-text-area")!, {
				dataTransfer,
				preventDefault: sandbox.stub(),
			})

			// Verify convertToMentionPath was called only for non-empty lines
			expect(convertToMentionPath.calledTwice).toBe(true)

			// Verify setInputValue was called with the correct value
			expect(setInputValue.calledWith("@/file1.js @/file2.js Initial text")).toBe(true)
		})

		it("should correctly update cursor position after adding multiple mentions", () => {
			const setInputValue = sandbox.stub()
			const initialCursorPosition = 5

			const { container } = render(
				<ChatTextArea {...buildDefaultProps()} setInputValue={setInputValue} inputValue="Hello world" />,
			)

			// Set the cursor position manually
			const textArea = container.querySelector("textarea")
			if (textArea) {
				textArea.selectionStart = initialCursorPosition
				textArea.selectionEnd = initialCursorPosition
			}

			// Create a mock dataTransfer object with text data
			const dataTransfer = {
				getData: sandbox.stub().returns("/Users/test/project/file1.js\n/Users/test/project/file2.js"),
				files: [],
			}

			// Simulate drop event
			fireEvent.drop(container.querySelector(".chat-text-area")!, {
				dataTransfer,
				preventDefault: sandbox.stub(),
			})

			// The cursor position should be updated based on the implementation in the component
			expect(setInputValue.calledWith("@/file1.js @/file2.js Hello world")).toBe(true)
		})

		it("should handle very long file paths correctly", () => {
			const setInputValue = sandbox.stub()

			const { container } = render(<ChatTextArea {...buildDefaultProps()} setInputValue={setInputValue} inputValue="" />)

			// Create a very long file path
			const longPath =
				"/Users/test/project/very/long/path/with/many/nested/directories/and/a/very/long/filename/with/extension.typescript"

			// Create a mock dataTransfer object with the long path
			const dataTransfer = {
				getData: sandbox.stub().returns(longPath),
				files: [],
			}

			// Simulate drop event
			fireEvent.drop(container.querySelector(".chat-text-area")!, {
				dataTransfer,
				preventDefault: sandbox.stub(),
			})

			// Verify convertToMentionPath was called with the long path
			expect(convertToMentionPath.calledWith(longPath, mockCwd, "unix")).toBe(true)

			// The mock implementation will convert it to @/very/long/path/...
			expect(
				setInputValue.calledWith(
					"@/very/long/path/with/many/nested/directories/and/a/very/long/filename/with/extension.typescript ",
				),
			).toBe(true)
		})

		it("should handle paths with special characters correctly", () => {
			const setInputValue = sandbox.stub()

			const { container } = render(<ChatTextArea {...buildDefaultProps()} setInputValue={setInputValue} inputValue="" />)

			// Create paths with special characters
			const specialPath1 = "/Users/test/project/file with spaces.js"
			const specialPath2 = "/Users/test/project/file-with-dashes.js"
			const specialPath3 = "/Users/test/project/file_with_underscores.js"
			const specialPath4 = "/Users/test/project/file.with.dots.js"

			// Create a mock dataTransfer object with the special paths
			const dataTransfer = {
				getData: sandbox
					.stub()
					.returns(`${specialPath1}\n${specialPath2}\n${specialPath3}\n${specialPath4}`),
				files: [],
			}

			// Simulate drop event
			fireEvent.drop(container.querySelector(".chat-text-area")!, {
				dataTransfer,
				preventDefault: sandbox.stub(),
			})

			// Verify convertToMentionPath was called for each path
			expect(convertToMentionPath.callCount).toBe(4)
			expect(convertToMentionPath.calledWith(specialPath1, mockCwd, "unix")).toBe(true)
			expect(convertToMentionPath.calledWith(specialPath2, mockCwd, "unix")).toBe(true)
			expect(convertToMentionPath.calledWith(specialPath3, mockCwd, "unix")).toBe(true)
			expect(convertToMentionPath.calledWith(specialPath4, mockCwd, "unix")).toBe(true)

			// Verify setInputValue was called with the correct value
			expect(
				setInputValue.calledWith(
					"@/file with spaces.js @/file-with-dashes.js @/file_with_underscores.js @/file.with.dots.js ",
				),
			).toBe(true)
		})

		it("should handle paths outside the current working directory", () => {
			const setInputValue = sandbox.stub()

			const { container } = render(<ChatTextArea {...buildDefaultProps()} setInputValue={setInputValue} inputValue="" />)

			// Create paths outside the current working directory
			const outsidePath = "/Users/other/project/file.js"

			// Mock the convertToMentionPath function to return the original path for paths outside cwd
			convertToMentionPath.onCall(0).callsFake((path: string) => path)

			// Create a mock dataTransfer object with the outside path
			const dataTransfer = {
				getData: sandbox.stub().returns(outsidePath),
				files: [],
			}

			// Simulate drop event
			fireEvent.drop(container.querySelector(".chat-text-area")!, {
				dataTransfer,
				preventDefault: sandbox.stub(),
			})

			// Verify convertToMentionPath was called with the outside path
			expect(convertToMentionPath.calledWith(outsidePath, mockCwd, "unix")).toBe(true)

			// Verify setInputValue was called with the original path
			expect(setInputValue.calledWith("/Users/other/project/file.js ")).toBe(true)
		})

		it("should do nothing when dropped text is empty", () => {
			const setInputValue = sandbox.stub()

			const { container } = render(
				<ChatTextArea {...buildDefaultProps()} setInputValue={setInputValue} inputValue="Initial text" />,
			)

			// Create a mock dataTransfer object with empty text
			const dataTransfer = {
				getData: sandbox.stub().returns(""),
				files: [],
			}

			// Simulate drop event
			fireEvent.drop(container.querySelector(".chat-text-area")!, {
				dataTransfer,
				preventDefault: sandbox.stub(),
			})

			// Verify convertToMentionPath was not called
			expect(convertToMentionPath.called).toBe(false)

			// Verify setInputValue was not called
			expect(setInputValue.called).toBe(false)
		})
	})
})
