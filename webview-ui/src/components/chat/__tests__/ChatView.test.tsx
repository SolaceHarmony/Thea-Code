import React from "react"
import { render, waitFor } from "@testing-library/react"
import sinon from "sinon"
import proxyquire from "proxyquire"
import { ExtensionStateContextProvider } from "../../../context/ExtensionStateContext"

// Define minimal types needed for testing
interface ClineMessage {
	type: "say" | "ask"
	say?: string
	ask?: string
	ts: number
	text?: string
	partial?: boolean
}

interface ExtensionState {
	version: string
	clineMessages: ClineMessage[]
	taskHistory: Array<Record<string, unknown>>
	shouldShowAnnouncement: boolean
	allowedCommands: string[]
	alwaysAllowExecute: boolean
	[key: string]: unknown
}

const proxyquireStrict = proxyquire.noPreserveCache()

const MockBrowserSessionRow = ({ messages }: { messages: ClineMessage[] }) => (
	<div data-testid="browser-session">{JSON.stringify(messages)}</div>
)

const MockChatRow = ({ message }: { message: ClineMessage }) => (
	<div data-testid="chat-row">{JSON.stringify(message)}</div>
)

interface ChatTextAreaProps {
	onSend: (value: string) => void
	inputValue?: string
	textAreaDisabled?: boolean
	placeholderText?: string
	selectedImages?: string[]
	shouldDisableImages?: boolean
}

const MockChatTextArea = React.forwardRef(function MockChatTextArea(
	props: ChatTextAreaProps,
	ref: React.ForwardedRef<HTMLInputElement>,
) {
	return (
		<div data-testid="chat-textarea">
			<input ref={ref} type="text" onChange={(e) => props.onSend(e.target.value)} />
		</div>
	)
})

const MockTaskHeader = ({ task }: { task: ClineMessage }) => <div data-testid="task-header">{JSON.stringify(task)}</div>

const MockVSCodeButton = ({
	children,
	onClick,
	appearance,
}: {
	children: React.ReactNode
	onClick?: () => void
	appearance?: string
}) => (
	<button onClick={onClick} data-appearance={appearance}>
		{children}
	</button>
)

const MockVSCodeTextField = ({
	value,
	onInput,
	placeholder,
}: {
	value?: string
	onInput?: (e: { target: { value: string } }) => void
	placeholder?: string
}) => (
	<input
		type="text"
		value={value}
		onChange={(e) => onInput?.({ target: { value: e.target.value } })}
		placeholder={placeholder}
	/>
)

const MockVSCodeLink = ({ children, href }: { children: React.ReactNode; href?: string }) => (
	<a href={href}>{children}</a>
)

let sandbox: sinon.SinonSandbox
let ChatView: typeof import("../ChatView").default
let vscode: { postMessage: sinon.SinonStub }

// Mock window.postMessage to trigger state hydration
const mockPostMessage = (state: Partial<ExtensionState>) => {
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
				...state,
			},
		},
		"*",
	)
}

beforeEach(() => {
	sandbox = sinon.createSandbox()
	vscode = { postMessage: sandbox.stub() }
	ChatView = proxyquireStrict("../ChatView", {
		"../../../utils/vscode": { vscode },
		"../BrowserSessionRow": { __esModule: true, default: MockBrowserSessionRow },
		"../ChatRow": { __esModule: true, default: MockChatRow },
		"../AutoApproveMenu": { __esModule: true, default: () => null },
		"../ChatTextArea": { __esModule: true, default: MockChatTextArea },
		"../TaskHeader": { __esModule: true, default: MockTaskHeader },
		"@/components/ui/vscode-components": {
			VSCodeButton: MockVSCodeButton,
			VSCodeTextField: MockVSCodeTextField,
			VSCodeLink: MockVSCodeLink,
		},
	}).default
})

afterEach(() => {
	sandbox.restore()
})

describe("ChatView - Auto Approval Tests", () => {
	beforeEach(() => {
		sandbox.resetHistory()
	})

	it("does not auto-approve any actions when autoApprovalEnabled is false", () => {
		render(
			<ExtensionStateContextProvider>
				<ChatView
					isHidden={false}
					showAnnouncement={false}
					hideAnnouncement={() => {}}
					showHistoryView={() => {}}
				/>
			</ExtensionStateContextProvider>,
		)

		// First hydrate state with initial task
		mockPostMessage({
			autoApprovalEnabled: false,
			alwaysAllowBrowser: true,
			alwaysAllowReadOnly: true,
			alwaysAllowWrite: true,
			alwaysAllowExecute: true,
			allowedCommands: ["npm test"],
			clineMessages: [
				{
					type: "say",
					say: "task",
					ts: Date.now() - 2000,
					text: "Initial task",
				},
			],
		})

		// Test various types of actions that should not be auto-approved
		const testCases = [
			{
				ask: "browser_action_launch",
				text: JSON.stringify({ action: "launch", url: "http://example.com" }),
			},
			{
				ask: "tool",
				text: JSON.stringify({ tool: "readFile", path: "test.txt" }),
			},
			{
				ask: "tool",
				text: JSON.stringify({ tool: "editedExistingFile", path: "test.txt" }),
			},
			{
				ask: "command",
				text: "npm test",
			},
		]

		testCases.forEach((testCase) => {
			mockPostMessage({
				autoApprovalEnabled: false,
				alwaysAllowBrowser: true,
				alwaysAllowReadOnly: true,
				alwaysAllowWrite: true,
				alwaysAllowExecute: true,
				allowedCommands: ["npm test"],
				clineMessages: [
					{
						type: "say",
						say: "task",
						ts: Date.now() - 2000,
						text: "Initial task",
					},
					{
						type: "ask",
						ask: testCase.ask,
						ts: Date.now(),
						text: testCase.text,
						partial: false,
					},
				],
			})

			// Verify no auto-approval message was sent
			expect(
				vscode.postMessage.calledWith({
					type: "askResponse",
					askResponse: "yesButtonClicked",
				}),
			).toBe(false)
		})
	})

	it("auto-approves browser actions when alwaysAllowBrowser is enabled", async () => {
		render(
			<ExtensionStateContextProvider>
				<ChatView
					isHidden={false}
					showAnnouncement={false}
					hideAnnouncement={() => {}}
					showHistoryView={() => {}}
				/>
			</ExtensionStateContextProvider>,
		)

		// First hydrate state with initial task
		mockPostMessage({
			autoApprovalEnabled: true,
			alwaysAllowBrowser: true,
			clineMessages: [
				{
					type: "say",
					say: "task",
					ts: Date.now() - 2000,
					text: "Initial task",
				},
			],
		})

		// Then send the browser action ask message
		mockPostMessage({
			autoApprovalEnabled: true,
			alwaysAllowBrowser: true,
			clineMessages: [
				{
					type: "say",
					say: "task",
					ts: Date.now() - 2000,
					text: "Initial task",
				},
				{
					type: "ask",
					ask: "browser_action_launch",
					ts: Date.now(),
					text: JSON.stringify({ action: "launch", url: "http://example.com" }),
					partial: false,
				},
			],
		})

		// Wait for the auto-approval message
		await waitFor(() => {
			expect(
				vscode.postMessage.calledWith({
					type: "askResponse",
					askResponse: "yesButtonClicked",
				}),
			).toBe(true)
		})
	})

	it("auto-approves read-only tools when alwaysAllowReadOnly is enabled", async () => {
		render(
			<ExtensionStateContextProvider>
				<ChatView
					isHidden={false}
					showAnnouncement={false}
					hideAnnouncement={() => {}}
					showHistoryView={() => {}}
				/>
			</ExtensionStateContextProvider>,
		)

		// First hydrate state with initial task
		mockPostMessage({
			autoApprovalEnabled: true,
			alwaysAllowReadOnly: true,
			clineMessages: [
				{
					type: "say",
					say: "task",
					ts: Date.now() - 2000,
					text: "Initial task",
				},
			],
		})

		// Then send the read-only tool ask message
		mockPostMessage({
			autoApprovalEnabled: true,
			alwaysAllowReadOnly: true,
			clineMessages: [
				{
					type: "say",
					say: "task",
					ts: Date.now() - 2000,
					text: "Initial task",
				},
				{
					type: "ask",
					ask: "tool",
					ts: Date.now(),
					text: JSON.stringify({ tool: "readFile", path: "test.txt" }),
					partial: false,
				},
			],
		})

		// Wait for the auto-approval message
		await waitFor(() => {
			expect(
				vscode.postMessage.calledWith({
					type: "askResponse",
					askResponse: "yesButtonClicked",
				}),
			).toBe(true)
		})
	})

	describe("Write Tool Auto-Approval Tests", () => {
		it("auto-approves write tools when alwaysAllowWrite is enabled and message is a tool request", async () => {
			render(
				<ExtensionStateContextProvider>
					<ChatView
						isHidden={false}
						showAnnouncement={false}
						hideAnnouncement={() => {}}
						showHistoryView={() => {}}
					/>
				</ExtensionStateContextProvider>,
			)

			// First hydrate state with initial task
			mockPostMessage({
				autoApprovalEnabled: true,
				alwaysAllowWrite: true,
				writeDelayMs: 0,
				clineMessages: [
					{
						type: "say",
						say: "task",
						ts: Date.now() - 2000,
						text: "Initial task",
					},
				],
			})

			// Then send the write tool ask message
			mockPostMessage({
				autoApprovalEnabled: true,
				alwaysAllowWrite: true,
				writeDelayMs: 0,
				clineMessages: [
					{
						type: "say",
						say: "task",
						ts: Date.now() - 2000,
						text: "Initial task",
					},
					{
						type: "ask",
						ask: "tool",
						ts: Date.now(),
						text: JSON.stringify({ tool: "editedExistingFile", path: "test.txt" }),
						partial: false,
					},
				],
			})

			// Wait for the auto-approval message
			await waitFor(() => {
				expect(
					vscode.postMessage.calledWith({
						type: "askResponse",
						askResponse: "yesButtonClicked",
					}),
				).toBe(true)
			})
		})

		it("does not auto-approve write operations when alwaysAllowWrite is enabled but message is not a tool request", () => {
			render(
				<ExtensionStateContextProvider>
					<ChatView
						isHidden={false}
						showAnnouncement={false}
						hideAnnouncement={() => {}}
						showHistoryView={() => {}}
					/>
				</ExtensionStateContextProvider>,
			)

			// First hydrate state with initial task
			mockPostMessage({
				autoApprovalEnabled: true,
				alwaysAllowWrite: true,
				clineMessages: [
					{
						type: "say",
						say: "task",
						ts: Date.now() - 2000,
						text: "Initial task",
					},
				],
			})

			// Then send a non-tool write operation message
			mockPostMessage({
				autoApprovalEnabled: true,
				alwaysAllowWrite: true,
				clineMessages: [
					{
						type: "say",
						say: "task",
						ts: Date.now() - 2000,
						text: "Initial task",
					},
					{
						type: "ask",
						ask: "write_operation",
						ts: Date.now(),
						text: JSON.stringify({ path: "test.txt", content: "test content" }),
						partial: false,
					},
				],
			})

			// Verify no auto-approval message was sent
			expect(
				vscode.postMessage.calledWith({
					type: "askResponse",
					askResponse: "yesButtonClicked",
				}),
			).toBe(false)
		})
	})

	it("auto-approves allowed commands when alwaysAllowExecute is enabled", async () => {
		render(
			<ExtensionStateContextProvider>
				<ChatView
					isHidden={false}
					showAnnouncement={false}
					hideAnnouncement={() => {}}
					showHistoryView={() => {}}
				/>
			</ExtensionStateContextProvider>,
		)

		// First hydrate state with initial task
		mockPostMessage({
			autoApprovalEnabled: true,
			alwaysAllowExecute: true,
			allowedCommands: ["npm test"],
			clineMessages: [
				{
					type: "say",
					say: "task",
					ts: Date.now() - 2000,
					text: "Initial task",
				},
			],
		})

		// Then send the command ask message
		mockPostMessage({
			autoApprovalEnabled: true,
			alwaysAllowExecute: true,
			allowedCommands: ["npm test"],
			clineMessages: [
				{
					type: "say",
					say: "task",
					ts: Date.now() - 2000,
					text: "Initial task",
				},
				{
					type: "ask",
					ask: "command",
					ts: Date.now(),
					text: "npm test",
					partial: false,
				},
			],
		})

		// Wait for the auto-approval message
		await waitFor(() => {
			expect(
				vscode.postMessage.calledWith({
					type: "askResponse",
					askResponse: "yesButtonClicked",
				}),
			).toBe(true)
		})
	})

	it("does not auto-approve disallowed commands even when alwaysAllowExecute is enabled", () => {
		render(
			<ExtensionStateContextProvider>
				<ChatView
					isHidden={false}
					showAnnouncement={false}
					hideAnnouncement={() => {}}
					showHistoryView={() => {}}
				/>
			</ExtensionStateContextProvider>,
		)

		// First hydrate state with initial task
		mockPostMessage({
			autoApprovalEnabled: true,
			alwaysAllowExecute: true,
			allowedCommands: ["npm test"],
			clineMessages: [
				{
					type: "say",
					say: "task",
					ts: Date.now() - 2000,
					text: "Initial task",
				},
			],
		})

		// Then send the disallowed command ask message
		mockPostMessage({
			autoApprovalEnabled: true,
			alwaysAllowExecute: true,
			allowedCommands: ["npm test"],
			clineMessages: [
				{
					type: "say",
					say: "task",
					ts: Date.now() - 2000,
					text: "Initial task",
				},
				{
					type: "ask",
					ask: "command",
					ts: Date.now(),
					text: "rm -rf /",
					partial: false,
				},
			],
		})

		// Verify no auto-approval message was sent
		expect(
			vscode.postMessage.calledWith({
				type: "askResponse",
				askResponse: "yesButtonClicked",
			}),
		).toBe(false)
	})

	describe("Command Chaining Tests", () => {
		it("auto-approves chained commands when all parts are allowed", async () => {
			render(
				<ExtensionStateContextProvider>
					<ChatView
						isHidden={false}
						showAnnouncement={false}
						hideAnnouncement={() => {}}
						showHistoryView={() => {}}
					/>
				</ExtensionStateContextProvider>,
			)

			// Test various allowed command chaining scenarios
			const allowedChainedCommands = [
				"npm test && npm run build",
				"npm test; npm run build",
				"npm test || npm run build",
				"npm test | npm run build",
				// Add test for quoted pipes which should be treated as part of the command, not as a chain operator
				'echo "hello | world"',
				'npm test "param with | inside" && npm run build',
				// PowerShell command with Select-String
				'npm test 2>&1 | Select-String -NotMatch "node_modules" | Select-String "FAIL|Error"',
			]

			for (const command of allowedChainedCommands) {
				sandbox.resetHistory()

				// First hydrate state with initial task
				mockPostMessage({
					autoApprovalEnabled: true,
					alwaysAllowExecute: true,
					allowedCommands: ["npm test", "npm run build", "echo", "Select-String"],
					clineMessages: [
						{
							type: "say",
							say: "task",
							ts: Date.now() - 2000,
							text: "Initial task",
						},
					],
				})

				// Then send the chained command ask message
				mockPostMessage({
					autoApprovalEnabled: true,
					alwaysAllowExecute: true,
					allowedCommands: ["npm test", "npm run build", "echo", "Select-String"],
					clineMessages: [
						{
							type: "say",
							say: "task",
							ts: Date.now() - 2000,
							text: "Initial task",
						},
						{
							type: "ask",
							ask: "command",
							ts: Date.now(),
							text: command,
							partial: false,
						},
					],
				})

				// Wait for the auto-approval message
				await waitFor(() => {
					expect(
						vscode.postMessage.calledWith({
							type: "askResponse",
							askResponse: "yesButtonClicked",
						}),
					).toBe(true)
				})
			}
		})

		it("does not auto-approve chained commands when any part is disallowed", () => {
			render(
				<ExtensionStateContextProvider>
					<ChatView
						isHidden={false}
						showAnnouncement={false}
						hideAnnouncement={() => {}}
						showHistoryView={() => {}}
					/>
				</ExtensionStateContextProvider>,
			)

			// Test various command chaining scenarios with disallowed parts
			const disallowedChainedCommands = [
				"npm test && rm -rf /",
				"npm test; rm -rf /",
				"npm test || rm -rf /",
				"npm test | rm -rf /",
				// Test subshell execution using $() and backticks
				"npm test $(echo dangerous)",
				"npm test `echo dangerous`",
				// Test unquoted pipes with disallowed commands
				"npm test | rm -rf /",
				// Test PowerShell command with disallowed parts
				'npm test 2>&1 | Select-String -NotMatch "node_modules" | rm -rf /',
			]

			disallowedChainedCommands.forEach((command) => {
				// First hydrate state with initial task
				mockPostMessage({
					alwaysAllowExecute: true,
					allowedCommands: ["npm test", "Select-String"],
					clineMessages: [
						{
							type: "say",
							say: "task",
							ts: Date.now() - 2000,
							text: "Initial task",
						},
					],
				})

				// Then send the chained command ask message
				mockPostMessage({
					autoApprovalEnabled: true,
					alwaysAllowExecute: true,
					allowedCommands: ["npm test", "Select-String"],
					clineMessages: [
						{
							type: "say",
							say: "task",
							ts: Date.now() - 2000,
							text: "Initial task",
						},
						{
							type: "ask",
							ask: "command",
							ts: Date.now(),
							text: command,
							partial: false,
						},
					],
				})

				// Verify no auto-approval message was sent for chained commands with disallowed parts
				expect(
					vscode.postMessage.calledWith({
						type: "askResponse",
						askResponse: "yesButtonClicked",
					}),
				).toBe(false)
			})
		})

		it("handles complex PowerShell command chains correctly", async () => {
			render(
				<ExtensionStateContextProvider>
					<ChatView
						isHidden={false}
						showAnnouncement={false}
						hideAnnouncement={() => {}}
						showHistoryView={() => {}}
					/>
				</ExtensionStateContextProvider>,
			)

			// Test PowerShell specific command chains
			const powershellCommands = {
				allowed: [
					'npm test 2>&1 | Select-String -NotMatch "node_modules"',
					'npm test 2>&1 | Select-String "FAIL|Error"',
					'npm test 2>&1 | Select-String -NotMatch "node_modules" | Select-String "FAIL|Error"',
				],
				disallowed: [
					'npm test 2>&1 | Select-String -NotMatch "node_modules" | rm -rf /',
					'npm test 2>&1 | Select-String "FAIL|Error" && del /F /Q *',
					'npm test 2>&1 | Select-String -NotMatch "node_modules" | Remove-Item -Recurse',
				],
			}

			// Test allowed PowerShell commands
			for (const command of powershellCommands.allowed) {
				sandbox.resetHistory()

				mockPostMessage({
					autoApprovalEnabled: true,
					alwaysAllowExecute: true,
					allowedCommands: ["npm test", "Select-String"],
					clineMessages: [
						{
							type: "say",
							say: "task",
							ts: Date.now() - 2000,
							text: "Initial task",
						},
					],
				})

				mockPostMessage({
					autoApprovalEnabled: true,
					alwaysAllowExecute: true,
					allowedCommands: ["npm test", "Select-String"],
					clineMessages: [
						{
							type: "say",
							say: "task",
							ts: Date.now() - 2000,
							text: "Initial task",
						},
						{
							type: "ask",
							ask: "command",
							ts: Date.now(),
							text: command,
							partial: false,
						},
					],
				})

				await waitFor(() => {
					expect(
						vscode.postMessage.calledWith({
							type: "askResponse",
							askResponse: "yesButtonClicked",
						}),
					).toBe(true)
				})
			}

			// Test disallowed PowerShell commands
			for (const command of powershellCommands.disallowed) {
				sandbox.resetHistory()

				mockPostMessage({
					autoApprovalEnabled: true,
					alwaysAllowExecute: true,
					allowedCommands: ["npm test", "Select-String"],
					clineMessages: [
						{
							type: "say",
							say: "task",
							ts: Date.now() - 2000,
							text: "Initial task",
						},
					],
				})

				mockPostMessage({
					autoApprovalEnabled: true,
					alwaysAllowExecute: true,
					allowedCommands: ["npm test", "Select-String"],
					clineMessages: [
						{
							type: "say",
							say: "task",
							ts: Date.now() - 2000,
							text: "Initial task",
						},
						{
							type: "ask",
							ask: "command",
							ts: Date.now(),
							text: command,
							partial: false,
						},
					],
				})

				expect(
					vscode.postMessage.calledWith({
						type: "askResponse",
						askResponse: "yesButtonClicked",
					}),
				).toBe(false)
			}
		})
	})
})

describe("ChatView - Sound Playing Tests", () => {
	beforeEach(() => {
		sandbox.resetHistory()
	})

	it("does not play sound for auto-approved browser actions", async () => {
		render(
			<ExtensionStateContextProvider>
				<ChatView
					isHidden={false}
					showAnnouncement={false}
					hideAnnouncement={() => {}}
					showHistoryView={() => {}}
				/>
			</ExtensionStateContextProvider>,
		)

		// First hydrate state with initial task and streaming
		mockPostMessage({
			autoApprovalEnabled: true,
			alwaysAllowBrowser: true,
			clineMessages: [
				{
					type: "say",
					say: "task",
					ts: Date.now() - 2000,
					text: "Initial task",
				},
				{
					type: "say",
					say: "api_req_started",
					ts: Date.now() - 1000,
					text: JSON.stringify({}),
					partial: true,
				},
			],
		})

		// Then send the browser action ask message (streaming finished)
		mockPostMessage({
			autoApprovalEnabled: true,
			alwaysAllowBrowser: true,
			clineMessages: [
				{
					type: "say",
					say: "task",
					ts: Date.now() - 2000,
					text: "Initial task",
				},
				{
					type: "ask",
					ask: "browser_action_launch",
					ts: Date.now(),
					text: JSON.stringify({ action: "launch", url: "http://example.com" }),
					partial: false,
				},
			],
		})

		// Verify no sound was played
		expect(
			vscode.postMessage.calledWithMatch({
				type: "playSound",
				audioType: sinon.match.string,
			}),
		).toBe(false)
	})

	it("plays notification sound for non-auto-approved browser actions", async () => {
		render(
			<ExtensionStateContextProvider>
				<ChatView
					isHidden={false}
					showAnnouncement={false}
					hideAnnouncement={() => {}}
					showHistoryView={() => {}}
				/>
			</ExtensionStateContextProvider>,
		)

		// First hydrate state with initial task and streaming
		mockPostMessage({
			autoApprovalEnabled: true,
			alwaysAllowBrowser: false,
			clineMessages: [
				{
					type: "say",
					say: "task",
					ts: Date.now() - 2000,
					text: "Initial task",
				},
				{
					type: "say",
					say: "api_req_started",
					ts: Date.now() - 1000,
					text: JSON.stringify({}),
					partial: true,
				},
			],
		})

		// Then send the browser action ask message (streaming finished)
		mockPostMessage({
			autoApprovalEnabled: true,
			alwaysAllowBrowser: false,
			clineMessages: [
				{
					type: "say",
					say: "task",
					ts: Date.now() - 2000,
					text: "Initial task",
				},
				{
					type: "ask",
					ask: "browser_action_launch",
					ts: Date.now(),
					text: JSON.stringify({ action: "launch", url: "http://example.com" }),
					partial: false,
				},
			],
		})

		// Verify notification sound was played
		await waitFor(() => {
			expect(
				vscode.postMessage.calledWith({
					type: "playSound",
					audioType: "notification",
				}),
			).toBe(true)
		})
	})

	it("plays celebration sound for completion results", async () => {
		render(
			<ExtensionStateContextProvider>
				<ChatView
					isHidden={false}
					showAnnouncement={false}
					hideAnnouncement={() => {}}
					showHistoryView={() => {}}
				/>
			</ExtensionStateContextProvider>,
		)

		// First hydrate state with initial task and streaming
		mockPostMessage({
			clineMessages: [
				{
					type: "say",
					say: "task",
					ts: Date.now() - 2000,
					text: "Initial task",
				},
				{
					type: "say",
					say: "api_req_started",
					ts: Date.now() - 1000,
					text: JSON.stringify({}),
					partial: true,
				},
			],
		})

		// Then send the completion result message (streaming finished)
		mockPostMessage({
			clineMessages: [
				{
					type: "say",
					say: "task",
					ts: Date.now() - 2000,
					text: "Initial task",
				},
				{
					type: "ask",
					ask: "completion_result",
					ts: Date.now(),
					text: "Task completed successfully",
					partial: false,
				},
			],
		})

		// Verify celebration sound was played
		await waitFor(() => {
			expect(
				vscode.postMessage.calledWith({
					type: "playSound",
					audioType: "celebration",
				}),
			).toBe(true)
		})
	})

	it("plays progress_loop sound for api failures", async () => {
		render(
			<ExtensionStateContextProvider>
				<ChatView
					isHidden={false}
					showAnnouncement={false}
					hideAnnouncement={() => {}}
					showHistoryView={() => {}}
				/>
			</ExtensionStateContextProvider>,
		)

		// First hydrate state with initial task and streaming
		mockPostMessage({
			clineMessages: [
				{
					type: "say",
					say: "task",
					ts: Date.now() - 2000,
					text: "Initial task",
				},
				{
					type: "say",
					say: "api_req_started",
					ts: Date.now() - 1000,
					text: JSON.stringify({}),
					partial: true,
				},
			],
		})

		// Then send the api failure message (streaming finished)
		mockPostMessage({
			clineMessages: [
				{
					type: "say",
					say: "task",
					ts: Date.now() - 2000,
					text: "Initial task",
				},
				{
					type: "ask",
					ask: "api_req_failed",
					ts: Date.now(),
					text: "API request failed",
					partial: false,
				},
			],
		})

		// Verify progress_loop sound was played
		await waitFor(() => {
			expect(
				vscode.postMessage.calledWith({
					type: "playSound",
					audioType: "progress_loop",
				}),
			).toBe(true)
		})
	})
})
