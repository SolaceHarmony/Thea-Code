import type { McpHub as McpHubType } from "../management/McpHub"
import type { TheaProvider } from "../../../core/webview/TheaProvider" // Renamed import
import type { ExtensionContext, Uri, Extension, Memento, SecretStorage, EnvironmentVariableCollection } from "vscode"
import type { Client } from "@modelcontextprotocol/sdk/client/index.js"
import type { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js"
import type { McpConnection } from "../management/McpHub"
import { ServerConfigSchema } from "../management/McpHub"
import { expect } from "chai"
import * as sinon from "sinon"
import proxyquire from "proxyquire"

interface TestSettings {
	mcpServers: Record<
		string,
		{
			type: string
			command: string
			args?: string[]
			alwaysAllow?: string[]
			disabled?: boolean
			timeout?: number
		}
	>
}

describe("McpHub", () => {
	let mcpHub: McpHubType
	let mockProvider: Partial<TheaProvider> // Renamed type
	let fsMock: any
	let vscodeMock: any
	let McpHub: typeof McpHubType

	beforeEach(() => {
		sinon.restore()

		vscodeMock = {
			workspace: {
				createFileSystemWatcher: sinon.stub().returns({
					onDidChange: sinon.stub(),
					onDidCreate: sinon.stub(),
					onDidDelete: sinon.stub(),
					dispose: sinon.stub(),
				}),
				onDidSaveTextDocument: sinon.stub(),
				onDidChangeWorkspaceFolders: sinon.stub(),
				workspaceFolders: [],
			},
			window: {
				showErrorMessage: sinon.stub(),
				showInformationMessage: sinon.stub(),
				showWarningMessage: sinon.stub(),
			},
			Disposable: {
				from: sinon.stub(),
			},
			Uri: {
				file: (path: string) => ({ fsPath: path, with: sinon.stub(), toJSON: sinon.stub() })
			}
		}

		fsMock = {
			readFile: sinon.stub(),
			writeFile: sinon.stub(),
		}

		// Load McpHub with mocks
		const module = proxyquire("../management/McpHub", {
			"vscode": vscodeMock,
			"fs/promises": fsMock,
			"../../../core/webview/TheaProvider": {} 
		})
		McpHub = module.McpHub

		const mockUri: Uri = {
			scheme: "file",
			authority: "",
			path: "/test/path",
			query: "",
			fragment: "",
			fsPath: "/test/path",
			with: sinon.stub(),
			toJSON: sinon.stub(),
		}

		mockProvider = {
			ensureSettingsDirectoryExists: sinon.stub().resolves("/mock/settings/path"),
			ensureMcpServersDirectoryExists: sinon.stub().resolves("/mock/settings/path"),
			postMessageToWebview: sinon.stub(),
			context: {
				subscriptions: [],
				workspaceState: {} as unknown as Memento,
				globalState: {} as unknown as Memento,
				secrets: {} as unknown as SecretStorage,
				extensionUri: mockUri,
				extensionPath: "/test/path",
				storagePath: "/test/storage",
				globalStoragePath: "/test/global-storage",
				environmentVariableCollection: {} as unknown as EnvironmentVariableCollection,
				extension: {
					id: "test-extension",
					extensionUri: mockUri,
					extensionPath: "/test/path",
					extensionKind: 1,
					isActive: true,
					packageJSON: {
						version: "1.0.0",
					},
					activate: sinon.stub(),
					exports: undefined,
				} as unknown as Extension<unknown>,
				asAbsolutePath: (path: string) => path,
				storageUri: mockUri,
				globalStorageUri: mockUri,
				logUri: mockUri,
				extensionMode: 1,
				logPath: "/test/path",
				languageModelAccessInformation: {} as unknown,
			} as ExtensionContext,
		}

		// Mock fs.readFile for initial settings
		fsMock.readFile.resolves(
			JSON.stringify({
				mcpServers: {
					"test-server": {
						type: "stdio",
						command: "node",
						args: ["test.js"],
						alwaysAllow: ["allowed-tool"],
					},
				},
			}),
		)

		mcpHub = new McpHub(mockProvider as TheaProvider) // Renamed type assertion
	})

	describe("toggleToolAlwaysAllow", () => {
		it("should add tool to always allow list when enabling", async () => {
			const mockConfig = {
				mcpServers: {
					"test-server": {
						type: "stdio",
						command: "node",
						args: ["test.js"],
						alwaysAllow: [],
					},
				},
			}

			// Mock reading initial config
			fsMock.readFile.resolves(JSON.stringify(mockConfig))

			await mcpHub.toggleToolAlwaysAllow("test-server", "global", "new-tool", true)

			// Verify the config was updated correctly
			const writeCall = fsMock.writeFile.getCall(0).args
			const writtenConfig = JSON.parse(writeCall[1]) as TestSettings
			expect(writtenConfig.mcpServers["test-server"].alwaysAllow).to.include("new-tool")
		})

		it("should remove tool from always allow list when disabling", async () => {
			const mockConfig = {
				mcpServers: {
					"test-server": {
						type: "stdio",
						command: "node",
						args: ["test.js"],
						alwaysAllow: ["existing-tool"],
					},
				},
			}

			// Mock reading initial config
			fsMock.readFile.resolves(JSON.stringify(mockConfig))

			await mcpHub.toggleToolAlwaysAllow("test-server", "global", "existing-tool", false)

			// Verify the config was updated correctly
			const writeCall = fsMock.writeFile.getCall(0).args
			const writtenConfig = JSON.parse(writeCall[1]) as TestSettings
			expect(writtenConfig.mcpServers["test-server"].alwaysAllow).not.to.include("existing-tool")
		})

		it("should initialize alwaysAllow if it does not exist", async () => {
			const mockConfig = {
				mcpServers: {
					"test-server": {
						type: "stdio",
						command: "node",
						args: ["test.js"],
					},
				},
			}

			// Mock reading initial config
			fsMock.readFile.resolves(JSON.stringify(mockConfig))

			await mcpHub.toggleToolAlwaysAllow("test-server", "global", "new-tool", true)

			// Verify the config was updated with initialized alwaysAllow
			const writeCall = fsMock.writeFile.getCall(0).args
			const writtenConfig = JSON.parse(writeCall[1]) as TestSettings
			expect(writtenConfig.mcpServers["test-server"].alwaysAllow).to.exist
			expect(writtenConfig.mcpServers["test-server"].alwaysAllow).to.include("new-tool")
		})
	})

	describe("server disabled state", () => {
		it("should toggle server disabled state", async () => {
			const mockConfig = {
				mcpServers: {
					"test-server": {
						type: "stdio",
						command: "node",
						args: ["test.js"],
						disabled: false,
					},
				},
			}

			// Mock reading initial config
			fsMock.readFile.resolves(JSON.stringify(mockConfig))

			await mcpHub.toggleServerDisabled("test-server", true)

			// Verify the config was updated correctly
			const writeCall = fsMock.writeFile.getCall(0).args
			const writtenConfig = JSON.parse(writeCall[1]) as TestSettings
			expect(writtenConfig.mcpServers["test-server"].disabled).to.be.true
		})

		it("should filter out disabled servers from getServers", () => {
			const mockConnections: McpConnection[] = [
				{
					server: {
						name: "enabled-server",
						config: "{}",
						status: "connected",
						disabled: false,
					},
					client: {} as unknown as Client,
					transport: {} as unknown as StdioClientTransport,
				},
				{
					server: {
						name: "disabled-server",
						config: "{}",
						status: "connected",
						disabled: true,
					},
					client: {} as unknown as Client,
					transport: {} as unknown as StdioClientTransport,
				},
			]

			mcpHub.connections = mockConnections
			const servers = mcpHub.getServers()

			expect(servers.length).to.equal(1)
			expect(servers[0].name).to.equal("enabled-server")
		})

		it("should prevent calling tools on disabled servers", async () => {
			const mockConnection: McpConnection = {
				server: {
					name: "disabled-server",
					config: "{}",
					status: "connected",
					disabled: true,
				},
				client: {
					request: sinon.stub().resolves({ result: "success" }),
				} as unknown as Client,
				transport: {} as unknown as StdioClientTransport,
			}

			mcpHub.connections = [mockConnection]

			try {
				await mcpHub.callTool("disabled-server", "some-tool", {})
				expect.fail("Should have thrown error")
			} catch (error: any) {
				expect(error.message).to.include('Server "disabled-server" is disabled and cannot be used')
			}
		})

		it("should prevent reading resources from disabled servers", async () => {
			const mockConnection: McpConnection = {
				server: {
					name: "disabled-server",
					config: "{}",
					status: "connected",
					disabled: true,
				},
				client: {
					request: sinon.stub(),
				} as unknown as Client,
				transport: {} as unknown as StdioClientTransport,
			}

			mcpHub.connections = [mockConnection]

			try {
				await mcpHub.readResource("disabled-server", "some/uri")
				expect.fail("Should have thrown error")
			} catch (error: any) {
				expect(error.message).to.include('Server "disabled-server" is disabled')
			}
		})
	})

	describe("callTool", () => {
		it("should execute tool successfully", async () => {
			// Mock the connection with a minimal client implementation
			const mockConnection: McpConnection = {
				server: {
					name: "test-server",
					config: JSON.stringify({}),
					status: "connected" as const,
				},
				client: {
					request: sinon.stub().resolves({ result: "success" }),
				} as unknown as Client,
				transport: {
					start: sinon.stub(),
					close: sinon.stub(),
					stderr: { on: sinon.stub() },
				} as unknown as StdioClientTransport,
			}

			mcpHub.connections = [mockConnection]

			await mcpHub.callTool("test-server", "some-tool", {})

			// Verify the request was made with correct parameters
			const requestStub = mockConnection.client.request as sinon.SinonStub
			expect(requestStub.calledWith(
				{
					method: "tools/call",
					params: {
						name: "some-tool",
						arguments: {},
					},
				},
				sinon.match.any,
				sinon.match({ timeout: 60000 }), // Default 60 second timeout
			)).to.be.true
		})

		it("should throw error if server not found", async () => {
			try {
				await mcpHub.callTool("non-existent-server", "some-tool", {})
				expect.fail("Should have thrown error")
			} catch (error: any) {
				expect(error.message).to.include("No connection found for server: non-existent-server")
			}
		})

		describe("timeout configuration", () => {
			it("should validate timeout values", () => {
				// Test valid timeout values
				const validConfig = {
					type: "stdio",
					command: "test",
					timeout: 60,
				}
				expect(() => ServerConfigSchema.parse(validConfig)).not.to.throw()

				// Test invalid timeout values
				const invalidConfigs = [
					{ type: "stdio", command: "test", timeout: 0 }, // Too low
					{ type: "stdio", command: "test", timeout: 3601 }, // Too high
					{ type: "stdio", command: "test", timeout: -1 }, // Negative
				]

				invalidConfigs.forEach((config) => {
					expect(() => ServerConfigSchema.parse(config)).to.throw()
				})
			})

			it("should use default timeout of 60 seconds if not specified", async () => {
				const mockConnection: McpConnection = {
					server: {
						name: "test-server",
						config: JSON.stringify({ type: "stdio", command: "test" }), // No timeout specified
						status: "connected",
					},
					client: {
						request: sinon.stub().resolves({ content: [] }),
					} as unknown as Client,
					transport: {} as unknown as StdioClientTransport,
				}

				mcpHub.connections = [mockConnection]
				await mcpHub.callTool("test-server", "test-tool")

				const requestStub = mockConnection.client.request as sinon.SinonStub
				expect(requestStub.calledWith(
					sinon.match.any,
					sinon.match.any,
					sinon.match({ timeout: 60000 }), // 60 seconds in milliseconds
				)).to.be.true
			})

			it("should apply configured timeout to tool calls", async () => {
				const mockConnection: McpConnection = {
					server: {
						name: "test-server",
						config: JSON.stringify({ type: "stdio", command: "test", timeout: 120 }), // 2 minutes
						status: "connected",
					},
					client: {
						request: sinon.stub().resolves({ content: [] }),
					} as unknown as Client,
					transport: {} as unknown as StdioClientTransport,
				}

				mcpHub.connections = [mockConnection]
				await mcpHub.callTool("test-server", "test-tool")

				const requestStub = mockConnection.client.request as sinon.SinonStub
				expect(requestStub.calledWith(
					sinon.match.any,
					sinon.match.any,
					sinon.match({ timeout: 120000 }), // 120 seconds in milliseconds
				)).to.be.true
			})
		})

		describe("updateServerTimeout", () => {
			it("should update server timeout in settings file", async () => {
				const mockConfig = {
					mcpServers: {
						"test-server": {
							type: "stdio",
							command: "node",
							args: ["test.js"],
							timeout: 60,
						},
					},
				}

				// Mock reading initial config
				fsMock.readFile.resolves(JSON.stringify(mockConfig))

				await mcpHub.updateServerTimeout("test-server", 120)

				// Verify the config was updated correctly
				const writeCall = fsMock.writeFile.getCall(0).args
				const writtenConfig = JSON.parse(writeCall[1]) as TestSettings
				expect(writtenConfig.mcpServers["test-server"].timeout).to.equal(120)
			})

			it("should fallback to default timeout when config has invalid timeout", async () => {
				const mockConfig = {
					mcpServers: {
						"test-server": {
							type: "stdio",
							command: "node",
							args: ["test.js"],
							timeout: 60,
						},
					},
				}

				// Mock initial read
				fsMock.readFile.resolves(JSON.stringify(mockConfig))

				// Update with invalid timeout
				await mcpHub.updateServerTimeout("test-server", 3601)

				// Config is written
				expect(fsMock.writeFile.called).to.be.true

				// Setup connection with invalid timeout
				const mockConnection: McpConnection = {
					server: {
						name: "test-server",
						config: JSON.stringify({
							type: "stdio",
							command: "node",
							args: ["test.js"],
							timeout: 3601, // Invalid timeout
						}),
						status: "connected",
					},
					client: {
						request: sinon.stub().resolves({ content: [] }),
					} as unknown as Client,
					transport: {} as unknown as StdioClientTransport,
				}

				mcpHub.connections = [mockConnection]

				// Call tool - should use default timeout
				await mcpHub.callTool("test-server", "test-tool")

				// Verify default timeout was used
				const requestStub = mockConnection.client.request as sinon.SinonStub
				expect(requestStub.calledWith(
					sinon.match.any,
					sinon.match.any,
					sinon.match({ timeout: 60000 }), // Default 60 seconds
				)).to.be.true
			})

			it("should accept valid timeout values", async () => {
				const mockConfig = {
					mcpServers: {
						"test-server": {
							type: "stdio",
							command: "node",
							args: ["test.js"],
							timeout: 60,
						},
					},
				}

				fsMock.readFile.resolves(JSON.stringify(mockConfig))

				// Test valid timeout values
				const validTimeouts = [1, 60, 3600]
				for (const timeout of validTimeouts) {
					await mcpHub.updateServerTimeout("test-server", timeout)
					expect(fsMock.writeFile.called).to.be.true
					fsMock.writeFile.resetHistory() // Reset for next iteration
					fsMock.readFile.resolves(JSON.stringify(mockConfig))
				}
			})

			it("should notify webview after updating timeout", async () => {
				const mockConfig = {
					mcpServers: {
						"test-server": {
							type: "stdio",
							command: "node",
							args: ["test.js"],
							timeout: 60,
						},
					},
				}

				fsMock.readFile.resolves(JSON.stringify(mockConfig))

				await mcpHub.updateServerTimeout("test-server", 120)

				expect(mockProvider.postMessageToWebview.calledWith(
					sinon.match({
						type: "mcpServers",
					}),
				)).to.be.true
			})
		})
	})
})
