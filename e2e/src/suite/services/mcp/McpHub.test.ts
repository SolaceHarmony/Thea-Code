import type { McpHub as McpHubType } from "../management/McpHub"
import type { TheaProvider } from "../../../core/webview/TheaProvider" // Renamed import
import type { ExtensionContext, Uri, Extension, Memento, SecretStorage, EnvironmentVariableCollection } from "vscode"
import type { Client } from "@modelcontextprotocol/sdk/client/index.js"
import type { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js"
import type { McpConnection } from "../management/McpHub"
import fs from "fs/promises"
import { ServerConfigSchema, McpHub } from "../management/McpHub"
import * as assert from 'assert'
import * as sinon from 'sinon'

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

// TODO: Use proxyquire for module mocking
		// Mock for "vscode" needed here
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
// Mock cleanup
// TODO: Mock setup needs manual migration for "fs/promises"
// TODO: Mock setup needs manual migration for "../../../core/webview/TheaProvider" // Updated mock path

suite("McpHub", () => {
	let mcpHub: McpHubType
	let mockProvider: Partial<TheaProvider> // Renamed type

	setup(() => {
		sinon.restore()

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
		;(fs.readFile as sinon.SinonStub).resolves(
			JSON.stringify({
				mcpServers: {
					"test-server": {
						type: "stdio",
						command: "node",
						args: ["test.js"],
						alwaysAllow: ["allowed-tool"],
					},
				},
			})

		mcpHub = new McpHub(mockProvider as TheaProvider) // Renamed type assertion
	})

	suite("toggleToolAlwaysAllow", () => {
		test("should add tool to always allow list when enabling", async () => {
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
			;(fs.readFile as sinon.SinonStub).mockResolvedValueOnce(JSON.stringify(mockConfig))

			await mcpHub.toggleToolAlwaysAllow("test-server", "global", "new-tool", true)

			// Verify the config was updated correctly
			const writeCall = (fs.writeFile as sinon.SinonStub).mock.calls[0] as [string, string]
			const writtenConfig = JSON.parse(writeCall[1]) as TestSettings
			assert.ok(writtenConfig.mcpServers["test-server"].alwaysAllow.includes("new-tool"))
		})

		test("should remove tool from always allow list when disabling", async () => {
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
			;(fs.readFile as sinon.SinonStub).mockResolvedValueOnce(JSON.stringify(mockConfig))

			await mcpHub.toggleToolAlwaysAllow("test-server", "global", "existing-tool", false)

			// Verify the config was updated correctly
			const writeCall = (fs.writeFile as sinon.SinonStub).mock.calls[0] as [string, string]
			const writtenConfig = JSON.parse(writeCall[1]) as TestSettings
			assert.ok(!writtenConfig.mcpServers["test-server"].alwaysAllow.includes("existing-tool"))
		})

		test("should initialize alwaysAllow if it does not exist", async () => {
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
			;(fs.readFile as sinon.SinonStub).mockResolvedValueOnce(JSON.stringify(mockConfig))

			await mcpHub.toggleToolAlwaysAllow("test-server", "global", "new-tool", true)

			// Verify the config was updated with initialized alwaysAllow
			const writeCall = (fs.writeFile as sinon.SinonStub).mock.calls[0] as [string, string]
			const writtenConfig = JSON.parse(writeCall[1]) as TestSettings
			assert.notStrictEqual(writtenConfig.mcpServers["test-server"].alwaysAllow, undefined)
			assert.ok(writtenConfig.mcpServers["test-server"].alwaysAllow.includes("new-tool"))
		})
	})

	suite("server disabled state", () => {
		test("should toggle server disabled state", async () => {
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
			;(fs.readFile as sinon.SinonStub).mockResolvedValueOnce(JSON.stringify(mockConfig))

			await mcpHub.toggleServerDisabled("test-server", true)

			// Verify the config was updated correctly
			const writeCall = (fs.writeFile as sinon.SinonStub).mock.calls[0] as [string, string]
			const writtenConfig = JSON.parse(writeCall[1]) as TestSettings
			assert.strictEqual(writtenConfig.mcpServers["test-server"].disabled, true)
		})

		test("should filter out disabled servers from getServers", () => {
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

			assert.strictEqual(servers.length, 1)
			assert.strictEqual(servers[0].name, "enabled-server")
		})

		test("should prevent calling tools on disabled servers", async () => {
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

			await expect(mcpHub.callTool("disabled-server", "some-tool", {})).rejects.toThrow(
				'Server "disabled-server" is disabled and cannot be used',
			)
		})

		test("should prevent reading resources from disabled servers", async () => {
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

			await expect(mcpHub.readResource("disabled-server", "some/uri")).rejects.toThrow(
				'Server "disabled-server" is disabled',
			)
		})
	})

	suite("callTool", () => {
		test("should execute tool successfully", async () => {
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
			// eslint-disable-next-line @typescript-eslint/unbound-method
			assert.ok(mockConnection.client.request as sinon.SinonStub.calledWith(
				{
					method: "tools/call",
					params: {
						name: "some-tool",
						arguments: {},
					},
				},
				expect.any(Object)),
				// TODO: Object partial match - { timeout: 60000 }), // Default 60 second timeout
			)
		})

		test("should throw error if server not found", async () => {
			await expect(mcpHub.callTool("non-existent-server", "some-tool", {})).rejects.toThrow(
				"No connection found for server: non-existent-server",
			)
		})

		suite("timeout configuration", () => {
			test("should validate timeout values", () => {
				// Test valid timeout values
				const validConfig = {
					type: "stdio",
					command: "test",
					timeout: 60,
				}
				expect(() => ServerConfigSchema.parse(validConfig)).not.toThrow()

				// Test invalid timeout values
				const invalidConfigs = [
					{ type: "stdio", command: "test", timeout: 0 }, // Too low
					{ type: "stdio", command: "test", timeout: 3601 }, // Too high
					{ type: "stdio", command: "test", timeout: -1 }, // Negative
				]

				invalidConfigs.forEach((config) => {
					expect(() => ServerConfigSchema.parse(config)).toThrow()
				})
			})

			test("should use default timeout of 60 seconds if not specified", async () => {
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

				// eslint-disable-next-line @typescript-eslint/unbound-method
				assert.ok(mockConnection.client.request as sinon.SinonStub.calledWith(
					expect.anything()),
					expect.anything(),
					// TODO: Object partial match - { timeout: 60000 }), // 60 seconds in milliseconds
				)
			})

			test("should apply configured timeout to tool calls", async () => {
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

				// eslint-disable-next-line @typescript-eslint/unbound-method
				assert.ok(mockConnection.client.request as sinon.SinonStub.calledWith(
					expect.anything()),
					expect.anything(),
					// TODO: Object partial match - { timeout: 120000 }), // 120 seconds in milliseconds
				)
			})
		})

		suite("updateServerTimeout", () => {
			test("should update server timeout in settings file", async () => {
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
				;(fs.readFile as sinon.SinonStub).mockResolvedValueOnce(JSON.stringify(mockConfig))

				await mcpHub.updateServerTimeout("test-server", 120)

				// Verify the config was updated correctly
				const writeCall = (fs.writeFile as sinon.SinonStub).mock.calls[0] as [string, string]
				const writtenConfig = JSON.parse(writeCall[1]) as TestSettings
				assert.strictEqual(writtenConfig.mcpServers["test-server"].timeout, 120)
			})

			test("should fallback to default timeout when config has invalid timeout", async () => {
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
				;(fs.readFile as sinon.SinonStub).mockResolvedValueOnce(JSON.stringify(mockConfig))

				// Update with invalid timeout
				await mcpHub.updateServerTimeout("test-server", 3601)

				// Config is written
				assert.ok(fs.writeFile.called)

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
				// eslint-disable-next-line @typescript-eslint/unbound-method
				assert.ok(mockConnection.client.request as sinon.SinonStub.calledWith(
					expect.anything()),
					expect.anything(),
					// TODO: Object partial match - { timeout: 60000 }), // Default 60 seconds
				)
			})

			test("should accept valid timeout values", async () => {
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

				;(fs.readFile as sinon.SinonStub).mockResolvedValueOnce(JSON.stringify(mockConfig))

				// Test valid timeout values
				const validTimeouts = [1, 60, 3600]
				for (const timeout of validTimeouts) {
					await mcpHub.updateServerTimeout("test-server", timeout)
					assert.ok(fs.writeFile.called)
					sinon.restore() // Reset for next iteration
					;(fs.readFile as sinon.SinonStub).mockResolvedValueOnce(JSON.stringify(mockConfig))
				}
			})

			test("should notify webview after updating timeout", async () => {
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

				;(fs.readFile as sinon.SinonStub).mockResolvedValueOnce(JSON.stringify(mockConfig))

				await mcpHub.updateServerTimeout("test-server", 120)

				assert.ok(mockProvider.postMessageToWebview.calledWith({
						type: "mcpServers",
					})),
				)
			})
		})
	})
// Mock cleanup
