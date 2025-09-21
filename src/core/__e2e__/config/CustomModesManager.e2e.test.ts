import * as assert from 'assert'
import { expect } from 'chai'
import * as sinon from 'sinon'
import * as vscode from "vscode"
import * as path from "path"
import * as fs from "fs/promises"
import { ModeConfig } from "../../../shared/modes"
import { CustomModesManager } from "../../config/CustomModesManager"
import { arePathsEqual } from "../../../utils/path"
import { getWorkspacePath } from "../../../utils/path-vscode"
import { fileExistsAtPath } from "../../../utils/fs"
import { GLOBAL_FILENAMES as BRANDED_FILENAMES } from "../../../shared/config/thea-config"
import { GlobalFileNames } from "../../../shared/globalFileNames"
// Mock needs manual implementation
// Mock needs manual implementation

suite("CustomModesManager", () => {
	let manager: CustomModesManager
	let mockContext: vscode.ExtensionContext
	let mockOnUpdate: sinon.SinonStub
	let mockWorkspaceFolders: { uri: { fsPath: string } }[]

	// Use path.sep to ensure correct path separators for the current platform
	const mockStoragePath = `${path.sep}mock${path.sep}settings`
	const mockSettingsPath = path.join(mockStoragePath, "settings", GlobalFileNames.customModes)
	const mockProjectModesPath = `${path.sep}mock${path.sep}workspace${path.sep}${BRANDED_FILENAMES.MODES_FILENAME}`

	setup(() => {
		mockOnUpdate = sinon.stub()
		mockContext = {
			globalState: {
				get: sinon.stub(),
				update: sinon.stub(),
			},
			globalStorageUri: {
				fsPath: mockStoragePath,
			},
		} as unknown as vscode.ExtensionContext

		mockWorkspaceFolders = [{ 
			uri: { 
				fsPath: "/mock/workspace",
				scheme: "file",
				authority: "",
				path: "/mock/workspace",
				query: "",
				fragment: "",
				with: sinon.stub(),
				toJSON: sinon.stub(),
				toString: sinon.stub(() => "file:///mock/workspace")
			} as sinon.SinonStubbedInstance<vscode.Uri>
		}]
		Object.defineProperty(vscode.workspace, 'workspaceFolders', {
			value: mockWorkspaceFolders,
			writable: true
		})
		;(vscode.workspace.onDidSaveTextDocument as sinon.SinonStub).returns({ dispose: sinon.stub() })
		;(getWorkspacePath as sinon.SinonStub).returns("/mock/workspace")
		;(fileExistsAtPath as sinon.SinonStub).callsFake((path: string) => {
			return Promise.resolve(path === mockSettingsPath || path === mockProjectModesPath)
		})
		;(fs.mkdir as sinon.SinonStub).resolves(undefined)
		;(fs.readFile as sinon.SinonStub).callsFake((path: string) => {
			if (path === mockSettingsPath) {
				return Promise.resolve(JSON.stringify({ customModes: [] }))
			}
			return Promise.reject(new Error("File not found"))
		})

		manager = new CustomModesManager(mockContext, mockOnUpdate)
	})

	teardown(() => {
		sinon.restore()
	})

	suite("getCustomModes", () => {
		test(`should merge modes with ${BRANDED_FILENAMES.MODES_FILENAME} taking precedence`, async () => {
			const settingsModes = [
				{ slug: "mode1", name: "Mode 1", roleDefinition: "Role 1", groups: ["read"] },
				{ slug: "mode2", name: "Mode 2", roleDefinition: "Role 2", groups: ["read"] },
			]

			const projectModes = [
				{ slug: "mode2", name: "Mode 2 Override", roleDefinition: "Role 2 Override", groups: ["read"] },
				{ slug: "mode3", name: "Mode 3", roleDefinition: "Role 3", groups: ["read"] },
			]

			;(fs.readFile as sinon.SinonStub).callsFake((path: string) => {
				if (path === mockSettingsPath) {
					return Promise.resolve(JSON.stringify({ customModes: settingsModes }))
				}
				if (path === mockProjectModesPath) {
					return Promise.resolve(JSON.stringify({ customModes: projectModes }))
				}
				return Promise.reject(new Error("File not found"))
			})

			const modes = await manager.getCustomModes()

			// Should contain 3 modes (mode1 from settings, mode2 and mode3 from roomodes)
			assert.strictEqual(modes.length, 3)
			expect(modes.map((m) => m.slug)).toEqual(["mode2", "mode3", "mode1"])

			// mode2 should come from `${BRANDED_FILENAMES.MODES_FILENAME}` since it takes precedence
			const mode2 = modes.find((m) => m.slug === "mode2")
			assert.strictEqual(mode2?.name, "Mode 2 Override")
			assert.strictEqual(mode2?.roleDefinition, "Role 2 Override")
		})

		test(`"should handle missing ${BRANDED_FILENAMES.MODES_FILENAME} file"`, async () => {
			const settingsModes = [{ slug: "mode1", name: "Mode 1", roleDefinition: "Role 1", groups: ["read"] }]

			;(fileExistsAtPath as sinon.SinonStub).callsFake((path: string) => {
				return Promise.resolve(path === mockSettingsPath)
			})
			;(fs.readFile as sinon.SinonStub).callsFake((path: string) => {
				if (path === mockSettingsPath) {
					return Promise.resolve(JSON.stringify({ customModes: settingsModes }))
				}
				return Promise.reject(new Error("File not found"))
			})

			const modes = await manager.getCustomModes()

			assert.strictEqual(modes.length, 1)
			assert.strictEqual(modes[0].slug, "mode1")
		})

		test(`"should handle invalid JSON in ${BRANDED_FILENAMES.MODES_FILENAME}"`, async () => {
			const settingsModes = [{ slug: "mode1", name: "Mode 1", roleDefinition: "Role 1", groups: ["read"] }]

			;(fs.readFile as sinon.SinonStub).callsFake((path: string) => {
				if (path === mockSettingsPath) {
					return Promise.resolve(JSON.stringify({ customModes: settingsModes }))
				}
				if (path === mockProjectModesPath) {
					return Promise.resolve("invalid json")
				}
				return Promise.reject(new Error("File not found"))
			})

			const modes = await manager.getCustomModes()

			// Should fall back to settings modes when `${BRANDED_FILENAMES.MODES_FILENAME}` is invalid
			assert.strictEqual(modes.length, 1)
			assert.strictEqual(modes[0].slug, "mode1")
		})
	})

	suite("updateCustomMode", () => {
		test(`"should update mode in settings file while preserving ${BRANDED_FILENAMES.MODES_FILENAME} precedence"`, async () => {
			const newMode: ModeConfig = {
				slug: "mode1",
				name: "Updated Mode 1",
				roleDefinition: "Updated Role 1",
				groups: ["read"],
				source: "global",
			}

			const projectModes = [
				{
					slug: "mode1",
					name: "Theamodes Mode 1",
					roleDefinition: "Role 1",
					groups: ["read"],
					source: "project",
				},
			]

			const existingModes = [
				{ slug: "mode2", name: "Mode 2", roleDefinition: "Role 2", groups: ["read"], source: "global" },
			]

			let settingsContent = { customModes: existingModes }
			let roomodesContent = { customModes: projectModes }

			;(fs.readFile as sinon.SinonStub).callsFake((filePath: string) => {
				if (filePath === mockProjectModesPath) {
					return Promise.resolve(JSON.stringify(roomodesContent))
				}
				if (filePath === mockSettingsPath) {
					return Promise.resolve(JSON.stringify(settingsContent))
				}
				return Promise.reject(new Error("File not found"))
			})
			;(fs.writeFile as sinon.SinonStub).callsFake(
				(filePath: string, content: string) => {
					if (filePath === mockSettingsPath) {
						settingsContent = JSON.parse(content) as Record<string, unknown>
					}
					if (filePath === mockProjectModesPath) {
						roomodesContent = JSON.parse(content) as Record<string, unknown>
					}
					return Promise.resolve()
				},
			)

			await manager.updateCustomMode("mode1", newMode)

			// Should write to settings file
			assert.ok(fs.writeFile.calledWith(mockSettingsPath, sinon.match.instanceOf(String)), "utf-8")

			// Verify the content of the write
			const writeCall = (fs.writeFile as sinon.SinonStub).mock.calls[0] as [string, string, string?]
			const content = JSON.parse(writeCall[1]) as { customModes: unknown[] }
			assert.ok(content.customModes.some(x => JSON.stringify(x) === JSON.stringify(// TODO: Object partial match - {
					slug: "mode1",
					name: "Updated Mode 1",
					roleDefinition: "Updated Role 1",
					source: "global",
				})))

			// Should update global state with merged modes where `${BRANDED_FILENAMES.MODES_FILENAME}` takes precedence
			const expectedCall = (mockContext.globalState.update as sinon.SinonStub).mock.calls.find(
				(call: unknown[]) => Array.isArray(call) && call[0] === "customModes"
			) as [string, unknown[]] | undefined
			assert.notStrictEqual(expectedCall, undefined)
			assert.deepStrictEqual(expectedCall![1], 
				// TODO: Array partial match - [
					// TODO: Object partial match - {
						slug: "mode1",
						name: "Theamodes Mode 1", // `${BRANDED_FILENAMES.MODES_FILENAME}` version should take precedence
						source: "project",
					}),
				]),
			)

			// Should trigger onUpdate
			assert.ok(mockOnUpdate.called)
		})

		test(`"creates ${BRANDED_FILENAMES.MODES_FILENAME} file when adding project-specific mode"`, async () => {
			const projectMode: ModeConfig = {
				slug: "project-mode",
				name: "Project Mode",
				roleDefinition: "Project Role",
				groups: ["read"],
				source: "project",
			}

			// Mock `${BRANDED_FILENAMES.MODES_FILENAME}` to not exist initially
			let roomodesContent: { customModes: unknown[] } | null = null
			;(fileExistsAtPath as sinon.SinonStub).callsFake((filePath: string) => {
				return Promise.resolve(filePath === mockSettingsPath)
			})
			;(fs.readFile as sinon.SinonStub).callsFake((filePath: string) => {
				if (filePath === mockSettingsPath) {
					return Promise.resolve(JSON.stringify({ customModes: [] }))
				}
				if (filePath === mockProjectModesPath) {
					if (!roomodesContent) {
						return Promise.reject(new Error("File not found"))
					}
					return Promise.resolve(JSON.stringify(roomodesContent))
				}
				return Promise.reject(new Error("File not found"))
			})
			;(fs.writeFile as sinon.SinonStub).callsFake((filePath: string, content: string) => {
				if (filePath === mockProjectModesPath) {
					roomodesContent = JSON.parse(content) as { customModes: unknown[] }
				}
				return Promise.resolve()
			})

			await manager.updateCustomMode("project-mode", projectMode)

			// Verify `${BRANDED_FILENAMES.MODES_FILENAME}` was created with the project mode
			assert.ok(fs.writeFile.calledWith(
				sinon.match.instanceOf(String)), // Don't check exact path as it may have different separators on different platforms
				// TODO: String contains check - "project-mode"),
				"utf-8",
			)

			// Verify the path is correct regardless of separators
			const writeCall = (fs.writeFile as sinon.SinonStub).mock.calls[0] as [string, string, string?]
			expect(path.normalize(writeCall[0])).toBe(path.normalize(mockProjectModesPath))

			// Verify the content written to `${BRANDED_FILENAMES.MODES_FILENAME}`
			assert.deepStrictEqual(roomodesContent, {
				customModes: [
					// TODO: Object partial match - {
						slug: "project-mode",
						name: "Project Mode",
						roleDefinition: "Project Role",
						source: "project",
					}),
				],
			})
		})

		test("queues write operations", async () => {
			const mode1: ModeConfig = {
				slug: "mode1",
				name: "Mode 1",
				roleDefinition: "Role 1",
				groups: ["read"],
				source: "global",
			}
			const mode2: ModeConfig = {
				slug: "mode2",
				name: "Mode 2",
				roleDefinition: "Role 2",
				groups: ["read"],
				source: "global",
			}

			let settingsContent = { customModes: [] as unknown[] }
			;(fs.readFile as sinon.SinonStub).callsFake((filePath: string) => {
				if (filePath === mockSettingsPath) {
					return Promise.resolve(JSON.stringify(settingsContent))
				}
				return Promise.reject(new Error("File not found"))
			})
			;(fs.writeFile as sinon.SinonStub).callsFake(
				(filePath: string, content: string) => {
					if (filePath === mockSettingsPath) {
						settingsContent = JSON.parse(content) as { customModes: unknown[] }
					}
					return Promise.resolve()
				},
			)

			// Start both updates simultaneously
			await Promise.all([manager.updateCustomMode("mode1", mode1), manager.updateCustomMode("mode2", mode2)])

			// Verify final state in settings file
			assert.strictEqual(settingsContent.customModes.length, 2)
			expect((settingsContent.customModes as Array<{ name: string }>).map((m) => m.name)).toContain("Mode 1")
			expect((settingsContent.customModes as Array<{ name: string }>).map((m) => m.name)).toContain("Mode 2")

			// Verify global state was updated
			const allCalls = (mockContext.globalState.update as sinon.SinonStub).mock.calls.filter(
				(call: unknown[]) => Array.isArray(call) && call[0] === "customModes"
			) as [string, unknown[]][]
			assert.ok(allCalls.length > 0)
			
			// Check the final call has both modes
			const finalCall = allCalls[allCalls.length - 1]
			assert.deepStrictEqual(finalCall[1], 
				// TODO: Array partial match - [
					// TODO: Object partial match - {
						slug: "mode1",
						name: "Mode 1",
						source: "global",
					}),
					// TODO: Object partial match - {
						slug: "mode2",
						name: "Mode 2",
						source: "global",
					}),
				]),
			)

			// Should trigger onUpdate
			assert.ok(mockOnUpdate.called)
		})
	})
	suite("File Operations", () => {
		test("creates settings directory if it doesn't exist", async () => {
			const settingsPath = path.join(mockStoragePath, "settings", GlobalFileNames.customModes)
			await manager.getCustomModesFilePath()

			assert.ok(fs.mkdir.calledWith(path.dirname(settingsPath)), { recursive: true })
		})

		test("creates default config if file doesn't exist", async () => {
			const settingsPath = path.join(mockStoragePath, "settings", GlobalFileNames.customModes)

			// Mock fileExists to return false first time, then true
			let firstCall = true
			;(fileExistsAtPath as sinon.SinonStub).callsFake(() => {
				if (firstCall) {
					firstCall = false
					return Promise.resolve(false)
				}
				return Promise.resolve(true)
			})

			await manager.getCustomModesFilePath()

			assert.ok(fs.writeFile.calledWith(
				settingsPath,
				// TODO: String regex match - /^\{\s+"customModes":\s+\[\s*\]\s*\}$/)),
			)
		})

		test("watches file for changes", async () => {
			const configPath = path.join(mockStoragePath, "settings", GlobalFileNames.customModes)

			;(fs.readFile as sinon.SinonStub).resolves(JSON.stringify({ customModes: [] }))
			;(arePathsEqual as sinon.SinonStub).callsFake((path1: string, path2: string) => {
				return path.normalize(path1) === path.normalize(path2)
			})
			// Get the registered callback
			const registerCall = (vscode.workspace.onDidSaveTextDocument as sinon.SinonStub).mock.calls[0] as [
				(document: { uri: { fsPath: string } }) => Promise<void>
			]
			assert.notStrictEqual(registerCall, undefined)
			const callback = registerCall[0]

			// Simulate file save event
			const mockDocument = {
				uri: { fsPath: configPath },
			}
			await callback(mockDocument)

			// Verify file was processed
			const expectedCall = (fs.readFile as sinon.SinonStub).mock.calls.find(
				(call: unknown[]) => Array.isArray(call) && call[0] === configPath && call[1] === "utf-8"
			) as [string, string] | undefined
			assert.notStrictEqual(expectedCall, undefined)
			const globalStateCall = (mockContext.globalState.update as sinon.SinonStub).mock.calls.find(
				(call: unknown[]) => Array.isArray(call) && call[0] === "customModes"
			) as [string, unknown] | undefined
			assert.notStrictEqual(globalStateCall, undefined)
			assert.ok(mockOnUpdate.called)
		})
	})

	suite("deleteCustomMode", () => {
		test("deletes mode from settings file", async () => {
			const existingMode = {
				slug: "mode-to-delete",
				name: "Mode To Delete",
				roleDefinition: "Test role",
				groups: ["read"],
				source: "global",
			}

			let settingsContent = { customModes: [existingMode] as unknown[] }
			;(fs.readFile as sinon.SinonStub).callsFake((filePath: string) => {
				if (filePath === mockSettingsPath) {
					return Promise.resolve(JSON.stringify(settingsContent))
				}
				return Promise.reject(new Error("File not found"))
			})
			;(fs.writeFile as sinon.SinonStub).callsFake(
				(filePath: string, content: string) => {
					if (filePath === mockSettingsPath) {
						settingsContent = JSON.parse(content) as { customModes: unknown[] }
					}
					return Promise.resolve()
				},
			)

			// Mock the global state update to actually update the settingsContent
			;(mockContext.globalState.update as sinon.SinonStub).callsFake((key: string, value: unknown) => {
				if (key === "customModes") {
					settingsContent.customModes = value as unknown[]
				}
				return Promise.resolve()
			})

			await manager.deleteCustomMode("mode-to-delete")

			// Verify mode was removed from settings file
			assert.strictEqual(settingsContent.customModes.length, 0)

			// Verify global state was updated
			const expectedCall = (mockContext.globalState.update as sinon.SinonStub).mock.calls.find(
				(call: unknown[]) => Array.isArray(call) && call[0] === "customModes" && Array.isArray(call[1]) && (call[1] as unknown[]).length === 0
			) as [string, unknown[]] | undefined
			assert.notStrictEqual(expectedCall, undefined)

			// Should trigger onUpdate
			assert.ok(mockOnUpdate.called)
		})

		test("handles errors gracefully", async () => {
			const mockShowError = sinon.stub()
			;(vscode.window.showErrorMessage as sinon.SinonStub) = mockShowError
			;(fs.writeFile as sinon.SinonStub).rejects(new Error("Write error"))

			await manager.deleteCustomMode("non-existent-mode")

			assert.ok(mockShowError.calledWith(// TODO: String contains check - "Write error")))
		})
	})

	suite("updateModesInFile", () => {
		test("handles corrupted JSON content gracefully", async () => {
			const corruptedJson = "{ invalid json content"
			;(fs.readFile as sinon.SinonStub).resolves(corruptedJson)

			const newMode: ModeConfig = {
				slug: "test-mode",
				name: "Test Mode",
				roleDefinition: "Test Role",
				groups: ["read"],
				source: "global",
			}

			await manager.updateCustomMode("test-mode", newMode)

			// Verify that a valid JSON structure was written
			const writeCall = (fs.writeFile as sinon.SinonStub).mock.calls[0] as [string, string, string?]
			const writtenContent = JSON.parse(writeCall[1]) as { customModes: unknown[] }
			assert.deepStrictEqual(writtenContent, {
				customModes: [
					// TODO: Object partial match - {
						slug: "test-mode",
						name: "Test Mode",
						roleDefinition: "Test Role",
					}),
				],
			})
		})
	})
// Mock cleanup
