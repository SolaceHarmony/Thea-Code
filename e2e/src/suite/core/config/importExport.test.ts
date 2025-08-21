import * as assert from 'assert'
import * as sinon from 'sinon'
/* eslint-disable @typescript-eslint/unbound-method */
/* eslint-disable @typescript-eslint/no-unsafe-assignment */

import fs from "fs/promises"
import * as path from "path"

import * as vscode from "vscode"

import type { ProviderName } from "../../../schemas"
import { SPECIFIC_STRINGS } from "../../../shared/config/thea-config"
import { ProviderSettingsManager } from "../ProviderSettingsManager"
import { importSettings, exportSettings } from "../importExport"
import { ContextProxy } from "../ContextProxy"

// Mock VSCode modules
// TODO: Use proxyquire for module mocking
		// Mock for "vscode" needed here
	window: {
		showOpenDialog: sinon.stub(),
		showSaveDialog: sinon.stub(),
	},
	Uri: {
		file: sinon.stub((filePath: string) => ({ fsPath: filePath })),
	},
// Mock cleanup

// Mock fs/promises
// TODO: Use proxyquire for module mocking
		// Mock for "fs/promises" needed here
	readFile: sinon.stub(),
	mkdir: sinon.stub(),
	writeFile: sinon.stub(),
// Mock cleanup needed

// Mock os module
// TODO: Use proxyquire for module mocking
		// Mock for "os" needed here
	homedir: sinon.stub(() => "/mock/home"),
// Mock cleanup needed

suite("importExport", () => {
	let mockProviderSettingsManager: sinon.SinonStubbedInstance<ProviderSettingsManager>
	let mockContextProxy: sinon.SinonStubbedInstance<ContextProxy>
	let mockExtensionContext: sinon.SinonStubbedInstance<vscode.ExtensionContext>

	setup(() => {
		// Reset all mocks
		sinon.restore()

		// Setup providerSettingsManager mock
		mockProviderSettingsManager = {
			export: sinon.stub(),
			import: sinon.stub(),
			listConfig: sinon.stub(),
		} as unknown as sinon.SinonStubbedInstance<ProviderSettingsManager>

		// Setup contextProxy mock with properly typed export method
		mockContextProxy = {
			setValues: sinon.stub(),
			setValue: sinon.stub(),
			export: sinon.stub().callsFake(() => Promise.resolve({})),
		} as unknown as sinon.SinonStubbedInstance<ContextProxy>

		const map = new Map<string, string>()

		mockExtensionContext = {
			secrets: {
				get: sinon.stub().callsFake((key: string) => map.get(key)),
				store: sinon.stub().callsFake((key: string, value: string) => map.set(key, value)),
			},
		} as unknown as sinon.SinonStubbedInstance<vscode.ExtensionContext>
	})

	suite("importSettings", () => {
		test("should return success: false when user cancels file selection", async () => {
			// Mock user canceling file selection
			;(vscode.window.showOpenDialog as sinon.SinonStub).resolves(undefined)

			const result = await importSettings({
				providerSettingsManager: mockProviderSettingsManager,
				contextProxy: mockContextProxy,
			})

			assert.deepStrictEqual(result, { success: false })
			assert.ok(vscode.window.showOpenDialog.calledWith({
				filters: { JSON: ["json"] },
				canSelectMany: false,
			}))
			assert.ok(!fs.readFile.called)
			assert.ok(!mockProviderSettingsManager.import.called)
			assert.ok(!mockContextProxy.setValues.called)
		})

		test("should import settings successfully from a valid file", async () => {
			// Mock successful file selection
			;(vscode.window.showOpenDialog as sinon.SinonStub).resolves([{ fsPath: "/mock/path/settings.json" }])

			// Valid settings content
			const mockFileContent = JSON.stringify({
				providerProfiles: {
					currentApiConfigName: "test",
					apiConfigs: {
						test: {
							apiProvider: "openai" as ProviderName,
							apiKey: "test-key",
							id: "test-id",
						},
					},
				},
				globalSettings: {
					mode: "code",
					autoApprovalEnabled: true,
				},
			})

			// Mock reading file
			;(fs.readFile as sinon.SinonStub).resolves(mockFileContent)

			// Mock export returning previous provider profiles
			const previousProviderProfiles = {
				currentApiConfigName: "default",
				apiConfigs: {
					default: {
						apiProvider: "anthropic" as ProviderName,
						id: "default-id",
					},
				},
			}
			mockProviderSettingsManager.export.resolves(previousProviderProfiles)

			// Mock listConfig
			mockProviderSettingsManager.listConfig.resolves([
				{ name: "test", id: "test-id", apiProvider: "openai" as ProviderName },
				{ name: "default", id: "default-id", apiProvider: "anthropic" as ProviderName },
			])

			// Mock contextProxy.export
			mockContextProxy.export.resolves({
				mode: "code",
			})

			const result = await importSettings({
				providerSettingsManager: mockProviderSettingsManager,
				contextProxy: mockContextProxy,
			})

			assert.strictEqual(result.success, true)
			assert.ok(fs.readFile.calledWith("/mock/path/settings.json", "utf-8"))
			assert.ok(mockProviderSettingsManager.export.called)
			assert.ok(mockProviderSettingsManager.import.calledWith({
				...previousProviderProfiles,
				currentApiConfigName: "test",
				apiConfigs: {
					test: {
						apiProvider: "openai" as ProviderName,
						apiKey: "test-key",
						id: "test-id",
					},
				},
			})
			assert.ok(mockContextProxy.setValues.calledWith({
				mode: "code",
				autoApprovalEnabled: true,
			}))
			assert.ok(mockContextProxy.setValue.calledWith("currentApiConfigName", "test"))
			assert.ok(mockContextProxy.setValue.calledWith("listApiConfigMeta", [
				{ name: "test", id: "test-id", apiProvider: "openai" as ProviderName },
				{ name: "default", id: "default-id", apiProvider: "anthropic" as ProviderName },
			]))
		})

		test("should return success: false when file content is invalid", async () => {
			// Mock successful file selection
			;(vscode.window.showOpenDialog as sinon.SinonStub).resolves([{ fsPath: "/mock/path/settings.json" }])

			// Invalid content (missing required fields)
			const mockInvalidContent = JSON.stringify({
				providerProfiles: { apiConfigs: {} },
				globalSettings: {},
			})

			// Mock reading file
			;(fs.readFile as sinon.SinonStub).resolves(mockInvalidContent)

			const result = await importSettings({
				providerSettingsManager: mockProviderSettingsManager,
				contextProxy: mockContextProxy,
			})

			assert.deepStrictEqual(result, { success: false })
			assert.ok(fs.readFile.calledWith("/mock/path/settings.json", "utf-8"))
			assert.ok(!mockProviderSettingsManager.import.called)
			assert.ok(!mockContextProxy.setValues.called)
		})

		test("should return success: false when file content is not valid JSON", async () => {
			// Mock successful file selection
			;(vscode.window.showOpenDialog as sinon.SinonStub).resolves([{ fsPath: "/mock/path/settings.json" }])

			// Invalid JSON
			const mockInvalidJson = "{ this is not valid JSON }"

			// Mock reading file
			;(fs.readFile as sinon.SinonStub).resolves(mockInvalidJson)

			const result = await importSettings({
				providerSettingsManager: mockProviderSettingsManager,
				contextProxy: mockContextProxy,
			})

			assert.deepStrictEqual(result, { success: false })
			assert.ok(fs.readFile.calledWith("/mock/path/settings.json", "utf-8"))
			assert.ok(!mockProviderSettingsManager.import.called)
			assert.ok(!mockContextProxy.setValues.called)
		})

		test("should return success: false when reading file fails", async () => {
			// Mock successful file selection
			;(vscode.window.showOpenDialog as sinon.SinonStub).resolves([{ fsPath: "/mock/path/settings.json" }])

			// Mock file read error
			;(fs.readFile as sinon.SinonStub).rejects(new Error("File read error"))

			const result = await importSettings({
				providerSettingsManager: mockProviderSettingsManager,
				contextProxy: mockContextProxy,
			})

			assert.deepStrictEqual(result, { success: false })
			assert.ok(fs.readFile.calledWith("/mock/path/settings.json", "utf-8"))
			assert.ok(!mockProviderSettingsManager.import.called)
			assert.ok(!mockContextProxy.setValues.called)
		})

		test("should not clobber existing api configs", async () => {
			const providerSettingsManager = new ProviderSettingsManager(mockExtensionContext)
			await providerSettingsManager.saveConfig("openai", { apiProvider: "openai", id: "openai" })

			const configs = await providerSettingsManager.listConfig()
			assert.strictEqual(configs[0].name, "default")
			assert.strictEqual(configs[1].name, "openai")
			;(vscode.window.showOpenDialog as sinon.SinonStub).resolves([{ fsPath: "/mock/path/settings.json" }])

			const mockFileContent = JSON.stringify({
				globalSettings: { mode: "code" },
				providerProfiles: {
					currentApiConfigName: "anthropic",
					apiConfigs: { default: { apiProvider: "anthropic" as const, id: "anthropic" } },
				},
			})

			;(fs.readFile as sinon.SinonStub).resolves(mockFileContent)

			mockContextProxy.export.resolves({ mode: "code" })

			const result = await importSettings({
				providerSettingsManager,
				contextProxy: mockContextProxy,
			})

			assert.strictEqual(result.success, true)
			assert.notStrictEqual(result.providerProfiles?.apiConfigs["openai"], undefined)
			assert.notStrictEqual(result.providerProfiles?.apiConfigs["default"], undefined)
			assert.strictEqual(result.providerProfiles?.apiConfigs["default"].apiProvider, "anthropic")
		})
	})

	suite("exportSettings", () => {
		test("should not export settings when user cancels file selection", async () => {
			// Mock user canceling file selection
			;(vscode.window.showSaveDialog as sinon.SinonStub).resolves(undefined)

			await exportSettings({
				providerSettingsManager: mockProviderSettingsManager,
				contextProxy: mockContextProxy,
			})

			assert.ok(vscode.window.showSaveDialog.calledWith({
				filters: { JSON: ["json"] },
				defaultUri: expect.anything()),
			})
			assert.ok(!mockProviderSettingsManager.export.called)
			assert.ok(!mockContextProxy.export.called)
			assert.ok(!fs.writeFile.called)
		})

		test("should export settings to the selected file location", async () => {
			// Mock successful file location selection
			;(vscode.window.showSaveDialog as sinon.SinonStub).resolves({
				fsPath: "/mock/path/thea-code-settings.json",
			})

			// Mock providerProfiles data
			const mockProviderProfiles = {
				currentApiConfigName: "test",
				apiConfigs: {
					test: {
						apiProvider: "openai" as ProviderName,
						id: "test-id",
					},
				},
			}
			mockProviderSettingsManager.export.resolves(mockProviderProfiles)

			// Mock globalSettings data
			const mockGlobalSettings = {
				mode: "code",
				autoApprovalEnabled: true,
			}
			mockContextProxy.export.resolves(mockGlobalSettings)

			await exportSettings({
				providerSettingsManager: mockProviderSettingsManager,
				contextProxy: mockContextProxy,
			})

			assert.ok(vscode.window.showSaveDialog.calledWith({
				filters: { JSON: ["json"] },
				defaultUri: expect.anything()),
			})
			assert.ok(mockProviderSettingsManager.export.called)
			assert.ok(mockContextProxy.export.called)
			assert.ok(fs.mkdir.calledWith("/mock/path", { recursive: true }))
			assert.ok(fs.writeFile.calledWith(
				"/mock/path/thea-code-settings.json",
				JSON.stringify(
					{
						providerProfiles: mockProviderProfiles,
						globalSettings: mockGlobalSettings,
					},
					null,
					2,
				)),
				"utf-8",
			)
		})

		test("should handle errors during the export process", async () => {
			// Mock successful file location selection
			;(vscode.window.showSaveDialog as sinon.SinonStub).resolves({
				fsPath: "/mock/path/thea-code-settings.json",
			})

			// Mock provider profiles
			mockProviderSettingsManager.export.resolves({
				currentApiConfigName: "test",
				apiConfigs: {
					test: {
						apiProvider: "openai" as ProviderName,
						id: "test-id",
					},
				},
			})

			// Mock global settings
			mockContextProxy.export.resolves({
				mode: "code",
			})

			// Mock file write error
			;(fs.writeFile as sinon.SinonStub).rejects(new Error("Write error"))

			// The function catches errors internally and doesn't throw or return anything
			await exportSettings({
				providerSettingsManager: mockProviderSettingsManager,
				contextProxy: mockContextProxy,
			})

			assert.ok(vscode.window.showSaveDialog.called)
			assert.ok(mockProviderSettingsManager.export.called)
			assert.ok(mockContextProxy.export.called)
			assert.ok(fs.mkdir.calledWith("/mock/path", { recursive: true }))
			assert.ok(fs.writeFile.called)
			// The error is caught and the function exits silently
		})

		test("should handle errors during directory creation", async () => {
			// Mock successful file location selection
			;(vscode.window.showSaveDialog as sinon.SinonStub).resolves({
				fsPath: "/mock/path/thea-code-settings.json",
			})

			// Mock provider profiles
			mockProviderSettingsManager.export.resolves({
				currentApiConfigName: "test",
				apiConfigs: {
					test: {
						apiProvider: "openai" as ProviderName,
						id: "test-id",
					},
				},
			})

			// Mock global settings
			mockContextProxy.export.resolves({
				mode: "code",
			})

			// Mock directory creation error
			;(fs.mkdir as sinon.SinonStub).rejects(new Error("Directory creation error"))

			// The function catches errors internally and doesn't throw or return anything
			await exportSettings({
				providerSettingsManager: mockProviderSettingsManager,
				contextProxy: mockContextProxy,
			})

			assert.ok(vscode.window.showSaveDialog.called)
			assert.ok(mockProviderSettingsManager.export.called)
			assert.ok(mockContextProxy.export.called)
			assert.ok(fs.mkdir.called)
			assert.ok(!fs.writeFile.called) // Should not be called since mkdir failed
		})

		test("should use the correct default save location", async () => {
			// Mock user cancels to avoid full execution
			;(vscode.window.showSaveDialog as sinon.SinonStub).resolves(undefined)

			// Call the function
			await exportSettings({
				providerSettingsManager: mockProviderSettingsManager,
				contextProxy: mockContextProxy,
			})

			// Verify the default save location
			assert.ok(vscode.window.showSaveDialog.calledWith({
				filters: { JSON: ["json"] },
				defaultUri: expect.anything()),
			})

			// Verify Uri.file was called with the correct path
			assert.ok(vscode.Uri.file.calledWith(
				path.join("/mock/home", "Documents", SPECIFIC_STRINGS.SETTINGS_FILE_NAME)),
			)
		})
	})
// Mock cleanup
