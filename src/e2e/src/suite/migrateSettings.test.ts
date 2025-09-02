import * as vscode from "vscode"
import * as path from "path"
import * as fs from "fs/promises"
import { fileExistsAtPath } from "../utils/fs"
import { GlobalFileNames } from "../shared/globalFileNames"
import { migrateSettings } from "../utils/migrateSettings"

// Mock dependencies
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// We're testing the real migrateSettings function

import * as assert from 'assert'
import * as sinon from 'sinon'
import * as vscode from 'vscode'
declare global {
	var outputChannel: vscode.OutputChannel

suite("Settings Migration", () => {
	let mockContext: vscode.ExtensionContext
	let mockOutputChannel: vscode.OutputChannel
	const mockStoragePath = "/mock/storage"
	const mockSettingsDir = path.join(mockStoragePath, "settings")

	// Legacy file names
	const legacyCustomModesPath = path.join(mockSettingsDir, "cline_custom_modes.json")
	const legacyMcpSettingsPath = path.join(mockSettingsDir, "cline_mcp_settings.json")

	// New file names
	const newCustomModesPath = path.join(mockSettingsDir, GlobalFileNames.customModes)
	const newMcpSettingsPath = path.join(mockSettingsDir, GlobalFileNames.mcpSettings)

	setup(() => {
		sinon.restore()

		// Mock output channel
		mockOutputChannel = {
			appendLine: sinon.stub(),
			append: sinon.stub(),
			clear: sinon.stub(),
			show: sinon.stub(),
			hide: sinon.stub(),
			dispose: sinon.stub(),
		} as unknown as vscode.OutputChannel

		// Mock extension context
		mockContext = {
			globalStorageUri: { fsPath: mockStoragePath },
		} as unknown as vscode.ExtensionContext

		// The fs/promises mock is already set up in src/__mocks__/fs/promises.ts
		// We don't need to manually mock these methods

		// Set global outputChannel for all tests
		global.outputChannel = mockOutputChannel

	test("should migrate custom modes file if old file exists and new file doesn't", async () => {
		// Mock file existence checks
		;(fileExistsAtPath as sinon.SinonStub).callsFake((path: string) => {
			if (path === mockSettingsDir) return true
			if (path === legacyCustomModesPath) return true
			if (path === newCustomModesPath) return false
			return false

		await migrateSettings(mockContext, mockOutputChannel)

		// Verify file was renamed
		assert.ok(fs.rename.calledWith(legacyCustomModesPath, newCustomModesPath))

	test("should migrate MCP settings file if old file exists and new file doesn't", async () => {
		// Mock file existence checks
		;(fileExistsAtPath as sinon.SinonStub).callsFake((path: string) => {
			if (path === mockSettingsDir) return true
			if (path === legacyMcpSettingsPath) return true
			if (path === newMcpSettingsPath) return false
			return false

		await migrateSettings(mockContext, mockOutputChannel)

		// Verify file was renamed
		assert.ok(fs.rename.calledWith(legacyMcpSettingsPath, newMcpSettingsPath))

	test("should not migrate if new file already exists", async () => {
		// Mock file existence checks
		;(fileExistsAtPath as sinon.SinonStub).callsFake((path: string) => {
			if (path === mockSettingsDir) return true
			if (path === legacyCustomModesPath) return true
			if (path === newCustomModesPath) return true
			if (path === legacyMcpSettingsPath) return true
			if (path === newMcpSettingsPath) return true
			return false

		await migrateSettings(mockContext, mockOutputChannel)

		// Verify no files were renamed
		assert.ok(!fs.rename.called)

	test("should handle errors gracefully", async () => {
		// Mock file existence checks to throw an error
		;(fileExistsAtPath as sinon.SinonStub).rejects(new Error("Test error"))

		// Set the global outputChannel for the test
		global.outputChannel = mockOutputChannel

		await migrateSettings(mockContext, mockOutputChannel)

		// Verify error was logged
		expect(mockOutputChannel.appendLine.bind(mockOutputChannel)).toHaveBeenCalledWith(
			sinon.match.string.and(sinon.match("Error migrating settings files")),
