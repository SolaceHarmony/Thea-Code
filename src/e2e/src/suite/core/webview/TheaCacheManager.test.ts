import * as vscode from "vscode"
import * as path from "path"
import fs from "fs/promises"
import { TheaCacheManager } from "../cache/TheaCacheManager" // Updated import
import { fileExistsAtPath } from "../../../utils/fs"
import { ModelInfo } from "../../../shared/api"

// Mock dependencies
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
// TODO: Mock setup needs manual migration
import * as assert from 'assert'
import * as sinon from 'sinon'
import * as vscode from 'vscode'
// TODO: Fix mock - needs proxyquire
/*
=> {
	return {
		getCacheDirectoryPath: sinon.stub().callsFake((storagePath: string) => {
			return path.join(storagePath, "cache")
		}),
		getSettingsDirectoryPath: sinon.stub().callsFake((storagePath: string) => {
			return path.join(storagePath, "settings")
		}),

})*/

suite("TheaCacheManager", () => {
	// Updated describe block
	let cacheManager: TheaCacheManager // Updated type
	let mockContext: vscode.ExtensionContext

	setup(() => {
		// Reset mocks
		sinon.restore()

		// Mock context
		mockContext = {
			extensionPath: "/test/path",
			extensionUri: {} as vscode.Uri,
			globalStorageUri: {
				fsPath: "/test/storage/path",
			},
		} as unknown as vscode.ExtensionContext

		// Create instance of TheaCacheManager
		cacheManager = new TheaCacheManager(mockContext) // Updated instantiation

		// Mock console.error to prevent test output noise
		sinon.spy(console, "error").callsFake(() => {})
		sinon.spy(console, "log").callsFake(() => {})

		// Mock fs.readdir
		;(fs.readdir as sinon.SinonStub) = sinon.stub()

	teardown(() => {
		// Restore console.error mock
		sinon.restore()

	test("ensureCacheDirectoryExists creates and returns cache directory path", async () => {
		// Setup
		const expectedCachePath = path.join("/test/storage/path", "cache")

		// Execute
		const result = await cacheManager.ensureCacheDirectoryExists()

		// Verify
		assert.ok(fs.mkdir.calledWith(expectedCachePath, { recursive: true }))
		assert.strictEqual(result, expectedCachePath)

	test("ensureCacheDirectoryExists handles errors gracefully", async () => {
		// Setup
		const expectedCachePath = path.join("/test/storage/path", "cache")
		const mockError = new Error("Directory creation failed")
		;(fs.mkdir as sinon.SinonStub).rejects(mockError)

		// Execute
		const result = await cacheManager.ensureCacheDirectoryExists()

		// Verify
		assert.ok(fs.mkdir.calledWith(expectedCachePath, { recursive: true }))
		assert.ok(console.error.called)
		assert.strictEqual(result, expectedCachePath)

	test("ensureSettingsDirectoryExists creates and returns settings directory path", async () => {
		// Setup
		const expectedSettingsPath = path.join("/test/storage/path", "settings")

		// Execute
		const result = await cacheManager.ensureSettingsDirectoryExists()

		// Verify
		assert.ok(fs.mkdir.calledWith(expectedSettingsPath, { recursive: true }))
		assert.strictEqual(result, expectedSettingsPath)

	test("ensureSettingsDirectoryExists handles errors gracefully", async () => {
		// Setup
		const expectedSettingsPath = path.join("/test/storage/path", "settings")
		const mockError = new Error("Directory creation failed")
		;(fs.mkdir as sinon.SinonStub).rejects(mockError)

		// Execute
		const result = await cacheManager.ensureSettingsDirectoryExists()

		// Verify
		assert.ok(fs.mkdir.calledWith(expectedSettingsPath, { recursive: true }))
		assert.ok(console.error.called)
		assert.strictEqual(result, expectedSettingsPath)

	test("readModelsFromCache reads and parses file content", async () => {
		// Setup
		const expectedCachePath = path.join("/test/storage/path", "cache")
		const filename = "models.json"
		const expectedFilePath = path.join(expectedCachePath, filename)
		const mockModelData: Record<string, ModelInfo> = {
			"model-1": {
				contextWindow: 4000,
				supportsPromptCache: true,
				maxTokens: 1000,
			},

		;(fileExistsAtPath as sinon.SinonStub).resolves(true)
		;(fs.readFile as sinon.SinonStub).resolves(JSON.stringify(mockModelData))

		// Execute
		const result = await cacheManager.readModelsFromCache(filename)

		// Verify
		assert.ok(fileExistsAtPath.calledWith(expectedFilePath))
		assert.ok(fs.readFile.calledWith(expectedFilePath, "utf8"))
		assert.deepStrictEqual(result, mockModelData)

	test("readModelsFromCache returns undefined if file doesn't exist", async () => {
		// Setup
		const filename = "non-existent.json"
		;(fileExistsAtPath as sinon.SinonStub).resolves(false)

		// Execute
		const result = await cacheManager.readModelsFromCache(filename)

		// Verify
		assert.strictEqual(result, undefined)
		assert.ok(!fs.readFile.called)

	test("readModelsFromCache handles read/parse errors gracefully", async () => {
		// Setup
		const filename = "corrupt.json"
		;(fileExistsAtPath as sinon.SinonStub).resolves(true)
		;(fs.readFile as sinon.SinonStub).rejects(new Error("Read error"))

		// Execute
		const result = await cacheManager.readModelsFromCache(filename)

		// Verify
		assert.strictEqual(result, undefined)
		assert.ok(console.error.called)

	test("writeModelsToCache writes model data to file", async () => {
		// Setup
		const expectedCachePath = path.join("/test/storage/path", "cache")
		const filename = "models.json"
		const expectedFilePath = path.join(expectedCachePath, filename)
		const mockModelData: Record<string, ModelInfo> = {
			"model-1": {
				contextWindow: 4000,
				supportsPromptCache: true,
				maxTokens: 1000,
			},

		// Execute
		await cacheManager.writeModelsToCache(filename, mockModelData)

		// Verify
		assert.ok(fs.writeFile.calledWith(expectedFilePath, JSON.stringify(mockModelData, null, 2)))

	test("writeModelsToCache handles write errors gracefully", async () => {
		// Setup
		const filename = "error.json"
		const mockModelData: Record<string, ModelInfo> = {
			"model-1": {
				contextWindow: 4000,
				supportsPromptCache: true,
				maxTokens: 1000,
			},

		const mockError = new Error("Write error")
		;(fs.writeFile as sinon.SinonStub).rejects(mockError)

		// Execute
		await cacheManager.writeModelsToCache(filename, mockModelData)

		// Verify
		assert.ok(console.error.called)

	test("clearCache deletes all files in cache directory", async () => {
		// Setup
		const expectedCachePath = path.join("/test/storage/path", "cache")
		const mockFiles = ["file1.json", "file2.json"]
		;(fs.readdir as sinon.SinonStub).resolves(mockFiles)
		;(fs.unlink as sinon.SinonStub) = sinon.stub().resolves(undefined)

		// Execute
		await cacheManager.clearCache()

		// Verify
		assert.ok(fs.readdir.calledWith(expectedCachePath))
		assert.strictEqual(fs.unlink.callCount, 2)
		assert.ok(fs.unlink.calledWith(path.join(expectedCachePath, "file1.json")))
		assert.ok(fs.unlink.calledWith(path.join(expectedCachePath, "file2.json")))

	test("clearCache handles errors gracefully", async () => {
		// Setup
		const mockError = new Error("Read directory error")
		;(fs.readdir as sinon.SinonStub).rejects(mockError)

		// Execute
		await cacheManager.clearCache()

		// Verify
		assert.ok(console.error.called)
