import { defaultModeSlug, modes } from "../../../shared/modes"
import { SYSTEM_PROMPT } from "../../prompts/system"
import * as vscode from "vscode"
import * as fs from "fs/promises"
import { EXTENSION_CONFIG_DIR } from "../../../shared/config/thea-config"
import { toPosix } from "../../prompts/__tests__/utils"
import * as assert from 'assert'
import * as sinon from 'sinon'

// Mock the fs/promises module
// TODO: Use proxyquire for module mocking
		// Mock for "fs/promises" needed here
	readFile: sinon.stub(),
	mkdir: sinon.stub().resolves(undefined),
	access: sinon.stub().resolves(undefined),
// Mock cleanup needed

// Get the mocked fs module
const mockedFs = fs as sinon.SinonStubbedInstance<typeof fs>

// Mock the fileExistsAtPath function
// TODO: Use proxyquire for module mocking
		// Mock for "../../../utils/fs" needed here
	fileExistsAtPath: sinon.stub().resolves(true),
	createDirectoriesForFile: sinon.stub().resolves([]),
// Mock cleanup needed

// Create a mock ExtensionContext with relative paths instead of absolute paths
const mockContext = {
	extensionPath: "mock/extension/path",
	globalStoragePath: "mock/storage/path",
	storagePath: "mock/storage/path",
	logPath: "mock/log/path",
	subscriptions: [],
	workspaceState: {
		get: () => undefined,
		update: () => Promise.resolve(),
	},
	globalState: {
		get: () => undefined,
		update: () => Promise.resolve(),
		setKeysForSync: () => {},
	},
	extensionUri: { fsPath: "mock/extension/path" },
	globalStorageUri: { fsPath: "mock/settings/path" },
	asAbsolutePath: (relativePath: string) => `mock/extension/path/${relativePath}`,
	extension: {
		packageJSON: {
			version: "1.0.0",
		},
	},
} as unknown as vscode.ExtensionContext

suite("File-Based Custom System Prompt", () => {
	setup(() => {
		// Reset mocks before each test
		sinon.restore()

		// Default behavior: file doesn't exist
		mockedFs.readFile.rejects({ code: "ENOENT" })
	})

	test("should use default generation when no file-based system prompt is found", async () => {
		const customModePrompts = {
			[defaultModeSlug]: {
				roleDefinition: "Test role definition",
			},
		}

		const prompt = await SYSTEM_PROMPT(
			mockContext,
			"test/path", // Using a relative path without leading slash
			false, // supportsComputerUse
			undefined, // mcpHub
			undefined, // diffStrategy
			undefined, // browserViewportSize
			defaultModeSlug, // mode
			customModePrompts, // customModePrompts
			undefined, // customModes
			undefined, // globalCustomInstructions
			undefined, // diffEnabled
			undefined, // experiments
			true, // enableMcpServerCreation
		)

		// Should contain default sections
		assert.ok(prompt.includes("TOOL USE"))
		assert.ok(prompt.includes("CAPABILITIES"))
		assert.ok(prompt.includes("MODES"))
		assert.ok(prompt.includes("Test role definition"))
	})

	test("should use file-based custom system prompt when available", async () => {
		// Mock the readFile to return content from a file
		const fileCustomSystemPrompt = "Custom system prompt from file"
		// When called with utf-8 encoding, return a string
		mockedFs.readFile.callsFake((filePath, options) => {
			if (
				toPosix(filePath).includes(`${EXTENSION_CONFIG_DIR}/system-prompt-${defaultModeSlug}`) &&
				options === "utf-8"
			) {
				return Promise.resolve(fileCustomSystemPrompt)
			}
			return Promise.reject(new Error("ENOENT"))
		})

		const prompt = await SYSTEM_PROMPT(
			mockContext,
			"test/path", // Using a relative path without leading slash
			false, // supportsComputerUse
			undefined, // mcpHub
			undefined, // diffStrategy
			undefined, // browserViewportSize
			defaultModeSlug, // mode
			undefined, // customModePrompts
			undefined, // customModes
			undefined, // globalCustomInstructions
			undefined, // diffEnabled
			undefined, // experiments
			true, // enableMcpServerCreation
		)

		// Should contain role definition and file-based system prompt
		assert.ok(prompt.includes(modes[0].roleDefinition))
		assert.ok(prompt.includes(fileCustomSystemPrompt))

		// Should not contain any of the default sections
		assert.ok(!prompt.includes("CAPABILITIES"))
		assert.ok(!prompt.includes("MODES"))
	})

	test("should combine file-based system prompt with role definition and custom instructions", async () => {
		// Mock the readFile to return content from a file
		const fileCustomSystemPrompt = "Custom system prompt from file"
		mockedFs.readFile.callsFake((filePath, options) => {
			if (
				toPosix(filePath).includes(`${EXTENSION_CONFIG_DIR}/system-prompt-${defaultModeSlug}`) &&
				options === "utf-8"
			) {
				return Promise.resolve(fileCustomSystemPrompt)
			}
			return Promise.reject(new Error("ENOENT"))
		})

		// Define custom role definition
		const customRoleDefinition = "Custom role definition"
		const customModePrompts = {
			[defaultModeSlug]: {
				roleDefinition: customRoleDefinition,
			},
		}

		const prompt = await SYSTEM_PROMPT(
			mockContext,
			"test/path", // Using a relative path without leading slash
			false, // supportsComputerUse
			undefined, // mcpHub
			undefined, // diffStrategy
			undefined, // browserViewportSize
			defaultModeSlug, // mode
			customModePrompts, // customModePrompts
			undefined, // customModes
			undefined, // globalCustomInstructions
			undefined, // diffEnabled
			undefined, // experiments
			true, // enableMcpServerCreation
		)

		// Should contain custom role definition and file-based system prompt
		assert.ok(prompt.includes(customRoleDefinition))
		assert.ok(prompt.includes(fileCustomSystemPrompt))

		// Should not contain any of the default sections
		assert.ok(!prompt.includes("CAPABILITIES"))
		assert.ok(!prompt.includes("MODES"))
	})
// Mock cleanup
