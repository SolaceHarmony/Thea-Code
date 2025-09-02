import * as assert from 'assert'
import * as sinon from 'sinon'
import * as vscode from 'vscode'
/* eslint-disable @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access, @typescript-eslint/no-unsafe-return */
import * as vscode from "vscode"

import { SYSTEM_PROMPT } from "../system"
import { McpHub } from "../../../services/mcp/management/McpHub"
import { defaultModeSlug, modes, ModeConfig } from "../../../shared/modes"
import "../../../utils/path" // Import path utils to get access to toPosix string extension.
import { addCustomInstructions } from "../sections/custom-instructions"
import { EXPERIMENT_IDS } from "../../../shared/experiments"

// Mock the sections
// TODO: Mock setup needs manual migration
// TODO: Mock needs manual migration
// TODO: Implement proper mock with proxyquire

// Mock the custom instructions
// TODO: Mock setup needs manual migration
// TODO: Fix mock - needs proxyquire
/*
=> {
	const addCustomInstructions = sinon.stub()
	return {
		addCustomInstructions,
		__setMockImplementation: (impl: (...args: unknown[]) => unknown) => {
			// Cast is safe in test context
			addCustomInstructions.callsFake(impl as typeof addCustomInstructions)
		},

})*/

// Set up default mock implementation
const { __setMockImplementation } = // TODO: requireMock("../sections/custom-instructions")
__setMockImplementation(
	(
		modeCustomInstructions: string,
		globalCustomInstructions: string,
		cwd: string,
		mode: string,
		options?: { language?: string },
	) => {
		const sections = []

		// Add language preference if provided
		if (options?.language) {
			sections.push(
				`Language Preference:\nYou should always speak and think in the "${options.language}" language.`,

		// Add global instructions first
		if (globalCustomInstructions?.trim()) {
			sections.push(`Global Instructions:\n${globalCustomInstructions.trim()}`)

		// Add mode-specific instructions after
		if (modeCustomInstructions?.trim()) {
			sections.push(`Mode-specific Instructions:\n${modeCustomInstructions}`)

		// Add rules
		const rules = []
		if (mode) {
			rules.push(`# Rules from .Thearules-${mode}:\nMock mode-specific rules`)

		rules.push(`# Rules from .Thearules:\nMock generic rules`)

		if (rules.length > 0) {
			sections.push(`Rules:\n${rules.join("\n")}`)

		const joinedSections = sections.join("\n\n")
		return joinedSections
			? `\n====\n\nUSER'S CUSTOM INSTRUCTIONS\n\nThe following additional instructions are provided by the user, and should be followed to the best of your ability without interfering with the TOOL USE guidelines.\n\n${joinedSections}`
			: ""
	},

// Mock environment-specific values for consistent tests
// TODO: Mock setup needs manual migration
// TODO: Mock needs manual migration
// TODO: Implement proper mock with proxyquire

// TODO: Mock setup needs manual migration
=> "/bin/zsh")

// TODO: Mock setup needs manual migration
// TODO: Mock needs manual migration
// ) => "Linux")

// Mock vscode language
// TODO: Mock setup needs manual migration
// TODO: Mock needs manual migration
// TODO: Implement proper mock with proxyquire

// TODO: Mock setup needs manual migration
// TODO: Mock needs manual migration
// TODO: Implement proper mock with proxyquire

// Create a mock ExtensionContext
const mockContext = {
	extensionPath: "/mock/extension/path",
	globalStoragePath: "/mock/storage/path",
	storagePath: "/mock/storage/path",
	logPath: "/mock/log/path",
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
	extensionUri: { fsPath: "/mock/extension/path" },
	globalStorageUri: { fsPath: "/mock/settings/path" },
	asAbsolutePath: (relativePath: string) => `/mock/extension/path/${relativePath}`,
	extension: {
		packageJSON: {
			version: "1.0.0",
		},
	},
} as unknown as vscode.ExtensionContext

// Instead of extending McpHub, create a mock that implements just what we need
const createMockMcpHub = (): McpHub =>
	({
		getServers: () => [],
		getMcpServersPath: () => "/mock/mcp/path",
		getMcpSettingsFilePath: () => "/mock/settings/path",
		dispose: () => {},
		// Add other required public methods with no-op implementations
		restartConnection: () => {},
		readResource: () => ({ contents: [] }),
		callTool: () => ({ content: [] }),
		toggleServerDisabled: () => {},
		toggleToolAlwaysAllow: () => {},
		isConnecting: false,
		connections: [],
	}) as unknown as McpHub

suite("SYSTEM_PROMPT", () => {
	let mockMcpHub: McpHub
	let experiments: Record<string, boolean> | undefined

	suiteSetup(() => {
		// Ensure fs mock is properly initialized
		const mockFs = // TODO: requireMock("fs/promises")
		mockFs._setInitialMockData()

		// Initialize all required directories
		const dirs = [
			"/mock",
			"/mock/extension",
			"/mock/extension/path",
			"/mock/storage",
			"/mock/storage/path",
			"/mock/settings",
			"/mock/settings/path",
			"/mock/mcp",
			"/mock/mcp/path",

		dirs.forEach((dir) => mockFs._mockDirectories.add(dir))

	setup(() => {
		// Reset experiments before each test to ensure they're disabled by default
		experiments = {
			[EXPERIMENT_IDS.SEARCH_AND_REPLACE]: false,
			[EXPERIMENT_IDS.INSERT_BLOCK]: false,

	setup(() => {
		sinon.restore()

	teardown(async () => {
		// Clean up any McpHub instances
		if (mockMcpHub) {
			await mockMcpHub.dispose()

	test("should maintain consistent system prompt", async () => {
		const prompt = await SYSTEM_PROMPT(
			mockContext,
			"/test/path",
			false, // supportsComputerUse
			undefined, // mcpHub
			undefined, // diffStrategy
			undefined, // browserViewportSize
			defaultModeSlug, // mode
			undefined, // customModePrompts
			undefined, // customModes
			undefined, // globalCustomInstructions
			undefined, // diffEnabled
			experiments,
			true, // enableMcpServerCreation

		expect(prompt).toMatchSnapshot()

	test("should include browser actions when supportsComputerUse is true", async () => {
		const prompt = await SYSTEM_PROMPT(
			mockContext,
			"/test/path",
			true, // supportsComputerUse
			undefined, // mcpHub
			undefined, // diffStrategy
			"1280x800", // browserViewportSize
			defaultModeSlug, // mode
			undefined, // customModePrompts
			undefined, // customModes,
			undefined, // globalCustomInstructions
			undefined, // diffEnabled
			experiments,
			true, // enableMcpServerCreation

		expect(prompt).toMatchSnapshot()

	test("should include MCP server info when mcpHub is provided", async () => {
		mockMcpHub = createMockMcpHub()

		const prompt = await SYSTEM_PROMPT(
			mockContext,
			"/test/path",
			false, // supportsComputerUse
			mockMcpHub, // mcpHub
			undefined, // diffStrategy
			undefined, // browserViewportSize
			defaultModeSlug, // mode
			undefined, // customModePrompts
			undefined, // customModes,
			undefined, // globalCustomInstructions
			undefined, // diffEnabled
			experiments,
			true, // enableMcpServerCreation

		expect(prompt).toMatchSnapshot()

	test("should explicitly handle undefined mcpHub", async () => {
		const prompt = await SYSTEM_PROMPT(
			mockContext,
			"/test/path",
			false, // supportsComputerUse
			undefined, // explicitly undefined mcpHub
			undefined, // diffStrategy
			undefined, // browserViewportSize
			defaultModeSlug, // mode
			undefined, // customModePrompts
			undefined, // customModes,
			undefined, // globalCustomInstructions
			undefined, // diffEnabled
			experiments,
			true, // enableMcpServerCreation

		expect(prompt).toMatchSnapshot()

	test("should handle different browser viewport sizes", async () => {
		const prompt = await SYSTEM_PROMPT(
			mockContext,
			"/test/path",
			true, // supportsComputerUse
			undefined, // mcpHub
			undefined, // diffStrategy
			"900x600", // different viewport size
			defaultModeSlug, // mode
			undefined, // customModePrompts
			undefined, // customModes,
			undefined, // globalCustomInstructions
			undefined, // diffEnabled
			experiments,
			true, // enableMcpServerCreation

		expect(prompt).toMatchSnapshot()

	test("should include diff strategy tool description when diffEnabled is true", async () => {
		// Create a mock diff strategy that returns a Promise<DiffResult>
		const mockDiffStrategy = {
			getName: () => "MockStrategy",
			getToolDescription: () => "apply_diff tool description",
			applyDiff: sinon.stub().resolves({ success: true, content: "mock result" }),

		const prompt = await SYSTEM_PROMPT(
			mockContext,
			"/test/path",
			false, // supportsComputerUse
			undefined, // mcpHub
			mockDiffStrategy, // Use mock diff strategy that returns Promise
			undefined, // browserViewportSize
			defaultModeSlug, // mode
			undefined, // customModePrompts
			undefined, // customModes
			undefined, // globalCustomInstructions
			true, // diffEnabled
			experiments,
			true, // enableMcpServerCreation

		assert.ok(prompt.includes("apply_diff"))
		expect(prompt).toMatchSnapshot()

	test("should exclude diff strategy tool description when diffEnabled is false", async () => {
		// Create a mock diff strategy that returns a Promise<DiffResult>
		const mockDiffStrategy = {
			getName: () => "MockStrategy",
			getToolDescription: () => "apply_diff tool description",
			applyDiff: sinon.stub().resolves({ success: true, content: "mock result" }),

		const prompt = await SYSTEM_PROMPT(
			mockContext,
			"/test/path",
			false, // supportsComputerUse
			undefined, // mcpHub
			mockDiffStrategy, // Use mock diff strategy that returns Promise
			undefined, // browserViewportSize
			defaultModeSlug, // mode
			undefined, // customModePrompts
			undefined, // customModes
			undefined, // globalCustomInstructions
			false, // diffEnabled
			experiments,
			true, // enableMcpServerCreation

		expect(prompt).not.toContain("apply_diff")
		expect(prompt).toMatchSnapshot()

	test("should exclude diff strategy tool description when diffEnabled is undefined", async () => {
		// Create a mock diff strategy that returns a Promise<DiffResult>
		const mockDiffStrategy = {
			getName: () => "MockStrategy",
			getToolDescription: () => "apply_diff tool description",
			applyDiff: sinon.stub().resolves({ success: true, content: "mock result" }),

		const prompt = await SYSTEM_PROMPT(
			mockContext,
			"/test/path",
			false, // supportsComputerUse
			undefined, // mcpHub
			mockDiffStrategy, // Use mock diff strategy that returns Promise
			undefined, // browserViewportSize
			defaultModeSlug, // mode
			undefined, // customModePrompts
			undefined, // customModes
			undefined, // globalCustomInstructions
			undefined, // diffEnabled
			experiments,
			true, // enableMcpServerCreation

		expect(prompt).not.toContain("apply_diff")
		expect(prompt).toMatchSnapshot()

	test("should include vscode language in custom instructions", async () => {
		// Mock vscode.env.language
		const vscode = // TODO: requireMock("vscode")
		vscode.env = { language: "es" }

		const prompt = await SYSTEM_PROMPT(
			mockContext,
			"/test/path",
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

		assert.ok(prompt.includes("Language Preference:"))
		assert.ok(prompt.includes('You should always speak and think in the "es" language'))

		// Reset mock
		vscode.env = { language: "en" }

	test("should include custom mode role definition at top and instructions at bottom", async () => {
		const modeCustomInstructions = "Custom mode instructions"

		const customModes: ModeConfig[] = [
			{
				slug: "custom-mode",
				name: "Custom Mode",
				roleDefinition: "Custom role definition",
				customInstructions: modeCustomInstructions,
				groups: ["read"] as const,
			},

		const prompt = await SYSTEM_PROMPT(
			mockContext,
			"/test/path",
			false, // supportsComputerUse
			undefined, // mcpHub
			undefined, // diffStrategy
			undefined, // browserViewportSize
			"custom-mode", // mode
			undefined, // customModePrompts
			customModes, // customModes
			"Global instructions", // globalCustomInstructions
			undefined, // diffEnabled
			experiments,
			true, // enableMcpServerCreation

		// Role definition should be at the top
		expect(prompt.indexOf("Custom role definition")).toBeLessThan(prompt.indexOf("TOOL USE"))

		// Custom instructions should be at the bottom
		const customInstructionsIndex = prompt.indexOf("Custom mode instructions")
		const userInstructionsHeader = prompt.indexOf("USER'S CUSTOM INSTRUCTIONS")
		assert.ok(customInstructionsIndex > -1)
		assert.ok(userInstructionsHeader > -1)
		assert.ok(customInstructionsIndex > userInstructionsHeader)

	test("should use promptComponent roleDefinition when available", async () => {
		const customModePrompts = {
			[defaultModeSlug]: {
				roleDefinition: "Custom prompt role definition",
				customInstructions: "Custom prompt instructions",
			},

		const prompt = await SYSTEM_PROMPT(
			mockContext,
			"/test/path",
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
			false, // enableMcpServerCreation

		// Role definition from promptComponent should be at the top
		expect(prompt.indexOf("Custom prompt role definition")).toBeLessThan(prompt.indexOf("TOOL USE"))
		// Should not contain the default mode's role definition
		expect(prompt).not.toContain(modes[0].roleDefinition)

	test("should fallback to modeConfig roleDefinition when promptComponent has no roleDefinition", async () => {
		const customModePrompts = {
			[defaultModeSlug]: {
				customInstructions: "Custom prompt instructions",
				// No roleDefinition provided
			},

		const prompt = await SYSTEM_PROMPT(
			mockContext,
			"/test/path",
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
			false, // enableMcpServerCreation

		// Should use the default mode's role definition
		expect(prompt.indexOf(modes[0].roleDefinition)).toBeLessThan(prompt.indexOf("TOOL USE"))

	suite("experimental tools", () => {
		test("should disable experimental tools by default", async () => {
			// Set experiments to explicitly disable experimental tools
			const experimentsConfig = {
				[EXPERIMENT_IDS.SEARCH_AND_REPLACE]: false,
				[EXPERIMENT_IDS.INSERT_BLOCK]: false,

			// Reset experiments
			experiments = experimentsConfig

			const prompt = await SYSTEM_PROMPT(
				mockContext,
				"/test/path",
				false, // supportsComputerUse
				undefined, // mcpHub
				undefined, // diffStrategy
				undefined, // browserViewportSize
				defaultModeSlug, // mode
				undefined, // customModePrompts
				undefined, // customModes
				undefined, // globalCustomInstructions
				undefined, // diffEnabled
				experimentsConfig, // Explicitly disable experimental tools
				true, // enableMcpServerCreation

			// Check that experimental tool sections are not included
			const toolSections = prompt.split("\n## ").slice(1)
			const toolNames = toolSections.map((section) => section.split("\n")[0].trim())
			expect(toolNames).not.toContain("search_and_replace")
			expect(toolNames).not.toContain("insert_content")
			expect(prompt).toMatchSnapshot()

		test("should enable experimental tools when explicitly enabled", async () => {
			// Set experiments for testing experimental features
			const experimentsEnabled = {
				[EXPERIMENT_IDS.SEARCH_AND_REPLACE]: true,
				[EXPERIMENT_IDS.INSERT_BLOCK]: true,

			// Reset default experiments
			experiments = undefined

			const prompt = await SYSTEM_PROMPT(
				mockContext,
				"/test/path",
				false, // supportsComputerUse
				undefined, // mcpHub
				undefined, // diffStrategy
				undefined, // browserViewportSize
				defaultModeSlug, // mode
				undefined, // customModePrompts
				undefined, // customModes
				undefined, // globalCustomInstructions
				undefined, // diffEnabled
				experimentsEnabled, // Use the enabled experiments
				true, // enableMcpServerCreation

			// Get all tool sections
			const toolSections = prompt.split("## ").slice(1) // Split by section headers and remove first non-tool part
			const toolNames = toolSections.map((section) => section.split("\n")[0].trim())

			// Verify experimental tools are included in the prompt when enabled
			assert.ok(toolNames.includes("search_and_replace"))
			assert.ok(toolNames.includes("insert_content"))
			expect(prompt).toMatchSnapshot()

		test("should selectively enable experimental tools", async () => {
			// Set experiments for testing selective enabling
			const experimentsSelective = {
				[EXPERIMENT_IDS.SEARCH_AND_REPLACE]: true,
				[EXPERIMENT_IDS.INSERT_BLOCK]: false,

			// Reset default experiments
			experiments = undefined

			const prompt = await SYSTEM_PROMPT(
				mockContext,
				"/test/path",
				false, // supportsComputerUse
				undefined, // mcpHub
				undefined, // diffStrategy
				undefined, // browserViewportSize
				defaultModeSlug, // mode
				undefined, // customModePrompts
				undefined, // customModes
				undefined, // globalCustomInstructions
				undefined, // diffEnabled
				experimentsSelective, // Use the selective experiments
				true, // enableMcpServerCreation

			// Get all tool sections
			const toolSections = prompt.split("## ").slice(1) // Split by section headers and remove first non-tool part
			const toolNames = toolSections.map((section) => section.split("\n")[0].trim())

			// Verify only enabled experimental tools are included
			assert.ok(toolNames.includes("search_and_replace"))
			expect(toolNames).not.toContain("insert_content")
			expect(prompt).toMatchSnapshot()

		test("should list all available editing tools in base instruction", async () => {
			const experiments = {
				[EXPERIMENT_IDS.SEARCH_AND_REPLACE]: true,
				[EXPERIMENT_IDS.INSERT_BLOCK]: true,

			// Create a mock diff strategy that returns a Promise<DiffResult>
			const mockDiffStrategy = {
				getName: () => "MockStrategy",
				getToolDescription: () => "apply_diff tool description",
				applyDiff: sinon.stub().resolves({ success: true, content: "mock result" }),

			const prompt = await SYSTEM_PROMPT(
				mockContext,
				"/test/path",
				false,
				undefined,
				mockDiffStrategy, // Use mock diff strategy that returns Promise
				undefined,
				defaultModeSlug,
				undefined,
				undefined,
				undefined,
				true, // diffEnabled
				experiments, // experiments
				true, // enableMcpServerCreation

			// Verify base instruction lists all available tools
			assert.ok(prompt.includes("apply_diff (for replacing lines in existing files))")
			assert.ok(prompt.includes("write_to_file (for creating new files or complete file rewrites))")
			assert.ok(prompt.includes("insert_content (for adding lines to existing files))")
			assert.ok(prompt.includes("search_and_replace (for finding and replacing individual pieces of text))")

		test("should provide detailed instructions for each enabled tool", async () => {
			const experiments = {
				[EXPERIMENT_IDS.SEARCH_AND_REPLACE]: true,
				[EXPERIMENT_IDS.INSERT_BLOCK]: true,

			// Create a mock diff strategy that returns a Promise<DiffResult>
			const mockDiffStrategy = {
				getName: () => "MockStrategy",
				getToolDescription: () => "apply_diff tool description",
				applyDiff: sinon.stub().resolves({ success: true, content: "mock result" }),

			const prompt = await SYSTEM_PROMPT(
				mockContext,
				"/test/path",
				false,
				undefined,
				mockDiffStrategy, // Use mock diff strategy that returns Promise
				undefined,
				defaultModeSlug,
				undefined,
				undefined,
				undefined,
				true, // diffEnabled
				experiments,
				true, // enableMcpServerCreation

			// Verify detailed instructions for each tool
			assert.ok(prompt.includes(
				"You should always prefer using other editing tools over write_to_file when making changes to existing files since write_to_file is much slower and cannot handle large files.",
			assert.ok(prompt.includes("The insert_content tool adds lines of text to files"))
			assert.ok(prompt.includes("The search_and_replace tool finds and replaces text or regex in files"))

	suiteTeardown(() => {
		sinon.restore()

suite("addCustomInstructions", () => {
	suiteSetup(() => {
		// Ensure fs mock is properly initialized
		const mockFs = // TODO: requireMock("fs/promises")
		mockFs._setInitialMockData()
		mockFs.mkdir.callsFake(async (path: string) => {
			if (path.startsWith("/test")) {
				mockFs._mockDirectories.add(path)
				return Promise.resolve()

			throw new Error(`ENOENT: no such file or directory, mkdir '${path}'`)

	setup(() => {
		sinon.restore()

	test("should generate correct prompt for architect mode", async () => {
		const prompt = await SYSTEM_PROMPT(
			mockContext,
			"/test/path",
			false, // supportsComputerUse
			undefined, // mcpHub
			undefined, // diffStrategy
			undefined, // browserViewportSize
			"architect", // mode
			undefined, // customModePrompts
			undefined, // customModes
			undefined, // globalCustomInstructions
			undefined, // diffEnabled
			undefined, // experiments
			true, // enableMcpServerCreation

		expect(prompt).toMatchSnapshot()

	test("should generate correct prompt for ask mode", async () => {
		const prompt = await SYSTEM_PROMPT(
			mockContext,
			"/test/path",
			false, // supportsComputerUse
			undefined, // mcpHub
			undefined, // diffStrategy
			undefined, // browserViewportSize
			"ask", // mode
			undefined, // customModePrompts
			undefined, // customModes
			undefined, // globalCustomInstructions
			undefined, // diffEnabled
			undefined, // experiments
			true, // enableMcpServerCreation

		expect(prompt).toMatchSnapshot()

	test("should include MCP server creation info when enabled", async () => {
		const mockMcpHub = createMockMcpHub()

		const prompt = await SYSTEM_PROMPT(
			mockContext,
			"/test/path",
			false, // supportsComputerUse
			mockMcpHub, // mcpHub
			undefined, // diffStrategy
			undefined, // browserViewportSize
			defaultModeSlug, // mode
			undefined, // customModePrompts
			undefined, // customModes,
			undefined, // globalCustomInstructions
			undefined, // diffEnabled
			undefined, // experiments
			true, // enableMcpServerCreation

		assert.ok(prompt.includes("Creating an MCP Server"))
		expect(prompt).toMatchSnapshot()

	test("should exclude MCP server creation info when disabled", async () => {
		const mockMcpHub = createMockMcpHub()

		const prompt = await SYSTEM_PROMPT(
			mockContext,
			"/test/path",
			false, // supportsComputerUse
			mockMcpHub, // mcpHub
			undefined, // diffStrategy
			undefined, // browserViewportSize
			defaultModeSlug, // mode
			undefined, // customModePrompts
			undefined, // customModes,
			undefined, // globalCustomInstructions
			undefined, // diffEnabled
			undefined, // experiments
			false, // enableMcpServerCreation

		expect(prompt).not.toContain("Creating an MCP Server")
		expect(prompt).toMatchSnapshot()

	test("should prioritize mode-specific rules for code mode", async () => {
		const instructions = await addCustomInstructions("", "", "/test/path", defaultModeSlug)
		expect(instructions).toMatchSnapshot()

	test("should prioritize mode-specific rules for ask mode", async () => {
		const instructions = await addCustomInstructions("", "", "/test/path", modes[2].slug)
		expect(instructions).toMatchSnapshot()

	test("should prioritize mode-specific rules for architect mode", async () => {
		const instructions = await addCustomInstructions("", "", "/test/path", modes[1].slug)
		expect(instructions).toMatchSnapshot()

	test("should prioritize mode-specific rules for test engineer mode", async () => {
		const instructions = await addCustomInstructions("", "", "/test/path", "test")
		expect(instructions).toMatchSnapshot()

	test("should prioritize mode-specific rules for code reviewer mode", async () => {
		const instructions = await addCustomInstructions("", "", "/test/path", "review")
		expect(instructions).toMatchSnapshot()

	test("should fall back to generic rules when mode-specific rules not found", async () => {
		const instructions = await addCustomInstructions("", "", "/test/path", defaultModeSlug)
		expect(instructions).toMatchSnapshot()

	test("should include preferred language when provided", async () => {
		const instructions = await addCustomInstructions("", "", "/test/path", defaultModeSlug, {
			language: "es",

		expect(instructions).toMatchSnapshot()

	test("should include custom instructions when provided", async () => {
		const instructions = await addCustomInstructions("Custom test instructions", "", "/test/path", defaultModeSlug)
		expect(instructions).toMatchSnapshot()

	test("should combine all custom instructions", async () => {
		const instructions = await addCustomInstructions(
			"Custom test instructions",
			"",
			"/test/path",
			defaultModeSlug,
			{ language: "fr" },

		expect(instructions).toMatchSnapshot()

	test("should handle undefined mode-specific instructions", async () => {
		const instructions = await addCustomInstructions("", "", "/test/path", defaultModeSlug)
		expect(instructions).toMatchSnapshot()

	test("should trim mode-specific instructions", async () => {
		const instructions = await addCustomInstructions(
			"  Custom mode instructions  ",
			"",
			"/test/path",
			defaultModeSlug,

		expect(instructions).toMatchSnapshot()

	test("should handle empty mode-specific instructions", async () => {
		const instructions = await addCustomInstructions("", "", "/test/path", defaultModeSlug)
		expect(instructions).toMatchSnapshot()

	test("should combine global and mode-specific instructions", async () => {
		const instructions = await addCustomInstructions(
			"Mode-specific instructions",
			"Global instructions",
			"/test/path",
			defaultModeSlug,

		expect(instructions).toMatchSnapshot()

	test("should prioritize mode-specific instructions after global ones", async () => {
		const instructions = await addCustomInstructions(
			"Second instruction",
			"First instruction",
			"/test/path",
			defaultModeSlug,

		const instructionParts = instructions.split("\n\n")
		const globalIndex = instructionParts.findIndex((part) => part.includes("First instruction"))
		const modeSpecificIndex = instructionParts.findIndex((part) => part.includes("Second instruction"))

		assert.ok(globalIndex < modeSpecificIndex)
		expect(instructions).toMatchSnapshot()

	suiteTeardown(() => {
		sinon.restore()
