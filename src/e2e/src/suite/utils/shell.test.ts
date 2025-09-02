import * as assert from 'assert'
import * as sinon from 'sinon'
import * as vscode from 'vscode'
import * as proxyquire from 'proxyquire'

suite("Shell Detection Tests", () => {
    let sandbox: sinon.SinonSandbox
    let originalPlatform: string
    let originalEnv: NodeJS.ProcessEnv
    let mockUserInfo: sinon.SinonStub
    let getShell: () => string
    let mockGetConfiguration: sinon.SinonStub

    // Helper to mock VS Code configuration
    function mockVsCodeConfig(
        platformKey: string,
        defaultProfileName: string | null,
        profiles: Record<string, unknown>,
    ) {
        mockGetConfiguration.returns({
            get: (key: string) => {
                if (key === `defaultProfile.${platformKey}`) {
                    return defaultProfileName

                if (key === `profiles.${platformKey}`) {
                    return profiles

                return undefined
            },

    // Helper function to create a properly typed mock for vscode.workspace.getConfiguration
    function createConfigMock(returnValue: unknown = undefined) {
        mockGetConfiguration.returns({
            get: () => returnValue

    setup(() => {
        sandbox = sinon.createSandbox()
        
        // Store original references
        originalPlatform = process.platform
        originalEnv = { ...process.env }

        // Clear environment variables for a clean test
        delete process.env.SHELL
        delete process.env.COMSPEC

        // Create stubs
        mockUserInfo = sandbox.stub()
        mockUserInfo.returns({
            shell: null,
            username: "testuser",
            uid: 1000,
            gid: 1000,
            homedir: "/home/testuser"

        mockGetConfiguration = sandbox.stub()
        
        // Mock vscode.workspace.getConfiguration
        sandbox.stub(vscode.workspace, 'getConfiguration').callsFake(mockGetConfiguration)

        // Use proxyquire to load the module with mocked os
        const shellModule = proxyquire.noCallThru()('../../../../utils/shell', {
            'os': {
                userInfo: mockUserInfo

        getShell = shellModule.getShell

    teardown(() => {
        // Restore everything
        sandbox.restore()
        Object.defineProperty(process, "platform", { value: originalPlatform })
        process.env = originalEnv

    // --------------------------------------------------------------------------
    // Windows Shell Detection
    // --------------------------------------------------------------------------
    suite("Windows Shell Detection", () => {
        setup(() => {
            Object.defineProperty(process, "platform", { value: "win32" })

        test("uses explicit PowerShell 7 path from VS Code config (profile path)", () => {
            mockVsCodeConfig("windows", "PowerShell", {
                PowerShell: { path: "C:\\Program Files\\PowerShell\\7\\pwsh.exe" },
            assert.strictEqual(getShell(), "C:\\Program Files\\PowerShell\\7\\pwsh.exe")

        test("uses PowerShell 7 path if source is 'PowerShell' but no explicit path", () => {
            mockVsCodeConfig("windows", "PowerShell", {
                PowerShell: { source: "PowerShell" },
            assert.strictEqual(getShell(), "C:\\Program Files\\PowerShell\\7\\pwsh.exe")

        test("falls back to legacy PowerShell if profile includes 'powershell' but no path/source", () => {
            mockVsCodeConfig("windows", "PowerShell", {
                PowerShell: {},
            assert.strictEqual(getShell(), "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe")

        test("uses WSL bash when profile indicates WSL source", () => {
            mockVsCodeConfig("windows", "WSL", {
                WSL: { source: "WSL" },
            assert.strictEqual(getShell(), "/bin/bash")

        test("uses WSL bash when profile name includes 'wsl'", () => {
            mockVsCodeConfig("windows", "Ubuntu WSL", {
                "Ubuntu WSL": {},
            assert.strictEqual(getShell(), "/bin/bash")

        test("defaults to cmd.exe if no special profile is matched", () => {
            mockVsCodeConfig("windows", "CommandPrompt", {
                CommandPrompt: {},
            assert.strictEqual(getShell(), "C:\\Windows\\System32\\cmd.exe")

        test("handles undefined profile gracefully", () => {
            // Mock a case where defaultProfileName exists but the profile doesn't
            mockVsCodeConfig("windows", "NonexistentProfile", {})
            assert.strictEqual(getShell(), "C:\\Windows\\System32\\cmd.exe")

        test("respects userInfo() if no VS Code config is available", () => {
            createConfigMock(undefined)
            mockUserInfo.returns({
                shell: "C:\\Custom\\PowerShell.exe",
                username: "testuser",
                uid: 1000,
                gid: 1000,
                homedir: "/home/testuser"

            assert.strictEqual(getShell(), "C:\\Custom\\PowerShell.exe")

        test("respects an odd COMSPEC if no userInfo shell is available", () => {
            createConfigMock(undefined)
            process.env.COMSPEC = "D:\\CustomCmd\\cmd.exe"

            assert.strictEqual(getShell(), "D:\\CustomCmd\\cmd.exe")

    // --------------------------------------------------------------------------
    // macOS Shell Detection
    // --------------------------------------------------------------------------
    suite("macOS Shell Detection", () => {
        setup(() => {
            Object.defineProperty(process, "platform", { value: "darwin" })

        test("uses VS Code profile path if available", () => {
            mockVsCodeConfig("osx", "MyCustomShell", {
                MyCustomShell: { path: "/usr/local/bin/fish" },
            assert.strictEqual(getShell(), "/usr/local/bin/fish")

        test("falls back to userInfo().shell if no VS Code config is available", () => {
            createConfigMock(undefined)
            mockUserInfo.returns({
                shell: "/opt/homebrew/bin/zsh",
                username: "testuser",
                uid: 1000,
                gid: 1000,
                homedir: "/home/testuser"
            assert.strictEqual(getShell(), "/opt/homebrew/bin/zsh")

        test("falls back to SHELL env var if no userInfo shell is found", () => {
            createConfigMock(undefined)
            process.env.SHELL = "/usr/local/bin/zsh"
            assert.strictEqual(getShell(), "/usr/local/bin/zsh")

        test("falls back to /bin/zsh if no config, userInfo, or env variable is set", () => {
            createConfigMock(undefined)
            assert.strictEqual(getShell(), "/bin/zsh")

    // --------------------------------------------------------------------------
    // Linux Shell Detection
    // --------------------------------------------------------------------------
    suite("Linux Shell Detection", () => {
        setup(() => {
            Object.defineProperty(process, "platform", { value: "linux" })

        test("uses VS Code profile path if available", () => {
            mockVsCodeConfig("linux", "CustomProfile", {
                CustomProfile: { path: "/usr/bin/fish" },
            assert.strictEqual(getShell(), "/usr/bin/fish")

        test("falls back to userInfo().shell if no VS Code config is available", () => {
            createConfigMock(undefined)
            mockUserInfo.returns({
                shell: "/usr/bin/zsh",
                username: "testuser",
                uid: 1000,
                gid: 1000,
                homedir: "/home/testuser"
            assert.strictEqual(getShell(), "/usr/bin/zsh")

        test("falls back to SHELL env var if no userInfo shell is found", () => {
            createConfigMock(undefined)
            process.env.SHELL = "/usr/bin/fish"
            assert.strictEqual(getShell(), "/usr/bin/fish")

        test("falls back to /bin/bash if nothing is set", () => {
            createConfigMock(undefined)
            assert.strictEqual(getShell(), "/bin/bash")

    // --------------------------------------------------------------------------
    // Unknown Platform & Error Handling
    // --------------------------------------------------------------------------
    suite("Unknown Platform / Error Handling", () => {
        test("falls back to /bin/sh for unknown platforms", () => {
            Object.defineProperty(process, "platform", { value: "sunos" })
            createConfigMock(undefined)
            assert.strictEqual(getShell(), "/bin/sh")

        test("handles VS Code config errors gracefully, falling back to userInfo shell if present", () => {
            Object.defineProperty(process, "platform", { value: "linux" })
            mockGetConfiguration.throws(new Error("Configuration error"))
            mockUserInfo.returns({
                shell: "/bin/bash",
                username: "testuser",
                uid: 1000,
                gid: 1000,
                homedir: "/home/testuser"
            assert.strictEqual(getShell(), "/bin/bash")

        test("handles userInfo errors gracefully, falling back to environment variable if present", () => {
            Object.defineProperty(process, "platform", { value: "darwin" })
            createConfigMock(undefined)
            mockUserInfo.throws(new Error("userInfo error"))
            process.env.SHELL = "/bin/zsh"
            assert.strictEqual(getShell(), "/bin/zsh")

        test("falls back fully to default shell paths if everything fails", () => {
            Object.defineProperty(process, "platform", { value: "linux" })
            mockGetConfiguration.throws(new Error("Configuration error"))
            mockUserInfo.throws(new Error("userInfo error"))
            assert.strictEqual(getShell(), "/bin/bash")
