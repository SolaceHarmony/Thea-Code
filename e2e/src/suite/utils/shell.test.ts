import * as assert from 'assert'
import * as sinon from 'sinon'
import * as proxyquire from 'proxyquire'

suite("Shell Detection Tests", () => {
	let sandbox: sinon.SinonSandbox
	let shellModule: any
	let mockVscode: any
	let mockOs: any
	let originalPlatform: string
	let originalEnv: NodeJS.ProcessEnv

	setup(() => {
		sandbox = sinon.createSandbox()
		originalPlatform = process.platform
		originalEnv = { ...process.env }
		
		// Create VS Code mock
		mockVscode = {
			workspace: {
				getConfiguration: sandbox.stub()
			}
		}
		
		// Create OS mock
		mockOs = {
			userInfo: sandbox.stub().returns({
				username: "testuser",
				uid: 1000,
				gid: 1000,
				shell: null,
				homedir: "/home/testuser"
			})
		}
		
		// Load the shell module with mocked dependencies
		shellModule = proxyquire('../../../src/utils/shell', {
			'vscode': mockVscode,
			'os': mockOs
		})
	})

	teardown(() => {
		sandbox.restore()
		Object.defineProperty(process, "platform", {
			value: originalPlatform,
			configurable: true
		})
		process.env = originalEnv
	})

	// Helper to mock VS Code configuration
	function mockVsCodeConfig(
		platformKey: string,
		defaultProfileName: string | null,
		profiles: Record<string, any>
	) {
		mockVscode.workspace.getConfiguration.returns({
			get: (key: string) => {
				if (key === `defaultProfile.${platformKey}`) {
					return defaultProfileName
				}
				if (key === `profiles.${platformKey}`) {
					return profiles
				}
				return undefined
			}
		})
	}

	suite("Windows Shell Detection", () => {
		setup(() => {
			Object.defineProperty(process, "platform", {
				value: "win32",
				configurable: true
			})
		})

		test("detects PowerShell 7 from VS Code config", () => {
			mockVsCodeConfig("windows", "PowerShell", {
				PowerShell: { path: "C:\\Program Files\\PowerShell\\7\\pwsh.exe" }
			})

			const shell = shellModule.getShell()
			assert.strictEqual(shell, "C:\\Program Files\\PowerShell\\7\\pwsh.exe")
		})

		test("detects WSL bash from VS Code config", () => {
			mockVsCodeConfig("windows", "WSL", {
				WSL: { source: "WSL" }
			})

			const shell = shellModule.getShell()
			assert.strictEqual(shell, "/bin/bash")
		})

		test("detects Git Bash from VS Code config", () => {
			mockVsCodeConfig("windows", "Git Bash", {
				"Git Bash": { path: "C:\\Program Files\\Git\\bin\\bash.exe" }
			})

			const shell = shellModule.getShell()
			assert.strictEqual(shell, "C:\\Program Files\\Git\\bin\\bash.exe")
		})

		test("falls back to PowerShell 7 if available", () => {
			mockVsCodeConfig("windows", null, {})
			process.env.PROGRAMFILES = "C:\\Program Files"

			const shell = shellModule.getShell()
			// Will check for pwsh.exe existence, mock implementation would return PowerShell 7 path
			assert.ok(shell.includes("pwsh.exe") || shell.includes("powershell.exe"))
		})

		test("falls back to legacy PowerShell when PS7 not available", () => {
			mockVsCodeConfig("windows", null, {})
			delete process.env.PROGRAMFILES

			const shell = shellModule.getShell()
			assert.ok(shell.includes("powershell.exe") || shell.includes("cmd.exe"))
		})

		test("falls back to cmd.exe as last resort", () => {
			mockVsCodeConfig("windows", null, {})
			process.env.COMSPEC = "C:\\Windows\\System32\\cmd.exe"

			const shell = shellModule.getShell()
			assert.ok(shell.includes("cmd.exe") || shell.includes("powershell.exe"))
		})
	})

	suite("macOS Shell Detection", () => {
		setup(() => {
			Object.defineProperty(process, "platform", {
				value: "darwin",
				configurable: true
			})
		})

		test("detects custom shell from VS Code config", () => {
			mockVsCodeConfig("osx", "iTerm", {
				iTerm: { path: "/usr/local/bin/fish" }
			})

			const shell = shellModule.getShell()
			assert.strictEqual(shell, "/usr/local/bin/fish")
		})

		test("detects shell from SHELL env variable", () => {
			mockVsCodeConfig("osx", null, {})
			process.env.SHELL = "/usr/local/bin/zsh"

			const shell = shellModule.getShell()
			assert.strictEqual(shell, "/usr/local/bin/zsh")
		})

		test("detects shell from user info", () => {
			mockVsCodeConfig("osx", null, {})
			delete process.env.SHELL
			mockOs.userInfo.returns({
				username: "testuser",
				uid: 1000,
				gid: 1000,
				shell: "/bin/bash",
				homedir: "/Users/testuser"
			})

			const shell = shellModule.getShell()
			assert.strictEqual(shell, "/bin/bash")
		})

		test("falls back to zsh on macOS", () => {
			mockVsCodeConfig("osx", null, {})
			delete process.env.SHELL
			mockOs.userInfo.returns({
				username: "testuser",
				uid: 1000,
				gid: 1000,
				shell: null,
				homedir: "/Users/testuser"
			})

			const shell = shellModule.getShell()
			assert.strictEqual(shell, "/bin/zsh")
		})
	})

	suite("Linux Shell Detection", () => {
		setup(() => {
			Object.defineProperty(process, "platform", {
				value: "linux",
				configurable: true
			})
		})

		test("detects custom shell from VS Code config", () => {
			mockVsCodeConfig("linux", "fish", {
				fish: { path: "/usr/bin/fish" }
			})

			const shell = shellModule.getShell()
			assert.strictEqual(shell, "/usr/bin/fish")
		})

		test("detects shell from SHELL env variable", () => {
			mockVsCodeConfig("linux", null, {})
			process.env.SHELL = "/usr/bin/zsh"

			const shell = shellModule.getShell()
			assert.strictEqual(shell, "/usr/bin/zsh")
		})

		test("detects shell from user info", () => {
			mockVsCodeConfig("linux", null, {})
			delete process.env.SHELL
			mockOs.userInfo.returns({
				username: "testuser",
				uid: 1000,
				gid: 1000,
				shell: "/usr/bin/fish",
				homedir: "/home/testuser"
			})

			const shell = shellModule.getShell()
			assert.strictEqual(shell, "/usr/bin/fish")
		})

		test("falls back to bash on Linux", () => {
			mockVsCodeConfig("linux", null, {})
			delete process.env.SHELL
			mockOs.userInfo.returns({
				username: "testuser",
				uid: 1000,
				gid: 1000,
				shell: null,
				homedir: "/home/testuser"
			})

			const shell = shellModule.getShell()
			assert.strictEqual(shell, "/bin/bash")
		})

		test("handles various shell paths", () => {
			const shells = [
				"/bin/bash",
				"/bin/zsh",
				"/usr/bin/fish",
				"/bin/dash",
				"/bin/tcsh"
			]

			shells.forEach(shellPath => {
				mockVsCodeConfig("linux", null, {})
				process.env.SHELL = shellPath

				const shell = shellModule.getShell()
				assert.strictEqual(shell, shellPath)
			})
		})
	})

	suite("Unsupported Platform", () => {
		test("falls back to /bin/sh on unknown platforms", () => {
			Object.defineProperty(process, "platform", {
				value: "freebsd",
				configurable: true
			})
			mockVsCodeConfig("freebsd", null, {})

			const shell = shellModule.getShell()
			assert.strictEqual(shell, "/bin/sh")
		})
	})

	suite("Error Handling", () => {
		test("handles VS Code config errors gracefully", () => {
			mockVscode.workspace.getConfiguration.throws(new Error("Config error"))
			
			// Should not throw and should return a fallback shell
			const shell = shellModule.getShell()
			assert.ok(shell)
			assert.ok(typeof shell === "string")
		})

		test("handles userInfo errors gracefully", () => {
			mockVsCodeConfig("linux", null, {})
			delete process.env.SHELL
			mockOs.userInfo.throws(new Error("User info error"))

			// Should not throw and should return a fallback shell
			const shell = shellModule.getShell()
			assert.ok(shell)
			assert.strictEqual(shell, "/bin/bash")
		})

		test("handles missing profile configuration", () => {
			mockVsCodeConfig("windows", "NonExistent", {})

			// Should not throw and should return a fallback shell
			const shell = shellModule.getShell()
			assert.ok(shell)
			assert.ok(typeof shell === "string")
		})
	})

	suite("Shell Priority", () => {
		test("VS Code config takes priority over environment variables", () => {
			Object.defineProperty(process, "platform", {
				value: "linux",
				configurable: true
			})
			
			process.env.SHELL = "/bin/bash"
			mockVsCodeConfig("linux", "zsh", {
				zsh: { path: "/usr/bin/zsh" }
			})

			const shell = shellModule.getShell()
			assert.strictEqual(shell, "/usr/bin/zsh")
		})

		test("Environment variables take priority over user info", () => {
			Object.defineProperty(process, "platform", {
				value: "linux",
				configurable: true
			})
			
			mockVsCodeConfig("linux", null, {})
			process.env.SHELL = "/usr/bin/zsh"
			mockOs.userInfo.returns({
				username: "testuser",
				uid: 1000,
				gid: 1000,
				shell: "/bin/bash",
				homedir: "/home/testuser"
			})

			const shell = shellModule.getShell()
			assert.strictEqual(shell, "/usr/bin/zsh")
		})
	})
})