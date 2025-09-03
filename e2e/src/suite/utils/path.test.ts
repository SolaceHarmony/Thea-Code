import * as assert from 'assert'
import * as sinon from 'sinon'
import * as proxyquire from 'proxyquire'
import * as os from "os"
import * as path from "path"
import '../../../../src/utils/path' // Import to get String.prototype.toPosix

suite("Path Utilities", () => {
	let sandbox: sinon.SinonSandbox
	let pathModule: any
	let mockVscode: any
	
	setup(() => {
		sandbox = sinon.createSandbox()
		
		// Create VS Code mock
		mockVscode = {
			window: {
				activeTextEditor: undefined
			},
			workspace: {
				workspaceFolders: undefined,
				getWorkspaceFolder: sandbox.stub()
			}
		}
		
		// Load the path module with mocked VS Code
		pathModule = proxyquire('../../../src/utils/path', {
			'vscode': mockVscode,
			'../shared/formatPath': {
				formatPath: (p: string) => p  // Simple pass-through for testing
			}
		})
		
		// Add toPosix to String prototype (since the module extends it)
		String.prototype.toPosix = function(this: string): string {
			const isExtendedLengthPath = this.startsWith("\\\\?\\")
			if (isExtendedLengthPath) {
				return this
			}
			return this.replace(/\\/g, "/")
		}
	})

	teardown(() => {
		sandbox.restore()
	})

	suite("String.prototype.toPosix", () => {
		test("converts backslashes to forward slashes", () => {
			const windowsPath = "C:\\Users\\test\\file.txt"
			assert.strictEqual(windowsPath.toPosix(), "C:/Users/test/file.txt")
		})

		test("does not modify paths with forward slashes", () => {
			const unixPath = "/home/user/file.txt"
			assert.strictEqual(unixPath.toPosix(), "/home/user/file.txt")
		})

		test("preserves extended-length Windows paths", () => {
			const extendedPath = "\\\\?\\C:\\Very\\Long\\Path"
			assert.strictEqual(extendedPath.toPosix(), "\\\\?\\C:\\Very\\Long\\Path")
		})

		test("handles mixed separators", () => {
			const mixedPath = "C:\\Users/test\\file.txt"
			assert.strictEqual(mixedPath.toPosix(), "C:/Users/test/file.txt")
		})

		test("handles empty string", () => {
			assert.strictEqual("".toPosix(), "")
		})
	})

	suite("getWorkspacePath", () => {
		test("returns the workspace folder path when available", () => {
			mockVscode.workspace.workspaceFolders = [
				{
					uri: { fsPath: "/test/workspace" },
					name: "test",
					index: 0
				}
			]

			const result = pathModule.getWorkspacePath()
			assert.strictEqual(result, "/test/workspace")
		})

		test("returns path from active editor when no workspace folders", () => {
			mockVscode.workspace.workspaceFolders = undefined
			mockVscode.window.activeTextEditor = {
				document: {
					uri: { fsPath: "/test/workspaceFolder/file.ts" }
				}
			}
			mockVscode.workspace.getWorkspaceFolder.returns({
				uri: { fsPath: "/test/workspaceFolder" }
			})

			const result = pathModule.getWorkspacePath()
			assert.strictEqual(result, "/test/workspaceFolder")
		})

		test("returns home directory when no workspace or active editor", () => {
			mockVscode.workspace.workspaceFolders = undefined
			mockVscode.window.activeTextEditor = undefined

			const result = pathModule.getWorkspacePath()
			assert.strictEqual(result, os.homedir())
		})

		test("returns home directory when getWorkspaceFolder returns undefined", () => {
			mockVscode.workspace.workspaceFolders = undefined
			mockVscode.window.activeTextEditor = {
				document: {
					uri: { fsPath: "/test/file.ts" }
				}
			}
			mockVscode.workspace.getWorkspaceFolder.returns(undefined)

			const result = pathModule.getWorkspacePath()
			assert.strictEqual(result, os.homedir())
		})
	})

	suite("arePathsEqual", () => {
		const originalPlatform = process.platform

		teardown(() => {
			Object.defineProperty(process, "platform", {
				value: originalPlatform,
				configurable: true
			})
		})

		test("returns true for identical paths", () => {
			const result = pathModule.arePathsEqual("/test/path", "/test/path")
			assert.strictEqual(result, true)
		})

		test("returns true when both paths are undefined", () => {
			const result = pathModule.arePathsEqual(undefined, undefined)
			assert.strictEqual(result, true)
		})

		test("returns false when one path is undefined", () => {
			assert.strictEqual(pathModule.arePathsEqual("/test/path", undefined), false)
			assert.strictEqual(pathModule.arePathsEqual(undefined, "/test/path"), false)
		})

		test("handles different separators on Windows", () => {
			Object.defineProperty(process, "platform", {
				value: "win32",
				configurable: true
			})

			const result = pathModule.arePathsEqual("C:\\test\\path", "C:/test/path")
			assert.strictEqual(result, true)
		})

		test("normalizes paths before comparison", () => {
			const result = pathModule.arePathsEqual("/test/./path/../path", "/test/path")
			assert.strictEqual(result, true)
		})

		test("is case-insensitive on Windows", () => {
			Object.defineProperty(process, "platform", {
				value: "win32",
				configurable: true
			})

			const result = pathModule.arePathsEqual("C:\\Test\\Path", "c:\\test\\path")
			assert.strictEqual(result, true)
		})

		test("is case-sensitive on Unix", () => {
			Object.defineProperty(process, "platform", {
				value: "linux",
				configurable: true
			})

			const result = pathModule.arePathsEqual("/Test/Path", "/test/path")
			assert.strictEqual(result, false)
		})

		test("handles trailing slashes", () => {
			const result = pathModule.arePathsEqual("/test/path/", "/test/path")
			assert.strictEqual(result, true)
		})

		test("returns false for different paths", () => {
			const result = pathModule.arePathsEqual("/test/path1", "/test/path2")
			assert.strictEqual(result, false)
		})
	})

	suite("getReadablePath", () => {
		test("returns relative path when inside workspace", () => {
			mockVscode.workspace.workspaceFolders = [
				{
					uri: { fsPath: "/test/workspace" },
					name: "test",
					index: 0
				}
			]

			const result = pathModule.getReadablePath("/test/workspace/src/file.ts")
			assert.strictEqual(result, "src/file.ts")
		})

		test("returns absolute path when outside workspace", () => {
			mockVscode.workspace.workspaceFolders = [
				{
					uri: { fsPath: "/test/workspace" },
					name: "test",
					index: 0
				}
			]

			const result = pathModule.getReadablePath("/other/path/file.ts")
			assert.strictEqual(result, "/other/path/file.ts")
		})

		test("returns absolute path when no workspace", () => {
			mockVscode.workspace.workspaceFolders = undefined

			const result = pathModule.getReadablePath("/test/path/file.ts")
			assert.strictEqual(result, "/test/path/file.ts")
		})

		test("converts backslashes to forward slashes", () => {
			mockVscode.workspace.workspaceFolders = [
				{
					uri: { fsPath: "C:\\test\\workspace" },
					name: "test",
					index: 0
				}
			]

			const result = pathModule.getReadablePath("C:\\test\\workspace\\src\\file.ts")
			assert.strictEqual(result, "src/file.ts")
		})

		test("handles paths at workspace root", () => {
			mockVscode.workspace.workspaceFolders = [
				{
					uri: { fsPath: "/test/workspace" },
					name: "test",
					index: 0
				}
			]

			const result = pathModule.getReadablePath("/test/workspace/file.ts")
			assert.strictEqual(result, "file.ts")
		})

		test("returns dot for workspace root path", () => {
			mockVscode.workspace.workspaceFolders = [
				{
					uri: { fsPath: "/test/workspace" },
					name: "test",
					index: 0
				}
			]

			const result = pathModule.getReadablePath("/test/workspace")
			assert.strictEqual(result, ".")
		})

		test("handles normalized paths", () => {
			mockVscode.workspace.workspaceFolders = [
				{
					uri: { fsPath: "/test/workspace" },
					name: "test",
					index: 0
				}
			]

			const result = pathModule.getReadablePath("/test/workspace/./src/../src/file.ts")
			assert.strictEqual(result, "src/file.ts")
		})
	})
// Mock cleanup
