import * as assert from 'assert'
import * as sinon from 'sinon'
import * as proxyquire from 'proxyquire'
import * as path from 'path'

/**
 * Tests for applyDiffTool - validates Thea's file modification capabilities
 * This tests diff application, error handling, file validation, and user approval flows
 */

suite("applyDiffTool", () => {
	let sandbox: sinon.SinonSandbox
	let applyDiffTool: any
	let mockTheaTask: any
	let mockAskApproval: sinon.SinonStub
	let mockHandleError: sinon.SinonStub
	let mockPushToolResult: sinon.SinonStub
	let mockRemoveClosingTag: sinon.SinonStub
	
	// File system mocks
	let fileExistsStub: sinon.SinonStub
	let fsStub: any
	
	// Path utility mocks
	let getReadablePathStub: sinon.SinonStub

	setup(() => {
		sandbox = sinon.createSandbox()
		
		// Setup file system mocks
		fileExistsStub = sandbox.stub()
		fsStub = {
			readFile: sandbox.stub(),
			writeFile: sandbox.stub(),
			mkdir: sandbox.stub(),
			unlink: sandbox.stub()
		}
		
		// Setup path mocks
		getReadablePathStub = sandbox.stub().callsFake((cwd: string, relPath: string) => {
			return relPath // Simple implementation for testing
		})
		
		// Setup TheaTask mock
		mockTheaTask = {
			cwd: "/test/workspace",
			consecutiveMistakeCount: 0,
			consecutiveMistakeCountForApplyDiff: new Map(),
			webviewCommunicator: {
				ask: sandbox.stub().resolves(),
				say: sandbox.stub().resolves()
			},
			diffViewProvider: {
				open: sandbox.stub(),
				update: sandbox.stub(),
				scrollToFirstDiff: sandbox.stub(),
				revertChanges: sandbox.stub(),
				saveChanges: sandbox.stub().resolves({
					newProblemsMessage: "",
					userEdits: undefined,
					finalContent: "Modified content"
				}),
				reset: sandbox.stub()
			},
			diffStrategy: {
				applyDiff: sandbox.stub().resolves({
					success: true,
					content: "Modified content"
				}),
				getProgressStatus: sandbox.stub().returns("in_progress")
			},
			theaIgnoreController: {
				validateAccess: sandbox.stub().returns({ allowed: true })
			},
			sayAndCreateMissingParamError: sandbox.stub().resolves("Missing parameter error"),
			didEditFile: false
		}
		
		// Setup tool operation mocks
		mockAskApproval = sandbox.stub().resolves(true)
		mockHandleError = sandbox.stub().resolves()
		mockPushToolResult = sandbox.stub()
		mockRemoveClosingTag = sandbox.stub().callsFake((tag: string, content: string) => content)
		
		// Load applyDiffTool with mocked dependencies
		const module = proxyquire('../../../src/core/tools/applyDiffTool', {
			'fs/promises': fsStub,
			'../../utils/fs': { fileExistsAtPath: fileExistsStub },
			'../../utils/path': { getReadablePath: getReadablePathStub }
		})
		
		applyDiffTool = module.applyDiffTool
	})
	
	teardown(() => {
		sandbox.restore()
	})
	
	suite("parameter validation", () => {
		test("handles missing path parameter", async () => {
			const block = {
				type: "tool_use",
				name: "apply_diff",
				params: { diff: "--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+new" },
				partial: false
			}
			
			await applyDiffTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag
			)
			
			assert.ok(mockTheaTask.sayAndCreateMissingParamError.calledWith("apply_diff", "path"))
			assert.ok(mockPushToolResult.calledWith("Missing parameter error"))
			assert.strictEqual(mockTheaTask.consecutiveMistakeCount, 1)
			assert.ok(!mockAskApproval.called)
		})
		
		test("handles missing diff parameter", async () => {
			const block = {
				type: "tool_use",
				name: "apply_diff",
				params: { path: "file.txt" },
				partial: false
			}
			
			await applyDiffTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag
			)
			
			assert.ok(mockTheaTask.sayAndCreateMissingParamError.calledWith("apply_diff", "diff"))
			assert.ok(mockPushToolResult.calledWith("Missing parameter error"))
			assert.strictEqual(mockTheaTask.consecutiveMistakeCount, 1)
			assert.ok(!mockAskApproval.called)
		})
	})
	
	suite("file validation", () => {
		test("handles non-existent files", async () => {
			fileExistsStub.resolves(false)
			
			const block = {
				type: "tool_use",
				name: "apply_diff",
				params: {
					path: "nonexistent.txt",
					diff: "--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+new"
				},
				partial: false
			}
			
			await applyDiffTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag
			)
			
			assert.ok(fileExistsStub.called)
			assert.ok(mockTheaTask.webviewCommunicator.say.called)
			assert.ok(mockPushToolResult.called)
			assert.strictEqual(mockTheaTask.consecutiveMistakeCount, 1)
			// Ensure no attempt to read file or apply diff
			assert.ok(!fsStub.readFile.called)
			assert.ok(!mockTheaTask.diffStrategy.applyDiff.called)
		})
		
		test("validates file access permissions", async () => {
			fileExistsStub.resolves(true)
			mockTheaTask.theaIgnoreController.validateAccess.returns({
				allowed: false,
				reason: "File is in ignored directory"
			})
			
			const block = {
				type: "tool_use",
				name: "apply_diff",
				params: {
					path: "ignored/file.txt",
					diff: "--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+new"
				},
				partial: false
			}
			
			await applyDiffTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag
			)
			
			assert.ok(mockTheaTask.theaIgnoreController.validateAccess.called)
			assert.ok(mockPushToolResult.called)
			assert.ok(!mockTheaTask.diffStrategy.applyDiff.called)
		})
	})
	
	suite("partial updates", () => {
		test("handles partial block updates", async () => {
			const block = {
				type: "tool_use",
				name: "apply_diff",
				params: { path: "file.txt", diff: "partial diff content" },
				partial: true
			}
			
			await applyDiffTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag
			)
			
			assert.ok(mockTheaTask.webviewCommunicator.ask.called)
			assert.ok(!mockAskApproval.called)
			assert.ok(!mockTheaTask.diffViewProvider.open.called)
			// Partial blocks should update UI but not apply changes
			assert.ok(!mockTheaTask.diffStrategy.applyDiff.called)
		})
		
		test("includes progress status in partial updates", async () => {
			mockTheaTask.diffStrategy.getProgressStatus.returns("applying_hunks")
			
			const block = {
				type: "tool_use",
				name: "apply_diff",
				params: { path: "file.txt", diff: "partial" },
				partial: true
			}
			
			await applyDiffTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag
			)
			
			assert.ok(mockTheaTask.diffStrategy.getProgressStatus.called)
			const askCall = mockTheaTask.webviewCommunicator.ask.firstCall
			assert.ok(askCall)
			assert.strictEqual(askCall.args[3], "applying_hunks")
		})
	})
	
	suite("diff application", () => {
		test("successfully applies diff to existing file", async () => {
			fileExistsStub.resolves(true)
			fsStub.readFile.resolves("Original content")
			mockAskApproval.resolves(true)
			
			const block = {
				type: "tool_use",
				name: "apply_diff",
				params: {
					path: "src/file.txt",
					diff: "--- a/src/file.txt\n+++ b/src/file.txt\n@@ -1 +1 @@\n-Original content\n+Modified content"
				},
				partial: false
			}
			
			await applyDiffTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag
			)
			
			assert.ok(fileExistsStub.called)
			assert.ok(fsStub.readFile.called)
			assert.ok(mockAskApproval.called)
			assert.ok(mockTheaTask.diffStrategy.applyDiff.called)
			assert.ok(mockTheaTask.diffViewProvider.open.called)
			assert.strictEqual(mockTheaTask.didEditFile, true)
			assert.strictEqual(mockTheaTask.consecutiveMistakeCount, 0)
		})
		
		test("handles diff application failure", async () => {
			fileExistsStub.resolves(true)
			fsStub.readFile.resolves("Original content")
			mockAskApproval.resolves(true)
			mockTheaTask.diffStrategy.applyDiff.resolves({
				success: false,
				error: "Failed to apply hunk"
			})
			
			const block = {
				type: "tool_use",
				name: "apply_diff",
				params: {
					path: "src/file.txt",
					diff: "invalid diff"
				},
				partial: false
			}
			
			await applyDiffTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag
			)
			
			assert.ok(mockTheaTask.diffStrategy.applyDiff.called)
			assert.ok(mockPushToolResult.called)
			// Should increment mistake count on failure
			const mistakeCount = mockTheaTask.consecutiveMistakeCountForApplyDiff.get("src/file.txt")
			assert.ok(mistakeCount > 0)
		})
	})
	
	suite("user approval", () => {
		test("requests approval before applying diff", async () => {
			fileExistsStub.resolves(true)
			fsStub.readFile.resolves("Original content")
			mockAskApproval.resolves(true)
			
			const block = {
				type: "tool_use",
				name: "apply_diff",
				params: {
					path: "file.txt",
					diff: "--- a/file.txt\n+++ b/file.txt"
				},
				partial: false
			}
			
			await applyDiffTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag
			)
			
			assert.ok(mockAskApproval.called)
			assert.ok(mockAskApproval.calledBefore(mockTheaTask.diffStrategy.applyDiff))
		})
		
		test("handles approval rejection", async () => {
			fileExistsStub.resolves(true)
			fsStub.readFile.resolves("Original content")
			mockAskApproval.resolves(false)
			
			const block = {
				type: "tool_use",
				name: "apply_diff",
				params: {
					path: "file.txt",
					diff: "--- a/file.txt\n+++ b/file.txt"
				},
				partial: false
			}
			
			await applyDiffTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag
			)
			
			assert.ok(mockAskApproval.called)
			assert.ok(!mockTheaTask.diffStrategy.applyDiff.called)
			assert.ok(mockPushToolResult.called)
			// Should report rejection
			const resultCall = mockPushToolResult.firstCall
			assert.ok(resultCall.args[0].includes("rejected") || resultCall.args[0].includes("denied"))
		})
		
		test("handles approval with user edits", async () => {
			fileExistsStub.resolves(true)
			fsStub.readFile.resolves("Original content")
			mockAskApproval.resolves(true)
			
			// Simulate user making edits in diff view
			mockTheaTask.diffViewProvider.saveChanges.resolves({
				newProblemsMessage: "",
				userEdits: "User edited content",
				finalContent: "User edited content"
			})
			
			const block = {
				type: "tool_use",
				name: "apply_diff",
				params: {
					path: "file.txt",
					diff: "--- a/file.txt\n+++ b/file.txt"
				},
				partial: false
			}
			
			await applyDiffTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag
			)
			
			assert.ok(mockTheaTask.diffViewProvider.saveChanges.called)
			assert.ok(mockPushToolResult.called)
			// Should indicate user edits were applied
			const resultCall = mockPushToolResult.firstCall
			assert.ok(resultCall.args[0].includes("edit") || resultCall.args[0].includes("modified"))
		})
	})
	
	suite("consecutive mistake tracking", () => {
		test("tracks consecutive mistakes per file", async () => {
			fileExistsStub.resolves(true)
			fsStub.readFile.resolves("Original content")
			mockAskApproval.resolves(true)
			mockTheaTask.diffStrategy.applyDiff.resolves({
				success: false,
				error: "Failed to apply"
			})
			
			const block = {
				type: "tool_use",
				name: "apply_diff",
				params: {
					path: "problem.txt",
					diff: "bad diff"
				},
				partial: false
			}
			
			// First attempt
			await applyDiffTool(mockTheaTask, block, mockAskApproval, mockHandleError, mockPushToolResult, mockRemoveClosingTag)
			assert.strictEqual(mockTheaTask.consecutiveMistakeCountForApplyDiff.get("problem.txt"), 1)
			
			// Second attempt
			await applyDiffTool(mockTheaTask, block, mockAskApproval, mockHandleError, mockPushToolResult, mockRemoveClosingTag)
			assert.strictEqual(mockTheaTask.consecutiveMistakeCountForApplyDiff.get("problem.txt"), 2)
		})
		
		test("resets mistake count on success", async () => {
			fileExistsStub.resolves(true)
			fsStub.readFile.resolves("Original content")
			mockAskApproval.resolves(true)
			
			// Set initial mistake count
			mockTheaTask.consecutiveMistakeCountForApplyDiff.set("file.txt", 3)
			
			const block = {
				type: "tool_use",
				name: "apply_diff",
				params: {
					path: "file.txt",
					diff: "--- a/file.txt\n+++ b/file.txt"
				},
				partial: false
			}
			
			await applyDiffTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag
			)
			
			// Should reset on success
			assert.strictEqual(mockTheaTask.consecutiveMistakeCountForApplyDiff.get("file.txt"), 0)
		})
	})
	
	suite("diff view integration", () => {
		test("opens diff view for user review", async () => {
			fileExistsStub.resolves(true)
			fsStub.readFile.resolves("Original content")
			mockAskApproval.resolves(true)
			
			const block = {
				type: "tool_use",
				name: "apply_diff",
				params: {
					path: "file.txt",
					diff: "--- a/file.txt\n+++ b/file.txt"
				},
				partial: false
			}
			
			await applyDiffTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag
			)
			
			assert.ok(mockTheaTask.diffViewProvider.open.called)
			assert.ok(mockTheaTask.diffViewProvider.scrollToFirstDiff.called)
		})
		
		test("updates diff view on subsequent changes", async () => {
			fileExistsStub.resolves(true)
			fsStub.readFile.resolves("Original content")
			mockAskApproval.resolves(true)
			
			// Mark diff view as already open
			mockTheaTask.diffViewProvider.isOpen = true
			
			const block = {
				type: "tool_use",
				name: "apply_diff",
				params: {
					path: "file.txt",
					diff: "updated diff"
				},
				partial: false
			}
			
			await applyDiffTool(
				mockTheaTask,
				block,
				mockAskApproval,
				mockHandleError,
				mockPushToolResult,
				mockRemoveClosingTag
			)
			
			assert.ok(mockTheaTask.diffViewProvider.update.called)
		})
	})
// Mock cleanup