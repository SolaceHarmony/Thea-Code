import * as assert from 'assert'
import * as sinon from 'sinon'
import fs from "fs/promises"
import path from "path"
import os from "os"
import { EventEmitter } from "events"

import { simpleGit, SimpleGit } from "simple-git"
import { EXTENSION_DISPLAY_NAME, AUTHOR_EMAIL } from "../../../shared/config/thea-config"
import { fileExistsAtPath } from "../../../utils/fs"

import { ShadowCheckpointService } from "../ShadowCheckpointService"
import { RepoPerTaskCheckpointService } from "../RepoPerTaskCheckpointService"
import { RepoPerWorkspaceCheckpointService } from "../RepoPerWorkspaceCheckpointService"
import type { CheckpointEventMap } from "../types"
import { globby } from "globby"

// TODO: Use proxyquire for module mocking - "globby", () => ({
	globby: sinon.stub().resolves([]),
// Mock cleanup needed

const tmpDir = path.join(os.tmpdir(), "CheckpointService")

const initWorkspaceRepo = async ({
	workspaceDir,
	userName = EXTENSION_DISPLAY_NAME as string,
	userEmail = AUTHOR_EMAIL as string,
	testFileName = "test.txt",
	textFileContent = "Hello, world!",
}: {
	workspaceDir: string
	userName?: string
	userEmail?: string
	testFileName?: string
	textFileContent?: string
}) => {
	// Create a temporary directory for testing.
	await fs.mkdir(workspaceDir, { recursive: true })

	// Initialize git repo.
	const git = simpleGit(workspaceDir)
	await git.init()
	await git.addConfig("user.name", userName)
	await git.addConfig("user.email", userEmail)

	// Create test file.
	const testFile = path.join(workspaceDir, testFileName)
	await fs.writeFile(testFile, textFileContent)

	// Create initial commit.
	await git.add(".")
	await git.commit("Initial commit")
// Mock removed - needs manual implementation
// 
describe.each([
	[RepoPerTaskCheckpointService, "RepoPerTaskCheckpointService"],
	[RepoPerWorkspaceCheckpointService, "RepoPerWorkspaceCheckpointService"],
])("CheckpointService", (klass, prefix) => {
	const taskId = "test-task"

	let workspaceGit: SimpleGit
	let testFile: string
	let service: RepoPerTaskCheckpointService | RepoPerWorkspaceCheckpointService

	setup(async () => {
		globby.resetHistory().resolves([])

		const shadowDir = path.join(tmpDir, `${prefix}-${Date.now()}`)
		const workspaceDir = path.join(tmpDir, `workspace-${Date.now()}`)
		const repo = await initWorkspaceRepo({ workspaceDir })

		workspaceGit = repo.git
		testFile = repo.testFile

		service = klass.create({ taskId, shadowDir, workspaceDir, log: () => {} })
		await service.initShadowGit()
	})

	teardown(() => {
		sinon.restore()
	})

	suiteTeardown(async () => {
		await fs.rm(tmpDir, { recursive: true, force: true })
	})

	suite(`${klass.name}#getDiff`, () => {
		test("returns the correct diff between commits", async () => {
			await fs.writeFile(testFile, "Ahoy, world!")
			const commit1 = await service.saveCheckpoint("Ahoy, world!")
			assert.ok(commit1?.commit)

			await fs.writeFile(testFile, "Goodbye, world!")
			const commit2 = await service.saveCheckpoint("Goodbye, world!")
			assert.ok(commit2?.commit)

			const diff1 = await service.getDiff({ to: commit1!.commit })
			assert.strictEqual(diff1.length, 1)
			assert.strictEqual(diff1[0].paths.relative, "test.txt")
			assert.strictEqual(diff1[0].paths.absolute, testFile)
			assert.strictEqual(diff1[0].content.before, "Hello, world!")
			assert.strictEqual(diff1[0].content.after, "Ahoy, world!")

			const diff2 = await service.getDiff({ from: service.baseHash, to: commit2!.commit })
			assert.strictEqual(diff2.length, 1)
			assert.strictEqual(diff2[0].paths.relative, "test.txt")
			assert.strictEqual(diff2[0].paths.absolute, testFile)
			assert.strictEqual(diff2[0].content.before, "Hello, world!")
			assert.strictEqual(diff2[0].content.after, "Goodbye, world!")

			const diff12 = await service.getDiff({ from: commit1!.commit, to: commit2!.commit })
			assert.strictEqual(diff12.length, 1)
			assert.strictEqual(diff12[0].paths.relative, "test.txt")
			assert.strictEqual(diff12[0].paths.absolute, testFile)
			assert.strictEqual(diff12[0].content.before, "Ahoy, world!")
			assert.strictEqual(diff12[0].content.after, "Goodbye, world!")
		})

		test("handles new files in diff", async () => {
			const newFile = path.join(service.workspaceDir, "new.txt")
			await fs.writeFile(newFile, "New file content")
			const commit = await service.saveCheckpoint("Add new file")
			assert.ok(commit?.commit)

			const changes = await service.getDiff({ to: commit!.commit })
			const change = changes.find((c) => c.paths.relative === "new.txt")
			assert.notStrictEqual(change, undefined)
			assert.strictEqual(change?.content.before, "")
			assert.strictEqual(change?.content.after, "New file content")
		})

		test("handles deleted files in diff", async () => {
			const fileToDelete = path.join(service.workspaceDir, "new.txt")
			await fs.writeFile(fileToDelete, "New file content")
			const commit1 = await service.saveCheckpoint("Add file")
			assert.ok(commit1?.commit)

			await fs.unlink(fileToDelete)
			const commit2 = await service.saveCheckpoint("Delete file")
			assert.ok(commit2?.commit)

			const changes = await service.getDiff({ from: commit1!.commit, to: commit2!.commit })
			const change = changes.find((c) => c.paths.relative === "new.txt")
			assert.notStrictEqual(change, undefined)
			assert.strictEqual(change!.content.before, "New file content")
			assert.strictEqual(change!.content.after, "")
		})
	})

	suite(`${klass.name}#saveCheckpoint`, () => {
		test("creates a checkpoint if there are pending changes", async () => {
			await fs.writeFile(testFile, "Ahoy, world!")
			const commit1 = await service.saveCheckpoint("First checkpoint")
			assert.ok(commit1?.commit)
			const details1 = await service.getDiff({ to: commit1!.commit })
			assert.ok(details1[0].content.before.includes("Hello, world!"))
			assert.ok(details1[0].content.after.includes("Ahoy, world!"))

			await fs.writeFile(testFile, "Hola, world!")
			const commit2 = await service.saveCheckpoint("Second checkpoint")
			assert.ok(commit2?.commit)
			const details2 = await service.getDiff({ from: commit1!.commit, to: commit2!.commit })
			assert.ok(details2[0].content.before.includes("Ahoy, world!"))
			assert.ok(details2[0].content.after.includes("Hola, world!"))

			// Switch to checkpoint 1.
			await service.restoreCheckpoint(commit1!.commit)
			expect(await fs.readFile(testFile, "utf-8")).toBe("Ahoy, world!")

			// Switch to checkpoint 2.
			await service.restoreCheckpoint(commit2!.commit)
			expect(await fs.readFile(testFile, "utf-8")).toBe("Hola, world!")

			// Switch back to initial commit.
			assert.ok(service.baseHash)
			await service.restoreCheckpoint(service.baseHash!)
			expect(await fs.readFile(testFile, "utf-8")).toBe("Hello, world!")
		})

		test("preserves workspace and index state after saving checkpoint", async () => {
			// Create three files with different states: staged, unstaged, and mixed.
			const unstagedFile = path.join(service.workspaceDir, "unstaged.txt")
			const stagedFile = path.join(service.workspaceDir, "staged.txt")
			const mixedFile = path.join(service.workspaceDir, "mixed.txt")

			await fs.writeFile(unstagedFile, "Initial unstaged")
			await fs.writeFile(stagedFile, "Initial staged")
			await fs.writeFile(mixedFile, "Initial mixed")
			await workspaceGit.add(["."])
			const result = await workspaceGit.commit("Add initial files")
			assert.ok(result?.commit)

			await fs.writeFile(unstagedFile, "Modified unstaged")

			await fs.writeFile(stagedFile, "Modified staged")
			await workspaceGit.add([stagedFile])

			await fs.writeFile(mixedFile, "Modified mixed - staged")
			await workspaceGit.add([mixedFile])
			await fs.writeFile(mixedFile, "Modified mixed - unstaged")

			// Save checkpoint.
			const commit = await service.saveCheckpoint("Test checkpoint")
			assert.ok(commit?.commit)

			// Verify workspace state is preserved.
			const status = await workspaceGit.status()

			// All files should be modified.
			assert.ok(status.modified.includes("unstaged.txt"))
			assert.ok(status.modified.includes("staged.txt"))
			assert.ok(status.modified.includes("mixed.txt"))

			// Only staged and mixed files should be staged.
			assert.ok(!status.staged.includes("unstaged.txt"))
			assert.ok(status.staged.includes("staged.txt"))
			assert.ok(status.staged.includes("mixed.txt"))

			// Verify file contents.
			expect(await fs.readFile(unstagedFile, "utf-8")).toBe("Modified unstaged")
			expect(await fs.readFile(stagedFile, "utf-8")).toBe("Modified staged")
			expect(await fs.readFile(mixedFile, "utf-8")).toBe("Modified mixed - unstaged")

			// Verify staged changes (--cached shows only staged changes).
			const stagedDiff = await workspaceGit.diff(["--cached", "mixed.txt"])
			assert.ok(stagedDiff.includes("-Initial mixed"))
			assert.ok(stagedDiff.includes("+Modified mixed - staged"))

			// Verify unstaged changes (shows working directory changes).
			const unstagedDiff = await workspaceGit.diff(["mixed.txt"])
			assert.ok(unstagedDiff.includes("-Modified mixed - staged"))
			assert.ok(unstagedDiff.includes("+Modified mixed - unstaged"))
		})

		test("does not create a checkpoint if there are no pending changes", async () => {
			const commit0 = await service.saveCheckpoint("Zeroth checkpoint")
			assert.ok(!commit0?.commit)

			await fs.writeFile(testFile, "Ahoy, world!")
			const commit1 = await service.saveCheckpoint("First checkpoint")
			assert.ok(commit1?.commit)

			const commit2 = await service.saveCheckpoint("Second checkpoint")
			assert.ok(!commit2?.commit)
		})

		test("includes untracked files in checkpoints", async () => {
			// Create an untracked file.
			const untrackedFile = path.join(service.workspaceDir, "untracked.txt")
			await fs.writeFile(untrackedFile, "I am untracked!")

			// Save a checkpoint with the untracked file.
			const commit1 = await service.saveCheckpoint("Checkpoint with untracked file")
			assert.ok(commit1?.commit)

			// Verify the untracked file was included in the checkpoint.
			const details = await service.getDiff({ to: commit1!.commit })
			assert.ok(details[0].content.before.includes(""))
			assert.ok(details[0].content.after.includes("I am untracked!"))

			// Create another checkpoint with a different state.
			await fs.writeFile(testFile, "Changed tracked file")
			const commit2 = await service.saveCheckpoint("Second checkpoint")
			assert.ok(commit2?.commit)

			// Restore first checkpoint and verify untracked file is preserved.
			await service.restoreCheckpoint(commit1!.commit)
			expect(await fs.readFile(untrackedFile, "utf-8")).toBe("I am untracked!")
			expect(await fs.readFile(testFile, "utf-8")).toBe("Hello, world!")

			// Restore second checkpoint and verify untracked file remains (since
			// restore preserves untracked files)
			await service.restoreCheckpoint(commit2!.commit)
			expect(await fs.readFile(untrackedFile, "utf-8")).toBe("I am untracked!")
			expect(await fs.readFile(testFile, "utf-8")).toBe("Changed tracked file")
		})

		test("handles file deletions correctly", async () => {
			await fs.writeFile(testFile, "I am tracked!")
			const untrackedFile = path.join(service.workspaceDir, "new.txt")
			await fs.writeFile(untrackedFile, "I am untracked!")
			const commit1 = await service.saveCheckpoint("First checkpoint")
			assert.ok(commit1?.commit)

			await fs.unlink(testFile)
			await fs.unlink(untrackedFile)
			const commit2 = await service.saveCheckpoint("Second checkpoint")
			assert.ok(commit2?.commit)

			// Verify files are gone.
			await expect(fs.readFile(testFile, "utf-8")).rejects.toThrow()
			await expect(fs.readFile(untrackedFile, "utf-8")).rejects.toThrow()

			// Restore first checkpoint.
			await service.restoreCheckpoint(commit1!.commit)
			expect(await fs.readFile(testFile, "utf-8")).toBe("I am tracked!")
			expect(await fs.readFile(untrackedFile, "utf-8")).toBe("I am untracked!")

			// Restore second checkpoint.
			await service.restoreCheckpoint(commit2!.commit)
			await expect(fs.readFile(testFile, "utf-8")).rejects.toThrow()
			await expect(fs.readFile(untrackedFile, "utf-8")).rejects.toThrow()
		})

		test("does not create a checkpoint for ignored files", async () => {
			// Create a file that matches an ignored pattern (e.g., .log file).
			const ignoredFile = path.join(service.workspaceDir, "ignored.log")
			await fs.writeFile(ignoredFile, "Initial ignored content")

			const commit = await service.saveCheckpoint("Ignored file checkpoint")
			assert.ok(!commit?.commit)

			await fs.writeFile(ignoredFile, "Modified ignored content")

			const commit2 = await service.saveCheckpoint("Ignored file modified checkpoint")
			assert.ok(!commit2?.commit)

			expect(await fs.readFile(ignoredFile, "utf-8")).toBe("Modified ignored content")
		})

		test("does not create a checkpoint for LFS files", async () => {
			// Create a .gitattributes file with LFS patterns.
			const gitattributesPath = path.join(service.workspaceDir, ".gitattributes")
			await fs.writeFile(gitattributesPath, "*.lfs filter=lfs diff=lfs merge=lfs -text")

			// Re-initialize the service to trigger a write to .git/info/exclude.
			service = new klass(service.taskId, service.checkpointsDir, service.workspaceDir, () => {})
			const excludesPath = path.join(service.checkpointsDir, ".git", "info", "exclude")
			expect((await fs.readFile(excludesPath, "utf-8")).split("\n")).not.toContain("*.lfs")
			await service.initShadowGit()
			expect((await fs.readFile(excludesPath, "utf-8")).split("\n")).toContain("*.lfs")

			const commit0 = await service.saveCheckpoint("Add gitattributes")
			assert.ok(commit0?.commit)

			// Create a file that matches an LFS pattern.
			const lfsFile = path.join(service.workspaceDir, "foo.lfs")
			await fs.writeFile(lfsFile, "Binary file content simulation")

			const commit = await service.saveCheckpoint("LFS file checkpoint")
			assert.ok(!commit?.commit)

			await fs.writeFile(lfsFile, "Modified binary content")

			const commit2 = await service.saveCheckpoint("LFS file modified checkpoint")
			assert.ok(!commit2?.commit)

			expect(await fs.readFile(lfsFile, "utf-8")).toBe("Modified binary content")
		})
	})

	suite(`${klass.name}#create`, () => {
		test("initializes a git repository if one does not already exist", async () => {
			const shadowDir = path.join(tmpDir, `${prefix}2-${Date.now()}`)
			const workspaceDir = path.join(tmpDir, `workspace2-${Date.now()}`)
			await fs.mkdir(workspaceDir)

			const newTestFile = path.join(workspaceDir, "test.txt")
			await fs.writeFile(newTestFile, "Hello, world!")
			expect(await fs.readFile(newTestFile, "utf-8")).toBe("Hello, world!")

			// Ensure the git repository was initialized.
			const newService = klass.create({ taskId, shadowDir, workspaceDir, log: () => {} })
			const { created } = await newService.initShadowGit()
			assert.ok(created)

			const gitDir = path.join(newService.checkpointsDir, ".git")
			expect(await fs.stat(gitDir)).toBeTruthy()

			// Save a new checkpoint: Ahoy, world!
			await fs.writeFile(newTestFile, "Ahoy, world!")
			const commit1 = await newService.saveCheckpoint("Ahoy, world!")
			assert.ok(commit1?.commit)
			expect(await fs.readFile(newTestFile, "utf-8")).toBe("Ahoy, world!")

			// Restore "Hello, world!"
			await newService.restoreCheckpoint(newService.baseHash!)
			expect(await fs.readFile(newTestFile, "utf-8")).toBe("Hello, world!")

			// Restore "Ahoy, world!"
			await newService.restoreCheckpoint(commit1!.commit)
			expect(await fs.readFile(newTestFile, "utf-8")).toBe("Ahoy, world!")

			await fs.rm(newService.checkpointsDir, { recursive: true, force: true })
			await fs.rm(newService.workspaceDir, { recursive: true, force: true })
		})
	})

	suite(`${klass.name}#renameNestedGitRepos`, () => {
		test("handles nested git repositories during initialization", async () => {
			// Create a new temporary workspace and service for this test.
			const shadowDir = path.join(tmpDir, `${prefix}-nested-git-${Date.now()}`)
			const workspaceDir = path.join(tmpDir, `workspace-nested-git-${Date.now()}`)

			// Create a primary workspace repo.
			await fs.mkdir(workspaceDir, { recursive: true })
			const mainGit = simpleGit(workspaceDir)
			await mainGit.init()
			await mainGit.addConfig("user.name", EXTENSION_DISPLAY_NAME as string)
			await mainGit.addConfig("user.email", AUTHOR_EMAIL as string)

			// Create a nested repo inside the workspace.
			const nestedRepoPath = path.join(workspaceDir, "nested-project")
			await fs.mkdir(nestedRepoPath, { recursive: true })
			const nestedGit = simpleGit(nestedRepoPath)
			await nestedGit.init()
			await nestedGit.addConfig("user.name", EXTENSION_DISPLAY_NAME as string)
			await nestedGit.addConfig("user.email", AUTHOR_EMAIL as string)

			// Add a file to the nested repo.
			const nestedFile = path.join(nestedRepoPath, "nested-file.txt")
			await fs.writeFile(nestedFile, "Content in nested repo")
			await nestedGit.add(".")
			await nestedGit.commit("Initial commit in nested repo")

			// Create a test file in the main workspace.
			const mainFile = path.join(workspaceDir, "main-file.txt")
			await fs.writeFile(mainFile, "Content in main repo")
			await mainGit.add(".")
			await mainGit.commit("Initial commit in main repo")

			// Confirm nested git directory exists before initialization.
			const nestedGitDir = path.join(nestedRepoPath, ".git")
			const nestedGitDisabledDir = `${nestedGitDir}_disabled`
			expect(await fileExistsAtPath(nestedGitDir)).toBe(true)
			expect(await fileExistsAtPath(nestedGitDisabledDir)).toBe(false)

			// Configure globby mock to return our nested git repository.
			const relativeGitPath = path.relative(workspaceDir, nestedGitDir)
			globby.callsFake((pattern: string | readonly string[]) => {
				if (pattern === "**/.git") {
					return Promise.resolve([relativeGitPath])
				} else if (pattern === "**/.git_disabled") {
					return Promise.resolve([`${relativeGitPath}_disabled`])
				}

				return Promise.resolve([])
			})

			// Create a spy on fs.rename to track when it's called.
			const renameSpy = sinon.spy(fs, "rename")

			// Initialize the shadow git service.
			const service = new klass(taskId, shadowDir, workspaceDir, () => {})

			// Override renameNestedGitRepos to track calls.
			const originalRenameMethod = service["renameNestedGitRepos"].bind(service)
			let disableCall = false
			let enableCall = false

			service["renameNestedGitRepos"] = async (disable: boolean) => {
				if (disable) {
					disableCall = true
} else {
					enableCall = true
				}

				return originalRenameMethod(disable)
			}

			// Initialize the shadow git repo.
			await service.initShadowGit()

			// Verify both disable and enable were called.
			assert.strictEqual(disableCall, true)
			assert.strictEqual(enableCall, true)

			// Verify rename was called with correct paths.
			const renameCallsArgs = renameSpy.mock.calls.map((call) => String(call[0]) + " -> " + String(call[1]))
			expect(
				renameCallsArgs.some((args) => args.includes(nestedGitDir) && args.includes(nestedGitDisabledDir)),
			).toBe(true)
			expect(
				renameCallsArgs.some((args) => args.includes(nestedGitDisabledDir) && args.includes(nestedGitDir)),
			).toBe(true)

			// Verify the nested git directory is back to normal after initialization.
			expect(await fileExistsAtPath(nestedGitDir)).toBe(true)
			expect(await fileExistsAtPath(nestedGitDisabledDir)).toBe(false)

			// Clean up.
			renameSpy.restore()
			await fs.rm(shadowDir, { recursive: true, force: true })
			await fs.rm(workspaceDir, { recursive: true, force: true })
		})
	})

	suite(`${klass.name}#events`, () => {
		test("emits initialize event when service is created", async () => {
			const shadowDir = path.join(tmpDir, `${prefix}3-${Date.now()}`)
			const workspaceDir = path.join(tmpDir, `workspace3-${Date.now()}`)
			await fs.mkdir(workspaceDir, { recursive: true })

			const newTestFile = path.join(workspaceDir, "test.txt")
			await fs.writeFile(newTestFile, "Testing events!")

			// Create a mock implementation of emit to track events.
			const emitSpy = sinon.spy(EventEmitter.prototype, "emit")

			// Create the service - this will trigger the initialize event.
			const newService = klass.create({ taskId, shadowDir, workspaceDir, log: () => {} })
			await newService.initShadowGit()

			// Find the initialize event in the emit calls.
			let initializeEvent: CheckpointEventMap["initialize"] | null = null

			for (let i = 0; i < emitSpy.mock.calls.length; i++) {
				const call = emitSpy.mock.calls[i]

				if (call[0] === "initialize") {
					initializeEvent = call[1] as CheckpointEventMap["initialize"]
					break
				}
			}

			// Restore the spy.
			emitSpy.restore()

			// Verify the event was emitted with the correct data.
			expect(initializeEvent).not.toBeNull()
			if (initializeEvent) {
				assert.strictEqual(initializeEvent.type, "initialize")
				assert.strictEqual(initializeEvent.workspaceDir, workspaceDir)
				assert.ok(initializeEvent.baseHash)
				assert.strictEqual(typeof initializeEvent.created, "boolean")
				assert.strictEqual(typeof initializeEvent.duration, "number")
			}

			// Clean up.
			await fs.rm(shadowDir, { recursive: true, force: true })
			await fs.rm(workspaceDir, { recursive: true, force: true })
		})

		test("emits checkpoint event when saving checkpoint", async () => {
			const checkpointHandler = sinon.stub()
			service.on("checkpoint", checkpointHandler)

			await fs.writeFile(testFile, "Changed content for checkpoint event test")
			const result = await service.saveCheckpoint("Test checkpoint event")
			assert.notStrictEqual(result?.commit, undefined)

			assert.strictEqual(checkpointHandler.callCount, 1)
			const eventData = checkpointHandler.mock.calls[0][0]
			assert.strictEqual(eventData.type, "checkpoint")
			assert.notStrictEqual(eventData.toHash, undefined)
			assert.strictEqual(eventData.toHash, result!.commit)
			assert.strictEqual(typeof eventData.duration, "number")
		})

		test("emits restore event when restoring checkpoint", async () => {
			// First create a checkpoint to restore.
			await fs.writeFile(testFile, "Content for restore test")
			const commit = await service.saveCheckpoint("Checkpoint for restore test")
			assert.ok(commit?.commit)

			// Change the file again.
			await fs.writeFile(testFile, "Changed after checkpoint")

			// Setup restore event listener.
			const restoreHandler = sinon.stub()
			service.on("restore", restoreHandler)

			// Restore the checkpoint.
			await service.restoreCheckpoint(commit!.commit)

			// Verify the event was emitted.
			assert.strictEqual(restoreHandler.callCount, 1)
			const eventData = restoreHandler.mock.calls[0][0]
			assert.strictEqual(eventData.type, "restore")
			assert.strictEqual(eventData.commitHash, commit!.commit)
			assert.strictEqual(typeof eventData.duration, "number")

			// Verify the file was actually restored.
			expect(await fs.readFile(testFile, "utf-8")).toBe("Content for restore test")
		})

		test("emits error event when an error occurs", async () => {
			const errorHandler = sinon.stub()
			service.on("error", errorHandler)

			// Force an error by providing an invalid commit hash.
			const invalidCommitHash = "invalid-commit-hash"

			// Try to restore an invalid checkpoint.
			try {
				await service.restoreCheckpoint(invalidCommitHash)
			} catch {
				// Expected to throw, we're testing the event emission.
			}

			// Verify the error event was emitted.
			assert.strictEqual(errorHandler.callCount, 1)
			const eventData = errorHandler.mock.calls[0][0]
			assert.strictEqual(eventData.type, "error")
			assert.ok(eventData.error instanceof Error)
		})

		test("supports multiple event listeners for the same event", async () => {
			const checkpointHandler1 = sinon.stub()
			const checkpointHandler2 = sinon.stub()

			service.on("checkpoint", checkpointHandler1)
			service.on("checkpoint", checkpointHandler2)

			await fs.writeFile(testFile, "Content for multiple listeners test")
			const result = await service.saveCheckpoint("Testing multiple listeners")

			// Verify both handlers were called with the same event data.
			assert.strictEqual(checkpointHandler1.callCount, 1)
			assert.strictEqual(checkpointHandler2.callCount, 1)

			const eventData1 = checkpointHandler1.mock.calls[0][0]
			const eventData2 = checkpointHandler2.mock.calls[0][0]

			assert.deepStrictEqual(eventData1, eventData2)
			assert.strictEqual(eventData1.type, "checkpoint")
			assert.strictEqual(eventData1.toHash, result?.commit)
		})

		test("allows removing event listeners", async () => {
			const checkpointHandler = sinon.stub()

			// Add the listener.
			service.on("checkpoint", checkpointHandler)

			// Make a change and save a checkpoint.
			await fs.writeFile(testFile, "Content for remove listener test - part 1")
			await service.saveCheckpoint("Testing listener - part 1")

			// Verify handler was called.
			assert.strictEqual(checkpointHandler.callCount, 1)
			checkpointHandler.resetHistory()

			// Remove the listener.
			service.off("checkpoint", checkpointHandler)

			// Make another change and save a checkpoint.
			await fs.writeFile(testFile, "Content for remove listener test - part 2")
			await service.saveCheckpoint("Testing listener - part 2")

			// Verify handler was not called after being removed.
			assert.ok(!checkpointHandler.called)
		})
	})
// Mock cleanup
suite("ShadowCheckpointService", () => {
	const taskId = "test-task-storage"
	const tmpDir = path.join(os.tmpdir(), "CheckpointService")
	const globalStorageDir = path.join(tmpDir, "global-storage-dir")
	const workspaceDir = path.join(tmpDir, "workspace-dir")
	const workspaceHash = ShadowCheckpointService.hashWorkspaceDir(workspaceDir)

	setup(async () => {
		await fs.mkdir(globalStorageDir, { recursive: true })
		await fs.mkdir(workspaceDir, { recursive: true })
	})

	teardown(async () => {
		await fs.rm(globalStorageDir, { recursive: true, force: true })
		await fs.rm(workspaceDir, { recursive: true, force: true })
	})

	suite("getTaskStorage", () => {
		test("returns 'task' when task repo exists", async () => {
			const service = RepoPerTaskCheckpointService.create({
				taskId,
				shadowDir: globalStorageDir,
				workspaceDir,
				log: () => {},
			})

			await service.initShadowGit()

			const storage = await ShadowCheckpointService.getTaskStorage({ taskId, globalStorageDir, workspaceDir })
			assert.strictEqual(storage, "task")
		})

		test("returns 'workspace' when workspace repo exists with task branch", async () => {
			const service = RepoPerWorkspaceCheckpointService.create({
				taskId,
				shadowDir: globalStorageDir,
				workspaceDir,
				log: () => {},
			})

			await service.initShadowGit()

			const storage = await ShadowCheckpointService.getTaskStorage({ taskId, globalStorageDir, workspaceDir })
			assert.strictEqual(storage, "workspace")
		})

		test("returns undefined when no repos exist", async () => {
			const storage = await ShadowCheckpointService.getTaskStorage({ taskId, globalStorageDir, workspaceDir })
			assert.strictEqual(storage, undefined)
		})

		test("returns undefined when workspace repo exists but has no task branch", async () => {
			// Setup: Create workspace repo without the task branch
			const workspaceRepoDir = path.join(globalStorageDir, "checkpoints", workspaceHash)
			await fs.mkdir(workspaceRepoDir, { recursive: true })

			// Create git repo without adding the specific branch
			const git = simpleGit(workspaceRepoDir)
			await git.init()
			await git.addConfig("user.name", EXTENSION_DISPLAY_NAME as string)
			await git.addConfig("user.email", AUTHOR_EMAIL as string)

			// We need to create a commit, but we won't create the specific branch
			const testFile = path.join(workspaceRepoDir, "test.txt")
			await fs.writeFile(testFile, "Test content")
			await git.add(".")
			await git.commit("Initial commit")

			const storage = await ShadowCheckpointService.getTaskStorage({
				taskId,
				globalStorageDir,
				workspaceDir,
			})

			assert.strictEqual(storage, undefined)
		})
	})
// Mock cleanup
// Mock cleanup
