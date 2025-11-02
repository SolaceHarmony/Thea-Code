// npx mocha src/services/checkpoints/__tests__/ShadowCheckpointService.test.ts

import fs from "fs/promises"
import path from "path"
import os from "os"
import assert from "node:assert/strict"

import { simpleGit, SimpleGit } from "simple-git"
import { EXTENSION_DISPLAY_NAME, AUTHOR_EMAIL } from "../../../shared/config/thea-config"
import { fileExistsAtPath } from "../../../utils/fs"

import { ShadowCheckpointService } from "../ShadowCheckpointService"
import { RepoPerTaskCheckpointService } from "../RepoPerTaskCheckpointService"
import { RepoPerWorkspaceCheckpointService } from "../RepoPerWorkspaceCheckpointService"
import type { CheckpointEventMap } from "../types"

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

	return { git, testFile }
}

const serviceClasses: Array<{
	Class: typeof RepoPerTaskCheckpointService | typeof RepoPerWorkspaceCheckpointService
	name: string
}> = [
	{ Class: RepoPerTaskCheckpointService, name: "RepoPerTaskCheckpointService" },
	{ Class: RepoPerWorkspaceCheckpointService, name: "RepoPerWorkspaceCheckpointService" },
]

for (const { Class: klass, name: prefix } of serviceClasses) {
	describe(`CheckpointService (${klass.name})`, () => {
		const taskId = "test-task"

		let workspaceGit: SimpleGit
		let testFile: string
		let service: RepoPerTaskCheckpointService | RepoPerWorkspaceCheckpointService

		beforeEach(async () => {
			const shadowDir = path.join(tmpDir, `${prefix}-${Date.now()}`)
			const workspaceDir = path.join(tmpDir, `workspace-${Date.now()}`)
			const repo = await initWorkspaceRepo({ workspaceDir })

			workspaceGit = repo.git
			testFile = repo.testFile

			service = klass.create({ taskId, shadowDir, workspaceDir, log: () => {} })
			await service.initShadowGit()
		})

		afterEach(async () => {
			// Clean up after each test
			try {
				await fs.rm(service.checkpointsDir, { recursive: true, force: true })
				await fs.rm(service.workspaceDir, { recursive: true, force: true })
			} catch {
				// Ignore cleanup errors
			}
		})

		describe(`#getDiff`, () => {
		it("returns the correct diff between commits", async () => {
			await fs.writeFile(testFile, "Ahoy, world!")
			const commit1 = await service.saveCheckpoint("Ahoy, world!")
			assert.ok(commit1?.commit, "First checkpoint should have a commit hash")

			await fs.writeFile(testFile, "Goodbye, world!")
			const commit2 = await service.saveCheckpoint("Goodbye, world!")
			assert.ok(commit2?.commit, "Second checkpoint should have a commit hash")

			const diff1 = await service.getDiff({ to: commit1!.commit })
			assert.strictEqual(diff1.length, 1, "Diff should have one change")
			assert.strictEqual(diff1[0].paths.relative, "test.txt", "Relative path should be test.txt")
			assert.strictEqual(diff1[0].paths.absolute, testFile, "Absolute path should match testFile")
			assert.strictEqual(diff1[0].content.before, "Hello, world!", "Before content should be initial content")
			assert.strictEqual(diff1[0].content.after, "Ahoy, world!", "After content should be new content")

			const diff2 = await service.getDiff({ from: service.baseHash, to: commit2!.commit })
			assert.strictEqual(diff2.length, 1, "Diff should have one change")
			assert.strictEqual(diff2[0].paths.relative, "test.txt", "Relative path should be test.txt")
			assert.strictEqual(diff2[0].paths.absolute, testFile, "Absolute path should match testFile")
			assert.strictEqual(diff2[0].content.before, "Hello, world!", "Before content should be initial content")
			assert.strictEqual(diff2[0].content.after, "Goodbye, world!", "After content should be new content")

			const diff12 = await service.getDiff({ from: commit1!.commit, to: commit2!.commit })
			assert.strictEqual(diff12.length, 1, "Diff should have one change")
			assert.strictEqual(diff12[0].paths.relative, "test.txt", "Relative path should be test.txt")
			assert.strictEqual(diff12[0].paths.absolute, testFile, "Absolute path should match testFile")
			assert.strictEqual(diff12[0].content.before, "Ahoy, world!", "Before content should be first change")
			assert.strictEqual(diff12[0].content.after, "Goodbye, world!", "After content should be final change")
		})

		it("handles new files in diff", async () => {
			const newFile = path.join(service.workspaceDir, "new.txt")
			await fs.writeFile(newFile, "New file content")
			const commit = await service.saveCheckpoint("Add new file")
			assert.ok(commit?.commit, "Checkpoint should have a commit hash")

			const changes = await service.getDiff({ to: commit!.commit })
			const change = changes.find((c) => c.paths.relative === "new.txt")
			assert.ok(change, "Should find the new file in changes")
			assert.strictEqual(change?.content.before, "", "Before content should be empty for new file")
			assert.strictEqual(change?.content.after, "New file content", "After content should be file content")
		})

		it("handles deleted files in diff", async () => {
			const fileToDelete = path.join(service.workspaceDir, "new.txt")
			await fs.writeFile(fileToDelete, "New file content")
			const commit1 = await service.saveCheckpoint("Add file")
			assert.ok(commit1?.commit, "First checkpoint should have a commit hash")

			await fs.unlink(fileToDelete)
			const commit2 = await service.saveCheckpoint("Delete file")
			assert.ok(commit2?.commit, "Second checkpoint should have a commit hash")

			const changes = await service.getDiff({ from: commit1!.commit, to: commit2!.commit })
			const change = changes.find((c) => c.paths.relative === "new.txt")
			assert.ok(change, "Should find the deleted file in changes")
			assert.strictEqual(change!.content.before, "New file content", "Before content should be file content")
			assert.strictEqual(change!.content.after, "", "After content should be empty for deleted file")
		})
	})

	describe(`#saveCheckpoint`, () => {
		it("creates a checkpoint if there are pending changes", async () => {
			await fs.writeFile(testFile, "Ahoy, world!")
			const commit1 = await service.saveCheckpoint("First checkpoint")
			assert.ok(commit1?.commit, "First checkpoint should have a commit hash")
			const details1 = await service.getDiff({ to: commit1!.commit })
			assert.ok(details1[0].content.before.includes("Hello, world!"), "Before content should include initial content")
			assert.ok(details1[0].content.after.includes("Ahoy, world!"), "After content should include new content")

			await fs.writeFile(testFile, "Hola, world!")
			const commit2 = await service.saveCheckpoint("Second checkpoint")
			assert.ok(commit2?.commit, "Second checkpoint should have a commit hash")
			const details2 = await service.getDiff({ from: commit1!.commit, to: commit2!.commit })
			assert.ok(details2[0].content.before.includes("Ahoy, world!"), "Before content should include first change")
			assert.ok(details2[0].content.after.includes("Hola, world!"), "After content should include second change")

			// Switch to checkpoint 1.
			await service.restoreCheckpoint(commit1!.commit)
			assert.strictEqual(await fs.readFile(testFile, "utf-8"), "Ahoy, world!", "File should be restored to first checkpoint")

			// Switch to checkpoint 2.
			await service.restoreCheckpoint(commit2!.commit)
			assert.strictEqual(await fs.readFile(testFile, "utf-8"), "Hola, world!", "File should be restored to second checkpoint")

			// Switch back to initial commit.
			assert.ok(service.baseHash, "Base hash should exist")
			await service.restoreCheckpoint(service.baseHash!)
			assert.strictEqual(await fs.readFile(testFile, "utf-8"), "Hello, world!", "File should be restored to initial state")
		})

		it("preserves workspace and index state after saving checkpoint", async () => {
			// Create three files with different states: staged, unstaged, and mixed.
			const unstagedFile = path.join(service.workspaceDir, "unstaged.txt")
			const stagedFile = path.join(service.workspaceDir, "staged.txt")
			const mixedFile = path.join(service.workspaceDir, "mixed.txt")

			await fs.writeFile(unstagedFile, "Initial unstaged")
			await fs.writeFile(stagedFile, "Initial staged")
			await fs.writeFile(mixedFile, "Initial mixed")
			await workspaceGit.add(["."])
			const result = await workspaceGit.commit("Add initial files")
			assert.ok(result?.commit, "Initial commit should have a commit hash")

			await fs.writeFile(unstagedFile, "Modified unstaged")

			await fs.writeFile(stagedFile, "Modified staged")
			await workspaceGit.add([stagedFile])

			await fs.writeFile(mixedFile, "Modified mixed - staged")
			await workspaceGit.add([mixedFile])
			await fs.writeFile(mixedFile, "Modified mixed - unstaged")

			// Save checkpoint.
			const commit = await service.saveCheckpoint("Test checkpoint")
			assert.ok(commit?.commit, "Checkpoint should have a commit hash")

			// Verify workspace state is preserved.
			const status = await workspaceGit.status()

			// All files should be modified.
			assert.ok(status.modified.includes("unstaged.txt"), "unstaged.txt should be modified")
			assert.ok(status.modified.includes("staged.txt"), "staged.txt should be modified")
			assert.ok(status.modified.includes("mixed.txt"), "mixed.txt should be modified")

			// Only staged and mixed files should be staged.
			assert.ok(!status.staged.includes("unstaged.txt"), "unstaged.txt should not be staged")
			assert.ok(status.staged.includes("staged.txt"), "staged.txt should be staged")
			assert.ok(status.staged.includes("mixed.txt"), "mixed.txt should be staged")

			// Verify file contents.
			assert.strictEqual(await fs.readFile(unstagedFile, "utf-8"), "Modified unstaged", "unstaged.txt should have modified content")
			assert.strictEqual(await fs.readFile(stagedFile, "utf-8"), "Modified staged", "staged.txt should have modified content")
			assert.strictEqual(await fs.readFile(mixedFile, "utf-8"), "Modified mixed - unstaged", "mixed.txt should have unstaged content")

			// Verify staged changes (--cached shows only staged changes).
			const stagedDiff = await workspaceGit.diff(["--cached", "mixed.txt"])
			assert.ok(stagedDiff.includes("-Initial mixed"), "Staged diff should show removed initial content")
			assert.ok(stagedDiff.includes("+Modified mixed - staged"), "Staged diff should show added staged content")

			// Verify unstaged changes (shows working directory changes).
			const unstagedDiff = await workspaceGit.diff(["mixed.txt"])
			assert.ok(unstagedDiff.includes("-Modified mixed - staged"), "Unstaged diff should show removed staged content")
			assert.ok(unstagedDiff.includes("+Modified mixed - unstaged"), "Unstaged diff should show added unstaged content")
		})

		it("does not create a checkpoint if there are no pending changes", async () => {
			const commit0 = await service.saveCheckpoint("Zeroth checkpoint")
			assert.strictEqual(commit0?.commit, undefined, "No checkpoint should be created if there are no changes")

			await fs.writeFile(testFile, "Ahoy, world!")
			const commit1 = await service.saveCheckpoint("First checkpoint")
			assert.ok(commit1?.commit, "Checkpoint should be created when there are changes")

			const commit2 = await service.saveCheckpoint("Second checkpoint")
			assert.strictEqual(commit2?.commit, undefined, "No checkpoint should be created if there are no changes after first checkpoint")
		})

		it("includes untracked files in checkpoints", async () => {
			// Create an untracked file.
			const untrackedFile = path.join(service.workspaceDir, "untracked.txt")
			await fs.writeFile(untrackedFile, "I am untracked!")

			// Save a checkpoint with the untracked file.
			const commit1 = await service.saveCheckpoint("Checkpoint with untracked file")
			assert.ok(commit1?.commit, "Checkpoint should have a commit hash")

			// Verify the untracked file was included in the checkpoint.
			const details = await service.getDiff({ to: commit1!.commit })
			assert.ok(details[0].content.before.includes(""), "Before content should be empty")
			assert.ok(details[0].content.after.includes("I am untracked!"), "After content should include untracked file")

			// Create another checkpoint with a different state.
			await fs.writeFile(testFile, "Changed tracked file")
			const commit2 = await service.saveCheckpoint("Second checkpoint")
			assert.ok(commit2?.commit, "Second checkpoint should have a commit hash")

			// Restore first checkpoint and verify untracked file is preserved.
			await service.restoreCheckpoint(commit1!.commit)
			assert.strictEqual(await fs.readFile(untrackedFile, "utf-8"), "I am untracked!", "Untracked file should be preserved")
			assert.strictEqual(await fs.readFile(testFile, "utf-8"), "Hello, world!", "Tracked file should be at initial state")

			// Restore second checkpoint and verify untracked file remains (since
			// restore preserves untracked files)
			await service.restoreCheckpoint(commit2!.commit)
			assert.strictEqual(await fs.readFile(untrackedFile, "utf-8"), "I am untracked!", "Untracked file should remain")
			assert.strictEqual(await fs.readFile(testFile, "utf-8"), "Changed tracked file", "Tracked file should be at changed state")
		})

		it("handles file deletions correctly", async () => {
			await fs.writeFile(testFile, "I am tracked!")
			const untrackedFile = path.join(service.workspaceDir, "new.txt")
			await fs.writeFile(untrackedFile, "I am untracked!")
			const commit1 = await service.saveCheckpoint("First checkpoint")
			assert.ok(commit1?.commit, "First checkpoint should have a commit hash")

			await fs.unlink(testFile)
			await fs.unlink(untrackedFile)
			const commit2 = await service.saveCheckpoint("Second checkpoint")
			assert.ok(commit2?.commit, "Second checkpoint should have a commit hash")

			// Verify files are gone.
			await assert.rejects(fs.readFile(testFile, "utf-8"), {}, "Tracked file should not exist")
			await assert.rejects(fs.readFile(untrackedFile, "utf-8"), {}, "Untracked file should not exist")

			// Restore first checkpoint.
			await service.restoreCheckpoint(commit1!.commit)
			assert.strictEqual(await fs.readFile(testFile, "utf-8"), "I am tracked!", "Tracked file should be restored")
			assert.strictEqual(await fs.readFile(untrackedFile, "utf-8"), "I am untracked!", "Untracked file should be restored")

			// Restore second checkpoint.
			await service.restoreCheckpoint(commit2!.commit)
			await assert.rejects(fs.readFile(testFile, "utf-8"), {}, "Tracked file should not exist after restore")
			await assert.rejects(fs.readFile(untrackedFile, "utf-8"), {}, "Untracked file should not exist after restore")
		})

		it("does not create a checkpoint for ignored files", async () => {
			// Create a file that matches an ignored pattern (e.g., .log file).
			const ignoredFile = path.join(service.workspaceDir, "ignored.log")
			await fs.writeFile(ignoredFile, "Initial ignored content")

			const commit = await service.saveCheckpoint("Ignored file checkpoint")
			assert.strictEqual(commit?.commit, undefined, "No checkpoint should be created for ignored files")

			await fs.writeFile(ignoredFile, "Modified ignored content")

			const commit2 = await service.saveCheckpoint("Ignored file modified checkpoint")
			assert.strictEqual(commit2?.commit, undefined, "No checkpoint should be created for modified ignored files")

			assert.strictEqual(await fs.readFile(ignoredFile, "utf-8"), "Modified ignored content", "Ignored file should still be modifiable")
		})

		it("does not create a checkpoint for LFS files", async () => {
			// Create a .gitattributes file with LFS patterns.
			const gitattributesPath = path.join(service.workspaceDir, ".gitattributes")
			await fs.writeFile(gitattributesPath, "*.lfs filter=lfs diff=lfs merge=lfs -text")

			// Re-initialize the service to trigger a write to .git/info/exclude.
			service = new klass(service.taskId, service.checkpointsDir, service.workspaceDir, () => {})
			const excludesPath = path.join(service.checkpointsDir, ".git", "info", "exclude")
			const excludesBefore = (await fs.readFile(excludesPath, "utf-8")).split("\n")
			assert.ok(!excludesBefore.includes("*.lfs"), "LFS pattern should not be in excludes before init")
			await service.initShadowGit()
			const excludesAfter = (await fs.readFile(excludesPath, "utf-8")).split("\n")
			assert.ok(excludesAfter.includes("*.lfs"), "LFS pattern should be in excludes after init")

			const commit0 = await service.saveCheckpoint("Add gitattributes")
			assert.ok(commit0?.commit, "Checkpoint should be created for gitattributes")

			// Create a file that matches an LFS pattern.
			const lfsFile = path.join(service.workspaceDir, "foo.lfs")
			await fs.writeFile(lfsFile, "Binary file content simulation")

			const commit = await service.saveCheckpoint("LFS file checkpoint")
			assert.strictEqual(commit?.commit, undefined, "No checkpoint should be created for LFS files")

			await fs.writeFile(lfsFile, "Modified binary content")

			const commit2 = await service.saveCheckpoint("LFS file modified checkpoint")
			assert.strictEqual(commit2?.commit, undefined, "No checkpoint should be created for modified LFS files")

			assert.strictEqual(await fs.readFile(lfsFile, "utf-8"), "Modified binary content", "LFS file should still be modifiable")
		})
	})

	describe(`#create`, () => {
		it("initializes a git repository if one does not already exist", async () => {
			const shadowDir = path.join(tmpDir, `${prefix}2-${Date.now()}`)
			const workspaceDir = path.join(tmpDir, `workspace2-${Date.now()}`)
			await fs.mkdir(workspaceDir)

			const newTestFile = path.join(workspaceDir, "test.txt")
			await fs.writeFile(newTestFile, "Hello, world!")
			assert.strictEqual(await fs.readFile(newTestFile, "utf-8"), "Hello, world!", "Test file should be created")

			// Ensure the git repository was initialized.
			const newService = klass.create({ taskId, shadowDir, workspaceDir, log: () => {} })
			const { created } = await newService.initShadowGit()
			assert.strictEqual(created, true, "Git repository should be newly created")

			const gitDir = path.join(newService.checkpointsDir, ".git")
			const stats = await fs.stat(gitDir)
			assert.ok(stats, "Git directory should exist")

			// Save a new checkpoint: Ahoy, world!
			await fs.writeFile(newTestFile, "Ahoy, world!")
			const commit1 = await newService.saveCheckpoint("Ahoy, world!")
			assert.ok(commit1?.commit, "First checkpoint should have a commit hash")
			assert.strictEqual(await fs.readFile(newTestFile, "utf-8"), "Ahoy, world!", "File should have new content")

			// Restore "Hello, world!"
			assert.ok(newService.baseHash, "Base hash should exist")
			await newService.restoreCheckpoint(newService.baseHash!)
			assert.strictEqual(await fs.readFile(newTestFile, "utf-8"), "Hello, world!", "File should be restored to initial state")

			// Restore "Ahoy, world!"
			await newService.restoreCheckpoint(commit1!.commit)
			assert.strictEqual(await fs.readFile(newTestFile, "utf-8"), "Ahoy, world!", "File should be restored to checkpoint state")

			await fs.rm(newService.checkpointsDir, { recursive: true, force: true })
			await fs.rm(newService.workspaceDir, { recursive: true, force: true })
		})
	})

	describe(`#renameNestedGitRepos`, () => {
		it("handles nested git repositories during initialization", async () => {
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
			assert.strictEqual(await fileExistsAtPath(nestedGitDir), true, "Nested git directory should exist before init")
			assert.strictEqual(await fileExistsAtPath(nestedGitDisabledDir), false, "Disabled nested git directory should not exist before init")

			// Initialize the shadow git service.
			const testService = new klass(taskId, shadowDir, workspaceDir, () => {})

			// Initialize the shadow git repo - this should handle the nested git repo.
			await testService.initShadowGit()

			// Verify the nested git directory is back to normal after initialization.
			// The service should have temporarily disabled and then re-enabled nested git repos.
			assert.strictEqual(await fileExistsAtPath(nestedGitDir), true, "Nested git directory should exist after init")
			assert.strictEqual(await fileExistsAtPath(nestedGitDisabledDir), false, "Disabled nested git directory should not exist after init")

			// Clean up.
			await fs.rm(shadowDir, { recursive: true, force: true })
			await fs.rm(workspaceDir, { recursive: true, force: true })
		})
	})

	describe(`#events`, () => {
		it("emits initialize event when service is created", async () => {
			const shadowDir = path.join(tmpDir, `${prefix}3-${Date.now()}`)
			const workspaceDir = path.join(tmpDir, `workspace3-${Date.now()}`)
			await fs.mkdir(workspaceDir, { recursive: true })

			const newTestFile = path.join(workspaceDir, "test.txt")
			await fs.writeFile(newTestFile, "Testing events!")

			// Track emitted events using real event listeners
			const emittedEvents: Array<CheckpointEventMap[keyof CheckpointEventMap]> = []

			// Create the service - this will trigger the initialize event.
			const newService = klass.create({ taskId, shadowDir, workspaceDir, log: () => {} })
			
			// Listen for all events
			newService.on("initialize", (event: CheckpointEventMap["initialize"]) => {
				emittedEvents.push(event)
			})

			await newService.initShadowGit()

			// Verify the event was emitted with the correct data.
			assert.strictEqual(emittedEvents.length, 1, "One initialize event should be emitted")
			const initializeEvent = emittedEvents[0] as CheckpointEventMap["initialize"]
			assert.strictEqual(initializeEvent.type, "initialize", "Event type should be initialize")
			assert.strictEqual(initializeEvent.workspaceDir, workspaceDir, "Workspace dir should match")
			assert.ok(initializeEvent.baseHash, "Base hash should exist")
			assert.strictEqual(typeof initializeEvent.created, "boolean", "Created should be boolean")
			assert.strictEqual(typeof initializeEvent.duration, "number", "Duration should be number")

			// Clean up.
			await fs.rm(shadowDir, { recursive: true, force: true })
			await fs.rm(workspaceDir, { recursive: true, force: true })
		})

		it("emits checkpoint event when saving checkpoint", async () => {
			const emittedEvents: Array<CheckpointEventMap["checkpoint"]> = []

			// Set up real event listener
			service.on("checkpoint", (event: CheckpointEventMap["checkpoint"]) => {
				emittedEvents.push(event)
			})

			await fs.writeFile(testFile, "Changed content for checkpoint event test")
			const result = await service.saveCheckpoint("Test checkpoint event")
			assert.ok(result?.commit, "Checkpoint should have a commit")

			assert.strictEqual(emittedEvents.length, 1, "One checkpoint event should be emitted")
			const eventData = emittedEvents[0]
			assert.strictEqual(eventData.type, "checkpoint", "Event type should be checkpoint")
			assert.ok(eventData.toHash, "toHash should be defined")
			assert.strictEqual(eventData.toHash, result!.commit, "toHash should match commit")
			assert.strictEqual(typeof eventData.duration, "number", "Duration should be number")
		})

		it("emits restore event when restoring checkpoint", async () => {
			// First create a checkpoint to restore.
			await fs.writeFile(testFile, "Content for restore test")
			const commit = await service.saveCheckpoint("Checkpoint for restore test")
			assert.ok(commit?.commit, "Checkpoint should have a commit hash")

			// Change the file again.
			await fs.writeFile(testFile, "Changed after checkpoint")

			// Setup real restore event listener.
			const emittedEvents: Array<CheckpointEventMap["restore"]> = []
			service.on("restore", (event: CheckpointEventMap["restore"]) => {
				emittedEvents.push(event)
			})

			// Restore the checkpoint.
			await service.restoreCheckpoint(commit!.commit)

			// Verify the event was emitted.
			assert.strictEqual(emittedEvents.length, 1, "One restore event should be emitted")
			const eventData = emittedEvents[0]
			assert.strictEqual(eventData.type, "restore", "Event type should be restore")
			assert.strictEqual(eventData.commitHash, commit!.commit, "Commit hash should match")
			assert.strictEqual(typeof eventData.duration, "number", "Duration should be number")

			// Verify the file was actually restored.
			assert.strictEqual(await fs.readFile(testFile, "utf-8"), "Content for restore test", "File should be restored")
		})

		it("emits error event when an error occurs", async () => {
			const emittedEvents: Array<CheckpointEventMap["error"]> = []

			service.on("error", (event: CheckpointEventMap["error"]) => {
				emittedEvents.push(event)
			})

			// Force an error by providing an invalid commit hash.
			const invalidCommitHash = "invalid-commit-hash"

			// Try to restore an invalid checkpoint.
			try {
				await service.restoreCheckpoint(invalidCommitHash)
			} catch {
				// Expected to throw, we're testing the event emission.
			}

			// Verify the error event was emitted.
			assert.ok(emittedEvents.length > 0, "At least one error event should be emitted")
			const eventData = emittedEvents[0]
			assert.strictEqual(eventData.type, "error", "Event type should be error")
			assert.ok(eventData.error instanceof Error, "Error should be an Error instance")
		})

		it("supports multiple event listeners for the same event", async () => {
			const checkpointEvents1: Array<CheckpointEventMap["checkpoint"]> = []
			const checkpointEvents2: Array<CheckpointEventMap["checkpoint"]> = []

			service.on("checkpoint", (event: CheckpointEventMap["checkpoint"]) => {
				checkpointEvents1.push(event)
			})
			service.on("checkpoint", (event: CheckpointEventMap["checkpoint"]) => {
				checkpointEvents2.push(event)
			})

			await fs.writeFile(testFile, "Content for multiple listeners test")
			const result = await service.saveCheckpoint("Testing multiple listeners")

			// Verify both handlers were called with the same event data.
			assert.strictEqual(checkpointEvents1.length, 1, "First listener should receive one event")
			assert.strictEqual(checkpointEvents2.length, 1, "Second listener should receive one event")

			const eventData1 = checkpointEvents1[0]
			const eventData2 = checkpointEvents2[0]

			assert.deepStrictEqual(eventData1, eventData2, "Both listeners should receive the same event data")
			assert.strictEqual(eventData1.type, "checkpoint", "Event type should be checkpoint")
			assert.strictEqual(eventData1.toHash, result?.commit, "toHash should match commit")
		})

		it("allows removing event listeners", async () => {
			const checkpointEvents: Array<CheckpointEventMap["checkpoint"]> = []

			const listener = (event: CheckpointEventMap["checkpoint"]) => {
				checkpointEvents.push(event)
			}

			// Add the listener.
			service.on("checkpoint", listener)

			// Make a change and save a checkpoint.
			await fs.writeFile(testFile, "Content for remove listener test - part 1")
			await service.saveCheckpoint("Testing listener - part 1")

			// Verify handler was called.
			assert.strictEqual(checkpointEvents.length, 1, "Listener should receive first event")
			checkpointEvents.length = 0 // Clear the array

			// Remove the listener.
			service.off("checkpoint", listener)

			// Make another change and save a checkpoint.
			await fs.writeFile(testFile, "Content for remove listener test - part 2")
			await service.saveCheckpoint("Testing listener - part 2")

			// Verify handler was not called after being removed.
			assert.strictEqual(checkpointEvents.length, 0, "Listener should not receive events after being removed")
		})
	})
	})
}

describe("ShadowCheckpointService", () => {
	const taskId = "test-task-storage"
	const tmpDir = path.join(os.tmpdir(), "CheckpointService")
	const globalStorageDir = path.join(tmpDir, "global-storage-dir")
	const workspaceDir = path.join(tmpDir, "workspace-dir")
	const workspaceHash = ShadowCheckpointService.hashWorkspaceDir(workspaceDir)

	beforeEach(async () => {
		await fs.mkdir(globalStorageDir, { recursive: true })
		await fs.mkdir(workspaceDir, { recursive: true })
	})

	afterEach(async () => {
		await fs.rm(globalStorageDir, { recursive: true, force: true })
		await fs.rm(workspaceDir, { recursive: true, force: true })
	})

	describe("getTaskStorage", () => {
		it("returns 'task' when task repo exists", async () => {
			const service = RepoPerTaskCheckpointService.create({
				taskId,
				shadowDir: globalStorageDir,
				workspaceDir,
				log: () => {},
			})

			await service.initShadowGit()

			const storage = await ShadowCheckpointService.getTaskStorage({ taskId, globalStorageDir, workspaceDir })
			assert.strictEqual(storage, "task", "Storage should be 'task'")
		})

		it("returns 'workspace' when workspace repo exists with task branch", async () => {
			const service = RepoPerWorkspaceCheckpointService.create({
				taskId,
				shadowDir: globalStorageDir,
				workspaceDir,
				log: () => {},
			})

			await service.initShadowGit()

			const storage = await ShadowCheckpointService.getTaskStorage({ taskId, globalStorageDir, workspaceDir })
			assert.strictEqual(storage, "workspace", "Storage should be 'workspace'")
		})

		it("returns undefined when no repos exist", async () => {
			const storage = await ShadowCheckpointService.getTaskStorage({ taskId, globalStorageDir, workspaceDir })
			assert.strictEqual(storage, undefined, "Storage should be undefined")
		})

		it("returns undefined when workspace repo exists but has no task branch", async () => {
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

			assert.strictEqual(storage, undefined, "Storage should be undefined")
		})
	})
})
