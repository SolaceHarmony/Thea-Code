import * as assert from 'assert'
import * as sinon from 'sinon'
import * as proxyquire from 'proxyquire'

suite("Git Utilities", () => {
	let sandbox: sinon.SinonSandbox
	let execStub: sinon.SinonStub
	let gitModule: any
	let truncateOutputStub: sinon.SinonStub

	setup(() => {
		sandbox = sinon.createSandbox()
		
		// Create the exec stub that mimics child_process.exec behavior
		execStub = sandbox.stub()
		
		// Create truncateOutput stub
		truncateOutputStub = sandbox.stub().callsFake((text) => text)
		
		// Load the git module with mocked dependencies
		gitModule = proxyquire('../../../src/utils/git', {
			'child_process': {
				exec: execStub
			},
			'util': {
				promisify: (fn: any) => {
					// Return a promisified version that uses our stub
					return async (command: string, options?: { cwd?: string }) => {
						return new Promise((resolve, reject) => {
							// Call the stub with command, options, and callback
							execStub(command, options || {}, (error: any, stdout: string, stderr: string) => {
								if (error) {
									reject(error)
} else {
									resolve({ stdout, stderr })
								}
							})
						})
					}
				}
			},
			'../integrations/misc/extract-text': {
				truncateOutput: truncateOutputStub
			}
		})
	})

	teardown(() => {
		sandbox.restore()
	})

	suite("searchCommits", () => {
		test("returns commits when git is installed and in a repo", async () => {
			// Setup stubs for git checks
			execStub.withArgs("git --version", {}).callsArgWith(2, null, "git version 2.34.1", "")
			execStub.withArgs("git rev-parse --git-dir", { cwd: "/test/repo" }).callsArgWith(2, null, ".git", "")
			
			// Setup stub for git log command
			const gitLogOutput = `abc123456789def
abc123
feat: Add new feature
John Doe
2024-01-15
def987654321abc
def987
fix: Fix bug
Jane Smith
2024-01-14`
			
			execStub.withArgs(
				sinon.match(/git log -n 10 --format="%H%n%h%n%s%n%an%n%ad"/),
				{ cwd: "/test/repo" }
			).callsArgWith(2, null, gitLogOutput, "")

			const commits = await gitModule.searchCommits("feature", "/test/repo")

			assert.strictEqual(commits.length, 2)
			assert.strictEqual(commits[0].hash, "abc123456789def")
			assert.strictEqual(commits[0].shortHash, "abc123")
			assert.strictEqual(commits[0].subject, "feat: Add new feature")
			assert.strictEqual(commits[0].author, "John Doe")
			assert.strictEqual(commits[0].date, "2024-01-15")
		})

		test("throws error when git is not installed", async () => {
			execStub.withArgs("git --version", {}).callsArgWith(2, new Error("Command not found"), null, null)

			try {
				await gitModule.searchCommits("query", "/test/repo")
				assert.fail("Should have thrown an error")
} catch (error) {
				assert.ok(error instanceof Error)
				assert.strictEqual(error.message, "Git is not installed.")
			}
		})

		test("throws error when not in a git repository", async () => {
			execStub.withArgs("git --version", {}).callsArgWith(2, null, "git version 2.34.1", "")
			execStub.withArgs("git rev-parse --git-dir", { cwd: "/test/repo" }).callsArgWith(2, new Error("Not a git repository"), null, null)

			try {
				await gitModule.searchCommits("query", "/test/repo")
				assert.fail("Should have thrown an error")
} catch (error) {
				assert.ok(error instanceof Error)
				assert.strictEqual(error.message, "Not a git repository.")
			}
		})

		test("searches by hash when query looks like a hash and grep returns nothing", async () => {
			// Setup git checks
			execStub.withArgs("git --version", {}).callsArgWith(2, null, "git version 2.34.1", "")
			execStub.withArgs("git rev-parse --git-dir", { cwd: "/test/repo" }).callsArgWith(2, null, ".git", "")
			
			// First search by grep returns empty
			execStub.withArgs(
				sinon.match(/--grep=/),
				{ cwd: "/test/repo" }
			).callsArgWith(2, null, "", "")
			
			// Second search by hash returns commit
			const hashSearchOutput = `abc123456789def
abc123
feat: Specific commit
John Doe
2024-01-15`
			
			execStub.withArgs(
				sinon.match(/--author-date-order abc123/),
				{ cwd: "/test/repo" }
			).callsArgWith(2, null, hashSearchOutput, "")

			const commits = await gitModule.searchCommits("abc123", "/test/repo")

			assert.strictEqual(commits.length, 1)
			assert.strictEqual(commits[0].hash, "abc123456789def")
		})

		test("returns empty array when no commits found", async () => {
			execStub.withArgs("git --version", {}).callsArgWith(2, null, "git version 2.34.1", "")
			execStub.withArgs("git rev-parse --git-dir", { cwd: "/test/repo" }).callsArgWith(2, null, ".git", "")
			execStub.withArgs(
				sinon.match(/git log/),
				{ cwd: "/test/repo" }
			).callsArgWith(2, null, "", "")

			const commits = await gitModule.searchCommits("nonexistent", "/test/repo")

			assert.strictEqual(commits.length, 0)
		})
	})

	suite("getCommitInfo", () => {
		test("returns commit info with diff", async () => {
			execStub.withArgs("git --version", {}).callsArgWith(2, null, "git version 2.34.1", "")
			execStub.withArgs("git rev-parse --git-dir", { cwd: "/test/repo" }).callsArgWith(2, null, ".git", "")
			
			const commitInfo = `commit abc123456789def
Author: John Doe <john@example.com>
Date:   Mon Jan 15 10:00:00 2024 +0000

    feat: Add new feature
    
    This is a longer description

diff --git a/file.js b/file.js
index 123..456 100644
--- a/file.js
+++ b/file.js
@@ -1,3 +1,3 @@
-old line
+new line`

			execStub.withArgs(
				"git show --stat --patch abc123",
				{ cwd: "/test/repo" }
			).callsArgWith(2, null, commitInfo, "")

			const result = await gitModule.getCommitInfo("abc123", "/test/repo")

			assert.ok(result.includes("feat: Add new feature"))
			assert.ok(result.includes("John Doe"))
			assert.ok(result.includes("diff --git"))
		})

		test("throws error when git is not installed", async () => {
			execStub.withArgs("git --version", {}).callsArgWith(2, new Error("Command not found"), null, null)

			try {
				await gitModule.getCommitInfo("abc123", "/test/repo")
				assert.fail("Should have thrown an error")
} catch (error) {
				assert.ok(error instanceof Error)
				assert.strictEqual(error.message, "Git is not installed.")
			}
		})

		test("throws error when commit not found", async () => {
			execStub.withArgs("git --version", {}).callsArgWith(2, null, "git version 2.34.1", "")
			execStub.withArgs("git rev-parse --git-dir", { cwd: "/test/repo" }).callsArgWith(2, null, ".git", "")
			execStub.withArgs(
				"git show --stat --patch nonexistent",
				{ cwd: "/test/repo" }
			).callsArgWith(2, new Error("fatal: bad object nonexistent"), null, null)

			try {
				await gitModule.getCommitInfo("nonexistent", "/test/repo")
				assert.fail("Should have thrown an error")
} catch (error) {
				assert.ok(error instanceof Error)
				assert.ok(error.message.includes("bad object"))
			}
		})

		test("truncates large output", async () => {
			execStub.withArgs("git --version", {}).callsArgWith(2, null, "git version 2.34.1", "")
			execStub.withArgs("git rev-parse --git-dir", { cwd: "/test/repo" }).callsArgWith(2, null, ".git", "")
			
			const largeOutput = "Large commit info\n" + "Line\n".repeat(1000)
			truncateOutputStub.returns("Truncated output")
			
			execStub.withArgs(
				"git show --stat --patch abc123",
				{ cwd: "/test/repo" }
			).callsArgWith(2, null, largeOutput, "")

			const result = await gitModule.getCommitInfo("abc123", "/test/repo")

			assert.strictEqual(result, "Truncated output")
			assert.ok(truncateOutputStub.calledWith(largeOutput, 500))
		})
	})

	suite("getWorkingState", () => {
		test("returns working state with staged and unstaged changes", async () => {
			execStub.withArgs("git --version", {}).callsArgWith(2, null, "git version 2.34.1", "")
			execStub.withArgs("git rev-parse --git-dir", { cwd: "/test/repo" }).callsArgWith(2, null, ".git", "")
			
			const statusOutput = `M  src/file1.js
 M src/file2.js
MM src/file3.js
?? new-file.txt`

			const diffOutput = `diff --git a/src/file1.js b/src/file1.js
index 123..456 100644
--- a/src/file1.js
+++ b/src/file1.js
@@ -1,3 +1,3 @@
-old line
+new line`

			execStub.withArgs("git status --short", { cwd: "/test/repo" }).callsArgWith(2, null, statusOutput, "")
			execStub.withArgs("git diff --cached", { cwd: "/test/repo" }).callsArgWith(2, null, diffOutput, "")
			execStub.withArgs("git diff", { cwd: "/test/repo" }).callsArgWith(2, null, "unstaged diff", "")

			const result = await gitModule.getWorkingState("/test/repo")

			assert.ok(result.includes("Working Directory Status"))
			assert.ok(result.includes("M  src/file1.js"))
			assert.ok(result.includes("Staged Changes"))
			assert.ok(result.includes("diff --git"))
			assert.ok(result.includes("Unstaged Changes"))
			assert.ok(result.includes("unstaged diff"))
		})

		test("returns clean state message when no changes", async () => {
			execStub.withArgs("git --version", {}).callsArgWith(2, null, "git version 2.34.1", "")
			execStub.withArgs("git rev-parse --git-dir", { cwd: "/test/repo" }).callsArgWith(2, null, ".git", "")
			execStub.withArgs("git status --short", { cwd: "/test/repo" }).callsArgWith(2, null, "", "")
			execStub.withArgs("git diff --cached", { cwd: "/test/repo" }).callsArgWith(2, null, "", "")
			execStub.withArgs("git diff", { cwd: "/test/repo" }).callsArgWith(2, null, "", "")

			const result = await gitModule.getWorkingState("/test/repo")

			assert.ok(result.includes("Working directory is clean"))
		})

		test("handles non-git directory gracefully", async () => {
			execStub.withArgs("git --version", {}).callsArgWith(2, null, "git version 2.34.1", "")
			execStub.withArgs("git rev-parse --git-dir", { cwd: "/test/repo" }).callsArgWith(2, new Error("Not a git repository"), null, null)

			const result = await gitModule.getWorkingState("/test/repo")

			assert.ok(result.includes("Not a git repository"))
		})

		test("truncates large diffs", async () => {
			execStub.withArgs("git --version", {}).callsArgWith(2, null, "git version 2.34.1", "")
			execStub.withArgs("git rev-parse --git-dir", { cwd: "/test/repo" }).callsArgWith(2, null, ".git", "")
			execStub.withArgs("git status --short", { cwd: "/test/repo" }).callsArgWith(2, null, "M file.js", "")
			
			const largeDiff = "diff\n" + "Line\n".repeat(1000)
			truncateOutputStub.withArgs(largeDiff, 500).returns("Truncated diff")
			
			execStub.withArgs("git diff --cached", { cwd: "/test/repo" }).callsArgWith(2, null, largeDiff, "")
			execStub.withArgs("git diff", { cwd: "/test/repo" }).callsArgWith(2, null, "", "")

			const result = await gitModule.getWorkingState("/test/repo")

			assert.ok(result.includes("Truncated diff"))
		})
	})
// Mock cleanup
