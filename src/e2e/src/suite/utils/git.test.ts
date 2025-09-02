import { searchCommits, getCommitInfo, getWorkingState } from "../git"
import { ExecException } from "child_process"

import * as assert from 'assert'
import * as sinon from 'sinon'
type ExecFunction = (
	command: string,
	options: { cwd?: string },
	callback: (error: ExecException | null, result?: { stdout: string; stderr: string }) => void,
) => void

type PromisifiedExec = (command: string, options?: { cwd?: string }) => Promise<{ stdout: string; stderr: string }>

type MockedChildProcessModule = {
	exec: sinon.SinonStub

// Mock child_process.exec
// TODO: Mock setup needs manual migration
: MockedChildProcessModule => ({
		exec: sinon.stub() as sinon.SinonStub,
	}),

// Mock util.promisify to return our own mock function
// TODO: Mock setup needs manual migration
// TODO: Mock needs manual migration
// TODO: Implement proper mock with proxyquire

// Mock extract-text
// TODO: Mock setup needs manual migration
// TODO: Mock needs manual migration
// TODO: Implement proper mock with proxyquire

suite("git utils", () => {
	const { exec }: MockedChildProcessModule = // TODO: requireMock("child_process")
	const cwd = "/test/path"

	setup(() => {
		sinon.restore()

	suite("searchCommits", () => {
		const mockCommitData = [
			"abc123def456",
			"abc123",
			"fix: test commit",
			"John Doe",
			"2024-01-06",
			"def456abc789",
			"def456",
			"feat: new feature",
			"Jane Smith",
			"2024-01-05",
		].join("\n")

		test("should return commits when git is installed and repo exists", async () => {
			// Set up mock responses
			const responses = new Map([
				["git --version", { stdout: "git version 2.39.2", stderr: "" }],
				["git rev-parse --git-dir", { stdout: ".git", stderr: "" }],
				[
					'git log -n 10 --format="%H%n%h%n%s%n%an%n%ad" --date=short --grep="test" --regexp-ignore-case',
					{ stdout: mockCommitData, stderr: "" },
				],
			])

			exec.callsFake(
				(command: string, options: { cwd?: string }, callback: Parameters<ExecFunction>[2]) => {
					// Find matching response
					for (const [cmd, response] of responses) {
						if (command === cmd) {
							callback(null, response)
							return

					callback(new Error(`Unexpected command: ${command}`))
				},

			const result = await searchCommits("test", cwd)

			// First verify the result is correct
			assert.strictEqual(result.length, 2)
			assert.deepStrictEqual(result[0], {
				hash: "abc123def456",
				shortHash: "abc123",
				subject: "fix: test commit",
				author: "John Doe",
				date: "2024-01-06",

			// Then verify all commands were called correctly
			assert.ok(exec.calledWith("git --version", {}, expect.any(Function)))
			assert.ok(exec.calledWith("git rev-parse --git-dir", { cwd }, expect.any(Function)))
			assert.ok(exec.calledWith(
				'git log -n 10 --format="%H%n%h%n%s%n%an%n%ad" --date=short --grep="test" --regexp-ignore-case',
				{ cwd },
				expect.any(Function)),

		test("should return empty array when git is not installed", async () => {
			exec.callsFake(
				(command: string, options: { cwd?: string }, callback: Parameters<ExecFunction>[2]) => {
					if (command === "git --version") {
						callback(new Error("git not found"))
						return

					callback(new Error("Unexpected command"))
				},

			const result = await searchCommits("test", cwd)
			assert.deepStrictEqual(result, [])
			assert.ok(exec.calledWith("git --version", {}, expect.any(Function)))

		test("should return empty array when not in a git repository", async () => {
			const responses = new Map([
				["git --version", { stdout: "git version 2.39.2", stderr: "" }],
				["git rev-parse --git-dir", null], // null indicates error should be called
			])

			exec.callsFake(
				(command: string, options: { cwd?: string }, callback: Parameters<ExecFunction>[2]) => {
					const response = responses.get(command)
					if (response === null) {
						callback(new Error("not a git repository"))
					} else if (response) {
						callback(null, response)
					} else {
						callback(new Error("Unexpected command"))

				},

			const result = await searchCommits("test", cwd)
			assert.deepStrictEqual(result, [])
			assert.ok(exec.calledWith("git --version", {}, expect.any(Function)))
			assert.ok(exec.calledWith("git rev-parse --git-dir", { cwd }, expect.any(Function)))

		test("should handle hash search when grep search returns no results", async () => {
			const responses = new Map([
				["git --version", { stdout: "git version 2.39.2", stderr: "" }],
				["git rev-parse --git-dir", { stdout: ".git", stderr: "" }],
				[
					'git log -n 10 --format="%H%n%h%n%s%n%an%n%ad" --date=short --grep="abc123" --regexp-ignore-case',
					{ stdout: "", stderr: "" },
				],
				[
					'git log -n 10 --format="%H%n%h%n%s%n%an%n%ad" --date=short --author-date-order abc123',
					{ stdout: mockCommitData, stderr: "" },
				],
			])

			exec.callsFake(
				(command: string, options: { cwd?: string }, callback: Parameters<ExecFunction>[2]) => {
					for (const [cmd, response] of responses) {
						if (command === cmd) {
							callback(null, response)
							return

					callback(new Error("Unexpected command"))
				},

			const result = await searchCommits("abc123", cwd)
			assert.strictEqual(result.length, 2)
			assert.deepStrictEqual(result[0], {
				hash: "abc123def456",
				shortHash: "abc123",
				subject: "fix: test commit",
				author: "John Doe",
				date: "2024-01-06",

	suite("getCommitInfo", () => {
		const mockCommitInfo = [
			"abc123def456",
			"abc123",
			"fix: test commit",
			"John Doe",
			"2024-01-06",
			"Detailed description",
		].join("\n")
		const mockStats = "1 file changed, 2 insertions(+), 1 deletion(-)"
		const mockDiff = "@@ -1,1 +1,2 @@\n-old line\n+new line"

		test("should return formatted commit info", async () => {
			const responses = new Map([
				["git --version", { stdout: "git version 2.39.2", stderr: "" }],
				["git rev-parse --git-dir", { stdout: ".git", stderr: "" }],
				[
					'git show --format="%H%n%h%n%s%n%an%n%ad%n%b" --no-patch abc123',
					{ stdout: mockCommitInfo, stderr: "" },
				],
				['git show --stat --format="" abc123', { stdout: mockStats, stderr: "" }],
				['git show --format="" abc123', { stdout: mockDiff, stderr: "" }],
			])

			exec.callsFake(
				(command: string, options: { cwd?: string }, callback: Parameters<ExecFunction>[2]) => {
					for (const [cmd, response] of responses) {
						if (command.startsWith(cmd)) {
							callback(null, response)
							return

					callback(new Error("Unexpected command"))
				},

			const result = await getCommitInfo("abc123", cwd)
			assert.ok(result.includes("Commit: abc123"))
			assert.ok(result.includes("Author: John Doe"))
			assert.ok(result.includes("Files Changed:"))
			assert.ok(result.includes("Full Changes:"))

		test("should return error message when git is not installed", async () => {
			exec.callsFake(
				(command: string, options: { cwd?: string }, callback: Parameters<ExecFunction>[2]) => {
					if (command === "git --version") {
						callback(new Error("git not found"))
						return

					callback(new Error("Unexpected command"))
				},

			const result = await getCommitInfo("abc123", cwd)
			assert.strictEqual(result, "Git is not installed")

		test("should return error message when not in a git repository", async () => {
			const responses = new Map([
				["git --version", { stdout: "git version 2.39.2", stderr: "" }],
				["git rev-parse --git-dir", null], // null indicates error should be called
			])

			exec.callsFake(
				(command: string, options: { cwd?: string }, callback: Parameters<ExecFunction>[2]) => {
					const response = responses.get(command)
					if (response === null) {
						callback(new Error("not a git repository"))
					} else if (response) {
						callback(null, response)
					} else {
						callback(new Error("Unexpected command"))

				},

			const result = await getCommitInfo("abc123", cwd)
			assert.strictEqual(result, "Not a git repository")

	suite("getWorkingState", () => {
		const mockStatus = " M src/file1.ts\n?? src/file2.ts"
		const mockDiff = "@@ -1,1 +1,2 @@\n-old line\n+new line"

		test("should return working directory changes", async () => {
			const responses = new Map([
				["git --version", { stdout: "git version 2.39.2", stderr: "" }],
				["git rev-parse --git-dir", { stdout: ".git", stderr: "" }],
				["git status --short", { stdout: mockStatus, stderr: "" }],
				["git diff HEAD", { stdout: mockDiff, stderr: "" }],
			])

			exec.callsFake(
				(command: string, options: { cwd?: string }, callback: Parameters<ExecFunction>[2]) => {
					for (const [cmd, response] of responses) {
						if (command === cmd) {
							callback(null, response)
							return

					callback(new Error("Unexpected command"))
				},

			const result = await getWorkingState(cwd)
			assert.ok(result.includes("Working directory changes:"))
			assert.ok(result.includes("src/file1.ts"))
			assert.ok(result.includes("src/file2.ts"))

		test("should return message when working directory is clean", async () => {
			const responses = new Map([
				["git --version", { stdout: "git version 2.39.2", stderr: "" }],
				["git rev-parse --git-dir", { stdout: ".git", stderr: "" }],
				["git status --short", { stdout: "", stderr: "" }],
			])

			exec.callsFake(
				(command: string, options: { cwd?: string }, callback: Parameters<ExecFunction>[2]) => {
					for (const [cmd, response] of responses) {
						if (command === cmd) {
							callback(null, response)
							return

					callback(new Error("Unexpected command"))
				},

			const result = await getWorkingState(cwd)
			assert.strictEqual(result, "No changes in working directory")

		test("should return error message when git is not installed", async () => {
			exec.callsFake(
				(command: string, options: { cwd?: string }, callback: Parameters<ExecFunction>[2]) => {
					if (command === "git --version") {
						callback(new Error("git not found"))
						return

					callback(new Error("Unexpected command"))
				},

			const result = await getWorkingState(cwd)
			assert.strictEqual(result, "Git is not installed")

		test("should return error message when not in a git repository", async () => {
			const responses = new Map([
				["git --version", { stdout: "git version 2.39.2", stderr: "" }],
				["git rev-parse --git-dir", null], // null indicates error should be called
			])

			exec.callsFake(((
				command: string,
				_options: { cwd?: string },
				callback: Parameters<ExecFunction>[2],
			) => {
				const response = responses.get(command)
				if (response === null) {
					callback(new Error("not a git repository"))
				} else if (response) {
					callback(null, response)
				} else {
					callback(new Error("Unexpected command"))

			}) as ExecFunction)

			const result = await getWorkingState(cwd)
			assert.strictEqual(result, "Not a git repository")
