import * as assert from "assert"
import * as sinon from "sinon"
import * as fs from "fs/promises"
import { join } from "path"

// Import the functions we're testing  
import { getExcludePatterns } from "../../../../services/checkpoints/excludes"
import { GIT_DISABLED_SUFFIX } from "../../../../services/checkpoints/constants"
import * as fsUtils from "../../../../utils/fs"

suite("Checkpoint Service - Exclude Patterns", () => {
	let sandbox: sinon.SinonSandbox
	const testWorkspacePath = "/test/workspace"

	setup(() => {
		sandbox = sinon.createSandbox()

	teardown(() => {
		sandbox.restore()

	suite("getExcludePatterns", () => {
		suite("getLfsPatterns", () => {
			test("Should include LFS patterns from .gitattributes when they exist", async () => {
				// Mock .gitattributes file exists
				const fileExistsStub = sandbox.stub(fsUtils, "fileExistsAtPath")
				fileExistsStub.resolves(true)

				// Mock .gitattributes file content with LFS patterns
				const gitAttributesContent = `*.psd filter=lfs diff=lfs merge=lfs -text
*.zip filter=lfs diff=lfs merge=lfs -text
# A comment line
*.mp4 filter=lfs diff=lfs merge=lfs -text
readme.md text
`
				const readFileStub = sandbox.stub(fs, "readFile")
				readFileStub.resolves(gitAttributesContent)

				// Expected LFS patterns
				const expectedLfsPatterns = ["*.psd", "*.zip", "*.mp4"]

				// Get exclude patterns
				const excludePatterns = await getExcludePatterns(testWorkspacePath)

				// Verify .gitattributes was checked at the correct path
				assert.ok(
					fileExistsStub.calledWith(join(testWorkspacePath, ".gitattributes")),
					"Should check for .gitattributes file"

				// Verify file was read
				assert.ok(
					readFileStub.calledWith(join(testWorkspacePath, ".gitattributes"), "utf8"),
					"Should read .gitattributes file"

				// Verify LFS patterns are included in result
				expectedLfsPatterns.forEach((pattern) => {
					assert.ok(
						excludePatterns.includes(pattern),
						`Should include LFS pattern: ${pattern}`

				// Verify all normal patterns also exist
				assert.ok(excludePatterns.includes(".git/"), "Should include .git/ pattern")
				assert.ok(
					excludePatterns.includes(`.git${GIT_DISABLED_SUFFIX}/`),
					"Should include disabled git pattern"

			test("Should handle .gitattributes with no LFS patterns", async () => {
				// Mock .gitattributes file exists
				const fileExistsStub = sandbox.stub(fsUtils, "fileExistsAtPath")
				fileExistsStub.resolves(true)

				// Mock .gitattributes file content with no LFS patterns
				const gitAttributesContent = `*.md text
*.txt text
*.js text eol=lf
`
				const readFileStub = sandbox.stub(fs, "readFile")
				readFileStub.resolves(gitAttributesContent)

				// Get exclude patterns
				const excludePatterns = await getExcludePatterns(testWorkspacePath)

				// Verify .gitattributes was checked
				assert.ok(
					fileExistsStub.calledWith(join(testWorkspacePath, ".gitattributes")),
					"Should check for .gitattributes file"

				// Verify file was read
				assert.ok(
					readFileStub.calledWith(join(testWorkspacePath, ".gitattributes"), "utf8"),
					"Should read .gitattributes file"

				// Verify LFS patterns are not included
				// Just ensure no lines from our mock gitAttributes are in the result
				const gitAttributesLines = gitAttributesContent
					.split("\n")
					.map((line) => line.split(" ")[0].trim())

				gitAttributesLines.forEach((line) => {
					if (line && !line.startsWith("#")) {
						assert.strictEqual(
							excludePatterns.includes(line),
							false,
							`Should not include non-LFS pattern: ${line}`

				// Verify default patterns are included
				assert.ok(excludePatterns.includes(".git/"), "Should include .git/ pattern")
				assert.ok(
					excludePatterns.includes(`.git${GIT_DISABLED_SUFFIX}/`),
					"Should include disabled git pattern"

			test("Should handle missing .gitattributes file", async () => {
				// Mock .gitattributes file doesn't exist
				const fileExistsStub = sandbox.stub(fsUtils, "fileExistsAtPath")
				fileExistsStub.resolves(false)

				const readFileStub = sandbox.stub(fs, "readFile")

				// Get exclude patterns
				const excludePatterns = await getExcludePatterns(testWorkspacePath)

				// Verify .gitattributes was checked
				assert.ok(
					fileExistsStub.calledWith(join(testWorkspacePath, ".gitattributes")),
					"Should check for .gitattributes file"

				// Verify file was not read
				assert.ok(
					readFileStub.notCalled,
					"Should not attempt to read non-existent file"

				// Verify standard patterns are included
				assert.ok(excludePatterns.includes(".git/"), "Should include .git/ pattern")
				assert.ok(
					excludePatterns.includes(`.git${GIT_DISABLED_SUFFIX}/`),
					"Should include disabled git pattern"

				// Verify we have standard patterns but no LFS patterns
				// Check for a few known patterns from different categories
				assert.ok(excludePatterns.includes("node_modules/"), "Should include node_modules/")
				assert.ok(excludePatterns.includes("*.jpg"), "Should include *.jpg")
				assert.ok(excludePatterns.includes("*.tmp"), "Should include *.tmp")
				assert.ok(excludePatterns.includes("*.env*"), "Should include *.env*")
				assert.ok(excludePatterns.includes("*.zip"), "Should include *.zip")
				assert.ok(excludePatterns.includes("*.db"), "Should include *.db")
				assert.ok(excludePatterns.includes("*.shp"), "Should include *.shp")
				assert.ok(excludePatterns.includes("*.log"), "Should include *.log")

			test("Should handle errors when reading .gitattributes", async () => {
				// Mock .gitattributes file exists
				const fileExistsStub = sandbox.stub(fsUtils, "fileExistsAtPath")
				fileExistsStub.resolves(true)

				// Mock readFile to throw error
				const readFileStub = sandbox.stub(fs, "readFile")
				readFileStub.rejects(new Error("File read error"))

				// Get exclude patterns
				const excludePatterns = await getExcludePatterns(testWorkspacePath)

				// Verify .gitattributes was checked
				assert.ok(
					fileExistsStub.calledWith(join(testWorkspacePath, ".gitattributes")),
					"Should check for .gitattributes file"

				// Verify file read was attempted
				assert.ok(
					readFileStub.calledWith(join(testWorkspacePath, ".gitattributes"), "utf8"),
					"Should attempt to read .gitattributes file"

				// Verify standard patterns are included despite error
				assert.ok(excludePatterns.includes(".git/"), "Should include .git/ pattern")
				assert.ok(
					excludePatterns.includes(`.git${GIT_DISABLED_SUFFIX}/`),
					"Should include disabled git pattern"

				// Verify we have standard patterns but no LFS patterns
				// Check for a few known patterns from different categories
				assert.ok(excludePatterns.includes("node_modules/"), "Should include node_modules/")
				assert.ok(excludePatterns.includes("*.jpg"), "Should include *.jpg")
				assert.ok(excludePatterns.includes("*.tmp"), "Should include *.tmp")
				assert.ok(excludePatterns.includes("*.env*"), "Should include *.env*")
				assert.ok(excludePatterns.includes("*.zip"), "Should include *.zip")
				assert.ok(excludePatterns.includes("*.db"), "Should include *.db")
				assert.ok(excludePatterns.includes("*.shp"), "Should include *.shp")
				assert.ok(excludePatterns.includes("*.log"), "Should include *.log")
