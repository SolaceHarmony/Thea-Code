import * as assert from 'assert'
import * as sinon from 'sinon'
import fs from "fs/promises"
import { join } from "path"

import { fileExistsAtPath } from "../../../utils/fs"

import { getExcludePatterns } from "../excludes"
import { GIT_DISABLED_SUFFIX } from "../constants" // Fixed: removed type-only import

// TODO: Mock setup needs manual migration for "fs/promises"

// TODO: Mock setup needs manual migration for "../../../utils/fs"

suite("getExcludePatterns", () => {
	const mockedFs = fs as sinon.SinonStubbedInstance<typeof fs>
	const mockedFileExistsAtPath = fileExistsAtPath as sinon.SinonStubbedInstanceFunction<typeof fileExistsAtPath>
	const testWorkspacePath = "/test/workspace"

	setup(() => {
		sinon.restore()
	})

	suite("getLfsPatterns", () => {
		test("should include LFS patterns from .gitattributes when they exist", async () => {
			// Mock .gitattributes file exists
			mockedFileExistsAtPath.resolves(true)

			// Mock .gitattributes file content with LFS patterns
			const gitAttributesContent = `*.psd filter=lfs diff=lfs merge=lfs -text
*.zip filter=lfs diff=lfs merge=lfs -text
# A comment line
*.mp4 filter=lfs diff=lfs merge=lfs -text
readme.md text
`
			mockedFs.readFile.resolves(gitAttributesContent)

			// Expected LFS patterns
			const expectedLfsPatterns = ["*.psd", "*.zip", "*.mp4"]

			// Get exclude patterns
			const excludePatterns = await getExcludePatterns(testWorkspacePath)

			// Verify .gitattributes was checked at the correct path
			assert.ok(mockedFileExistsAtPath.calledWith(join(testWorkspacePath, ".gitattributes")))

			// Verify file was read
			assert.ok(mockedFs.readFile.calledWith(join(testWorkspacePath, ".gitattributes")), "utf8")

			// Verify LFS patterns are included in result
			expectedLfsPatterns.forEach((pattern) => {
				assert.ok(excludePatterns.includes(pattern))
			})

			// Verify all normal patterns also exist
			assert.ok(excludePatterns.includes(".git/"))
			assert.ok(excludePatterns.includes(`.git${GIT_DISABLED_SUFFIX}/`))
		})

		test("should handle .gitattributes with no LFS patterns", async () => {
			// Mock .gitattributes file exists
			mockedFileExistsAtPath.resolves(true)

			// Mock .gitattributes file content with no LFS patterns
			const gitAttributesContent = `*.md text
*.txt text
*.js text eol=lf
`
			mockedFs.readFile.resolves(gitAttributesContent)

			// Get exclude patterns
			const excludePatterns = await getExcludePatterns(testWorkspacePath)

			// Verify .gitattributes was checked
			assert.ok(mockedFileExistsAtPath.calledWith(join(testWorkspacePath, ".gitattributes")))

			// Verify file was read
			assert.ok(mockedFs.readFile.calledWith(join(testWorkspacePath, ".gitattributes")), "utf8")

			// Verify LFS patterns are not included
			// Just ensure no lines from our mock gitAttributes are in the result
			const gitAttributesLines = gitAttributesContent.split("\n").map((line) => line.split(" ")[0].trim())

			gitAttributesLines.forEach((line) => {
				if (line && !line.startsWith("#")) {
					expect(excludePatterns.includes(line)).toBe(false)
				}
			})

			// Verify default patterns are included
			assert.ok(excludePatterns.includes(".git/"))
			assert.ok(excludePatterns.includes(`.git${GIT_DISABLED_SUFFIX}/`))
		})

		test("should handle missing .gitattributes file", async () => {
			// Mock .gitattributes file doesn't exist
			mockedFileExistsAtPath.resolves(false)

			// Get exclude patterns
			const excludePatterns = await getExcludePatterns(testWorkspacePath)

			// Verify .gitattributes was checked
			assert.ok(mockedFileExistsAtPath.calledWith(join(testWorkspacePath, ".gitattributes")))

			// Verify file was not read
			assert.ok(!mockedFs.readFile.called)

			// Verify standard patterns are included
			assert.ok(excludePatterns.includes(".git/"))
			assert.ok(excludePatterns.includes(`.git${GIT_DISABLED_SUFFIX}/`))

			// Verify we have standard patterns but no LFS patterns
			// Check for a few known patterns from different categories
			assert.ok(excludePatterns.includes("node_modules/")) // buildArtifact
			assert.ok(excludePatterns.includes("*.jpg")) // media
			assert.ok(excludePatterns.includes("*.tmp")) // cache
			assert.ok(excludePatterns.includes("*.env*")) // config
			assert.ok(excludePatterns.includes("*.zip")) // large data
			assert.ok(excludePatterns.includes("*.db")) // database
			assert.ok(excludePatterns.includes("*.shp")) // geospatial
			assert.ok(excludePatterns.includes("*.log")) // log
		})

		test("should handle errors when reading .gitattributes", async () => {
			// Mock .gitattributes file exists
			mockedFileExistsAtPath.resolves(true)

			// Mock readFile to throw error
			mockedFs.readFile.rejects(new Error("File read error"))

			// Get exclude patterns
			const excludePatterns = await getExcludePatterns(testWorkspacePath)

			// Verify .gitattributes was checked
			assert.ok(mockedFileExistsAtPath.calledWith(join(testWorkspacePath, ".gitattributes")))

			// Verify file read was attempted
			assert.ok(mockedFs.readFile.calledWith(join(testWorkspacePath, ".gitattributes")), "utf8")

			// Verify standard patterns are included
			assert.ok(excludePatterns.includes(".git/"))
			assert.ok(excludePatterns.includes(`.git${GIT_DISABLED_SUFFIX}/`))

			// Verify we have standard patterns but no LFS patterns
			// Check for a few known patterns from different categories
			assert.ok(excludePatterns.includes("node_modules/")) // buildArtifact
			assert.ok(excludePatterns.includes("*.jpg")) // media
			assert.ok(excludePatterns.includes("*.tmp")) // cache
			assert.ok(excludePatterns.includes("*.env*")) // config
			assert.ok(excludePatterns.includes("*.zip")) // large data
			assert.ok(excludePatterns.includes("*.db")) // database
			assert.ok(excludePatterns.includes("*.shp")) // geospatial
			assert.ok(excludePatterns.includes("*.log")) // log
		})
	})
// Mock cleanup
