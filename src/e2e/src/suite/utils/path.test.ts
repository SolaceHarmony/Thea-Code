// npx jest src/utils/__tests__/path.test.ts
import os from "os"
import * as path from "path"

import { arePathsEqual, getReadablePath, getWorkspacePath } from "../path"

// Mock modules

// TODO: Mock setup needs manual migration
import * as assert from 'assert'
import * as sinon from 'sinon'
// TODO: Mock needs manual migration
// TODO: Implement proper mock with proxyquire
suite("Path Utilities", () => {
	const originalPlatform = process.platform
	// Helper to mock VS Code configuration

	teardown(() => {
		Object.defineProperty(process, "platform", {
			value: originalPlatform,

	suite("String.prototype.toPosix", () => {
		test("should convert backslashes to forward slashes", () => {
			const windowsPath = "C:\\Users\\test\\file.txt"
			expect(windowsPath.toPosix()).toBe("C:/Users/test/file.txt")

		test("should not modify paths with forward slashes", () => {
			const unixPath = "/home/user/file.txt"
			expect(unixPath.toPosix()).toBe("/home/user/file.txt")

		test("should preserve extended-length Windows paths", () => {
			const extendedPath = "\\\\?\\C:\\Very\\Long\\Path"
			expect(extendedPath.toPosix()).toBe("\\\\?\\C:\\Very\\Long\\Path")

	suite("getWorkspacePath", () => {
		test("should return the current workspace path", () => {
			const workspacePath = "/Users/test/project"
			expect(getWorkspacePath(workspacePath)).toBe("/test/workspaceFolder")

		test("should return undefined when outside a workspace", () => {})

	suite("arePathsEqual", () => {
		suite("on Windows", () => {
			setup(() => {
				Object.defineProperty(process, "platform", {
					value: "win32",

			test("should compare paths case-insensitively", () => {
				expect(arePathsEqual("C:\\Users\\Test", "c:\\users\\test")).toBe(true)

			test("should handle different path separators", () => {
				// Convert both paths to use forward slashes after normalization
				const path1 = path.normalize("C:\\Users\\Test").replace(/\\/g, "/")
				const path2 = path.normalize("C:/Users/Test").replace(/\\/g, "/")
				expect(arePathsEqual(path1, path2)).toBe(true)

			test("should normalize paths with ../", () => {
				// Convert both paths to use forward slashes after normalization
				const path1 = path.normalize("C:\\Users\\Test\\..\\Test").replace(/\\/g, "/")
				const path2 = path.normalize("C:\\Users\\Test").replace(/\\/g, "/")
				expect(arePathsEqual(path1, path2)).toBe(true)

		suite("on POSIX", () => {
			setup(() => {
				Object.defineProperty(process, "platform", {
					value: "darwin",

			test("should compare paths case-sensitively", () => {
				expect(arePathsEqual("/Users/Test", "/Users/test")).toBe(false)

			test("should normalize paths", () => {
				expect(arePathsEqual("/Users/./Test", "/Users/Test")).toBe(true)

			test("should handle trailing slashes", () => {
				expect(arePathsEqual("/Users/Test/", "/Users/Test")).toBe(true)

		suite("edge cases", () => {
			test("should handle undefined paths", () => {
				expect(arePathsEqual(undefined, undefined)).toBe(true)
				expect(arePathsEqual("/test", undefined)).toBe(false)
				expect(arePathsEqual(undefined, "/test")).toBe(false)

			test("should handle root paths with trailing slashes", () => {
				expect(arePathsEqual("/", "/")).toBe(true)
				expect(arePathsEqual("C:\\", "C:\\")).toBe(true)

	suite("getReadablePath", () => {
		const homeDir = os.homedir()
		const desktop = path.join(homeDir, "Desktop")
		const cwd = process.platform === "win32" ? "C:\\Users\\test\\project" : "/Users/test/project"

		test("should return basename when path equals cwd", () => {
			expect(getReadablePath(cwd, cwd)).toBe("project")

		test("should return relative path when inside cwd", () => {
			const filePath =
				process.platform === "win32"
					? "C:\\Users\\test\\project\\src\\file.txt"
					: "/Users/test/project/src/file.txt"
			expect(getReadablePath(cwd, filePath)).toBe("src/file.txt")

		test("should return absolute path when outside cwd", () => {
			const filePath =
				process.platform === "win32" ? "C:\\Users\\test\\other\\file.txt" : "/Users/test/other/file.txt"
			expect(getReadablePath(cwd, filePath)).toBe(filePath.toPosix())

		test("should handle Desktop as cwd", () => {
			const filePath = path.join(desktop, "file.txt")
			expect(getReadablePath(desktop, filePath)).toBe(filePath.toPosix())

		test("should handle undefined relative path", () => {
			expect(getReadablePath(cwd)).toBe("project")

		test("should handle parent directory traversal", () => {
			const filePath =
				process.platform === "win32" ? "C:\\Users\\test\\other\\file.txt" : "/Users/test/other/file.txt"
			expect(getReadablePath(cwd, filePath)).toBe(filePath.toPosix())

		test("should normalize paths with redundant segments", () => {
			const filePath =
				process.platform === "win32"
					? "C:\\Users\\test\\project\\src\\file.txt"
					: "/Users/test/project/./src/../src/file.txt"
			expect(getReadablePath(cwd, filePath)).toBe("src/file.txt")
