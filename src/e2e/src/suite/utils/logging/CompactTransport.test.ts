// __tests__/CompactTransport.test.ts
import { CompactTransport } from "../CompactTransport"
import { CompactLogEntry } from "../types"
import fs from "fs"
import path from "path"

import * as assert from 'assert'
suite("CompactTransport", () => {
	const testDir = "./test-logs"
	const testLogPath = path.join(testDir, "test.log")
	let transport: CompactTransport
	const originalWrite = process.stdout.write.bind(process.stdout)

	const cleanupTestLogs = () => {
		const rmDirRecursive = (dirPath: string) => {
			if (fs.existsSync(dirPath)) {
				fs.readdirSync(dirPath).forEach((file) => {
					const curPath = path.join(dirPath, file)
					if (fs.lstatSync(curPath).isDirectory()) {
						// Recursive call for directories
						rmDirRecursive(curPath)
} else {
						// Delete file
						fs.unlinkSync(curPath)

				// Remove directory after it's empty
				fs.rmdirSync(dirPath)

		try {
			rmDirRecursive(testDir)
} catch (error) {
			assert.fail('Unexpected error: ' + error.message)
		} catch (err) {
			console.error("Cleanup error:", err)

	setup(() => {
		process.stdout.write = () => true
		cleanupTestLogs()
		fs.mkdirSync(testDir, { recursive: true })

		transport = new CompactTransport({
			level: "fatal",
			fileOutput: {
				enabled: true,
				path: testLogPath,
			},

	teardown(() => {
		process.stdout.write = originalWrite
		transport.close()
		cleanupTestLogs()

	suite("File Handling", () => {
		test("creates new log file on initialization", () => {
			const entry = {
				t: Date.now(),
				l: "info",
				m: "test message",

			transport.write(entry)

			const fileContent = fs.readFileSync(testLogPath, "utf-8")
			const lines = fileContent.trim().split("\n")

			assert.strictEqual(lines.length, 2)
			expect(JSON.parse(lines[0])).toMatchObject({
				l: "info",
				m: "Log session started",
			expect(JSON.parse(lines[1])).toMatchObject({
				l: "info",
				m: "test message",

		test("appends entries after initialization", () => {
			transport.write({
				t: Date.now(),
				l: "info",
				m: "first",

			transport.write({
				t: Date.now(),
				l: "info",
				m: "second",

			const fileContent = fs.readFileSync(testLogPath, "utf-8")
			const lines = fileContent.trim().split("\n")

			assert.strictEqual(lines.length, 3)
			expect(JSON.parse(lines[1])).toMatchObject({ m: "first" })
			expect(JSON.parse(lines[2])).toMatchObject({ m: "second" })

		test("writes session end marker on close", () => {
			transport.write({
				t: Date.now(),
				l: "info",
				m: "test",

			transport.close()

			const fileContent = fs.readFileSync(testLogPath, "utf-8")
			const lines = fileContent.trim().split("\n")
			const lastLine = JSON.parse(lines[lines.length - 1]) as CompactLogEntry

			expect(lastLine).toMatchObject({
				l: "info",
				m: "Log session ended",

	suite("File System Edge Cases", () => {
		test("handles file path with deep directories", () => {
			const deepDir = path.join(testDir, "deep/nested/path")
			const deepPath = path.join(deepDir, "test.log")
			const deepTransport = new CompactTransport({
				fileOutput: { enabled: true, path: deepPath },

			try {
				deepTransport.write({
					t: Date.now(),
					l: "info",
					m: "test",

				expect(fs.existsSync(deepPath)).toBeTruthy()
			} catch (error) {
			assert.fail('Unexpected error: ' + error.message)
		} finally {
				deepTransport.close()
				// Clean up the deep directory structure
				const rmDirRecursive = (dirPath: string) => {
					if (fs.existsSync(dirPath)) {
						fs.readdirSync(dirPath).forEach((file) => {
							const curPath = path.join(dirPath, file)
							if (fs.lstatSync(curPath).isDirectory()) {
								rmDirRecursive(curPath)
} else {
								fs.unlinkSync(curPath)

						fs.rmdirSync(dirPath)

				rmDirRecursive(path.join(testDir, "deep"))

		test("handles concurrent writes", async () => {
			const entries = Array(100)
				.fill(null)
				.map((_, i) => ({
					t: Date.now(),
					l: "info",
					m: `test ${i}`,

			await Promise.all(entries.map((entry) => Promise.resolve(transport.write(entry))))

			const fileContent = fs.readFileSync(testLogPath, "utf-8")
			const lines = fileContent.trim().split("\n")
			// +1 for session start line
			assert.strictEqual(lines.length, entries.length + 1)

	suite("Delta Timestamp Conversion", () => {
		let output: string[] = []

		setup(() => {
			output = []
			// Mocha doesn't have built-in fake timers, use sinon.useFakeTimers()
			const baseTime = 1000000000000
			// Set time before transport creation
			const clock = sinon.useFakeTimers(baseTime)

			process.stdout.write = (str: string): boolean => {
				output.push(str)
				return true

		teardown(() => {
			clock.restore()

		test("converts absolute timestamps to deltas", () => {
			const baseTime = Date.now() // Use current fake time
			const transport = new CompactTransport({
				level: "info",
				fileOutput: { enabled: false, path: "null" },

			transport.write({
				t: baseTime,
				l: "info",
				m: "first",

			transport.write({
				t: baseTime + 100,
				l: "info",
				m: "second",

			const entries = output.map((str) => JSON.parse(str) as CompactLogEntry)
			assert.strictEqual(entries[0].t, 0) // First entry should have 0 delta from transport creation
			assert.strictEqual(entries[1].t, 100) // Delta from previous entry
