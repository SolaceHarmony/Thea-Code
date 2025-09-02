import fs from "fs"
import * as readline from "readline"
import { countFileLines } from "../line-counter"

// Mock the fs module
// TODO: Mock setup needs manual migration
import * as assert from 'assert'
import * as sinon from 'sinon'
// TODO: Fix mock - needs proxyquire
/*
=> {
	const originalModule: typeof import("fs") = // TODO: requireActual("fs")
	return {
		...originalModule,
		createReadStream: sinon.stub(),
		promises: {
			access: sinon.stub(),
		},
	} as typeof import("fs")
})*/

// Mock readline
// TODO: Mock setup needs manual migration
// TODO: Mock needs manual migration
// TODO: Implement proper mock with proxyquire

suite("countFileLines", () => {
	setup(() => {
		sinon.restore()

	test("should throw error if file does not exist", async () => {
		// Setup
		;(fs.promises.access as sinon.SinonStub).rejectsOnce(new Error("File not found"))

		// Test & Assert
		await expect(countFileLines("non-existent-file.txt")).rejects.toThrow("File not found")

	test("should return the correct line count for a file", async () => {
		// Setup
		;(fs.promises.access as sinon.SinonStub).resolvesOnce(undefined)

		const mockEventEmitter = {
			on: sinon.stub().callsFake(function (
				this: { mockLines?: number },
				event: string,
				callback: () => void,
			) {
				if (event === "line") {
					for (let i = 0; i < 10; i++) {
						callback()

				if (event === "close") {
					callback()

				return this
			}),

		const mockReadStream = {
			on: sinon.stub().callsFake(function (this: unknown) {
				return this
			}),

		;(fs.createReadStream as sinon.SinonStub).returnsOnce(mockReadStream)
		const readlineMock = jest.mocked(readline)
		readlineMock.createInterface.returnsOnce(mockEventEmitter as unknown as readline.Interface)

		// Test
		const result = await countFileLines("test-file.txt")

		// Assert
		assert.strictEqual(result, 10)
		assert.ok(fs.promises.access.calledWith("test-file.txt", fs.constants.F_OK))
		assert.ok(fs.createReadStream.calledWith("test-file.txt"))

	test("should handle files with no lines", async () => {
		// Setup
		;(fs.promises.access as sinon.SinonStub).resolvesOnce(undefined)

		const mockEventEmitter = {
			on: sinon.stub().callsFake(function (
				this: { mockLines?: number },
				event: string,
				callback: () => void,
			) {
				if (event === "close") {
					callback()

				return this
			}),

		const mockReadStream = {
			on: sinon.stub().callsFake(function (this: unknown) {
				return this
			}),

		;(fs.createReadStream as sinon.SinonStub).returnsOnce(mockReadStream)
		const readlineMock = jest.mocked(readline)
		readlineMock.createInterface.returnsOnce(mockEventEmitter as unknown as readline.Interface)

		// Test
		const result = await countFileLines("empty-file.txt")

		// Assert
		assert.strictEqual(result, 0)

	test("should handle errors during reading", async () => {
		// Setup
		;(fs.promises.access as sinon.SinonStub).resolvesOnce(undefined)

		const mockEventEmitter = {
			on: sinon.stub().callsFake(function (
				this: { mockLines?: number },
				event: string,
				callback: (err?: Error) => void,
			) {
				if (event === "error") {
					callback(new Error("Read error"))

				return this
			}),

		const mockReadStream = {
			on: sinon.stub().callsFake(function (this: unknown) {
				return this
			}),

		;(fs.createReadStream as sinon.SinonStub).returnsOnce(mockReadStream)
		const readlineMock = jest.mocked(readline)
		readlineMock.createInterface.returnsOnce(mockEventEmitter as unknown as readline.Interface)

		// Test & Assert
		await expect(countFileLines("error-file.txt")).rejects.toThrow("Read error")
