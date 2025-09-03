// TODO: test.each patterns need manual migration
import * as assert from 'assert'
import * as sinon from 'sinon'
/* eslint-disable @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-member-access */
import { OllamaHandler } from "../ollama"
import { convertToOllamaHistory, convertToOllamaContentBlocks } from "../../transform/neutral-ollama-format"
import { NeutralConversationHistory, NeutralMessageContent } from "../../../shared/neutral-history"
import { XmlMatcher } from "../../../utils/xml-matcher"
import type { ApiStreamChunk } from "../../transform/stream"
import { Readable } from "stream"

// Mock the transform functions
// TODO: Mock setup needs manual migration
// TODO: Mock needs manual migration
// TODO: Implement proper mock with proxyquire

// TODO: Replace nock with fetch mocks as per architectural guidelines
// eslint-disable-next-line @typescript-eslint/no-require-imports
const nock = require("nock")

// Mock the XmlMatcher
// TODO: Mock setup needs manual migration
// TODO: Fix mock - needs proxyquire
/*
=> {
	return {
		XmlMatcher: sinon.stub().callsFake(() => ({
			update: sinon.stub().callsFake((text: string) => {
				if (text.includes("<think>")) {
					return [{ type: "reasoning", text: text.replace(/<\/?think>/g, "") }]

				return [{ type: "text", text }]
			}),
			final: sinon.stub().returns([]),
		})),

})*/

suite("OllamaHandler", () => {
	let handler: OllamaHandler
	// Define availableModels as a const directly for .each
	const availableModels: string[] = ["llama2", "mistral", "gemma"]

	setup(() => {
		sinon.restore()
		nock.cleanAll()

		// Create handler with mock options
		handler = new OllamaHandler({
			ollamaBaseUrl: "http://localhost:10000",
			ollamaModelId: "llama2", // Default model for tests

	teardown(() => {
		nock.cleanAll()

	suite("createMessage", () => {
		it.each(availableModels)(
			"should use convertToOllamaHistory to convert messages with %s model",
			async (modelId) => {
				// Update handler to use the current model
				handler = new OllamaHandler({
					ollamaBaseUrl: "http://localhost:10000",
					ollamaModelId: modelId,

				// Mock implementation
				;(convertToOllamaHistory as sinon.SinonStub).returns([{ role: "user", content: "Hello" }])

				// Create neutral history
				const neutralHistory: NeutralConversationHistory = [
					{ role: "user", content: [{ type: "text", text: "Hello" }] },

				let requestBody: Record<string, unknown> | undefined
				nock("http://localhost:10000")
					.post("/v1/chat/completions")
					.reply(function (_uri: string, body: unknown) {
						requestBody = body as Record<string, unknown>
						const deltas = [
							{ content: "Hello" },
							{ content: " world" },
							{ content: "<think>This is reasoning</think>" },

						const stream = new Readable({ read() {} })
						for (const delta of deltas) {
							const chunk = {
								id: "id",
								object: "chat.completion.chunk",
								created: 0,
								model: "test",
								choices: [{ delta, index: 0, finish_reason: null }],

							stream.push(`data: ${JSON.stringify(chunk)}\n\n`)

						stream.push("data: [DONE]\n\n")
						stream.push(null)
						return [200, stream]

				const stream = handler.createMessage("You are helpful.", neutralHistory)

				// Collect stream chunks
				const chunks: ApiStreamChunk[] = []
				for await (const chunk of stream) {
					chunks.push(chunk)

				// Verify transform function was called
				assert.ok(convertToOllamaHistory.calledWith(neutralHistory))

				const rb = requestBody as {
					model: string
					messages: unknown
					temperature: number
					stream: boolean

				assert.deepStrictEqual(rb, 
					sinon.match({
						model: modelId,
						messages: expect.arrayContaining([
							{ role: "system", content: "You are helpful." },
							{ role: "user", content: "Hello" },
						]) as unknown,
						temperature: 0,
						stream: true,
					}),

				// Verify stream chunks
				assert.ok(chunks.some(item => JSON.stringify(item) === JSON.stringify({ type: "text", text: "Hello" })))
				assert.ok(chunks.some(item => JSON.stringify(item) === JSON.stringify({ type: "text", text: " world" })))
				assert.ok(chunks.some(item => JSON.stringify(item) === JSON.stringify({ type: "reasoning", text: "This is reasoning" })))
			},

		it.each(availableModels)(
			"should not add system prompt if already included in messages with %s model",
			async (modelId) => {
				// Update handler to use the current model
				handler = new OllamaHandler({
					ollamaBaseUrl: "http://localhost:10000",
					ollamaModelId: modelId,

				// Mock implementation with system message already included
				;(convertToOllamaHistory as sinon.SinonStub).returns([
					{ role: "system", content: "Existing system prompt" },
					{ role: "user", content: "Hello" },
				])			// Create neutral history
			const neutralHistory: NeutralConversationHistory = [
				{ role: "system", content: [{ type: "text", text: "Existing system prompt" }] },
				{ role: "user", content: [{ type: "text", text: "Hello" }] },

			let requestBody: Record<string, unknown> | undefined
			nock("http://localhost:10000")
				.post("/v1/chat/completions")
				.reply((_uri: string, body: unknown) => {
					requestBody = body as Record<string, unknown>
					const stream = new Readable({ read() {} })
					const deltas = [{ content: "Hello" }, { content: " world" }]
					for (const d of deltas) {
						const chunk = {
							id: "id",
							object: "chat.completion.chunk",
							created: 0,
							model: "test",
							choices: [{ delta: d, index: 0, finish_reason: null }],

						stream.push(`data: ${JSON.stringify(chunk)}\n\n`)

					stream.push("data: [DONE]\n\n")
					stream.push(null)
					return [200, stream]

				const stream = handler.createMessage("You are helpful.", neutralHistory)

				// Collect stream chunks (just to complete the generator)
				// eslint-disable-next-line @typescript-eslint/no-unused-vars
				for await (const _chunk of stream) {
					// Do nothing - we're just consuming the stream

				assert.deepStrictEqual(requestBody, {
					model: modelId,
					messages: [
						{ role: "system", content: "Existing system prompt" },
						{ role: "user", content: "Hello" },
					],
					temperature: 0,
					stream: true,

			},

		it.each(availableModels)("should handle empty system prompt with %s model", async (modelId) => {
			// Update handler to use the current model
			handler = new OllamaHandler({
				ollamaBaseUrl: "http://localhost:10000",
				ollamaModelId: modelId,

			// Mock implementation
			;(convertToOllamaHistory as sinon.SinonStub).returns([{ role: "user", content: "Hello" }])

			// Create neutral history
			const neutralHistory: NeutralConversationHistory = [
				{ role: "user", content: [{ type: "text", text: "Hello" }] },

			let requestBody: Record<string, unknown> | undefined
			nock("http://localhost:10000")
				.post("/v1/chat/completions")
				.reply(function (_uri: string, body: unknown) {
					requestBody = body as Record<string, unknown>
					const stream = new Readable({ read() {} })
					stream.push("data: [DONE]\n\n")
					stream.push(null)
					return [200, stream]

			const stream = handler.createMessage("", neutralHistory)

			// Collect stream chunks (just to complete the generator)
			// eslint-disable-next-line @typescript-eslint/no-unused-vars
			for await (const _chunk of stream) {
				// Do nothing - we're just consuming the stream

			assert.deepStrictEqual(requestBody, {
				model: modelId,
				messages: [{ role: "user", content: "Hello" }],
				temperature: 0,
				stream: true,

		it.each(availableModels)("should use XmlMatcher for processing responses with %s model", async (modelId) => {
			// Update handler to use the current model
			handler = new OllamaHandler({
				ollamaBaseUrl: "http://localhost:10000",
				ollamaModelId: modelId,

			// Mock implementation
			;(convertToOllamaHistory as sinon.SinonStub).returns([{ role: "user", content: "Hello" }])

			// Create neutral history
			const neutralHistory: NeutralConversationHistory = [
				{ role: "user", content: [{ type: "text", text: "Hello" }] },

			// Call createMessage
			const stream = handler.createMessage("You are helpful.", neutralHistory)

			// Collect stream chunks
			const chunks: ApiStreamChunk[] = []
			for await (const chunk of stream) {
				chunks.push(chunk)

			// Verify XmlMatcher was created with the correct tag
			assert.ok(XmlMatcher.calledWith("think", sinon.match.func))

			// Verify XmlMatcher.update was called for each chunk
			const xmlMatcherInstance = (XmlMatcher as sinon.SinonStub).mock.results[0].value as {
				update: sinon.SinonStub
				final: sinon.SinonStub

			assert.ok(xmlMatcherInstance.update.calledWith("Hello"))
			assert.ok(xmlMatcherInstance.update.calledWith(" world"))

			// Verify XmlMatcher.final was called
			assert.ok(xmlMatcherInstance.final.called)

	suite("countTokens", () => {
		it.each(availableModels)("should use convertToOllamaContentBlocks with %s model", async (modelId) => {
			// Update handler to use the current model
			handler = new OllamaHandler({
				ollamaBaseUrl: "http://localhost:10000",
				ollamaModelId: modelId,

			// Mock implementation
			;(convertToOllamaContentBlocks as sinon.SinonStub).returns("Hello world")

			// Mock the base provider's countTokens method
			const mockSuperCountTokens = sinon.spy(Object.getPrototypeOf(OllamaHandler.prototype), "countTokens")
			mockSuperCountTokens.resolves(2) // 2 tokens for "Hello world"

			// Create neutral content
			const neutralContent: NeutralMessageContent = [{ type: "text", text: "Hello world" }]

			// Call countTokens
			const tokenCount = await handler.countTokens(neutralContent)

			// Verify transform function was called
			assert.ok(convertToOllamaContentBlocks.calledWith(neutralContent))

			// Verify base provider's countTokens was called with the converted content
			assert.ok(mockSuperCountTokens.calledWith([{ type: "text", text: "Hello world" }]))

			// Verify token count
			assert.strictEqual(tokenCount, 2)

			// Clean up
			mockSuperCountTokens.restore()

		it.each(availableModels)("should handle errors and use fallback with %s model", async (modelId) => {
			// Update handler to use the current model
			handler = new OllamaHandler({
				ollamaBaseUrl: "http://localhost:10000",
				ollamaModelId: modelId,

			// Mock implementation that throws an error
			;(convertToOllamaContentBlocks as sinon.SinonStub).callsFake(() => {
				throw new Error("Conversion error")

			// Mock console.warn
			const originalWarn = console.warn
			console.warn = sinon.stub()

			// Mock the base provider's countTokens method
			const mockSuperCountTokens = sinon.spy(Object.getPrototypeOf(OllamaHandler.prototype), "countTokens")
			mockSuperCountTokens.resolves(2) // 2 tokens for fallback

			// Create neutral content
			const neutralContent: NeutralMessageContent = [{ type: "text", text: "Hello world" }]

			// Call countTokens
			const tokenCount = await handler.countTokens(neutralContent)

			// Verify transform function was called
			assert.ok(convertToOllamaContentBlocks.calledWith(neutralContent))

			// Verify console.warn was called
			assert.ok(console.warn.called)

			// Verify base provider's countTokens was called with the original content
			assert.ok(mockSuperCountTokens.calledWith(neutralContent))

			// Verify token count
			assert.strictEqual(tokenCount, 2)

			// Clean up
			console.warn = originalWarn
			mockSuperCountTokens.restore()

	suite("completePrompt", () => {
		it.each(availableModels)(
			"should convert prompt to neutral format and use convertToOllamaHistory with %s model",
			async (modelId) => {
				// Update handler to use the current model
				handler = new OllamaHandler({
					ollamaBaseUrl: "http://localhost:10000",
					ollamaModelId: modelId,

				// Mock implementation
				;(convertToOllamaHistory as sinon.SinonStub).returns([{ role: "user", content: "Hello" }])

				let requestBody: Record<string, unknown> | undefined
				nock("http://localhost:10000")
					.post("/v1/chat/completions")
					.reply(function (_uri: string, body: unknown) {
						requestBody = body as Record<string, unknown>
						return [200, { choices: [{ message: { content: "Hello world" } }] }]

				// Call completePrompt
				const result = await handler.completePrompt("Hello")

				// Verify transform function was called with the correct neutral history
				assert.ok(convertToOllamaHistory.calledWith([
					{ role: "user", content: [{ type: "text", text: "Hello" }] },
				]))

				assert.deepStrictEqual(requestBody, {
					model: modelId,
					messages: [{ role: "user", content: "Hello" }],
					temperature: 0,
					stream: false,

				// Verify result
				assert.strictEqual(result, "Hello world")
			},

		it.each(availableModels)("should handle errors with %s model", async (modelId) => {
			// Update handler to use the current model
			handler = new OllamaHandler({
				ollamaBaseUrl: "http://localhost:10000",
				ollamaModelId: modelId,

			// Mock implementation
			;(convertToOllamaHistory as sinon.SinonStub).returns([{ role: "user", content: "Hello" }])

			nock("http://localhost:10000")
				.post("/v1/chat/completions")
				.reply(500, { error: { message: "API error" } })

			// Call completePrompt and expect it to throw
			await expect(handler.completePrompt("Hello")).rejects.toThrow("Ollama completion error: API error")
