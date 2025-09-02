import { VertexHandler } from "../vertex"
import { ApiHandlerOptions } from "../../../shared/api"
import { NeutralConversationHistory, NeutralMessageContent } from "../../../shared/neutral-history"
import * as neutralVertexFormat from "../../transform/neutral-vertex-format"

// Mock the Vertex AI SDK
// TODO: Mock setup needs manual migration
import * as assert from 'assert'
import * as sinon from 'sinon'
// TODO: Fix mock - needs proxyquire
/*
=> {
	return {
		VertexAI: sinon.stub().callsFake(() => ({
			getGenerativeModel: sinon.stub().callsFake(() => ({
				generateContentStream: sinon.stub().callsFake(() => {
					return {
						stream: {
							[Symbol.asyncIterator]: function* () {
								yield {
									candidates: [
										{
											content: {
												parts: [{ text: "Test response" }],
											},
										},
									],

							},
						},
						response: {
							usageMetadata: {
								promptTokenCount: 10,
								candidatesTokenCount: 5,
							},
						},

				}),
				generateContent: sinon.stub().resolves({
					response: {
						candidates: [
							{
								content: {
									parts: [{ text: "Test completion" }],
								},
							},
						],
					},
				}),
			})),
		})),

})*/

// Mock the neutral-vertex-format module
// TODO: Mock setup needs manual migration
// TODO: Mock needs manual migration
// TODO: Implement proper mock with proxyquire

suite("VertexHandler", () => {
	suite("Claude model", () => {
		const options: ApiHandlerOptions = {
			vertexProjectId: "test-project",
			vertexRegion: "us-central1",
			apiModelId: "claude-3-sonnet@20240229",

		let handler: VertexHandler

		setup(() => {
			sinon.restore()
			handler = new VertexHandler(options)

		suite("createMessage", () => {
			test("should convert neutral history to Vertex Claude format and stream response", async () => {
				const systemPrompt = "You are a helpful assistant"
				const messages: NeutralConversationHistory = [
					{
						role: "user",
						content: [{ type: "text", text: "Hello" }],
					},

				const stream = handler.createMessage(systemPrompt, messages)
				const chunks = []

				for await (const chunk of stream) {
					chunks.push(chunk)

				assert.ok(neutralVertexFormat.convertToVertexClaudeHistory.calledWith(messages))
				assert.ok(chunks.length > 0)
				assert.deepStrictEqual(chunks[0], {
					type: "usage",
					inputTokens: 10,
					outputTokens: 0,

				assert.deepStrictEqual(chunks[1], { type: "text", text: "Test response" })

		suite("countTokens", () => {
			test("should use the base provider's implementation", async () => {
				// Mock the base provider's countTokens method
				const baseCountTokens = jest
					.spyOn(Object.getPrototypeOf(Object.getPrototypeOf(handler)), "countTokens")
					.resolves(15)

				const content: NeutralMessageContent = [{ type: "text", text: "Hello" }]
				const result = await handler.countTokens(content)

				assert.ok(baseCountTokens.calledWith(content))
				assert.strictEqual(result, 15)

	suite("Gemini model", () => {
		const options: ApiHandlerOptions = {
			vertexProjectId: "test-project",
			vertexRegion: "us-central1",
			apiModelId: "gemini-1.5-pro",

		let handler: VertexHandler

		setup(() => {
			sinon.restore()
			handler = new VertexHandler(options)

		suite("createMessage", () => {
			test("should convert neutral history to Vertex Gemini format and stream response", async () => {
				const systemPrompt = "You are a helpful assistant"
				const messages: NeutralConversationHistory = [
					{
						role: "user",
						content: [{ type: "text", text: "Hello" }],
					},

				const stream = handler.createMessage(systemPrompt, messages)
				const chunks = []

				for await (const chunk of stream) {
					chunks.push(chunk)

				assert.ok(neutralVertexFormat.convertToVertexGeminiHistory.calledWith(messages))
				assert.ok(chunks.length > 0)
				assert.deepStrictEqual(chunks[0], { type: "text", text: "Test response" })
				assert.deepStrictEqual(chunks[1], {
					type: "usage",
					inputTokens: 10,
					outputTokens: 5,

		suite("completePrompt", () => {
			test("should complete a prompt and return the response", async () => {
				const result = await handler.completePrompt("Test prompt")
				assert.strictEqual(result, "Test completion")
