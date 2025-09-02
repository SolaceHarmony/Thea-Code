import { MistralHandler } from "../mistral"
import { ApiHandlerOptions } from "../../../shared/api"
import { NeutralConversationHistory, NeutralMessageContent } from "../../../shared/neutral-history"
import * as neutralMistralFormat from "../../transform/neutral-mistral-format"

// Mock the Mistral client
// TODO: Mock setup needs manual migration
import * as assert from 'assert'
import * as sinon from 'sinon'
// TODO: Fix mock - needs proxyquire
/*
=> {
	return {
		Mistral: sinon.stub().callsFake(() => ({
			chat: {
				stream: sinon.stub().callsFake(() => {
					return {
						[Symbol.asyncIterator]: function* () {
							yield {
								data: {
									choices: [
										{
											delta: {
												content: "Test response",
											},
										},
									],
									usage: {
										promptTokens: 10,
										completionTokens: 5,
									},
								},

						},

				}),
				complete: sinon.stub().resolves({
					choices: [
						{
							message: {
								content: "Test completion",
							},
						},
					],
				}),
			},
		})),

})*/

// Mock the neutral-mistral-format module
// TODO: Mock setup needs manual migration
// TODO: Mock needs manual migration
// TODO: Implement proper mock with proxyquire

suite("MistralHandler", () => {
	const options: ApiHandlerOptions = {
		mistralApiKey: "test-key",
		apiModelId: "mistral-medium",

	let handler: MistralHandler

	setup(() => {
		sinon.restore()
		handler = new MistralHandler(options)

	suite("createMessage", () => {
		test("should convert neutral history to Mistral format and stream response", async () => {
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

			assert.ok(neutralMistralFormat.convertToMistralMessages.calledWith(messages))
			assert.strictEqual(chunks.length, 2)
			assert.deepStrictEqual(chunks[0], { type: "text", text: "Test response" })
			assert.deepStrictEqual(chunks[1], {
				type: "usage",
				inputTokens: 10,
				outputTokens: 5,

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

	suite("completePrompt", () => {
		test("should complete a prompt and return the response", async () => {
			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "Test completion")
