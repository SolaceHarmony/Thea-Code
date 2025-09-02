import { AwsBedrockHandler } from "../bedrock"
import { ApiHandlerOptions } from "../../../shared/api"
import { NeutralConversationHistory } from "../../../shared/neutral-history"
import * as neutralBedrockFormat from "../../transform/neutral-bedrock-format"
import type { ApiStreamChunk } from "../../transform/stream" // Import ApiStreamChunk

// Explicitly import the mocked module to access its exports within the mock
import * as BedrockRuntimeClientMock from "@aws-sdk/client-bedrock-runtime"

// Mock the AWS SDK

// TODO: Mock setup needs manual migration
import * as assert from 'assert'
import * as sinon from 'sinon'
// TODO: Fix mock - needs proxyquire
/*
=> {
	const mockConverseStreamCommand = sinon.stub()
	const mockConverseCommand = sinon.stub()

	const mockSend = sinon.stub().callsFake((command) => {
		// Removed async
		// Check if the command is an instance of the mocked constructors from the imported mock
		if (command instanceof BedrockRuntimeClientMock.ConverseStreamCommand) {
			return Promise.resolve({
				// Added Promise.resolve
				stream: {
					[Symbol.asyncIterator]: function* () {
						// Removed async
						// Metadata event (object)
						yield {
							metadata: {
								usage: {
									inputTokens: 10,
									outputTokens: 5,
								},
							},

						// Content block start (object)
						yield {
							contentBlockStart: {
								start: {
									text: "Test response",
								},
							},

						// Message stop (object)
						yield {
							messageStop: {},

					},
				},

		} else if (command instanceof BedrockRuntimeClientMock.ConverseCommand) {
			return Promise.resolve({
				// Added Promise.resolve
				output: new TextEncoder().encode(JSON.stringify({ content: "Test completion" })),

		// Default return for any other command type
		return Promise.resolve({}) // Added Promise.resolve

	return {
		BedrockRuntimeClient: sinon.stub().callsFake(() => ({
			send: mockSend,
			config: {
				region: "us-east-1",
			},
		})),
		ConverseStreamCommand: mockConverseStreamCommand,
		ConverseCommand: mockConverseCommand,

})*/

// Mock the neutral-bedrock-format module
// TODO: Mock setup needs manual migration
// TODO: Mock needs manual migration
// TODO: Implement proper mock with proxyquire

suite("AwsBedrockHandler", () => {
	const options: ApiHandlerOptions = {
		awsAccessKey: "test-access-key",
		awsSecretKey: "test-secret-key",
		apiModelId: "anthropic.claude-v2",

	let handler: AwsBedrockHandler

	setup(() => {
		sinon.restore()
		handler = new AwsBedrockHandler(options)

	suite("createMessage", () => {
		test("should convert neutral history to Bedrock format and stream response", async () => {
			const systemPrompt = "You are a helpful assistant"
			const messages: NeutralConversationHistory = [
				{
					role: "user",
					content: [{ type: "text", text: "Hello" }],
				},

			const stream = handler.createMessage(systemPrompt, messages)
			const chunks: ApiStreamChunk[] = []

			for await (const chunk of stream) {
				chunks.push(chunk)

			assert.ok(neutralBedrockFormat.convertToBedrockConverseMessages.calledWith(messages))
			assert.strictEqual(chunks.length, 2)
			assert.deepStrictEqual(chunks[0], {
				type: "usage",
				inputTokens: 10,
				outputTokens: 5,

			assert.deepStrictEqual(chunks[1], { type: "text", text: "Test response" })

	suite("countTokens", () => {
		test("should use the base provider's implementation", async () => {
			// Mock the base provider's countTokens method
			const baseCountTokens = jest
				.spyOn(Object.getPrototypeOf(Object.getPrototypeOf(handler)), "countTokens")
				.resolves(15)

			const content: NeutralConversationHistory[0]["content"] = [{ type: "text", text: "Hello" }]
			const result = await handler.countTokens(content)

			assert.ok(baseCountTokens.calledWith(content))
			assert.strictEqual(result, 15)

	suite("completePrompt", () => {
		test("should complete a prompt and return the response", async () => {
			const result = await handler.completePrompt("Test prompt")
			assert.strictEqual(result, "Test completion")
