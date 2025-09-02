// Mock AWS SDK credential providers
// TODO: Mock setup needs manual migration
import * as assert from 'assert'
import * as sinon from 'sinon'
// TODO: Mock needs manual migration
// TODO: Implement proper mock with proxyquire

import { AwsBedrockHandler, StreamEvent } from "../bedrock"
import { ApiHandlerOptions } from "../../../shared/api"
import { BedrockRuntimeClient } from "@aws-sdk/client-bedrock-runtime"

suite("AwsBedrockHandler with invokedModelId", () => {
	let mockSend: sinon.SinonSpy

	setup(() => {
		// Mock the BedrockRuntimeClient.prototype.send method
		// eslint-disable-next-line @typescript-eslint/no-unsafe-argument
		mockSend = sinon.spy(BedrockRuntimeClient.prototype, "send").callsFake((() => {
			return Promise.resolve({
				stream: createMockStream([]),

		}) as any) // eslint-disable-line @typescript-eslint/no-explicit-any

	teardown(() => {
		mockSend.restore()

	// Helper function to create a mock async iterable stream
	function createMockStream(events: StreamEvent[]) {
		return {
			[Symbol.asyncIterator]: function* () {
				for (const event of events) {
					yield event

				// Always yield a metadata event at the end
				yield {
					metadata: {
						usage: {
							inputTokens: 100,
							outputTokens: 200,
						},
					},

			},

	test("should update costModelConfig when invokedModelId is present in the stream", async () => {
		// Create a handler with a custom ARN
		const mockOptions: ApiHandlerOptions = {
			//	apiModelId: "anthropic.claude-3-5-sonnet-20241022-v2:0",
			awsAccessKey: "test-access-key",
			awsSecretKey: "test-secret-key",
			awsRegion: "us-east-1",
			awsCustomArn: "arn:aws:bedrock:us-west-2:699475926481:default-prompt-router/anthropic.claude:1",

		const handler = new AwsBedrockHandler(mockOptions)

		// Create a spy on the getModel method before mocking it
		const getModelSpy = sinon.spy(handler, "getModelByName")

		// Mock the stream to include an event with invokedModelId and usage metadata
		mockSend.onFirstCall().callsFake(() => {
			return {
				stream: createMockStream([
					// First event with invokedModelId and usage metadata
					{
						trace: {
							promptRouter: {
								invokedModelId:
									"arn:aws:bedrock:us-west-2:699475926481:inference-profile/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
								usage: {
									inputTokens: 150,
									outputTokens: 250,
								},
							},
						},
						// Some content events
					},
					{
						contentBlockStart: {
							start: {
								text: "Hello",
							},
							contentBlockIndex: 0,
						},
					},
					{
						contentBlockDelta: {
							delta: {
								text: ", world!",
							},
							contentBlockIndex: 0,
						},
					},
				]),

		// Create a message generator
		const messageGenerator = handler.createMessage("system prompt", [{ role: "user", content: "user message" }])

		// Collect all yielded events to verify usage events
		const events = []
		for await (const event of messageGenerator) {
			events.push(event)

		// Verify that getModel was called with the correct model name
		assert.ok(getModelSpy.calledWith("anthropic.claude-3-5-sonnet-20240620-v1:0"))

		// Verify that getModel returns the updated model info
		const costModel = handler.getModel()
		assert.strictEqual(costModel.id, "anthropic.claude-3-5-sonnet-20240620-v1:0")
		assert.strictEqual(costModel.info.inputPrice, 3)

		// Verify that a usage event was emitted after updating the costModelConfig
		const usageEvents = events.filter((event) => event.type === "usage")
		assert.ok(usageEvents.length >= 1)

		// The last usage event should have the token counts from the metadata
		const lastUsageEvent = usageEvents[usageEvents.length - 1]
		assert.deepStrictEqual(lastUsageEvent, {
			type: "usage",
			inputTokens: 100,
			outputTokens: 200,

	test("should not update costModelConfig when invokedModelId is not present", async () => {
		// Create a handler with default settings
		const mockOptions: ApiHandlerOptions = {
			apiModelId: "anthropic.claude-3-5-sonnet-20241022-v2:0",
			awsAccessKey: "test-access-key",
			awsSecretKey: "test-secret-key",
			awsRegion: "us-east-1",

		const handler = new AwsBedrockHandler(mockOptions)

		// Mock the stream without an invokedModelId event
		mockSend.onFirstCall().callsFake(() => {
			return {
				stream: createMockStream([
					// Some content events but no invokedModelId
					{
						contentBlockStart: {
							start: {
								text: "Hello",
							},
							contentBlockIndex: 0,
						},
					},
					{
						contentBlockDelta: {
							delta: {
								text: ", world!",
							},
							contentBlockIndex: 0,
						},
					},
				]),

		// Mock getModel to return expected values
		const getModelSpy = sinon.spy(handler, "getModel").returns({
			id: "anthropic.claude-3-5-sonnet-20241022-v2:0",
			info: {
				maxTokens: 4096,
				contextWindow: 128_000,
				supportsPromptCache: false,
				supportsImages: true,
			},

		// Create a message generator
		const messageGenerator = handler.createMessage("system prompt", [{ role: "user", content: "user message" }])

		// Consume the generator
		// eslint-disable-next-line @typescript-eslint/no-unused-vars
		for await (const _ of messageGenerator) {
			// Just consume the messages

		// Verify that getModel returns the original model info
		const costModel = handler.getModel()
		assert.strictEqual(costModel.id, "anthropic.claude-3-5-sonnet-20241022-v2:0")

		// Verify getModel was not called with a model name parameter
		expect(getModelSpy).not.toHaveBeenCalledWith(sinon.match.string)

	test("should handle invalid invokedModelId format gracefully", async () => {
		// Create a handler with default settings
		const mockOptions: ApiHandlerOptions = {
			apiModelId: "anthropic.claude-3-5-sonnet-20241022-v2:0",
			awsAccessKey: "test-access-key",
			awsSecretKey: "test-secret-key",
			awsRegion: "us-east-1",

		const handler = new AwsBedrockHandler(mockOptions)

		// Mock the stream with an invalid invokedModelId
		mockSend.onFirstCall().callsFake(() => {
			return {
				stream: createMockStream([
					// Event with invalid invokedModelId format
					{
						trace: {
							promptRouter: {
								invokedModelId: "invalid-format-not-an-arn",
							},
						},
					},
					// Some content events
					{
						contentBlockStart: {
							start: {
								text: "Hello",
							},
							contentBlockIndex: 0,
						},
					},
				]),

		// Mock getModel to return expected values
		sinon.spy(handler, "getModel").returns({
			id: "anthropic.claude-3-5-sonnet-20241022-v2:0",
			info: {
				maxTokens: 4096,
				contextWindow: 128_000,
				supportsPromptCache: false,
				supportsImages: true,
			},

		// Create a message generator
		const messageGenerator = handler.createMessage("system prompt", [{ role: "user", content: "user message" }])

		// Consume the generator
		// eslint-disable-next-line @typescript-eslint/no-unused-vars
		for await (const _ of messageGenerator) {
			// Just consume the messages

		// Verify that getModel returns the original model info
		const costModel = handler.getModel()
		assert.strictEqual(costModel.id, "anthropic.claude-3-5-sonnet-20241022-v2:0")

	test("should handle errors during invokedModelId processing", async () => {
		// Create a handler with default settings
		const mockOptions: ApiHandlerOptions = {
			apiModelId: "anthropic.claude-3-5-sonnet-20241022-v2:0",
			awsAccessKey: "test-access-key",
			awsSecretKey: "test-secret-key",
			awsRegion: "us-east-1",

		const handler = new AwsBedrockHandler(mockOptions)

		// Mock the stream with a valid invokedModelId
		mockSend.onFirstCall().callsFake(() => {
			return {
				stream: createMockStream([
					// Event with valid invokedModelId
					{
						trace: {
							promptRouter: {
								invokedModelId:
									"arn:aws:bedrock:us-east-1:123456789:foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
							},
						},
					},
				]),

		// Mock getModel to throw an error when called with the model name
		sinon.spy(handler, "getModel").callsFake((modelName?: string) => {
			if (modelName === "anthropic.claude-3-sonnet-20240229-v1:0") {
				throw new Error("Test error during model lookup")

			// Default return value for initial call
			return {
				id: "anthropic.claude-3-5-sonnet-20241022-v2:0",
				info: {
					maxTokens: 4096,
					contextWindow: 128_000,
					supportsPromptCache: false,
					supportsImages: true,
				},

		// Create a message generator
		const messageGenerator = handler.createMessage("system prompt", [{ role: "user", content: "user message" }])

		// Consume the generator
		// eslint-disable-next-line @typescript-eslint/no-unused-vars
		for await (const _ of messageGenerator) {
			// Just consume the messages

		// Verify that getModel returns the original model info
		const costModel = handler.getModel()
		assert.strictEqual(costModel.id, "anthropic.claude-3-5-sonnet-20241022-v2:0")
